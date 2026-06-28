"""
PgVectorStore — drop-in replacement for MilvusVectorStore (src/database/milvus_db.py).

Implements the same public interface:
  manage_collection() (context manager),
  ingest_documents(), get_uploaded_filenames_for_session(),
  enhanced_search(), synthesize_answer_with_llm(),
  search_by_session(), vector_search(), get_collection_stats()

PG connection params are read from PG_* env vars inside __init__
so MILVUS_* credentials never reach this class.
"""
import asyncio
import atexit
import json
import logging
import os
import hashlib
import zlib
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import psycopg2
import psycopg2.extras
import psycopg2.pool
import redis
from google import genai
from google.genai import types
from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.llm.litellm_client import LitellmClient
from src.utils.reasoning_extractor import REASONING_SECTION_PROMPT

logger = logging.getLogger(__name__)

EMBEDDING_DIM   = int(os.getenv("EMBEDDING_DIM", "768"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "200"))
DEFAULT_TOP_K   = int(os.getenv("DEFAULT_TOP_K", "5"))

from src.database.secrets_loader import load_pgvector_secrets
load_pgvector_secrets()

# ── Embedding concurrency config ───────────────────────────────────────────────
# Mirrors the approach used in milvus_db.py: a shared thread pool offloads
# blocking Gemini SDK calls so the asyncio event loop is never stalled.
# wait=False + cancel_futures=True prevents Lambda from hanging on shutdown.
_EMBED_EXECUTOR    = ThreadPoolExecutor(max_workers=8, thread_name_prefix="pg_embed_worker")
atexit.register(lambda: _EMBED_EXECUTOR.shutdown(wait=False, cancel_futures=True))
_EMBED_BATCH_SIZE  = int(os.getenv("EMBED_BATCH_SIZE", "30"))   # texts per Gemini API call
_EMBED_CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "8"))   # max concurrent API calls
_EMBED_TIMEOUT     = int(os.getenv("EMBED_TIMEOUT", "60"))      # seconds per batch call

# ── Redis config ───────────────────────────────────────────────────────────────
REDIS_HOST     = os.getenv("REDIS_HOST_PGVECTOR", "localhost")
REDIS_PORT     = int(os.getenv("REDIS_PORT_PGVECTOR", "6379"))
REDIS_USERNAME = os.getenv("REDIS_USERNAME_PGVECTOR", "default")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD_PGVECTOR", None)
REDIS_SSL      = os.getenv("REDIS_SSL", "false").strip().lower() in ("true", "1", "yes")
TTL_EMBEDDING  = int(os.getenv("CACHE_TTL_EMBEDDING", str(86400 * 7)))  # 7 days
TTL_SEARCH     = int(os.getenv("CACHE_TTL_SEARCH", "3600"))              # 1 hour

# # Guard so _init_table DDL only runs once per Lambda process lifetime.
# # On warm invocations the table/index already exist — no need to re-check.
# _TABLE_INITIALIZED: bool = False


class PgVectorStore:
    """PostgreSQL + pgvector store for IDP.

    Drop-in replacement for MilvusVectorStore — same public interface.
    PG connection params are read from PG_* env vars (not MILVUS_*).
    """

    def __init__(self):
        collection_name = os.getenv("PG_TABLE", os.getenv("MILVUS_COLLECTION", "idp_collection"))
        self.collection_name = collection_name
        self._embedding_dim  = EMBEDDING_DIM

        self._genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.litellm_client = LitellmClient()
        self.text_splitter  = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

        # PG connection — reads only PG_* env vars
        self.pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=os.getenv("PG_HOST", "localhost"),
            port=int(os.getenv("PG_PORT", "5432")),
            user=os.getenv("PG_USER", ""),
            password=os.getenv("PG_PASSWORD", ""),
            database=os.getenv("PG_DATABASE", ""),
        )
        self._connect_redis()
        self._init_table()  # no-op on warm Lambda containers
        logger.info("[PgVectorStore] Ready: table=%s", collection_name)

    # =========================================================================
    # Redis helpers
    # =========================================================================

    def _connect_redis(self):
        try:
            self.redis = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                username=REDIS_USERNAME,
                password=REDIS_PASSWORD,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
                ssl=REDIS_SSL,
            )
            self.redis.ping()
            self.redis_available = True
            logger.info("[PgVectorStore][REDIS] Connected to Redis at %s:%d", REDIS_HOST, REDIS_PORT)
        except Exception as e:
            self.redis_available = False
            logger.warning("[PgVectorStore][REDIS] Could not connect: %s. Caching disabled.", e)

    def _redis_get(self, key: str):
        if not self.redis_available:
            return None
        try:
            return self.redis.get(key)
        except Exception:
            return None

    def _redis_setex(self, key: str, ttl: int, value):
        if not self.redis_available:
            return
        try:
            self.redis.setex(key, ttl, value)
        except Exception:
            pass

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _get_conn(self):
        return self.pool.getconn()

    def _put_conn(self, conn):
        self.pool.putconn(conn)

    def _init_table(self):
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.collection_name} (
                        chunk_id   VARCHAR(200) PRIMARY KEY,
                        session_id VARCHAR(255),
                        team_id    VARCHAR(255),
                        text       TEXT,
                        metadata   JSONB,
                        embedding  vector({self._embedding_dim}),
                        created_at VARCHAR(100)
                    );
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.collection_name}_hnsw_idx
                    ON {self.collection_name}
                    USING hnsw (embedding vector_ip_ops)
                    WITH (m = 16, ef_construction = 200);
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.collection_name}_session_idx
                    ON {self.collection_name} (session_id);
                """)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error("[PgVectorStore] Failed to initialize table: %s", e)
            raise
        finally:
            self._put_conn(conn)

    def generate_embedding(self, text: str) -> List[float]:
        """Embed a single text. Used during search (one query at a time).

        Retries on 429 with the same backoff as batch embedding so HyDE and
        multi-query searches don't silently fail when quota is temporarily
        exhausted right after a preceding LLM call.
        """
        import time
        _waits = [2, 4, 8, 16]
        last_exc = None
        for attempt, wait in enumerate([0] + _waits):
            if wait:
                logger.warning(
                    "[PgVectorStore] generate_embedding 429 (attempt %d/%d) — retrying in %ds",
                    attempt, len(_waits) + 1, wait,
                )
                time.sleep(wait)
            try:
                result = self._genai_client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type="SEMANTIC_SIMILARITY",
                        output_dimensionality=self._embedding_dim,
                    ),
                )
                return list(result.embeddings[0].values)
            except Exception as exc:
                exc_str = str(exc)
                if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                    last_exc = exc
                else:
                    raise
        raise last_exc

    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embed many texts concurrently using batched Gemini API calls.

        Sends up to _EMBED_BATCH_SIZE texts per API call and dispatches all
        batches concurrently (bounded by _EMBED_CONCURRENCY).  Total wall-clock
        time ≈ one single-batch latency instead of N × single-batch latency.

        Safe to call from both sync (ingest_documents) and async contexts.
        """
        if not texts:
            return []
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Called from inside a running async event loop — offload to a
                # fresh thread that can create its own loop (e.g. PDF streaming).
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(
                        asyncio.run, self._async_batch_embed(texts)
                    ).result()
            return loop.run_until_complete(self._async_batch_embed(texts))
        except RuntimeError:
            return asyncio.run(self._async_batch_embed(texts))

    async def _async_batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Async core of batch_generate_embeddings — do not call directly."""
        embed_cfg = types.EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY",
            output_dimensionality=self._embedding_dim,
        )
        batches = [
            (start, texts[start: start + _EMBED_BATCH_SIZE])
            for start in range(0, len(texts), _EMBED_BATCH_SIZE)
        ]
        sem  = asyncio.Semaphore(_EMBED_CONCURRENCY)
        loop = asyncio.get_event_loop()

        async def _one_batch(start: int, batch: List[str]):
            async with sem:
                # Retry up to 6 times with exponential backoff for 429 quota errors.
                # OCR image analysis saturates the shared Gemini API quota, so the
                # embedding call must wait for it to recover — up to ~60s in bad cases.
                # Sequence: 2s, 4s, 8s, 16s, 30s, 60s → max total wait ~120s
                last_exc = None
                _QUOTA_MAX_ATTEMPTS = 6
                _QUOTA_WAITS = [2, 4, 8, 16, 30, 60]
                for attempt in range(_QUOTA_MAX_ATTEMPTS):
                    try:
                        result = await asyncio.wait_for(
                            loop.run_in_executor(
                                _EMBED_EXECUTOR,
                                lambda b=batch: self._genai_client.models.embed_content(
                                    model=EMBEDDING_MODEL,
                                    contents=b,
                                    config=embed_cfg,
                                ),
                            ),
                            timeout=_EMBED_TIMEOUT,
                        )
                        return start, [list(e.values) for e in result.embeddings]
                    except Exception as exc:
                        last_exc = exc
                        exc_str = str(exc)
                        if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                            wait = _QUOTA_WAITS[min(attempt, len(_QUOTA_WAITS) - 1)]
                            logger.warning(
                                "[PgVectorStore] Embedding 429 on batch at %d "
                                "(attempt %d/%d) — retrying in %ds",
                                start, attempt + 1, _QUOTA_MAX_ATTEMPTS, wait,
                            )
                            await asyncio.sleep(wait)
                        else:
                            raise
                # All retries exhausted due to quota — tag the exception so callers
                # can distinguish a quota failure from a generic embed error.
                raise RuntimeError(
                    f"QUOTA_EXHAUSTED: Gemini embedding quota not recovered after "
                    f"{_QUOTA_MAX_ATTEMPTS} attempts. Original error: {last_exc}"
                )

        results = await asyncio.gather(*[_one_batch(s, b) for s, b in batches])

        # Re-assemble in original order (gather preserves insertion order)
        out: List[List[float]] = [None] * len(texts)  # type: ignore[list-item]
        for start, embs in results:
            for i, emb in enumerate(embs):
                out[start + i] = emb

        logger.info(
            "[PgVectorStore] Concurrent embedding complete: %d vectors in %d batches",
            len(out), len(batches),
        )
        return out

    # =========================================================================
    # Collection lifecycle (context manager)
    # =========================================================================

    @contextmanager
    def manage_collection(self):
        """No-op context manager — pgvector tables are always ready."""
        logger.info("[PgVectorStore] manage_collection() entered (no-op for pgvector)")
        try:
            yield
        finally:
            logger.info("[PgVectorStore] manage_collection() exited")

    # =========================================================================
    # Public interface (identical to MilvusVectorStore)
    # =========================================================================

    def get_uploaded_filenames_for_session(self, session_id: str) -> Set[str]:
        """Return the set of filenames already uploaded for this session."""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"SELECT metadata FROM {self.collection_name} WHERE session_id = %s",
                    (session_id,),
                )
                rows = cur.fetchall()
        finally:
            self._put_conn(conn)

        filenames: Set[str] = set()
        for row in rows:
            meta = row["metadata"] or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    continue
            fname = meta.get("filename")
            if fname:
                filenames.add(fname)
        logger.info(
            "[PgVectorStore] %d files already uploaded in session %s", len(filenames), session_id
        )
        return filenames

    def ingest_documents(
        self,
        documents: Dict[str, List[LangchainDocument]],
        session_id: str,
        team_id: str,
        batch_size: Optional[int] = None,
        generate_ids: Optional[bool] = None,
        attachment_url_map: Optional[Dict] = None,
        document_source_map: Optional[List] = None,
    ) -> Dict[str, Any]:
        """Embed and insert documents into pgvector."""
        if batch_size is None:
            batch_size = 100
        if generate_ids is None:
            generate_ids = True

        total_docs   = len(documents)
        total_chunks = 0
        failed_docs  = []
        timestamp    = datetime.now(timezone.utc).isoformat()

        conn = self._get_conn()
        try:
            for filename, chunks in documents.items():
                if not chunks:
                    continue
                file_type = filename.rsplit(".", 1)[-1].lower() if "." in filename else "unknown"
                texts_to_embed = [
                    c.page_content[:65535]
                    for c in chunks
                    if isinstance(c, LangchainDocument)
                ]
                if not texts_to_embed:
                    continue

                try:
                    embeddings = self.batch_generate_embeddings(texts_to_embed)
                except Exception as e:
                    err_str = str(e)
                    if "QUOTA_EXHAUSTED" in err_str:
                        # Re-raise quota errors so the upload endpoint returns a real
                        # failure response — prevents silent 200 with 0 docs uploaded
                        # which would cause a confusing "no documents found" on query.
                        logger.error(
                            "[PgVectorStore] Quota exhausted for %s — propagating to caller",
                            filename,
                        )
                        raise
                    logger.error("[PgVectorStore] Embedding failed for %s: %s", filename, e)
                    failed_docs.append(filename)
                    continue

                rows = []
                for chunk_idx, (chunk, emb) in enumerate(
                    zip(
                        (c for c in chunks if isinstance(c, LangchainDocument)),
                        embeddings,
                    )
                ):
                    chunk_id = (
                        f"{session_id}_{uuid.uuid4().hex[:16]}_{chunk_idx}"
                        if generate_ids
                        else f"{session_id}_{filename}_{chunk_idx}"
                    )
                    text = chunk.page_content[:65535]
                    meta = json.dumps({
                        "filename": filename,
                        "file_type": file_type,
                        **chunk.metadata,
                    })
                    emb_str = "[" + ",".join(str(float(v)) for v in emb) + "]"
                    rows.append((chunk_id, session_id, team_id, text, meta, emb_str, timestamp))
                    total_chunks += 1

                    if len(rows) >= batch_size:
                        self._insert_rows(conn, rows)
                        rows = []

                if rows:
                    self._insert_rows(conn, rows)

                logger.info("[PgVectorStore] Ingested %s: %d chunks", filename, total_chunks)
        finally:
            self._put_conn(conn)

        return {
            "total_documents":       total_docs,
            "successful_documents":  total_docs - len(failed_docs),
            "failed_documents":      len(failed_docs),
            "total_chunks_inserted": total_chunks,
            "failed_doc_details":    failed_docs,
            "session_id":            session_id,
            "team_id":               team_id,
            "batch_size_used":       batch_size,
        }

    def _insert_rows(self, conn, rows: list):
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                f"""
                INSERT INTO {self.collection_name}
                    (chunk_id, session_id, team_id, text, metadata, embedding, created_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector, %s)
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                rows,
            )
        conn.commit()

    def vector_search(
        self,
        query: str,
        vector_field: str = "embedding",
        filter_expr: Optional[str] = None,
        top_k: Optional[int] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Vector similarity search."""
        if top_k is None:
            top_k = DEFAULT_TOP_K
        if output_fields is None:
            output_fields = ["text", "metadata", "session_id", "team_id", "created_at"]

        # Redis search cache
        filter_hash = hashlib.sha256((filter_expr or "").encode()).hexdigest()
        cache_key = f"search:{hashlib.sha256(query.encode()).hexdigest()}:{filter_hash}:{top_k}"
        cached = self._redis_get(cache_key)
        if cached:
            logger.info("[PgVectorStore][CACHE] vector_search cache HIT")
            return json.loads(zlib.decompress(cached))

        emb = self.generate_embedding(query)
        emb_str = "[" + ",".join(str(float(v)) for v in emb) + "]"

        where_parts = []
        if filter_expr and "session_id" in filter_expr:
            import re
            m = re.search(r'session_id\s*==\s*"([^"]+)"', filter_expr)
            if m:
                where_parts.append(f"session_id = '{m.group(1)}'")
        where = "WHERE " + " AND ".join(where_parts) if where_parts else ""

        sql = f"""
            SELECT chunk_id AS id, session_id, team_id, text, metadata, created_at,
                   (embedding <#> %s::vector) * -1 AS score
            FROM   {self.collection_name}
            {where}
            ORDER  BY embedding <#> %s::vector
            LIMIT  %s
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (emb_str, emb_str, top_k))
                rows = cur.fetchall()
        finally:
            self._put_conn(conn)

        results = [
            {
                "score": float(row["score"]),
                "id":    row["id"],
                **{f: row.get(f) for f in output_fields},
            }
            for row in rows
        ]
        self._redis_setex(cache_key, TTL_SEARCH, zlib.compress(json.dumps(results, default=str).encode()))
        return results

    def search_by_session(
        self,
        query: str,
        session_id: str,
        vector_field: str = "embedding",
        top_k: Optional[int] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search documents for a specific session."""
        filter_expr = f'session_id == "{session_id}"'
        return self.vector_search(
            query=query,
            vector_field=vector_field,
            filter_expr=filter_expr,
            top_k=top_k,
            output_fields=output_fields or ["text", "metadata", "session_id", "team_id", "created_at"],
        )

    async def enhanced_search(
        self,
        query: str,
        session_id: str,
        llm_params: dict,
        auth_token: str,
        use_hyde: bool = True,
        use_multi_query: bool = True,
        top_k: Optional[int] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic + HyDE + multi-query search — all LLM calls and searches run in parallel."""
        if top_k is None:
            top_k = DEFAULT_TOP_K
        if output_fields is None:
            output_fields = ["text", "metadata", "session_id", "team_id", "created_at"]

        loop = asyncio.get_event_loop()
        seen_ids: set = set()

        def _dedup(results: List[Dict]) -> List[Dict]:
            out = []
            for r in results:
                rid = r.get("id")
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    out.append(r)
            return out

        # ── Phase 1: initial search + both LLM calls in parallel ──────────────
        initial_search_fut = loop.run_in_executor(
            _EMBED_EXECUTOR,
            lambda: self.search_by_session(query, session_id, top_k=top_k, output_fields=output_fields),
        )

        async def _hyde_llm():
            try:
                hyde_prompt = f"Write a detailed answer to: {query}"
                return await self.litellm_client.generate_response(
                    llm_params, [{"role": "user", "content": hyde_prompt}], auth_token
                )
            except Exception as e:
                logger.warning("[PgVectorStore] HyDE LLM failed (non-fatal): %s", e)
                return None

        async def _multi_query_llm():
            try:
                var_prompt = f"Generate 3 query variations for: {query}\n(one per line)"
                return await self.litellm_client.generate_response(
                    llm_params, [{"role": "user", "content": var_prompt}], auth_token
                )
            except Exception as e:
                logger.warning("[PgVectorStore] Multi-query LLM failed (non-fatal): %s", e)
                return None

        phase1_tasks = [initial_search_fut]
        hyde_task        = asyncio.create_task(_hyde_llm())        if use_hyde        else None
        multi_query_task = asyncio.create_task(_multi_query_llm()) if use_multi_query else None
        if hyde_task:        phase1_tasks.append(hyde_task)
        if multi_query_task: phase1_tasks.append(multi_query_task)

        phase1 = await asyncio.gather(*phase1_tasks, return_exceptions=True)

        initial_results  = phase1[0] if not isinstance(phase1[0], Exception) else []
        _idx             = 1
        hypo_doc         = None
        variations_text  = None
        if hyde_task:
            hypo_doc = phase1[_idx] if not isinstance(phase1[_idx], Exception) else None
            _idx += 1
        if multi_query_task:
            variations_text = phase1[_idx] if not isinstance(phase1[_idx], Exception) else None

        all_results = _dedup(initial_results)

        # ── Phase 2: follow-up searches all in parallel ───────────────────────
        follow_up_futs = []
        if hypo_doc:
            _hd = hypo_doc
            follow_up_futs.append(loop.run_in_executor(
                _EMBED_EXECUTOR,
                lambda: self.search_by_session(_hd, session_id, top_k=top_k, output_fields=output_fields),
            ))
        if variations_text:
            for var in variations_text.strip().split("\n")[:3]:
                var = var.strip()
                if var:
                    _v = var
                    follow_up_futs.append(loop.run_in_executor(
                        _EMBED_EXECUTOR,
                        lambda v=_v: self.search_by_session(
                            v, session_id, top_k=top_k // 2 or 3, output_fields=output_fields
                        ),
                    ))

        if follow_up_futs:
            follow_results = await asyncio.gather(*follow_up_futs, return_exceptions=True)
            for res in follow_results:
                if not isinstance(res, Exception):
                    all_results += _dedup(res)

        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:top_k * 2]

    async def synthesize_answer_with_llm(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        llm_params: dict,
        auth_token: str,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """Synthesize a formatted answer from retrieved chunks via LiteLLM."""
        context_parts = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            text = chunk.get("text", "")
            context_parts.append(f"{text}\n")
        context = "\n---\n".join(context_parts)

        system_prompt = f"""You are an expert document analyst. Provide comprehensive, well-formatted markdown responses. {REASONING_SECTION_PROMPT}"""
        user_prompt   = f"""Context:\n{context}\n\nQuestion: {query}\n\nProvide a comprehensive answer using the context above."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        answer = await self.litellm_client.generate_response(llm_params, messages, auth_token)

        sources = []
        if include_sources:
            seen = set()
            for chunk in retrieved_chunks:
                meta   = chunk.get("metadata") or {}
                source = meta.get("source", meta.get("filename", "Unknown"))
                page   = meta.get("page", meta.get("page_number", "N/A"))
                key    = f"{source}_{page}"
                if key not in seen:
                    sources.append({"source": source, "page": page, "score": chunk.get("score", 0.0)})
                    seen.add(key)

        return {
            "answer":          answer,
            "sources":         sources,
            "num_chunks_used": len(retrieved_chunks),
            "query":           query,
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.collection_name}")
                count = cur.fetchone()[0]
        finally:
            self._put_conn(conn)
        return {
            "collection_name": self.collection_name,
            "num_entities":    count,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim":   self._embedding_dim,
        }
