"""
semantic_cache.py

Semantic (similarity-based) cache for LLM responses.

Stores LLM answers in Redis keyed by the embedding of the user query.
On a new query, embeds it and searches all cached embeddings (within the
same namespace) using cosine distance.  Returns the cached answer if the
closest match is within distance_threshold.

Session-aware design
--------------------
IDP documents are per-session, so a namespace (typically the session_id)
is passed to check() / store().  This keeps different sessions' caches
isolated — the same question asked in session A won't return session B's
answer, even if their documents differ.

Redis layout (per namespace)
----------------------------
scache:{ns}:index          Redis Hash   entry_key → zlib-compressed JSON embedding
scache:{ns}:entry:<hash>   Redis String zlib-compressed JSON response payload (TTL-aware)

No redisvl or Redis Stack required — works with plain Redis.
"""
import hashlib
import json
import os
import zlib
from typing import Callable, List, Optional

import numpy as np
import redis


_DEFAULT_TTL       = int(os.getenv("SC_TTL", str(86400)))       # 24 h
_DEFAULT_THRESHOLD = float(os.getenv("SC_THRESHOLD", "0.15"))   # cosine distance


class SemanticCache:
    """
    Session-scoped semantic cache for LLM responses.

    Usage:
        cache = SemanticCache(embed_fn=store.generate_embedding)

        payload = cache.check(query, namespace=session_id)
        if payload:
            return payload          # cache hit — skip pgvector + LLM

        result = run_full_pipeline(query)
        cache.store(query, result, namespace=session_id)
    """

    def __init__(
        self,
        embed_fn: Callable[[str], List[float]],
        distance_threshold: float = _DEFAULT_THRESHOLD,
        ttl: int = _DEFAULT_TTL,
    ):
        self.embed_fn           = embed_fn
        self.distance_threshold = distance_threshold
        self.ttl                = ttl
        self._connect()

    # ── Redis connection ───────────────────────────────────────────────────────

    def _connect(self):
        host     = os.getenv("REDIS_HOST_PGVECTOR", "localhost")
        port     = int(os.getenv("REDIS_PORT_PGVECTOR", "6379"))
        username = os.getenv("REDIS_USERNAME_PGVECTOR", "default")
        password = os.getenv("REDIS_PASSWORD_PGVECTOR", None)
        ssl      = os.getenv("REDIS_SSL", "false").strip().lower() in ("true", "1", "yes")
        try:
            self._redis = redis.Redis(
                host=host,
                port=port,
                username=username,
                password=password,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
                ssl=ssl,
            )
            self._redis.ping()
            self.available = True
            print(f"[SEMANTIC_CACHE] Connected to Redis at {host}:{port}")
        except Exception as e:
            self.available = False
            print(f"[SEMANTIC_CACHE] Redis unavailable — semantic cache disabled. error={e}")

    # ── Keys ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _index_key(namespace: str) -> str:
        return f"scache:{namespace}:index"

    @staticmethod
    def _entry_key(namespace: str, query_hash: str) -> str:
        return f"scache:{namespace}:entry:{query_hash}"

    # ── Cosine distance ────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_distance(a: List[float], b: List[float]) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return 1.0
        return float(1.0 - np.dot(va, vb) / denom)

    # ── Public API ─────────────────────────────────────────────────────────────

    def check(self, query: str, namespace: str = "default") -> Optional[dict]:
        """Return cached payload if a semantically close query exists in this namespace."""
        if not self.available:
            return None
        try:
            query_emb = self.embed_fn(query)
            index_key = self._index_key(namespace)
            index = self._redis.hgetall(index_key)
            if not index:
                print(f"[SEMANTIC_CACHE] MISS — cache empty for namespace={namespace}")
                return None

            best_key  = None
            best_dist = float("inf")
            stale     = []

            for raw_key, emb_bytes in index.items():
                entry_key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
                if not self._redis.exists(entry_key):
                    stale.append(raw_key)
                    continue
                stored_emb = json.loads(zlib.decompress(emb_bytes))
                dist = self._cosine_distance(query_emb, stored_emb)
                if dist < best_dist:
                    best_dist = dist
                    best_key  = entry_key

            if stale:
                self._redis.hdel(index_key, *stale)
                print(f"[SEMANTIC_CACHE] Pruned {len(stale)} expired entries for namespace={namespace}")

            if best_key and best_dist <= self.distance_threshold:
                payload_bytes = self._redis.get(best_key)
                if payload_bytes:
                    print(
                        f"[SEMANTIC_CACHE] HIT  dist={best_dist:.4f} "
                        f"threshold={self.distance_threshold} namespace={namespace}"
                    )
                    return json.loads(zlib.decompress(payload_bytes))

            print(
                f"[SEMANTIC_CACHE] MISS best_dist="
                f"{'inf' if best_dist == float('inf') else f'{best_dist:.4f}'} "
                f"namespace={namespace}"
            )
            return None

        except Exception as e:
            print(f"[SEMANTIC_CACHE] check() error: {e}")
            return None

    def store(self, query: str, payload: dict, namespace: str = "default") -> None:
        """Embed query and persist payload in Redis under this namespace."""
        if not self.available:
            return
        try:
            query_emb  = self.embed_fn(query)
            key_hash   = hashlib.sha256(query.encode()).hexdigest()[:20]
            index_key  = self._index_key(namespace)
            entry_key  = self._entry_key(namespace, key_hash)

            self._redis.hset(
                index_key,
                entry_key,
                zlib.compress(json.dumps(query_emb).encode()),
            )
            self._redis.setex(
                entry_key,
                self.ttl,
                zlib.compress(json.dumps(payload, default=str).encode()),
            )
            print(f"[SEMANTIC_CACHE] STORED namespace={namespace} key={entry_key} ttl={self.ttl}s")
        except Exception as e:
            print(f"[SEMANTIC_CACHE] store() error: {e}")
