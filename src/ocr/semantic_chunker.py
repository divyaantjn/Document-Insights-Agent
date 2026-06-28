import asyncio
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

_CHUNKER_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chunker")

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_FALLBACK_MIN_SENTENCE_LEN = 10   # characters; discard trivially short sentences
_COSINE_ZERO_THRESHOLD = 1e-10    # treat vectors shorter than this as zero

# Production guards — documents exceeding either limit skip semantic embedding
# and fall back to the naive grouping splitter (zero API calls, O(n) memory).
MAX_INPUT_CHARS = 500_000         # ~500 KB of text; beyond this semantic chunking adds no value
MAX_SENTENCES = 2_000             # embedding 2000+ sentences in one call risks OOM + timeout


class SemanticChunker:
    """
    Async-first semantic chunker using batched embedding calls.

    Parameters
    ----------
    embedding_model : str
        Gemini embedding model identifier (e.g. ``"gemini-embedding-001"``).
    task_type : str
        Embedding task type passed to the Gemini API.
    output_dimensionality : int
        Dimensionality of the returned embedding vectors.
    batch_size : int
        Number of sentences embedded per API call.
    percentile_threshold : float
        Percentile (0–100) of similarity scores used as the breakpoint cutoff.
        Lower values produce more / finer chunks.
    max_chunk_size : int
        Hard character cap per output chunk.  Chunks that grow beyond this are
        split at the nearest sentence boundary.
    min_chunk_size : int
        Chunks with fewer characters than this are discarded.
    """

    def __init__(
        self,
        embedding_model: str = "gemini-embedding-001",
        task_type: str = "SEMANTIC_SIMILARITY",
        output_dimensionality: int = 768,
        batch_size: int = 32,
        percentile_threshold: float = 25.0,
        max_chunk_size: int = 2000,
        min_chunk_size: int = 100,
    ) -> None:
        self._embedding_model = embedding_model
        self._task_type = task_type
        self._output_dimensionality = output_dimensionality
        self._batch_size = batch_size
        self._percentile_threshold = percentile_threshold
        self._max_chunk_size = max_chunk_size
        self._min_chunk_size = min_chunk_size
        self._client = genai.Client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chunk(self, text: str) -> List[str]:
        """
        Split *text* into semantically coherent chunks.

        Short-circuits to fallback splitter when:
          - text is blank
          - text exceeds MAX_INPUT_CHARS (memory guard)
          - sentence count exceeds MAX_SENTENCES (API cost + memory guard)
          - embedding call fails for any reason

        Parameters
        ----------
        text : str
            Raw text content of a single document element.

        Returns
        -------
        List[str]
            Ordered list of text chunks. Empty list if *text* is blank.
        """
        stripped = text.strip() if text else ""
        if not stripped:
            return []

        # ── Input size guard ──────────────────────────────────────────
        if len(stripped) > MAX_INPUT_CHARS:
            logger.warning(
                "[SemanticChunker] Input too large (%d chars > %d limit) — "
                "using fallback splitter to avoid OOM.",
                len(stripped),
                MAX_INPUT_CHARS,
            )
            sentences = self._split_into_sentences(stripped)
            return self._filter_min_size(self._fallback_chunk(sentences))

        sentences = self._split_into_sentences(stripped)
        logger.info(
            "[SemanticChunker] chunk() invoked: %d chars → %d sentences.",
            len(stripped),
            len(sentences),
        )

        if len(sentences) <= 1:
            return [stripped] if stripped else []

        # ── Sentence count guard ──────────────────────────────────────
        if len(sentences) > MAX_SENTENCES:
            logger.warning(
                "[SemanticChunker] Sentence count %d exceeds limit %d — "
                "using fallback splitter.",
                len(sentences),
                MAX_SENTENCES,
            )
            return self._filter_min_size(self._fallback_chunk(sentences))

        try:
            embeddings = await self._embed_sentences_batched(sentences)
            logger.info(
                "[SemanticChunker] Embeddings created: %d vectors.", len(embeddings)
            )
            breakpoints = self._find_breakpoints(embeddings)
            chunks = self._merge_sentences(sentences, breakpoints)
            logger.info(
                "[SemanticChunker] Done: %d breakpoints → %d chunks.",
                len(breakpoints),
                len(chunks),
            )
        except Exception:  # noqa: BLE001 — intentional: any embed failure → fallback
            logger.warning(
                "[SemanticChunker] Embedding failed — falling back to sentence grouping.",
                exc_info=True,
            )
            chunks = self._fallback_chunk(sentences)

        return self._filter_min_size(chunks)

    # ------------------------------------------------------------------
    # Sentence splitting
    # ------------------------------------------------------------------

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split *text* into sentences using punctuation-aware regex.

        Filters out trivially short fragments that add no semantic value.
        """
        raw_parts = _SENTENCE_SPLIT_PATTERN.split(text.strip())
        return [
            s.strip()
            for s in raw_parts
            if s.strip() and len(s.strip()) >= _FALLBACK_MIN_SENTENCE_LEN
        ]

    # ------------------------------------------------------------------
    # Embedding (async, batched)
    # ------------------------------------------------------------------

    async def _embed_sentences_batched(self, sentences: List[str]) -> List[List[float]]:
        """Embed all sentences using bounded concurrent Gemini API calls."""
        batches = self._build_batches(sentences)
        max_concurrent = 6
        sem = asyncio.Semaphore(max_concurrent)

        async def _embed_with_sem(batch: List[str]) -> List[List[float]]:
            async with sem:
                return await self._embed_batch(batch)

        batch_results = await asyncio.gather(*[_embed_with_sem(b) for b in batches])
        # Flatten list-of-lists into a single list
        return [embedding for batch in batch_results for embedding in batch]

    def _build_batches(self, sentences: List[str]) -> List[List[str]]:
        """Partition *sentences* into sub-lists of length ``_batch_size``."""
        return [
            sentences[start: start + self._batch_size]
            for start in range(0, len(sentences), self._batch_size)
        ]

    async def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        """Offload a single Gemini embed_content call to the shared thread executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _CHUNKER_EXECUTOR,
            self._sync_embed_batch,
            batch,
        )

    def _sync_embed_batch(self, batch: List[str]) -> List[List[float]]:
        """Synchronous Gemini embedding call — executed off the event loop."""
        embed_config = types.EmbedContentConfig(
            task_type=self._task_type,
            output_dimensionality=self._output_dimensionality,
        )
        result = self._client.models.embed_content(
            model=self._embedding_model,
            contents=batch,
            config=embed_config,
        )
        return [embedding.values for embedding in result.embeddings]

    # ------------------------------------------------------------------
    # Breakpoint detection
    # ------------------------------------------------------------------

    def _find_breakpoints(self, embeddings: List[List[float]]) -> List[int]:
        """
        Return indices *i* where similarity(embeddings[i], embeddings[i+1])
        falls below the configured percentile threshold.
        """
        if len(embeddings) < 2:
            return []

        similarities = [
            self._cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        threshold = self._percentile(similarities, self._percentile_threshold)
        return [i for i, sim in enumerate(similarities) if sim < threshold]

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two dense float vectors."""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a < _COSINE_ZERO_THRESHOLD or norm_b < _COSINE_ZERO_THRESHOLD:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        """
        Compute the *percentile*-th percentile without numpy.
        Uses linear interpolation consistent with numpy's default behaviour.
        """
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        index = (percentile / 100.0) * (n - 1)
        lower = int(index)
        upper = min(lower + 1, n - 1)
        fraction = index - lower
        return sorted_vals[lower] + fraction * (sorted_vals[upper] - sorted_vals[lower])

    # ------------------------------------------------------------------
    # Sentence merging
    # ------------------------------------------------------------------

    def _merge_sentences(
        self,
        sentences: List[str],
        breakpoints: List[int],
    ) -> List[str]:
        """
        Join sentences into chunks delimited by *breakpoints*, respecting
        ``_max_chunk_size``.
        """
        if not sentences:
            return []

        breakpoint_set = set(breakpoints)
        chunks: List[str] = []
        current_sentences: List[str] = []
        current_length = 0

        for idx, sentence in enumerate(sentences):
            sentence_len = len(sentence)

            should_break = (
                (idx > 0 and (idx - 1) in breakpoint_set)
                or (current_length + sentence_len > self._max_chunk_size and current_sentences)
            )

            if should_break:
                chunk_text = " ".join(current_sentences)
                if chunk_text.strip():
                    chunks.append(chunk_text.strip())
                current_sentences = []
                current_length = 0

            current_sentences.append(sentence)
            current_length += sentence_len + 1  # +1 for the joining space

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())

        return chunks

    # ------------------------------------------------------------------
    # Fallback and filtering
    # ------------------------------------------------------------------

    def _fallback_chunk(self, sentences: List[str]) -> List[str]:
        """
        Naïve O(n) fallback: merge sentences up to ``_max_chunk_size``
        without relying on embeddings. Zero API calls, constant memory.
        """
        chunks: List[str] = []
        current_parts: List[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_length + sentence_len > self._max_chunk_size and current_parts:
                chunks.append(" ".join(current_parts).strip())
                current_parts = []
                current_length = 0
            current_parts.append(sentence)
            current_length += sentence_len + 1

        if current_parts:
            chunks.append(" ".join(current_parts).strip())

        return chunks

    def _filter_min_size(self, chunks: List[str]) -> List[str]:
        """Discard chunks shorter than ``_min_chunk_size`` characters."""
        return [c for c in chunks if len(c) >= self._min_chunk_size]

    # ------------------------------------------------------------------
    # Sync convenience wrapper
    # ------------------------------------------------------------------

    def chunk_sync(
        self,
        text: str,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> List[str]:
        """
        Synchronous wrapper around :meth:`chunk`.

        Use only from non-async contexts (e.g. Excel/CSV handlers).
        Falls back to :meth:`_fallback_chunk` if the event loop is unavailable.
        """
        try:
            if loop is None:
                loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, self.chunk(text))
                    return future.result()
            return loop.run_until_complete(self.chunk(text))
        except Exception:  # noqa: BLE001 — intentional: sync fallback on any loop error
            logger.warning("[SemanticChunker] chunk_sync fallback triggered.", exc_info=True)
            sentences = self._split_into_sentences(text.strip() if text else "")
            return self._filter_min_size(self._fallback_chunk(sentences))