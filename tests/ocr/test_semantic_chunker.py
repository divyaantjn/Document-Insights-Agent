import pytest
from unittest.mock import AsyncMock, patch

from src.ocr.semantic_chunker import SemanticChunker


@pytest.mark.asyncio
async def test_chunk_with_embeddings_breakpoints_and_filtering():
    text = """This is the first sentence. This is the second sentence.
    A very different topic starts here. Another sentence follows.
    Final sentence that should join with the previous one."""

    sc = SemanticChunker(percentile_threshold=50.0, max_chunk_size=500, min_chunk_size=10)

    sentences = sc._split_into_sentences(text)
    assert len(sentences) >= 3

    # High similarity between first two, low between 2nd and 3rd, moderate after
    embeddings = [
        [1.0, 0.0],  # s0
        [0.9, 0.1],  # s1 (similar to s0)
        [0.0, 1.0],  # s2 (different)
        [0.0, 0.8],  # s3 (similar to s2)
        [0.0, 0.7],  # s4 (similar to s2)
    ][: len(sentences)]

    with patch.object(sc, "_embed_sentences_batched", new=AsyncMock(return_value=embeddings)):
        chunks = await sc.chunk(text)

    assert isinstance(chunks, list) and chunks
    # Expect at least two chunks due to the low similarity breakpoint
    assert len(chunks) >= 2
    # Chunks should respect min size filter
    assert all(len(c) >= 10 for c in chunks)


def test_cosine_similarity_zero_vector():
    assert SemanticChunker._cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_percentile_empty_values():
    assert SemanticChunker._percentile([], 50.0) == 0.0


@pytest.mark.asyncio
async def test_chunk_fallback_on_embedding_error():
    text = "This is a sentence. Another related sentence. And one more." \
           " A different part begins here."
    sc = SemanticChunker(max_chunk_size=50, min_chunk_size=10)
    with patch.object(sc, "_embed_sentences_batched", new=AsyncMock(side_effect=RuntimeError("boom"))):
        chunks = await sc.chunk(text)
    assert isinstance(chunks, list) and chunks


def test_build_batches_partitioning():
    sc = SemanticChunker(batch_size=3)
    data = [str(i) for i in range(10)]
    batches = sc._build_batches(data)
    assert batches[0] == ["0", "1", "2"]
    assert batches[-1] == ["9"]