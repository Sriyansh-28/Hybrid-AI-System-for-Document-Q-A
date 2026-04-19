"""Tests for FAISSVectorStore: add, search, persistence, and IVF upgrade."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from hybrid_ai_system.config import FAISSConfig
from hybrid_ai_system.document_processor import Chunk
from hybrid_ai_system.vector_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 64


def random_chunks(n: int, dim: int = DIM) -> tuple:
    """Return (chunks, embeddings) for *n* random documents."""
    chunks = [
        Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", text=f"content of document {i}")
        for i in range(n)
    ]
    embs = np.random.randn(n, dim).astype(np.float32)
    # L2-normalise so inner-product ~ cosine similarity
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs /= norms + 1e-10
    return chunks, embs


def flat_store(dim: int = DIM) -> FAISSVectorStore:
    cfg = FAISSConfig(embedding_dim=dim, nlist=100)
    return FAISSVectorStore(cfg)


# ---------------------------------------------------------------------------
# Adding vectors
# ---------------------------------------------------------------------------

class TestAddChunks:
    def test_empty_store_has_zero_vectors(self):
        store = flat_store()
        assert store.num_vectors == 0

    def test_add_increases_count(self):
        store = flat_store()
        chunks, embs = random_chunks(5)
        store.add_chunks(chunks, embs)
        assert store.num_vectors == 5

    def test_add_multiple_batches(self):
        store = flat_store()
        chunks1, embs1 = random_chunks(3)
        chunks2, embs2 = random_chunks(4)
        store.add_chunks(chunks1, embs1)
        store.add_chunks(chunks2, embs2)
        assert store.num_vectors == 7

    def test_mismatch_raises_value_error(self):
        store = flat_store()
        chunks, embs = random_chunks(3)
        with pytest.raises(ValueError):
            store.add_chunks(chunks, embs[:2])

    def test_add_empty_does_nothing(self):
        store = flat_store()
        store.add_chunks([], np.empty((0, DIM), dtype=np.float32))
        assert store.num_vectors == 0


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------

class TestSearch:
    def test_returns_top_k_results(self):
        store = flat_store()
        chunks, embs = random_chunks(10)
        store.add_chunks(chunks, embs)
        results = store.search(embs[0], top_k=3)
        assert len(results) == 3

    def test_nearest_neighbour_is_self(self):
        store = flat_store()
        chunks, embs = random_chunks(10)
        store.add_chunks(chunks, embs)
        results = store.search(embs[0], top_k=1)
        assert results[0][0].chunk_id == chunks[0].chunk_id

    def test_scores_are_descending(self):
        store = flat_store()
        chunks, embs = random_chunks(20)
        store.add_chunks(chunks, embs)
        results = store.search(embs[5], top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_store_returns_empty(self):
        store = flat_store()
        q = np.random.randn(DIM).astype(np.float32)
        results = store.search(q, top_k=5)
        assert results == []

    def test_top_k_capped_at_store_size(self):
        store = flat_store()
        chunks, embs = random_chunks(3)
        store.add_chunks(chunks, embs)
        results = store.search(embs[0], top_k=10)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Chunk lookup
# ---------------------------------------------------------------------------

class TestChunkLookup:
    def test_get_chunk_by_id(self):
        store = flat_store()
        chunks, embs = random_chunks(5)
        store.add_chunks(chunks, embs)
        retrieved = store.get_chunk_by_id("c2")
        assert retrieved is not None
        assert retrieved.chunk_id == "c2"

    def test_get_missing_chunk_returns_none(self):
        store = flat_store()
        chunks, embs = random_chunks(5)
        store.add_chunks(chunks, embs)
        assert store.get_chunk_by_id("nonexistent") is None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_roundtrip(self):
        store = flat_store()
        chunks, embs = random_chunks(8)
        store.add_chunks(chunks, embs)

        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir)

            store2 = flat_store()
            store2.load(tmpdir)

        assert store2.num_vectors == 8
        results = store2.search(embs[0], top_k=1)
        assert results[0][0].chunk_id == chunks[0].chunk_id

    def test_load_missing_raises(self):
        store = flat_store()
        with pytest.raises(FileNotFoundError):
            store.load("/nonexistent/path")
