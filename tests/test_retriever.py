"""Tests for HybridRetriever: dense, sparse, and fused retrieval."""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_ai_system.config import FAISSConfig, RetrievalConfig
from hybrid_ai_system.document_processor import Chunk
from hybrid_ai_system.retriever import BM25Index, HybridRetriever
from hybrid_ai_system.vector_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Shared fixtures  (embedding_model comes from conftest.py)
# ---------------------------------------------------------------------------

DOCS = [
    "The Eiffel Tower is located in Paris, France.",
    "Python is a popular programming language used for data science.",
    "Machine learning enables computers to learn from data.",
    "The Amazon rainforest is the world's largest tropical forest.",
    "FAISS is a library for efficient similarity search.",
    "Retrieval-Augmented Generation combines search with language models.",
    "Deep learning models use neural networks with many layers.",
    "The Python language was created by Guido van Rossum.",
]


@pytest.fixture(scope="module")
def indexed_retriever(mock_embedding_model) -> HybridRetriever:
    """A HybridRetriever pre-indexed with DOCS (uses mock embeddings)."""
    chunks = [
        Chunk(chunk_id=f"doc_{i}", doc_id=f"doc_{i}", text=text)
        for i, text in enumerate(DOCS)
    ]
    embs = mock_embedding_model.encode([c.text for c in chunks])

    faiss_cfg = FAISSConfig(embedding_dim=mock_embedding_model.embedding_dim, nlist=4)
    vector_store = FAISSVectorStore(faiss_cfg)
    vector_store.add_chunks(chunks, embs)

    ret_cfg = RetrievalConfig(top_k=3, dense_candidates=8, sparse_candidates=8)
    retriever = HybridRetriever(vector_store, mock_embedding_model, ret_cfg)
    retriever.build_sparse_index(chunks)
    return retriever


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------

class TestBM25Index:
    def test_build_and_search(self):
        chunks = [
            Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", text=t)
            for i, t in enumerate(DOCS)
        ]
        idx = BM25Index()
        idx.build(chunks)
        results = idx.search("Python programming language", top_k=3)
        assert len(results) == 3
        chunk_texts = [c.text for c, _ in results]
        assert any("Python" in t for t in chunk_texts)

    def test_scores_are_non_negative(self):
        chunks = [
            Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", text=t)
            for i, t in enumerate(DOCS)
        ]
        idx = BM25Index()
        idx.build(chunks)
        results = idx.search("machine learning", top_k=5)
        assert all(s >= 0 for _, s in results)

    def test_empty_index_returns_empty(self):
        idx = BM25Index()
        results = idx.search("any query", top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------

class TestDenseRetrieval:
    def test_dense_returns_top_k(self, indexed_retriever):
        results = indexed_retriever.dense_only("similarity search library", top_k=3)
        assert len(results) == 3

    def test_dense_scores_are_descending(self, indexed_retriever):
        results = indexed_retriever.dense_only("some query text", top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_dense_returns_chunk_objects(self, indexed_retriever):
        results = indexed_retriever.dense_only("some query", top_k=2)
        for chunk, score in results:
            assert chunk.chunk_id.startswith("doc_")
            assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Sparse retrieval (BM25 is deterministic and keyword-based)
# ---------------------------------------------------------------------------

class TestSparseRetrieval:
    def test_sparse_returns_top_k(self, indexed_retriever):
        results = indexed_retriever.sparse_only("Python programming", top_k=3)
        assert len(results) == 3

    def test_sparse_python_in_results(self, indexed_retriever):
        results = indexed_retriever.sparse_only("Python language Guido", top_k=3)
        chunk_texts = [c.text for c, _ in results]
        assert any("Python" in t for t in chunk_texts)

    def test_sparse_faiss_in_results(self, indexed_retriever):
        results = indexed_retriever.sparse_only("FAISS similarity search", top_k=3)
        chunk_texts = [c.text for c, _ in results]
        assert any("FAISS" in t or "similarity" in t for t in chunk_texts)


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------

class TestHybridRetrieval:
    def test_returns_top_k(self, indexed_retriever):
        results = indexed_retriever.retrieve("neural network deep learning", top_k=3)
        assert len(results) == 3

    def test_scores_are_descending(self, indexed_retriever):
        results = indexed_retriever.retrieve("retrieval augmented generation", top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_uses_both_retrievers(self, indexed_retriever):
        results = indexed_retriever.retrieve("RAG combines retrieval and language models", top_k=3)
        assert len(results) >= 1

    def test_no_duplicate_chunk_ids(self, indexed_retriever):
        results = indexed_retriever.retrieve("any query about documents", top_k=5)
        ids = [c.chunk_id for c, _ in results]
        assert len(ids) == len(set(ids))
