"""Integration tests for the RAGPipeline (retrieval-only path, no LLM call)."""

from __future__ import annotations

import os
import tempfile

import pytest

from hybrid_ai_system.config import (
    ChunkingConfig,
    EmbeddingConfig,
    FAISSConfig,
    GeneratorConfig,
    RetrievalConfig,
    SystemConfig,
)
from hybrid_ai_system.pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Shared corpus
# ---------------------------------------------------------------------------

CORPUS = [
    "The Eiffel Tower was built in 1889 and stands in Paris, France.",
    "Python is a high-level programming language known for its readability.",
    "Machine learning is a branch of artificial intelligence.",
    "FAISS stands for Facebook AI Similarity Search and speeds up nearest-neighbour queries.",
    "Retrieval-Augmented Generation (RAG) combines a retriever and a language model.",
    "The Amazon River flows through Brazil and is the largest river by discharge.",
    "Deep learning uses multi-layered neural networks to learn complex patterns.",
    "BM25 is a probabilistic ranking function used in information retrieval.",
    "Transformers are neural network architectures based on self-attention mechanisms.",
    "Vector databases store high-dimensional embeddings for semantic search.",
]


def make_pipeline(mock_embedding_model) -> RAGPipeline:
    """Build a lightweight pipeline that uses mock embeddings and skips the LLM."""
    dim = mock_embedding_model.embedding_dim
    cfg = SystemConfig(
        chunking=ChunkingConfig(chunk_size=200, chunk_overlap=20, min_chunk_size=10),
        embedding=EmbeddingConfig(normalize_embeddings=True, cache_embeddings=False),
        faiss=FAISSConfig(embedding_dim=dim, nlist=4),
        retrieval=RetrievalConfig(top_k=3, dense_candidates=8, sparse_candidates=8),
        generator=GeneratorConfig(model_name="google/flan-t5-base"),
    )
    pipeline = RAGPipeline(cfg)
    # Inject the mock embedding model so no network download happens
    pipeline.embedding_model = mock_embedding_model
    pipeline.retriever.embedding_model = mock_embedding_model
    return pipeline


@pytest.fixture(scope="module")
def pipeline(mock_embedding_model) -> RAGPipeline:
    p = make_pipeline(mock_embedding_model)
    p.index_texts(CORPUS, doc_ids=[f"doc_{i}" for i in range(len(CORPUS))])
    return p


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

class TestIndexing:
    def test_index_texts_returns_chunk_count(self, mock_embedding_model):
        p = make_pipeline(mock_embedding_model)
        n = p.index_texts(CORPUS[:3])
        assert n >= 3

    def test_num_indexed_chunks_grows(self, mock_embedding_model):
        p = make_pipeline(mock_embedding_model)
        p.index_texts(CORPUS[:5])
        assert p.num_indexed_chunks >= 5

    def test_index_empty_list_returns_zero(self, mock_embedding_model):
        p = make_pipeline(mock_embedding_model)
        assert p.index_texts([]) == 0

    def test_index_directory(self, mock_embedding_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, text in enumerate(CORPUS[:4]):
                path = os.path.join(tmpdir, f"doc{i}.txt")
                with open(path, "w") as fh:
                    fh.write(text * 5)  # repeat to produce content
            p = make_pipeline(mock_embedding_model)
            n = p.index_directory(tmpdir)
            assert n >= 4


# ---------------------------------------------------------------------------
# Querying (retrieval only – no LLM)
# ---------------------------------------------------------------------------

class TestQuerying:
    def test_query_returns_result(self, pipeline):
        result = pipeline.query("What is FAISS?", use_generator=False)
        assert result.question == "What is FAISS?"
        assert result.answer

    def test_query_retrieves_relevant_chunks(self, pipeline):
        result = pipeline.query("machine learning artificial intelligence", use_generator=False)
        texts = [c.text for c, _ in result.retrieved_chunks]
        assert any("machine learning" in t.lower() or "artificial intelligence" in t.lower() for t in texts)

    def test_latency_is_recorded(self, pipeline):
        result = pipeline.query("deep learning neural networks", use_generator=False)
        assert result.latency_ms > 0

    def test_retrieved_chunks_respects_top_k(self, pipeline):
        result = pipeline.query("programming language Python", top_k=2, use_generator=False)
        assert len(result.retrieved_chunks) <= 2

    def test_no_duplicate_chunks_in_result(self, pipeline):
        result = pipeline.query("vector database semantic search", use_generator=False)
        ids = [c.chunk_id for c, _ in result.retrieved_chunks]
        assert len(ids) == len(set(ids))

    def test_scores_are_positive(self, pipeline):
        result = pipeline.query("BM25 ranking information retrieval", use_generator=False)
        assert all(s >= 0 for _, s in result.retrieved_chunks)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_preserves_vectors(self, pipeline, mock_embedding_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline.save(tmpdir)

            p2 = make_pipeline(mock_embedding_model)
            p2.load(tmpdir)
            assert p2.num_indexed_chunks == pipeline.num_indexed_chunks

    def test_loaded_pipeline_can_answer_queries(self, pipeline, mock_embedding_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline.save(tmpdir)
            p2 = make_pipeline(mock_embedding_model)
            p2.load(tmpdir)
            result = p2.query("Eiffel Tower Paris", use_generator=False)
            assert result.answer


# ---------------------------------------------------------------------------
# Utility metrics
# ---------------------------------------------------------------------------

class TestUtilsMetrics:
    def test_mrr_perfect(self):
        from hybrid_ai_system.utils import compute_mrr
        assert compute_mrr(["a", "b", "c"], ["a"]) == 1.0

    def test_mrr_miss(self):
        from hybrid_ai_system.utils import compute_mrr
        assert compute_mrr(["x", "y", "z"], ["a"]) == 0.0

    def test_recall_at_k(self):
        from hybrid_ai_system.utils import compute_recall_at_k
        assert compute_recall_at_k(["a", "b", "c"], ["a", "c"], k=2) == 0.5

    def test_precision_at_k(self):
        from hybrid_ai_system.utils import compute_precision_at_k
        assert compute_precision_at_k(["a", "b", "c"], ["a", "c"], k=2) == 0.5
