"""Tests for the FastAPI REST layer (hybrid_ai_system.api).

Uses the shared MockEmbeddingModel fixture so no network/model download is
required, and exercises the retrieval-only path (use_generator=False) for
the same reason the pipeline tests do.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hybrid_ai_system.api import app, get_pipeline
from hybrid_ai_system.config import (
    ChunkingConfig,
    EmbeddingConfig,
    FAISSConfig,
    RetrievalConfig,
    SystemConfig,
)
from hybrid_ai_system.pipeline import RAGPipeline

CORPUS = [
    "The Eiffel Tower was built in 1889 and stands in Paris, France.",
    "Python is a high-level programming language known for its readability.",
    "Machine learning is a branch of artificial intelligence.",
    "FAISS stands for Facebook AI Similarity Search and speeds up nearest-neighbour queries.",
    "Retrieval-Augmented Generation (RAG) combines a retriever and a language model.",
]


def _make_test_pipeline(mock_embedding_model) -> RAGPipeline:
    dim = mock_embedding_model.embedding_dim
    cfg = SystemConfig(
        chunking=ChunkingConfig(chunk_size=200, chunk_overlap=20, min_chunk_size=10),
        embedding=EmbeddingConfig(normalize_embeddings=True, cache_embeddings=False),
        faiss=FAISSConfig(embedding_dim=dim, nlist=4),
        retrieval=RetrievalConfig(top_k=3, dense_candidates=8, sparse_candidates=8),
    )
    pipeline = RAGPipeline(cfg)
    pipeline.embedding_model = mock_embedding_model
    pipeline.retriever.embedding_model = mock_embedding_model
    return pipeline


@pytest.fixture()
def client(mock_embedding_model):
    test_pipeline = _make_test_pipeline(mock_embedding_model)
    app.dependency_overrides[get_pipeline] = lambda: test_pipeline
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestHealthAndStats:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_stats_before_indexing(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        assert resp.json()["indexed_chunks"] == 0


class TestIndexTexts:
    def test_index_texts_returns_counts(self, client):
        resp = client.post("/index/texts", json={"texts": CORPUS})
        assert resp.status_code == 200
        body = resp.json()
        assert body["chunks_indexed"] >= len(CORPUS)
        assert body["total_chunks"] == body["chunks_indexed"]

    def test_index_texts_empty_list_rejected(self, client):
        resp = client.post("/index/texts", json={"texts": []})
        assert resp.status_code == 422  # pydantic min_length validation

    def test_stats_reflects_indexed_chunks(self, client):
        client.post("/index/texts", json={"texts": CORPUS})
        resp = client.get("/stats")
        assert resp.json()["indexed_chunks"] >= len(CORPUS)


class TestQuery:
    def test_query_without_index_returns_409(self, client):
        resp = client.post("/query", json={"question": "What is FAISS?"})
        assert resp.status_code == 409

    def test_query_retrieval_only(self, client):
        client.post("/index/texts", json={"texts": CORPUS})
        resp = client.post(
            "/query",
            json={"question": "What is FAISS?", "use_generator": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["question"] == "What is FAISS?"
        assert body["answer"]
        assert body["latency_ms"] > 0
        assert len(body["retrieved_chunks"]) > 0

    def test_query_respects_top_k(self, client):
        client.post("/index/texts", json={"texts": CORPUS})
        resp = client.post(
            "/query",
            json={"question": "machine learning", "top_k": 2, "use_generator": False},
        )
        assert resp.status_code == 200
        assert len(resp.json()["retrieved_chunks"]) <= 2

    def test_query_blank_question_rejected(self, client):
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == 422


class TestIndexUpload:
    def test_upload_unsupported_extension_rejected(self, client):
        resp = client.post(
            "/index/upload",
            files={"files": ("doc.exe", b"binary data", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_upload_txt_file_indexes(self, client):
        content = (CORPUS[0] + " ") * 10
        resp = client.post(
            "/index/upload",
            files={"files": ("doc.txt", content.encode(), "text/plain")},
        )
        assert resp.status_code == 200
        assert resp.json()["chunks_indexed"] >= 1
