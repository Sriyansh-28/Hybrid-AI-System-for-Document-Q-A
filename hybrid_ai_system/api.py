"""
FastAPI REST service for the Hybrid AI Document Q&A System.

Run locally:
    uvicorn hybrid_ai_system.api:app --reload

Endpoints
---------
GET  /health              liveness/readiness probe
GET  /stats                indexed chunk count + config summary
POST /index/texts          index raw text strings
POST /index/upload          index uploaded files (TXT, PDF, DOCX, MD)
POST /query                 ask a question against the indexed corpus
DELETE /index               reset the index (drop all indexed chunks)

Configuration is read from environment variables (see ``_config_from_env``)
so the same image can be deployed with different models/parameters without
code changes.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from typing import List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import (
    ChunkingConfig,
    EmbeddingConfig,
    FAISSConfig,
    GeneratorConfig,
    RetrievalConfig,
    SystemConfig,
)
from .pipeline import RAGPipeline

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class IndexTextsRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="Raw text passages to index")
    doc_ids: Optional[List[str]] = Field(None, description="Optional doc id per text")


class IndexResponse(BaseModel):
    chunks_indexed: int
    total_chunks: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(None, ge=1, le=50)
    use_generator: bool = True


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    latency_ms: float
    retrieved_chunks: List[RetrievedChunk]


class StatsResponse(BaseModel):
    indexed_chunks: int
    embedding_model: str
    generator_model: str


class HealthResponse(BaseModel):
    status: str = "ok"


# ---------------------------------------------------------------------------
# Pipeline singleton (lazy, overridable in tests via dependency_overrides)
# ---------------------------------------------------------------------------

def _config_from_env() -> SystemConfig:
    return SystemConfig(
        chunking=ChunkingConfig(
            chunk_size=int(os.environ.get("CHUNK_SIZE", 512)),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", 64)),
        ),
        embedding=EmbeddingConfig(
            model_name=os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        ),
        faiss=FAISSConfig(
            nlist=int(os.environ.get("FAISS_NLIST", 100)),
            nprobe=int(os.environ.get("FAISS_NPROBE", 10)),
        ),
        retrieval=RetrievalConfig(
            top_k=int(os.environ.get("TOP_K", 5)),
            dense_weight=float(os.environ.get("DENSE_WEIGHT", 0.6)),
            sparse_weight=1.0 - float(os.environ.get("DENSE_WEIGHT", 0.6)),
        ),
        generator=GeneratorConfig(
            model_name=os.environ.get("GENERATOR_MODEL", "google/flan-t5-base"),
            device=os.environ.get("DEVICE", "cpu"),
        ),
    )


_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Lazily construct a process-wide pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(_config_from_env())
    return _pipeline


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Hybrid AI Document Q&A API",
    description="Hybrid (BM25 + dense embedding) retrieval-augmented Q&A over uploaded documents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/stats", response_model=StatsResponse, tags=["ops"])
def stats(pipeline: RAGPipeline = Depends(get_pipeline)) -> StatsResponse:
    return StatsResponse(
        indexed_chunks=pipeline.num_indexed_chunks,
        embedding_model=pipeline.config.embedding.model_name,
        generator_model=pipeline.config.generator.model_name,
    )


@app.post("/index/texts", response_model=IndexResponse, tags=["indexing"])
def index_texts(
    request: IndexTextsRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> IndexResponse:
    try:
        n = pipeline.index_texts(request.texts, doc_ids=request.doc_ids)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to index texts")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IndexResponse(chunks_indexed=n, total_chunks=pipeline.num_indexed_chunks)


@app.post("/index/upload", response_model=IndexResponse, tags=["indexing"])
def index_upload(
    files: List[UploadFile] = File(...),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> IndexResponse:
    unsupported = [
        f.filename for f in files
        if os.path.splitext(f.filename or "")[1].lower() not in SUPPORTED_EXTENSIONS
    ]
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type(s): {unsupported}. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    tmp_dir = tempfile.mkdtemp(prefix="qa_upload_")
    try:
        saved_paths = []
        for f in files:
            dest = os.path.join(tmp_dir, f.filename)
            with open(dest, "wb") as fh:
                shutil.copyfileobj(f.file, fh)
            saved_paths.append(dest)

        try:
            n = pipeline.index_documents(saved_paths)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to index uploaded files")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return IndexResponse(chunks_indexed=n, total_chunks=pipeline.num_indexed_chunks)


@app.post("/query", response_model=QueryResponse, tags=["query"])
def query(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    if pipeline.num_indexed_chunks == 0:
        raise HTTPException(status_code=409, detail="No documents indexed yet. Call /index/texts or /index/upload first.")

    try:
        result = pipeline.query(
            request.question,
            top_k=request.top_k,
            use_generator=request.use_generator,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        latency_ms=result.latency_ms,
        retrieved_chunks=[
            RetrievedChunk(chunk_id=c.chunk_id, doc_id=c.doc_id, text=c.text, score=score)
            for c, score in result.retrieved_chunks
        ],
    )


@app.delete("/index", response_model=StatsResponse, tags=["indexing"])
def reset_index() -> StatsResponse:
    """Drop the current pipeline (and its index), forcing a fresh one on next use."""
    global _pipeline
    _pipeline = None
    pipeline = get_pipeline()
    return StatsResponse(
        indexed_chunks=pipeline.num_indexed_chunks,
        embedding_model=pipeline.config.embedding.model_name,
        generator_model=pipeline.config.generator.model_name,
    )
