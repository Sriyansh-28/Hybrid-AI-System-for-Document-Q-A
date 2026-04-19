"""Hybrid retriever combining dense (FAISS) and sparse (BM25) retrieval with RRF fusion."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import RetrievalConfig
from .document_processor import Chunk
from .embeddings import EmbeddingModel
from .vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class BM25Index:
    """
    Lightweight BM25 index built on top of the rank-bm25 library.

    Wraps BM25Okapi and provides a `search` method consistent with the
    FAISS vector store interface.
    """

    def __init__(self) -> None:
        self._bm25 = None
        self._chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        """Build the BM25 index over *chunks*."""
        try:
            from rank_bm25 import BM25Okapi  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Install rank-bm25: pip install rank-bm25") from exc

        self._chunks = chunks
        tokenised = [self._tokenise(chunk.text) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenised)
        logger.info("Built BM25 index over %d chunks", len(chunks))

    def search(self, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
        """Return top-k chunks by BM25 score."""
        if self._bm25 is None or not self._chunks:
            return []
        tokens = self._tokenise(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:top_k]:
            results.append((self._chunks[idx], float(score)))
        return results

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """Lower-case, alphanumeric tokenisation."""
        import re  # noqa: PLC0415

        return re.findall(r"\b\w+\b", text.lower())


class HybridRetriever:
    """
    Combines dense (FAISS) and sparse (BM25) retrieval via Reciprocal Rank
    Fusion (RRF).

    RRF Formula
    -----------
    For each candidate chunk *c* appearing at rank *r_dense* in the dense
    results and at rank *r_sparse* in the sparse results, the fused score is:

        score(c) = dense_weight / (rrf_k + r_dense)
                 + sparse_weight / (rrf_k + r_sparse)

    Chunks that appear in only one list receive 0 for the missing term.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_model: EmbeddingModel,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.config = config or RetrievalConfig()
        self._bm25: Optional[BM25Index] = None

    # ------------------------------------------------------------------
    # Build / update
    # ------------------------------------------------------------------

    def build_sparse_index(self, chunks: List[Chunk]) -> None:
        """Build the BM25 sparse index from *chunks*."""
        self._bm25 = BM25Index()
        self._bm25.build(chunks)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Chunk, float]]:
        """
        Retrieve the most relevant chunks for *query* using hybrid search.

        Returns
        -------
        List of (Chunk, fused_score) sorted by descending fused score,
        truncated to *top_k* results.
        """
        cfg = self.config
        k = top_k if top_k is not None else cfg.top_k

        dense_results = self._dense_search(query, cfg.dense_candidates)
        sparse_results = self._sparse_search(query, cfg.sparse_candidates)

        fused = self._reciprocal_rank_fusion(dense_results, sparse_results, cfg.rrf_k, cfg.dense_weight, cfg.sparse_weight)
        return fused[:k]

    def dense_only(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Chunk, float]]:
        """Retrieve using dense search only."""
        k = top_k if top_k is not None else self.config.top_k
        return self._dense_search(query, k)

    def sparse_only(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Chunk, float]]:
        """Retrieve using sparse (BM25) search only."""
        k = top_k if top_k is not None else self.config.top_k
        return self._sparse_search(query, k)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dense_search(self, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
        q_emb = self.embedding_model.encode_single(query)
        return self.vector_store.search(q_emb, top_k=top_k)

    def _sparse_search(self, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
        if self._bm25 is None:
            return []
        return self._bm25.search(query, top_k=top_k)

    @staticmethod
    def _reciprocal_rank_fusion(
        dense_results: List[Tuple[Chunk, float]],
        sparse_results: List[Tuple[Chunk, float]],
        rrf_k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> List[Tuple[Chunk, float]]:
        """
        Merge dense and sparse result lists using Reciprocal Rank Fusion.
        """
        scores: Dict[str, float] = {}
        chunks: Dict[str, Chunk] = {}

        for rank, (chunk, _) in enumerate(dense_results, start=1):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + dense_weight / (rrf_k + rank)
            chunks[cid] = chunk

        for rank, (chunk, _) in enumerate(sparse_results, start=1):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + sparse_weight / (rrf_k + rank)
            chunks[cid] = chunk

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(chunks[cid], score) for cid, score in ranked]
