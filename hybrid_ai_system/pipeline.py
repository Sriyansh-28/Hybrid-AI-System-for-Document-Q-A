"""End-to-end RAG pipeline that ties all components together."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import SystemConfig
from .document_processor import Chunk, DocumentProcessor
from .embeddings import EmbeddingModel
from .generator import AnswerGenerator
from .retriever import HybridRetriever
from .utils import deduplicate_chunks
from .vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Structured result returned by :meth:`RAGPipeline.query`."""

    question: str
    answer: str
    retrieved_chunks: List[Tuple[Chunk, float]] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: Dict = field(default_factory=dict)


class RAGPipeline:
    """
    Scalable Retrieval-Augmented Generation pipeline.

    Architecture
    ------------
    1. **Document ingestion**: raw files → cleaned text → overlapping chunks
       (``DocumentProcessor``).
    2. **Embedding**: chunks are encoded in batches with a sentence-transformer
       model (``EmbeddingModel``).
    3. **Indexing**: embeddings are stored in a FAISS IVF index for low-latency
       approximate nearest-neighbour lookup (``FAISSVectorStore``).  A BM25
       inverted index is built in parallel for sparse retrieval.
    4. **Hybrid retrieval**: at query time dense (FAISS) and sparse (BM25)
       ranked lists are merged with Reciprocal Rank Fusion (``HybridRetriever``).
    5. **Generation**: the top-k chunks are injected into a prompt and a
       seq2seq model produces a natural-language answer (``AnswerGenerator``).

    Usage
    -----
    >>> pipeline = RAGPipeline()
    >>> pipeline.index_documents(["doc1.txt", "doc2.pdf"])
    >>> result = pipeline.query("What is the capital of France?")
    >>> print(result.answer)
    """

    def __init__(self, config: Optional[SystemConfig] = None) -> None:
        self.config = config or SystemConfig()
        self.doc_processor = DocumentProcessor(self.config.chunking)
        self.embedding_model = EmbeddingModel(self.config.embedding)
        self.vector_store = FAISSVectorStore(self.config.faiss)
        self.retriever = HybridRetriever(
            self.vector_store,
            self.embedding_model,
            self.config.retrieval,
        )
        self.generator = AnswerGenerator(self.config.generator)
        self._all_chunks: List[Chunk] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_documents(self, file_paths: List[str], show_progress: bool = False) -> int:
        """
        Process and index a list of document file paths.

        Returns the total number of chunks indexed.
        """
        chunks: List[Chunk] = []
        for path in file_paths:
            try:
                new_chunks = self.doc_processor.process_file(path)
                chunks.extend(new_chunks)
                logger.info("Processed '%s': %d chunk(s)", path, len(new_chunks))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not process '%s': %s", path, exc)

        return self._index_chunks(chunks, show_progress)

    def index_directory(self, directory: str, show_progress: bool = False) -> int:
        """
        Recursively process and index all supported documents in *directory*.

        Returns the total number of chunks indexed.
        """
        chunks = self.doc_processor.process_directory(directory)
        return self._index_chunks(chunks, show_progress)

    def index_texts(
        self,
        texts: List[str],
        doc_ids: Optional[List[str]] = None,
        show_progress: bool = False,
    ) -> int:
        """
        Index raw text strings directly (useful for testing / programmatic use).
        """
        chunks: List[Chunk] = []
        for i, text in enumerate(texts):
            doc_id = (doc_ids or [])[i] if doc_ids and i < len(doc_ids) else f"text_{i}"
            chunks.extend(self.doc_processor.process_text(text, doc_id=doc_id))
        return self._index_chunks(chunks, show_progress)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        deduplicate: bool = True,
        use_generator: bool = True,
    ) -> QueryResult:
        """
        Answer *question* using the indexed documents.

        Parameters
        ----------
        question:
            Natural language question.
        top_k:
            Number of chunks to retrieve (defaults to ``RetrievalConfig.top_k``).
        deduplicate:
            Remove near-duplicate chunks before passing to the generator.
        use_generator:
            If False, return the retrieved chunks without running the LLM
            (useful for retrieval-only evaluation).

        Returns
        -------
        A :class:`QueryResult` containing the answer, retrieved chunks, and
        latency information.
        """
        start = time.perf_counter()

        retrieved = self.retriever.retrieve(question, top_k=top_k)

        if deduplicate:
            retrieved = deduplicate_chunks(retrieved)

        if use_generator:
            answer = self.generator.generate(question, retrieved)
        else:
            # Return the top-1 chunk text as a simple extractive answer
            answer = retrieved[0][0].text if retrieved else "No relevant documents found."

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info("Query answered in %.1f ms (retrieved %d chunks)", latency_ms, len(retrieved))

        return QueryResult(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Persist the vector store to *directory*."""
        self.vector_store.save(directory)

    def load(self, directory: str) -> None:
        """Restore the vector store from *directory* and rebuild the BM25 index."""
        self.vector_store.load(directory)
        # Rebuild BM25 from the loaded chunks
        chunks = self.vector_store.get_all_chunks()
        if chunks:
            self.retriever.build_sparse_index(chunks)
        logger.info("Pipeline state loaded from '%s'", directory)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def num_indexed_chunks(self) -> int:
        return self.vector_store.num_vectors

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _index_chunks(self, chunks: List[Chunk], show_progress: bool) -> int:
        if not chunks:
            logger.warning("No chunks to index.")
            return 0

        logger.info("Encoding %d chunks …", len(chunks))
        texts = [c.text for c in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress=show_progress)

        self.vector_store.add_chunks(chunks, embeddings)
        self._all_chunks.extend(chunks)

        # Rebuild BM25 over the full corpus (incremental BM25 is not supported
        # by rank-bm25, so we rebuild on every index call)
        self.retriever.build_sparse_index(self.vector_store.get_all_chunks())

        logger.info("Indexing complete. Total chunks in store: %d", self.vector_store.num_vectors)
        return len(chunks)
