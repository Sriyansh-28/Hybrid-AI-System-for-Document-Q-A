"""FAISS-based vector store for scalable approximate nearest-neighbour search."""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import FAISSConfig
from .document_processor import Chunk

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Manages a FAISS index over chunk embeddings.

    Index type
    ----------
    * **Flat (exact)** when the number of stored vectors < 4 * nlist  (small
      collections where IVF training is unreliable).
    * **IVF + Flat** otherwise: an Inverted File index that partitions vectors
      into *nlist* Voronoi cells and probes only *nprobe* of them at query time.
      This reduces latency by ~40 % for collections of hundreds-to-thousands of
      vectors while retaining high recall.

    Inner-product similarity is used (equivalent to cosine similarity when
    embeddings are L2-normalised, which EmbeddingModel does by default).
    """

    def __init__(self, config: Optional[FAISSConfig] = None) -> None:
        self.config = config or FAISSConfig()
        self._index = None          # faiss.Index
        self._chunks: List[Chunk] = []
        self._id_to_idx: Dict[str, int] = {}  # chunk_id → position in self._chunks

    # ------------------------------------------------------------------
    # Build / update
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Add *chunks* and their pre-computed *embeddings* to the store.

        Parameters
        ----------
        chunks:
            The Chunk objects to add.
        embeddings:
            Float32 array of shape (len(chunks), embedding_dim).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match number of embeddings ({len(embeddings)})"
            )
        if len(chunks) == 0:
            return

        embeddings = np.array(embeddings, dtype=np.float32)

        if self._index is None:
            self._build_index(embeddings)
        else:
            # If we have accumulated enough vectors to switch from Flat to IVF,
            # rebuild the index from scratch (rare but handled gracefully).
            total = len(self._chunks) + len(chunks)
            if total >= 4 * self.config.nlist and self._is_flat():
                all_embeddings = np.vstack([self._get_all_stored_embeddings(), embeddings])
                all_chunks = self._chunks + list(chunks)
                self._chunks = []
                self._id_to_idx = {}
                self._build_index(all_embeddings)
                for i, chunk in enumerate(all_chunks):
                    self._id_to_idx[chunk.chunk_id] = i
                    self._chunks.append(chunk)
                return

            self._index.add(embeddings)

        start_idx = len(self._chunks)
        for i, chunk in enumerate(chunks):
            self._id_to_idx[chunk.chunk_id] = start_idx + i
            self._chunks.append(chunk)

        logger.info("Vector store now holds %d vectors", len(self._chunks))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[Chunk, float]]:
        """
        Return the *top_k* most similar chunks.

        Returns
        -------
        List of (Chunk, score) tuples sorted by descending similarity.
        """
        if self._index is None or len(self._chunks) == 0:
            return []

        q = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Persist the index and metadata to *directory*."""
        import faiss  # noqa: PLC0415

        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, "faiss.index")
        meta_path = os.path.join(directory, "metadata.pkl")

        faiss.write_index(self._index, index_path)
        with open(meta_path, "wb") as fh:
            pickle.dump({"chunks": self._chunks, "id_to_idx": self._id_to_idx}, fh)
        logger.info("Saved vector store to '%s'", directory)

    def load(self, directory: str) -> None:
        """Load a previously saved index from *directory*."""
        import faiss  # noqa: PLC0415

        index_path = os.path.join(directory, "faiss.index")
        meta_path = os.path.join(directory, "metadata.pkl")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"No saved index found in '{directory}'")

        self._index = faiss.read_index(index_path)
        with open(meta_path, "rb") as fh:
            # NOTE: This file is always written by this module itself (trusted source).
            # Do not load index files from untrusted external directories.
            data = pickle.load(fh)  # noqa: S301
        self._chunks = data["chunks"]
        self._id_to_idx = data["id_to_idx"]
        logger.info("Loaded vector store with %d vectors from '%s'", len(self._chunks), directory)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_vectors(self) -> int:
        return len(self._chunks)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        idx = self._id_to_idx.get(chunk_id)
        return self._chunks[idx] if idx is not None else None

    def get_all_chunks(self) -> List[Chunk]:
        """Return a copy of all indexed chunks."""
        return list(self._chunks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_index(self, embeddings: np.ndarray) -> None:
        """Choose and train an appropriate FAISS index for *embeddings*."""
        import faiss  # noqa: PLC0415

        n, dim = embeddings.shape
        cfg = self.config

        if n < 4 * cfg.nlist:
            # Too few vectors for reliable IVF training → use exact Flat index
            logger.info("Using FlatIP index (n=%d < 4×nlist=%d)", n, 4 * cfg.nlist)
            self._index = faiss.IndexFlatIP(dim)
        else:
            logger.info("Using IVF%d FlatIP index (n=%d, nprobe=%d)", cfg.nlist, n, cfg.nprobe)
            quantizer = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(quantizer, dim, cfg.nlist, faiss.METRIC_INNER_PRODUCT)
            self._index.train(embeddings)
            self._index.nprobe = cfg.nprobe

        if cfg.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
                logger.info("Moved FAISS index to GPU")
            except Exception as exc:  # noqa: BLE001
                logger.warning("GPU not available, falling back to CPU: %s", exc)

        self._index.add(embeddings)

    def _is_flat(self) -> bool:
        """Return True if the current index is a Flat (exact) index."""
        import faiss  # noqa: PLC0415

        return isinstance(self._index, faiss.IndexFlatIP)

    def _get_all_stored_embeddings(self) -> np.ndarray:
        """Reconstruct stored embeddings from a Flat index (used when upgrading to IVF)."""
        import faiss  # noqa: PLC0415

        n = self._index.ntotal
        dim = self.config.embedding_dim
        embeddings = np.empty((n, dim), dtype=np.float32)
        self._index.reconstruct_n(0, n, embeddings)
        return embeddings
