"""Sentence-transformer based embedding model with batching and optional caching."""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Optional

import numpy as np

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wraps a sentence-transformers model to produce dense vector embeddings.

    Features
    --------
    - Batch processing for throughput efficiency (configurable batch size).
    - Optional in-memory LRU-style cache: repeated texts avoid re-encoding.
    - Normalized embeddings (unit vectors) by default, which is required for
      cosine-similarity comparisons with FAISS inner-product indices.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        self.config = config or EmbeddingConfig()
        self._model = None  # lazy-loaded
        self._cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the produced embeddings."""
        return self._get_model().get_sentence_embedding_dimension()

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Encode a list of texts into a (N, D) float32 numpy array.

        If caching is enabled, already-seen texts are served from the cache.
        New texts are encoded in batches and added to the cache.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        cfg = self.config

        if not cfg.cache_embeddings:
            return self._encode_batch(texts, show_progress)

        # Split into cached vs. uncached
        keys = [self._cache_key(t) for t in texts]
        uncached_indices = [i for i, k in enumerate(keys) if k not in self._cache]
        uncached_texts = [texts[i] for i in uncached_indices]

        if uncached_texts:
            new_embeddings = self._encode_batch(uncached_texts, show_progress)
            for i, emb in zip(uncached_indices, new_embeddings):
                self._cache[keys[i]] = emb

        result = np.stack([self._cache[k] for k in keys]).astype(np.float32)
        return result

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text and return a 1-D float32 array."""
        return self.encode([text])[0]

    def clear_cache(self) -> None:
        """Discard all cached embeddings."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "Install sentence-transformers: pip install sentence-transformers"
                ) from exc
            logger.info("Loading embedding model '%s' …", self.config.model_name)
            self._model = SentenceTransformer(self.config.model_name)
            if self.config.max_seq_length:
                self._model.max_seq_length = self.config.max_seq_length
            logger.info("Embedding model loaded (dim=%d)", self._model.get_sentence_embedding_dimension())
        return self._model

    def _encode_batch(self, texts: List[str], show_progress: bool) -> np.ndarray:
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
