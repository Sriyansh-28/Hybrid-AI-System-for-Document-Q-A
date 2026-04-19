"""
Shared pytest fixtures, including a lightweight mock EmbeddingModel that avoids
any network downloads while still producing consistent, deterministic embeddings.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from hybrid_ai_system.config import EmbeddingConfig
from hybrid_ai_system.embeddings import EmbeddingModel

_MOCK_DIM = 64


class MockEmbeddingModel(EmbeddingModel):
    """
    Drop-in replacement for EmbeddingModel that generates deterministic
    pseudo-random unit vectors without loading any neural network.

    The hash of each text is used as the random seed so that identical texts
    always produce identical vectors.
    """

    def __init__(self, dim: int = _MOCK_DIM) -> None:
        cfg = EmbeddingConfig(cache_embeddings=False, normalize_embeddings=True)
        # Bypass the parent __init__ so we don't touch sentence-transformers
        self.config = cfg
        self._cache: dict = {}
        self._model = None
        self._dim = dim

    @property
    def embedding_dim(self) -> int:  # type: ignore[override]
        return self._dim

    def encode(self, texts: list, show_progress: bool = False) -> np.ndarray:  # type: ignore[override]
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        vecs = np.stack([self._text_to_vec(t) for t in texts])
        return vecs.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:  # type: ignore[override]
        return self._text_to_vec(text).astype(np.float32)

    def _text_to_vec(self, text: str) -> np.ndarray:
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self._dim)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10)


@pytest.fixture(scope="session")
def mock_embedding_model() -> MockEmbeddingModel:
    """Session-scoped mock embedding model (no network required)."""
    return MockEmbeddingModel(dim=_MOCK_DIM)
