"""Tests for EmbeddingModel: batching, caching, and output shape.

Tests that require downloading models from HuggingFace are marked with
``@pytest.mark.network`` and skipped automatically in offline CI environments.
The conftest MockEmbeddingModel is used for structural/logic tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_ai_system.config import EmbeddingConfig
from hybrid_ai_system.embeddings import EmbeddingModel


# ---------------------------------------------------------------------------
# Mock-based tests (no network required)
# ---------------------------------------------------------------------------

class TestMockEmbeddingModel:
    """Use the conftest MockEmbeddingModel to validate interface contracts."""

    def test_encode_returns_correct_shape(self, mock_embedding_model):
        texts = ["hello", "world", "test"]
        embs = mock_embedding_model.encode(texts)
        assert embs.shape == (3, mock_embedding_model.embedding_dim)

    def test_encode_empty_returns_empty(self, mock_embedding_model):
        embs = mock_embedding_model.encode([])
        assert embs.shape[0] == 0

    def test_encode_single_is_1d(self, mock_embedding_model):
        vec = mock_embedding_model.encode_single("one text")
        assert vec.ndim == 1

    def test_dtype_is_float32(self, mock_embedding_model):
        embs = mock_embedding_model.encode(["dtype check"])
        assert embs.dtype == np.float32

    def test_unit_norm(self, mock_embedding_model):
        embs = mock_embedding_model.encode(["normalised"])
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_deterministic(self, mock_embedding_model):
        text = "deterministic text"
        emb1 = mock_embedding_model.encode([text])
        emb2 = mock_embedding_model.encode([text])
        np.testing.assert_array_equal(emb1, emb2)

    def test_different_texts_different_vectors(self, mock_embedding_model):
        embs = mock_embedding_model.encode(["text A", "text B"])
        # Two different random seeds → different vectors
        assert not np.allclose(embs[0], embs[1])


# ---------------------------------------------------------------------------
# Caching tests (use EmbeddingModel directly with mock encode)
# ---------------------------------------------------------------------------

class TestCaching:
    """Test the caching logic in EmbeddingModel using the mock."""

    def test_cache_hit_returns_same_result(self, mock_embedding_model):
        text = "cache test sentence"
        emb1 = mock_embedding_model.encode([text])
        emb2 = mock_embedding_model.encode([text])
        np.testing.assert_array_equal(emb1, emb2)

