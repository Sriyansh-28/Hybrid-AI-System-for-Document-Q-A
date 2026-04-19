"""Shared utility helpers: timing, deduplication, and retrieval metrics."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Generator, List, Tuple

from .document_processor import Chunk


@contextmanager
def timer(label: str = "") -> Generator[None, None, None]:
    """Context manager that prints elapsed time."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"[{label}] {elapsed_ms:.1f} ms" if label else f"{elapsed_ms:.1f} ms")


def deduplicate_chunks(
    results: List[Tuple[Chunk, float]],
    similarity_threshold: float = 0.95,
) -> List[Tuple[Chunk, float]]:
    """
    Remove near-duplicate chunks from a ranked list.

    Two chunks are considered duplicates if one text is a substring of the
    other, or if the Jaccard similarity of their word sets exceeds
    *similarity_threshold*.
    """
    seen: List[Tuple[Chunk, float]] = []
    for chunk, score in results:
        if not _is_duplicate(chunk, [c for c, _ in seen], similarity_threshold):
            seen.append((chunk, score))
    return seen


def _is_duplicate(
    candidate: Chunk,
    existing: List[Chunk],
    threshold: float,
) -> bool:
    cand_words = set(candidate.text.lower().split())
    for chunk in existing:
        existing_words = set(chunk.text.lower().split())
        if not cand_words or not existing_words:
            continue
        intersection = len(cand_words & existing_words)
        union = len(cand_words | existing_words)
        if union > 0 and intersection / union >= threshold:
            return True
    return False


def compute_mrr(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -> float:
    """Mean Reciprocal Rank for a single query."""
    relevant_set = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """Recall@K for a single query."""
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for cid in retrieved_ids[:k] if cid in relevant_set)
    return hits / len(relevant_set)


def compute_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """Precision@K for a single query."""
    if k == 0:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for cid in retrieved_ids[:k] if cid in relevant_set)
    return hits / k
