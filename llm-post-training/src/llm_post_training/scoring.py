"""Pure-Python scorers for the evaluation pipeline.

Kept dependency-free so the scoring logic is fast to unit test. The heavier
semantic-similarity scorer lives behind an optional import in evaluate.py; these
lexical scorers are the always-available floor.
"""

from __future__ import annotations

import re
from typing import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def token_f1(prediction: str, reference: str) -> float:
    """Token-overlap F1 between prediction and reference (0..1)."""
    pred = tokenize(prediction)
    ref = tokenize(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    ref_counts: dict[str, int] = {}
    for tok in ref:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1

    overlap = 0
    for tok in pred:
        if ref_counts.get(tok, 0) > 0:
            ref_counts[tok] -= 1
            overlap += 1

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def follows_instruction(prediction: str, required_phrases: Sequence[str]) -> bool:
    """Instruction-following check: did the output honour required constraints?

    A lightweight, rubric-style check. Real pipelines extend this with
    format validators (JSON parses, regex matches, length bounds, etc.).
    """
    if not (prediction or "").strip():
        return False
    lowered = prediction.lower()
    return all(phrase.lower() in lowered for phrase in required_phrases)


def aggregate(scores: list[dict]) -> dict:
    """Roll per-example scores up into the headline metrics."""
    if not scores:
        return {"n": 0, "correctness": 0.0, "instruction_following": 0.0, "mean_f1": 0.0}

    n = len(scores)
    correct = sum(1 for s in scores if s["correct"])
    followed = sum(1 for s in scores if s["followed"])
    mean_f1 = sum(s["f1"] for s in scores) / n
    return {
        "n": n,
        "correctness": correct / n,
        "instruction_following": followed / n,
        "mean_f1": mean_f1,
    }
