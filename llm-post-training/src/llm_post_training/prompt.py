"""Prompt formatting.

We keep the prompt template in one place so the exact string the model trains
on is the exact string we use at eval and serve time. A drift here is one of the
most common causes of "works in training, broken in prod", so it is deliberately
boring and centralised.
"""

from __future__ import annotations

from typing import Mapping

SYSTEM_PREAMBLE = (
    "You are a domain expert assistant. Follow the instruction precisely and "
    "answer using only the information provided."
)

# Marks the start of the model's answer. We split on this at eval time to pull
# the completion back out of the full decoded string.
RESPONSE_MARKER = "### Response:"


def build_prompt(instruction: str, context: str | None = None) -> str:
    """Render the instruction (and optional context) into the training prompt."""
    instruction = (instruction or "").strip()
    context = (context or "").strip()

    parts = [SYSTEM_PREAMBLE, "", "### Instruction:", instruction]
    if context:
        parts += ["", "### Input:", context]
    parts += ["", RESPONSE_MARKER, ""]
    return "\n".join(parts)


def build_example(record: Mapping[str, str]) -> str:
    """Render a full training example (prompt + reference answer)."""
    prompt = build_prompt(record.get("instruction", ""), record.get("input", ""))
    answer = (record.get("output", "") or "").strip()
    return prompt + answer


def extract_response(decoded: str) -> str:
    """Pull just the model's answer out of a full decoded generation."""
    if RESPONSE_MARKER in decoded:
        decoded = decoded.split(RESPONSE_MARKER, 1)[1]
    return decoded.strip()
