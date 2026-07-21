"""Automated evaluation pipeline.

Generates answers for a held-out eval set with the fine-tuned adapter, scores
each one on correctness and instruction-following, and writes a JSON scorecard.
Running this on every candidate adapter is what replaces ad-hoc manual spot
checks and lets you diff model versions to catch regressions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .config import PipelineConfig
from .data import read_jsonl
from .prompt import build_prompt, extract_response
from .scoring import aggregate, follows_instruction, token_f1


def _load_generator(cfg: PipelineConfig) -> Callable[[str], str]:
    """Build a prompt->answer callable backed by the fine-tuned adapter."""
    import torch
    from peft import PeftModel

    from .model import load_base_model, load_tokenizer

    tokenizer = load_tokenizer(cfg.model)
    base = load_base_model(cfg.model)
    model = PeftModel.from_pretrained(base, cfg.eval.adapter_path)
    model.eval()

    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=cfg.eval.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        return extract_response(decoded)

    return generate


def score_predictions(
    records: list[dict],
    predictions: list[str],
    similarity_threshold: float,
) -> dict:
    """Score already-generated predictions against references."""
    per_example = []
    for record, pred in zip(records, predictions):
        f1 = token_f1(pred, record.get("output", ""))
        required = record.get("required_phrases", [])
        per_example.append(
            {
                "instruction": record.get("instruction", ""),
                "prediction": pred,
                "reference": record.get("output", ""),
                "f1": f1,
                "correct": f1 >= similarity_threshold,
                "followed": follows_instruction(pred, required),
            }
        )
    return {"summary": aggregate(per_example), "examples": per_example}


def run_evaluation(cfg: PipelineConfig, generate: Callable[[str], str] | None = None) -> dict:
    """Generate + score the eval set and persist the report.

    Pass `generate` to inject a stub (used in tests); otherwise the fine-tuned
    adapter is loaded.
    """
    records = list(read_jsonl(cfg.eval.eval_path))
    if generate is None:
        generate = _load_generator(cfg)

    predictions = [
        generate(build_prompt(r.get("instruction", ""), r.get("input", "")))
        for r in records
    ]
    report = score_predictions(records, predictions, cfg.eval.similarity_threshold)

    out_path = Path(cfg.eval.report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report
