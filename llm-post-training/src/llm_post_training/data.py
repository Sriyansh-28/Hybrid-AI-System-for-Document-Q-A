"""Data cleaning and train/eval splitting.

Takes raw instruction records (JSONL) and turns them into a clean, deduplicated,
deterministically-split dataset ready for training. No model dependencies here so
it runs anywhere and is fast to test.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable, Iterator

from .config import DataConfig

REQUIRED_FIELDS = ("instruction", "output")
_WHITESPACE = re.compile(r"\s+")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield one dict per non-empty line, skipping unparseable lines."""
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A single bad line shouldn't sink a 50K-example run.
                print(f"  skipping malformed JSON on line {line_no}")


def _norm_text(value: str) -> str:
    return _WHITESPACE.sub(" ", (value or "").strip())


def clean_record(record: dict) -> dict | None:
    """Normalise one record, or return None if it should be dropped."""
    cleaned = {
        "instruction": _norm_text(record.get("instruction", "")),
        "input": _norm_text(record.get("input", "")),
        "output": _norm_text(record.get("output", "")),
    }
    for req in REQUIRED_FIELDS:
        if not cleaned[req]:
            return None
    return cleaned


def _dedup_key(record: dict) -> str:
    payload = f"{record['instruction']}||{record['input']}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def clean_dataset(records: Iterable[dict], cfg: DataConfig) -> list[dict]:
    """Clean, drop-too-short, and dedupe on (instruction, input)."""
    seen: set[str] = set()
    out: list[dict] = []
    for record in records:
        cleaned = clean_record(record)
        if cleaned is None:
            continue
        if len(cleaned["output"]) < cfg.min_output_chars:
            continue
        key = _dedup_key(cleaned)
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if cfg.max_examples and len(out) >= cfg.max_examples:
            break
    return out


def split_dataset(records: list[dict], cfg: DataConfig) -> tuple[list[dict], list[dict]]:
    """Shuffle deterministically and carve off an eval slice."""
    shuffled = list(records)
    random.Random(cfg.seed).shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * cfg.eval_fraction)) if shuffled else 0
    return shuffled[n_eval:], shuffled[:n_eval]


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare(cfg: DataConfig) -> dict:
    """End-to-end: read raw -> clean -> split -> write train/eval JSONL."""
    raw = list(read_jsonl(cfg.raw_path))
    cleaned = clean_dataset(raw, cfg)
    train, evalset = split_dataset(cleaned, cfg)

    processed = Path(cfg.processed_dir)
    write_jsonl(train, processed / "train.jsonl")
    write_jsonl(evalset, processed / "eval.jsonl")

    return {
        "raw": len(raw),
        "cleaned": len(cleaned),
        "dropped": len(raw) - len(cleaned),
        "train": len(train),
        "eval": len(evalset),
    }
