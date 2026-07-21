#!/usr/bin/env python
"""Merge a LoRA adapter into the base model for standalone serving.

Usage:
    python scripts/merge_and_export.py \
        --config configs/train.yaml \
        --adapter outputs/adapter \
        --output outputs/merged
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_post_training.config import load_config  # noqa: E402
from llm_post_training.export import merge_and_export  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--adapter", default="outputs/adapter")
    parser.add_argument("--output", default="outputs/merged")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = merge_and_export(cfg.model, args.adapter, args.output)
    print(f"Merged model saved to: {out}")


if __name__ == "__main__":
    main()
