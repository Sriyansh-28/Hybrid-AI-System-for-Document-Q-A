#!/usr/bin/env python
"""Clean and split raw instruction data into train/eval JSONL.

Usage:
    python scripts/prepare_data.py --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_post_training.config import load_config  # noqa: E402
from llm_post_training.data import prepare  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stats = prepare(cfg.data)

    print("Data prepared:")
    print(f"  raw examples      : {stats['raw']}")
    print(f"  after cleaning    : {stats['cleaned']} (dropped {stats['dropped']})")
    print(f"  train split       : {stats['train']}")
    print(f"  eval split        : {stats['eval']}")
    print(f"  written to        : {cfg.data.processed_dir}/")


if __name__ == "__main__":
    main()
