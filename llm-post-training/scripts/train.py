#!/usr/bin/env python
"""Run LoRA fine-tuning.

Usage:
    python scripts/train.py --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_post_training.config import load_config  # noqa: E402
from llm_post_training.train import run_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = run_training(cfg)
    print(f"Adapter saved to: {out}")


if __name__ == "__main__":
    main()
