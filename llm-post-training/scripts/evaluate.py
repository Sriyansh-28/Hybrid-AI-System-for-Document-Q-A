#!/usr/bin/env python
"""Score a fine-tuned adapter on the held-out eval set.

Usage:
    python scripts/evaluate.py --config configs/eval.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_post_training.config import load_config  # noqa: E402
from llm_post_training.evaluate import run_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    report = run_evaluation(cfg)
    summary = report["summary"]

    print("Evaluation summary:")
    print(f"  examples              : {summary['n']}")
    print(f"  correctness           : {summary['correctness']:.1%}")
    print(f"  instruction-following : {summary['instruction_following']:.1%}")
    print(f"  mean token-F1         : {summary['mean_f1']:.3f}")
    print(f"  full report           : {cfg.eval.report_path}")


if __name__ == "__main__":
    main()
