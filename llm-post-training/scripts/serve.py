#!/usr/bin/env python
"""Serve the fine-tuned model over HTTP.

Usage:
    MODEL_PATH=outputs/merged python scripts/serve.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    from llm_post_training.serve import build_app

    uvicorn.run(build_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
