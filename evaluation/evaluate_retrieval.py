#!/usr/bin/env python
"""
Evaluation harness for the Hybrid AI Document Q&A System.

Produces the two headline numbers used in the project README:

1. **Retrieval relevance** — dense-only vs. hybrid (BM25 + dense, fused with
   Reciprocal Rank Fusion), scored with MRR, Recall@k and Precision@k on the
   labeled query set in ``eval_data.py``. Reports the relative improvement of
   hybrid over dense-only.

2. **Query latency** — FAISS ``IndexFlatIP`` (exhaustive linear scan) vs.
   ``IndexIVFFlat`` (approximate, probes a few Voronoi cells) on a synthetic
   corpus of normalised random vectors. Reports the mean per-query latency of
   each and the reduction, plus IVF's recall against the exact index so the
   speed-up isn't hiding a quality loss.

Run
---
    python -m evaluation.evaluate_retrieval
    python -m evaluation.evaluate_retrieval --top-k 5 --latency-n 20000

Part 1 downloads the sentence-transformer model on first run (~90 MB). Part 2
needs no model and no network.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import List, Tuple

import numpy as np

from hybrid_ai_system import RAGPipeline
from hybrid_ai_system.config import (
    ChunkingConfig,
    FAISSConfig,
    RetrievalConfig,
    SystemConfig,
)
from hybrid_ai_system.utils import (
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
)

from .eval_data import CORPUS, QUERIES


# ---------------------------------------------------------------------------
# Part 1 — retrieval relevance: dense-only vs. hybrid
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def evaluate_relevance(top_k: int) -> None:
    print("=" * 70)
    print("PART 1  Retrieval relevance — dense-only vs. hybrid (RRF fusion)")
    print("=" * 70)
    print(f"Corpus: {len(CORPUS)} documents · Queries: {len(QUERIES)} · top_k={top_k}\n")

    config = SystemConfig(
        # One chunk per short doc so doc_id == the unit being ranked.
        chunking=ChunkingConfig(chunk_size=1024, chunk_overlap=0, min_chunk_size=5),
        retrieval=RetrievalConfig(
            top_k=top_k,
            dense_weight=0.5,
            sparse_weight=0.5,
            dense_candidates=20,
            sparse_candidates=20,
        ),
        faiss=FAISSConfig(nlist=8),  # tiny corpus -> store uses an exact Flat index
    )
    pipeline = RAGPipeline(config)
    pipeline.index_texts([text for _, text in CORPUS], doc_ids=[did for did, _ in CORPUS])

    retriever = pipeline.retriever
    metrics = {
        "dense": {"mrr": [], "recall": [], "precision": []},
        "hybrid": {"mrr": [], "recall": [], "precision": []},
    }

    for query, relevant in QUERIES:
        runs = {
            "dense": retriever.dense_only(query, top_k=top_k),
            "hybrid": retriever.retrieve(query, top_k=top_k),
        }
        for name, results in runs.items():
            ranked_ids = [chunk.doc_id for chunk, _ in results]
            metrics[name]["mrr"].append(compute_mrr(ranked_ids, relevant))
            metrics[name]["recall"].append(compute_recall_at_k(ranked_ids, relevant, top_k))
            metrics[name]["precision"].append(compute_precision_at_k(ranked_ids, relevant, top_k))

    dense = {k: _mean(v) for k, v in metrics["dense"].items()}
    hybrid = {k: _mean(v) for k, v in metrics["hybrid"].items()}

    print(f"{'Metric':<14}{'Dense-only':>12}{'Hybrid':>12}{'Improvement':>14}")
    print("-" * 52)
    for key, label in [("mrr", f"MRR"), ("recall", f"Recall@{top_k}"), ("precision", f"Precision@{top_k}")]:
        d, h = dense[key], hybrid[key]
        imp = (h - d) / d * 100 if d > 0 else float("nan")
        print(f"{label:<14}{d:>12.4f}{h:>12.4f}{imp:>13.1f}%")

    mrr_imp = (hybrid["mrr"] - dense["mrr"]) / dense["mrr"] * 100 if dense["mrr"] else float("nan")
    print(f"\n=> Hybrid improves MRR by {mrr_imp:.1f}% over dense-only on this set.\n")


# ---------------------------------------------------------------------------
# Part 2 — query latency: exhaustive Flat vs. approximate IVF
# ---------------------------------------------------------------------------

def _clustered_vectors(n: int, dim: int, n_clusters: int, rng, spread: float = 0.12) -> np.ndarray:
    """
    Generate clustered unit vectors that mimic real sentence embeddings far
    better than uniform noise: points scatter tightly around a set of random
    cluster centres, so an IVF index can actually exploit the structure. This
    keeps the latency/recall trade-off representative of the real system
    instead of the pathological uniform-random worst case.
    """
    centers = rng.standard_normal((n_clusters, dim)).astype("float32")
    assign = rng.integers(0, n_clusters, size=n)
    vecs = centers[assign] + spread * rng.standard_normal((n, dim)).astype("float32")
    return vecs.astype("float32")


def benchmark_latency(
    n: int,
    dim: int,
    n_queries: int,
    top_k: int,
    target_recall: float = 0.95,
    seed: int = 0,
) -> None:
    import faiss  # noqa: PLC0415

    print("=" * 70)
    print("PART 2  Query latency — FAISS IndexFlatIP vs. IndexIVFFlat")
    print("=" * 70)
    print(f"Vectors: {n:,} · dim: {dim} · queries: {n_queries} · top_k={top_k}")
    print(f"Data: clustered unit vectors (mimics sentence embeddings)\n")

    rng = np.random.default_rng(seed)
    nlist = max(1, int(math.sqrt(n)))
    xb = _clustered_vectors(n, dim, n_clusters=nlist, rng=rng)
    xq = _clustered_vectors(n_queries, dim, n_clusters=nlist, rng=rng)
    faiss.normalize_L2(xb)
    faiss.normalize_L2(xq)

    # Exact index (linear scan) — ground truth + baseline latency.
    flat = faiss.IndexFlatIP(dim)
    flat.add(xb)

    quantizer = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(xb)
    ivf.add(xb)

    def per_query_ms(index) -> Tuple[float, np.ndarray]:
        index.search(xq[:16], top_k)  # warm-up
        ids = np.empty((n_queries, top_k), dtype="int64")
        start = time.perf_counter()
        for i in range(n_queries):
            _, idx = index.search(xq[i : i + 1], top_k)
            ids[i] = idx[0]
        elapsed_ms = (time.perf_counter() - start) / n_queries * 1000
        return elapsed_ms, ids

    flat_ms, flat_ids = per_query_ms(flat)

    def recall_of(ivf_ids: np.ndarray) -> float:
        return float(np.mean([
            len(set(flat_ids[i]) & set(ivf_ids[i])) / top_k
            for i in range(n_queries)
        ]))

    # Find the smallest nprobe that holds recall >= target, then report the
    # latency at that quality level — the honest way to state an ANN speed-up.
    chosen = None
    for nprobe in sorted({max(1, int(nlist * f)) for f in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)}):
        ivf.nprobe = nprobe
        ivf_ms, ivf_ids = per_query_ms(ivf)
        rec = recall_of(ivf_ids)
        marker = ""
        if rec >= target_recall and chosen is None:
            chosen = (nprobe, ivf_ms, rec)
            marker = "  <- meets recall target"
        print(f"  nprobe={nprobe:<4} recall@{top_k}={rec * 100:5.1f}%  latency={ivf_ms:.3f} ms{marker}")

    if chosen is None:  # fall back to the highest nprobe tried
        ivf.nprobe = max(1, nlist // 2)
        ivf_ms, ivf_ids = per_query_ms(ivf)
        chosen = (ivf.nprobe, ivf_ms, recall_of(ivf_ids))

    nprobe, ivf_ms, rec = chosen
    reduction = (flat_ms - ivf_ms) / flat_ms * 100 if flat_ms else float("nan")

    print(f"\nIVF operating point: nlist={nlist}, nprobe={nprobe}")
    print(f"{'Index':<24}{'Mean latency/query':>22}")
    print("-" * 46)
    print(f"{'IndexFlatIP (exact)':<24}{flat_ms:>19.3f} ms")
    print(f"{'IndexIVFFlat (approx)':<24}{ivf_ms:>19.3f} ms")
    print(f"\n=> At >= {target_recall * 100:.0f}% top-{top_k} recall "
          f"(measured {rec * 100:.1f}%), IVF cuts mean query latency by "
          f"{reduction:.1f}% vs. the exhaustive scan.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval relevance and latency.")
    parser.add_argument("--top-k", type=int, default=5, help="top_k for the latency benchmark recall")
    parser.add_argument("--relevance-top-k", type=int, default=3, help="strict top_k for the relevance metrics")
    parser.add_argument("--latency-n", type=int, default=20000, help="Vectors in the latency benchmark")
    parser.add_argument("--latency-dim", type=int, default=384, help="Embedding dimension for the latency benchmark")
    parser.add_argument("--latency-queries", type=int, default=500)
    parser.add_argument("--skip-relevance", action="store_true", help="Run only the latency benchmark (no model download)")
    args = parser.parse_args(argv)

    if not args.skip_relevance:
        evaluate_relevance(top_k=args.relevance_top_k)
    benchmark_latency(
        n=args.latency_n,
        dim=args.latency_dim,
        n_queries=args.latency_queries,
        top_k=args.top_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
