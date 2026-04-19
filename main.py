#!/usr/bin/env python
"""
Command-line interface for the Hybrid AI Document Q&A System.

Examples
--------
Index a directory and answer a question:

    python main.py --docs ./sample_docs --query "What is FAISS?"

Index and persist to disk for later reuse:

    python main.py --docs ./sample_docs --save ./index --query "How does RAG work?"

Load a previously saved index:

    python main.py --load ./index --query "What is BM25?"
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from hybrid_ai_system import RAGPipeline
from hybrid_ai_system.config import (
    ChunkingConfig,
    EmbeddingConfig,
    FAISSConfig,
    GeneratorConfig,
    RetrievalConfig,
    SystemConfig,
)


def build_config(args: argparse.Namespace) -> SystemConfig:
    return SystemConfig(
        chunking=ChunkingConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        ),
        embedding=EmbeddingConfig(
            model_name=args.embedding_model,
        ),
        faiss=FAISSConfig(
            nlist=args.nlist,
            nprobe=args.nprobe,
        ),
        retrieval=RetrievalConfig(
            top_k=args.top_k,
            dense_weight=args.dense_weight,
            sparse_weight=1.0 - args.dense_weight,
        ),
        generator=GeneratorConfig(
            model_name=args.generator_model,
            device=args.device,
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Hybrid AI System for Document Question Answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__),
    )

    # ---- I/O ----------------------------------------------------------------
    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument("--docs", metavar="DIR", help="Directory of documents to index")
    io_group.add_argument("--files", nargs="+", metavar="FILE", help="Individual files to index")
    io_group.add_argument("--save", metavar="DIR", help="Persist the index to this directory")
    io_group.add_argument("--load", metavar="DIR", help="Load a previously saved index")
    io_group.add_argument("--query", "-q", metavar="TEXT", help="Question to answer")

    # ---- Chunking -----------------------------------------------------------
    chunk_group = parser.add_argument_group("Chunking")
    chunk_group.add_argument("--chunk-size", type=int, default=512, metavar="N")
    chunk_group.add_argument("--chunk-overlap", type=int, default=64, metavar="N")

    # ---- Embedding ----------------------------------------------------------
    emb_group = parser.add_argument_group("Embedding")
    emb_group.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        metavar="NAME",
    )

    # ---- FAISS --------------------------------------------------------------
    faiss_group = parser.add_argument_group("FAISS")
    faiss_group.add_argument("--nlist", type=int, default=100, metavar="N",
                             help="Number of IVF Voronoi cells (default: 100)")
    faiss_group.add_argument("--nprobe", type=int, default=10, metavar="N",
                             help="IVF cells probed at query time (default: 10)")

    # ---- Retrieval ----------------------------------------------------------
    ret_group = parser.add_argument_group("Retrieval")
    ret_group.add_argument("--top-k", type=int, default=5, metavar="K")
    ret_group.add_argument("--dense-weight", type=float, default=0.6, metavar="W",
                           help="Dense retriever weight [0,1] (default: 0.6)")

    # ---- Generator ----------------------------------------------------------
    gen_group = parser.add_argument_group("Generator")
    gen_group.add_argument("--generator-model", default="google/flan-t5-base", metavar="NAME")
    gen_group.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    gen_group.add_argument("--no-generate", action="store_true",
                           help="Skip the LLM; return top-1 retrieved chunk as answer")

    args = parser.parse_args(argv)

    # ---- Validate -----------------------------------------------------------
    if not args.load and not args.docs and not args.files:
        parser.error("Provide --docs, --files, or --load to supply documents.")

    if not args.query:
        parser.error("Provide a --query to answer.")

    # ---- Build pipeline -----------------------------------------------------
    config = build_config(args)
    pipeline = RAGPipeline(config)

    if args.load:
        print(f"Loading index from '{args.load}' …")
        pipeline.load(args.load)

    if args.docs:
        print(f"Indexing documents from '{args.docs}' …")
        n = pipeline.index_directory(args.docs, show_progress=True)
        print(f"Indexed {n} chunks.")

    if args.files:
        print(f"Indexing {len(args.files)} file(s) …")
        n = pipeline.index_documents(args.files, show_progress=True)
        print(f"Indexed {n} chunks.")

    if args.save:
        print(f"Saving index to '{args.save}' …")
        pipeline.save(args.save)

    # ---- Answer -------------------------------------------------------------
    use_gen = not args.no_generate
    result = pipeline.query(args.query, use_generator=use_gen)

    print("\n" + "=" * 60)
    print(f"Question : {result.question}")
    print(f"Answer   : {result.answer}")
    print(f"Latency  : {result.latency_ms:.1f} ms")
    print("-" * 60)
    print("Retrieved chunks:")
    for rank, (chunk, score) in enumerate(result.retrieved_chunks, start=1):
        snippet = chunk.text[:120].replace("\n", " ")
        print(f"  [{rank}] (score={score:.4f}) [{chunk.chunk_id}] {snippet}…")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
