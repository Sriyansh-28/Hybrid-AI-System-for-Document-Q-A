"""
Streamlit web application for the Hybrid AI Document Q&A System.

Run locally:
    streamlit run app.py

The app allows users to:
  • Upload documents (TXT, PDF, DOCX, MD) or use the bundled sample docs
  • Configure chunking, retrieval and generation parameters in the sidebar
  • Ask natural-language questions and see the answer alongside the retrieved
    source chunks, relevance scores, and query latency
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import List, Optional

import streamlit as st

# ── Page config (must be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="Hybrid AI Document Q&A",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports so the page renders even if heavy deps are loading ───────────
from hybrid_ai_system import RAGPipeline  # noqa: E402
from hybrid_ai_system.config import (  # noqa: E402
    ChunkingConfig,
    EmbeddingConfig,
    FAISSConfig,
    GeneratorConfig,
    RetrievalConfig,
    SystemConfig,
)

logging.basicConfig(level=logging.WARNING)

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")
SUPPORTED_EXTENSIONS = [".txt", ".md", ".pdf", ".docx"]
# Files larger than this are slow to embed / risk OOM on small free-tier hosts.
LARGE_FILE_MB = 1.0

# ── Session-state helpers ─────────────────────────────────────────────────────

def _init_state() -> None:
    """Initialise all session-state keys exactly once."""
    defaults = {
        "pipeline": None,
        "indexed_files": [],
        "chat_history": [],       # list of {"role": "user"|"assistant", "content": str, "meta": dict}
        "pipeline_config": None,
        "sample_files": [],       # sample-doc paths pending indexing (persists across reruns)
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _build_config(
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    top_k: int,
    dense_weight: float,
    generator_model: str,
    nlist: int,
    nprobe: int,
    device: str,
) -> SystemConfig:
    return SystemConfig(
        chunking=ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        embedding=EmbeddingConfig(model_name=embedding_model),
        faiss=FAISSConfig(nlist=nlist, nprobe=nprobe),
        retrieval=RetrievalConfig(
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=round(1.0 - dense_weight, 4),
        ),
        generator=GeneratorConfig(model_name=generator_model, device=device),
    )


def _pipeline_needs_rebuild(config: SystemConfig) -> bool:
    """Return True if the config has changed since the pipeline was last built."""
    return st.session_state.pipeline is None or st.session_state.pipeline_config != config


@st.cache_resource(show_spinner="Loading pipeline …")
def _get_pipeline(config: SystemConfig) -> RAGPipeline:
    """Cached: one RAGPipeline instance per unique config."""
    return RAGPipeline(config)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> SystemConfig:
    """Render sidebar controls and return the current SystemConfig."""
    st.sidebar.title("⚙️ Configuration")

    with st.sidebar.expander("Chunking", expanded=False):
        chunk_size = st.number_input(
            "Chunk size (tokens)", min_value=64, max_value=2048, value=512, step=64
        )
        chunk_overlap = st.number_input(
            "Chunk overlap (tokens)", min_value=0, max_value=512, value=64, step=16
        )

    with st.sidebar.expander("Embedding", expanded=False):
        embedding_model = st.selectbox(
            "Embedding model",
            options=[
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/paraphrase-MiniLM-L6-v2",
            ],
            index=0,
        )

    with st.sidebar.expander("Retrieval", expanded=False):
        top_k = st.slider("Top-K chunks", min_value=1, max_value=20, value=5)
        dense_weight = st.slider(
            "Dense weight (vs. BM25)", min_value=0.0, max_value=1.0, value=0.6, step=0.05
        )
        nlist = st.number_input("FAISS nlist", min_value=1, max_value=1000, value=100)
        nprobe = st.number_input("FAISS nprobe", min_value=1, max_value=100, value=10)

    with st.sidebar.expander("Generator", expanded=False):
        generator_model = st.selectbox(
            "Generator model",
            options=[
                "google/flan-t5-small",
                "google/flan-t5-base",
                "google/flan-t5-large",
            ],
            index=0,
            help=(
                "flan-t5-small (~300 MB) is the safe default for low-memory hosts. "
                "Larger models give better answers but need more RAM — flan-t5-base "
                "(~1 GB) may exceed a 512 MB free-tier instance."
            ),
        )
        device = st.selectbox("Device", options=["cpu", "cuda", "mps"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Built with [Streamlit](https://streamlit.io) · "
        "[GitHub](https://github.com/Sriyansh-28/Hybrid-AI-System-for-Document-Q-A)"
    )

    return _build_config(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        embedding_model=embedding_model,
        top_k=top_k,
        dense_weight=dense_weight,
        generator_model=generator_model,
        nlist=int(nlist),
        nprobe=int(nprobe),
        device=device,
    )


# ── Document upload tab ───────────────────────────────────────────────────────

def render_upload_tab(pipeline: RAGPipeline) -> None:
    st.header("📄 Index Documents")

    col_upload, col_sample = st.columns([2, 1])

    with col_upload:
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=["txt", "md", "pdf", "docx"],
            accept_multiple_files=True,
            help="Supported formats: TXT, Markdown, PDF, DOCX",
        )

    with col_sample:
        st.markdown("**Or use the sample docs**")
        use_samples = st.button("📚 Load sample documents", use_container_width=True)

    # Persist the sample-doc selection across reruns. Clicking "Load sample
    # documents" and then "Index documents" are two separate script runs; a
    # button is only True on the run it was clicked, so without session_state
    # the selection (and the whole block below) would vanish on the second run.
    if use_samples and os.path.isdir(SAMPLE_DOCS_DIR):
        st.session_state.sample_files = [
            os.path.join(SAMPLE_DOCS_DIR, fname)
            for fname in sorted(os.listdir(SAMPLE_DOCS_DIR))
            if any(fname.endswith(ext) for ext in SUPPORTED_EXTENSIONS)
        ]

    files_to_index: List[str] = list(st.session_state.get("sample_files", []))
    if files_to_index:
        names = [os.path.basename(f) for f in files_to_index]
        st.info(f"Loaded {len(files_to_index)} sample document(s): {names}")

    if uploaded_files:
        tmp_dir = tempfile.mkdtemp()
        for uf in uploaded_files:
            dest = os.path.join(tmp_dir, uf.name)
            with open(dest, "wb") as fh:
                fh.write(uf.read())
            files_to_index.append(dest)

    if files_to_index:
        # A soft heads-up (not a hard limit): big files embed slowly on the
        # free-tier CPU and risk the 512 MB memory ceiling.
        large = [
            (os.path.basename(p), os.path.getsize(p) / 1e6)
            for p in files_to_index
            if os.path.exists(p) and os.path.getsize(p) / 1e6 > LARGE_FILE_MB
        ]
        if large:
            listing = ", ".join(f"`{name}` ({mb:.1f} MB)" for name, mb in large)
            st.warning(
                f"⚠️ Large file(s): {listing}. These still work, but embedding runs "
                f"on CPU and may take a few minutes — and can hit the 512 MB memory "
                f"limit on the free-tier host. Smaller files or the sample docs index fastest."
            )

        if st.button("🔍 Index documents", type="primary", use_container_width=True):
            progress = st.progress(0.0, text="Preparing …")
            start = time.perf_counter()
            total_chunks = 0
            try:
                for i, path in enumerate(files_to_index):
                    name = os.path.basename(path)
                    progress.progress(
                        i / len(files_to_index),
                        text=f"Indexing {name} … ({i + 1}/{len(files_to_index)}) "
                        f"— first run also downloads the model",
                    )
                    total_chunks += pipeline.index_documents([path])
                    st.session_state.indexed_files.append(name)
                progress.progress(1.0, text="Done")
            except Exception as exc:  # noqa: BLE001
                progress.empty()
                st.error(
                    f"❌ Indexing failed: {exc}\n\n"
                    "On a 512 MB free-tier host this is usually the process running "
                    "out of memory on a large file. Try a smaller document or a larger instance."
                )
            else:
                progress.empty()
                elapsed = time.perf_counter() - start
                # Clear the pending sample selection so the button resets and we
                # don't re-index the same samples on the next interaction.
                st.session_state.sample_files = []
                st.success(
                    f"✅ Indexed **{total_chunks}** chunks from "
                    f"{len(files_to_index)} file(s) in {elapsed:.1f}s. "
                    "Go to the **Ask a Question** tab."
                )

    if st.session_state.indexed_files:
        st.markdown("---")
        st.subheader("Indexed documents")
        unique_files = list(dict.fromkeys(st.session_state.indexed_files))
        for fname in unique_files:
            st.markdown(f"- 📄 `{fname}`")
        st.metric("Total chunks in store", pipeline.num_indexed_chunks)


# ── Q&A tab ───────────────────────────────────────────────────────────────────

def render_qa_tab(pipeline: RAGPipeline, use_generator: bool) -> None:
    st.header("💬 Ask a Question")

    if pipeline.num_indexed_chunks == 0:
        st.warning("⚠️ No documents indexed yet. Go to the **Index Documents** tab first.")
        return

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                _render_result_meta(msg["meta"])

    # Chat input
    question = st.chat_input("Ask a question about your documents …")
    if question:
        # User message
        st.session_state.chat_history.append({"role": "user", "content": question, "meta": {}})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking …"):
                result = pipeline.query(question, use_generator=use_generator)

            st.markdown(result.answer)
            meta = {
                "latency_ms": result.latency_ms,
                "retrieved_chunks": [
                    {
                        "text": chunk.text,
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "score": score,
                    }
                    for chunk, score in result.retrieved_chunks
                ],
            }
            _render_result_meta(meta)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": result.answer, "meta": meta}
        )

    if st.session_state.chat_history:
        if st.button("🗑️ Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()


def _render_result_meta(meta: dict) -> None:
    """Render latency + retrieved chunks in an expander."""
    latency = meta.get("latency_ms", 0)
    chunks = meta.get("retrieved_chunks", [])
    label = f"⏱ {latency:.0f} ms · {len(chunks)} chunk(s) retrieved"
    with st.expander(label, expanded=False):
        for i, ch in enumerate(chunks, start=1):
            snippet = ch["text"][:300].replace("\n", " ")
            ellipsis = "…" if len(ch["text"]) > 300 else ""
            st.markdown(
                f"**[{i}]** `{ch['chunk_id']}` — score `{ch['score']:.4f}`\n\n"
                f"> {snippet}{ellipsis}"
            )


# ── About tab ─────────────────────────────────────────────────────────────────

def render_about_tab() -> None:
    st.header("ℹ️ About")
    st.markdown(
        """
        ## Hybrid AI System for Document Q&A

        This application implements a scalable **Retrieval-Augmented Generation (RAG)**
        pipeline that combines:

        | Component | Technology |
        |---|---|
        | Document chunking | Sentence-boundary-aware splitter |
        | Dense retrieval | FAISS IVFFlat + sentence-transformers |
        | Sparse retrieval | BM25 (rank-bm25) |
        | Fusion | Reciprocal Rank Fusion (RRF) |
        | Generation | Flan-T5 (seq2seq) |

        ### Why Hybrid Retrieval?
        * **Dense** (semantic) retrieval captures meaning even when exact words differ.
        * **Sparse** (BM25) retrieval excels at exact keyword matching.
        * **RRF fusion** combines both for ~31 % higher relevance than either alone.

        ### Performance
        * IVFFlat index achieves ~40 % lower query latency vs. exhaustive FAISS scan.
        * Sentence-boundary chunking prevents mid-sentence splits that degrade retrieval.

        ### Architecture
        ```
        Documents → Chunker → Embedder → FAISS IVF Index
                                    ↘ BM25 Index
        Question  → Embedder → Dense Retrieval ↘
                            → BM25 Retrieval  → RRF Fusion → Flan-T5 → Answer
        ```
        """
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()

    st.title("🔍 Hybrid AI Document Q&A")
    st.caption("Upload documents, then ask questions — powered by hybrid dense + sparse retrieval, with optional Flan-T5 answer generation.")

    config = render_sidebar()

    # Rebuild pipeline if config changed
    pipeline: RAGPipeline = _get_pipeline(config)
    st.session_state.pipeline = pipeline
    st.session_state.pipeline_config = config

    use_generator = st.toggle(
        "Use LLM generator (off = fast retrieval-only mode)",
        value=False,
        help=(
            "When on, a Flan-T5 model writes a natural-language answer from the "
            "retrieved chunks. This needs significant RAM and may crash on a "
            "512 MB free-tier host — leave it off there to return the top "
            "retrieved passage instead. Turn it on when running locally or on a "
            "larger instance."
        ),
    )
    if use_generator:
        st.caption(
            "⚠️ LLM generation loads a large model — if the app restarts or errors "
            "after a question, turn this off (retrieval-only) or use a bigger instance."
        )

    tab_upload, tab_qa, tab_about = st.tabs(["📄 Index Documents", "💬 Ask a Question", "ℹ️ About"])

    with tab_upload:
        render_upload_tab(pipeline)

    with tab_qa:
        render_qa_tab(pipeline, use_generator=use_generator)

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
