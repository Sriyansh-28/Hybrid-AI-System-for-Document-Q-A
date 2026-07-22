# Hybrid AI System for Document Q&A

A scalable **Retrieval-Augmented Generation (RAG)** pipeline for automated document search and question answering, featuring hybrid dense + sparse retrieval and a FAISS-based vector store.

---

## Features

| Capability | Implementation |
|---|---|
| Multi-format document loading | TXT, Markdown, PDF (pdfplumber), DOCX (python-docx) |
| Smart chunking with overlap | Sentence-boundary-aware splits, configurable size & overlap |
| Dense retrieval | FAISS IVFFlat index — ~40 % lower query latency vs. linear scan |
| Sparse retrieval | BM25 (rank-bm25) for exact keyword matching |
| **Hybrid fusion** | Reciprocal Rank Fusion (RRF) combining both signals |
| Answer generation | Seq2seq LLM (default: Flan-T5) with retrieved context |
| Persistence | Save/load FAISS index & chunk metadata to disk |

---

## Architecture

```
Documents (TXT / PDF / DOCX)
        │
        ▼
┌─────────────────────┐
│  DocumentProcessor  │  sentence-boundary chunking + overlap
└─────────┬───────────┘
          │ Chunks
          ▼
┌─────────────────────┐        ┌─────────────────┐
│   EmbeddingModel    │──────▶ │  FAISSVectorStore│  IVFFlat index
│ (sentence-transformers)      │  (dense search)  │  nprobe ≪ nlist
└─────────────────────┘        └────────┬────────┘
                                         │
┌─────────────────────┐                 │  dense top-k
│     BM25Index       │                 │
│   (sparse search)   │─────────────────┤  sparse top-k
└─────────────────────┘                 │
                                         ▼
                               ┌─────────────────────┐
                               │   HybridRetriever   │  RRF fusion
                               └─────────┬───────────┘
                                         │ top-k chunks
                                         ▼
                               ┌─────────────────────┐
                               │   AnswerGenerator   │  Flan-T5 / any seq2seq
                               └─────────┬───────────┘
                                         │
                                         ▼
                                      Answer
```

### Retrieval relevance

Hybrid retrieval (RRF fusion of FAISS + BM25) combines:
- **Dense signals**: semantic similarity via L2-normalised sentence-transformer embeddings.
- **Sparse signals**: exact term matching via BM25 probabilistic scoring.

The `evaluation/` harness scores dense-only vs. hybrid on a labeled query set
(MRR / Recall@k / Precision@k). Hybrid matches or beats dense-only, with the
clearest gains on **exact-term queries** — error codes, model numbers,
identifiers — where lexical BM25 catches literal matches that pure semantic
search can drift away from. The size of the gain is corpus-dependent: a strong
dense model already saturates on a small, well-separated corpus, and the hybrid
advantage widens as the corpus grows and contains more lexically-confusable
passages. Run `python -m evaluation.evaluate_retrieval` to reproduce the numbers
on the bundled set.

### Query latency reduction

Using FAISS `IndexIVFFlat` (`nlist ≈ √N` Voronoi cells, `nprobe` tuned to hold
recall) instead of an exhaustive `IndexFlatIP` scan trades a little recall for
much lower latency. Benchmarked by the same harness on 20k clustered 384-d
vectors, IVF cuts **mean query latency by ~65–70 %** while **retaining ≥95 % of
the exact top-5 results** — reproduce with
`python -m evaluation.evaluate_retrieval --skip-relevance`.

---

## Project Structure

```
.
├── hybrid_ai_system/
│   ├── __init__.py
│   ├── config.py              # Dataclass configuration for all components
│   ├── document_processor.py  # Multi-format loading & smart chunking
│   ├── embeddings.py          # Sentence-transformer embeddings + caching
│   ├── vector_store.py        # FAISS IVFFlat index (build, search, save/load)
│   ├── retriever.py           # BM25Index + HybridRetriever (RRF fusion)
│   ├── generator.py           # Seq2seq answer generation (Flan-T5)
│   ├── pipeline.py            # End-to-end RAGPipeline
│   ├── api.py                 # FastAPI REST service
│   └── utils.py               # Timing, deduplication, MRR/Recall/Precision metrics
├── tests/
│   ├── conftest.py            # MockEmbeddingModel (no network required)
│   ├── test_document_processor.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_retriever.py
│   ├── test_pipeline.py
│   └── test_api.py
├── sample_docs/               # Example documents
├── .streamlit/
│   └── config.toml            # Streamlit server & theme settings
├── .github/workflows/ci.yml   # Test suite + Docker build on push/PR
├── app.py                     # Streamlit web UI
├── Dockerfile                 # Container image for the Streamlit UI
├── Dockerfile.api              # Container image for the FastAPI service
├── docker-compose.yml         # Run API + UI together locally
├── render.yaml                 # Render.com deployment blueprint
├── main.py                    # CLI entry-point
└── requirements.txt
```

---

## REST API

A FastAPI service (`hybrid_ai_system/api.py`) exposes the pipeline over HTTP for
chatbot-style integration — no Streamlit dependency required.

### Run locally

```bash
pip install -r requirements.txt
uvicorn hybrid_ai_system.api:app --reload
```

Interactive docs: http://localhost:8000/docs

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/stats` | Indexed chunk count + active model config |
| `POST` | `/index/texts` | Index raw text passages (JSON body) |
| `POST` | `/index/upload` | Index uploaded files (multipart, TXT/PDF/DOCX/MD) |
| `POST` | `/query` | Ask a question against the indexed corpus |
| `DELETE` | `/index` | Reset the in-memory index |

### Example

```bash
curl -X POST http://localhost:8000/index/texts \
  -H "Content-Type: application/json" \
  -d '{"texts": ["FAISS is a library for efficient similarity search."]}'

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FAISS?", "top_k": 3}'
```

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Dense embedding model |
| `GENERATOR_MODEL` | `google/flan-t5-base` | Seq2seq answer generator |
| `DEVICE` | `cpu` | `cpu`, `cuda`, or `mps` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `64` | Chunking parameters |
| `FAISS_NLIST` / `FAISS_NPROBE` | `100` / `10` | FAISS IVF parameters |
| `TOP_K` / `DENSE_WEIGHT` | `5` / `0.6` | Retrieval fusion parameters |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |

> The API holds a single in-memory index (process-wide singleton) — suitable
> for a demo/single-tenant deployment. For multi-tenant or persistent use,
> call `pipeline.save()`/`.load()` against a mounted volume or object store.

---

## Web Deployment

### Run locally with Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser.

The web UI provides:
- **Index Documents** tab — drag-and-drop TXT/PDF/DOCX/MD files or load the bundled sample docs
- **Ask a Question** tab — conversational chat interface with source-chunk attribution and latency display
- **Sidebar** — live controls for chunking, retrieval weights, FAISS parameters, and the generator model

### Run with Docker

```bash
# Build the image
docker build -t hybrid-ai-qa .

# Run (port 8501)
docker run -p 8501:8501 hybrid-ai-qa
```

### Deploy to Streamlit Community Cloud (free)

1. Fork this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select the forked repo → set **Main file path** to `app.py`.
4. Click **Deploy**. The app will be live at `https://<your-app>.streamlit.app`.

> **Note**: First load downloads the sentence-transformer and Flan-T5 models (~500 MB).
> Streamlit Community Cloud has a 2 GB RAM limit; use `flan-t5-small` if you hit memory limits.

### Deploy to Hugging Face Spaces

```bash
# Install the HF CLI
pip install huggingface_hub

# Create a Space (SDK: streamlit)
huggingface-cli repo create hybrid-ai-qa --type space --space_sdk streamlit

# Push
git remote add space https://huggingface.co/spaces/<your-username>/hybrid-ai-qa
git push space main
```

### Run API + UI together with Docker Compose

```bash
docker compose up --build
```

This builds and runs both the FastAPI service (`:8000`) and the Streamlit UI (`:8501`).

### Deploy the REST API to Render (free tier)

1. Fork this repository.
2. In the [Render dashboard](https://dashboard.render.com), click **New → Blueprint** and point it at your fork — Render reads `render.yaml` and provisions both the API and UI services automatically.
3. Alternatively, create a single **Web Service** manually: runtime **Docker**, Dockerfile path `Dockerfile.api`, health check path `/health`.
4. The API will be live at `https://<service-name>.onrender.com`; interactive docs at `/docs`.

> Free-tier instances have limited RAM — `GENERATOR_MODEL=google/flan-t5-small` (set in `render.yaml`) keeps memory usage low. Free instances also spin down on idle, so the first request after inactivity will be slow (cold start + model load).

### CI/CD

`.github/workflows/ci.yml` runs the full `pytest` suite and builds both Docker
images on every push/PR to `main`. Render (and most PaaS providers) can be
configured to auto-deploy on push once CI passes.

---



```bash
pip install -r requirements.txt
```

GPU support (optional):
```bash
pip install faiss-gpu
```

---

## Usage

### Python API

```python
from hybrid_ai_system import RAGPipeline
from hybrid_ai_system.config import SystemConfig, ChunkingConfig, RetrievalConfig

config = SystemConfig(
    chunking=ChunkingConfig(chunk_size=512, chunk_overlap=64),
    retrieval=RetrievalConfig(top_k=5, dense_weight=0.6, sparse_weight=0.4),
)

pipeline = RAGPipeline(config)

# Index documents
pipeline.index_directory("./my_documents")        # from a directory
# or
pipeline.index_documents(["doc1.pdf", "doc2.txt"]) # specific files
# or
pipeline.index_texts(["Raw text passage …"], doc_ids=["doc_0"])

# Persist index
pipeline.save("./my_index")

# Load saved index
pipeline.load("./my_index")

# Query
result = pipeline.query("What is the capital of France?")
print(result.answer)
print(f"Retrieved {len(result.retrieved_chunks)} chunks in {result.latency_ms:.1f} ms")
```

### Command-Line Interface

```bash
# Index a directory and answer a question
python main.py --docs ./sample_docs --query "What is FAISS?"

# Retrieval only (skip LLM generation)
python main.py --docs ./sample_docs --query "How does hybrid retrieval work?" --no-generate

# Persist index and reuse
python main.py --docs ./sample_docs --save ./index --query "Describe chunking strategies."
python main.py --load ./index --query "What is BM25?"

# Tune retrieval weights
python main.py --docs ./sample_docs --query "RAG pipeline" \
  --dense-weight 0.7 --top-k 3 --nlist 50 --nprobe 5
```

---

## Configuration

All parameters are exposed through dataclasses in `hybrid_ai_system/config.py`:

| Dataclass | Key parameters |
|---|---|
| `ChunkingConfig` | `chunk_size`, `chunk_overlap`, `min_chunk_size` |
| `EmbeddingConfig` | `model_name`, `batch_size`, `normalize_embeddings`, `cache_embeddings` |
| `FAISSConfig` | `nlist`, `nprobe`, `embedding_dim`, `use_gpu` |
| `RetrievalConfig` | `top_k`, `dense_weight`, `sparse_weight`, `rrf_k` |
| `GeneratorConfig` | `model_name`, `max_new_tokens`, `temperature`, `device` |

---

## Running Tests

```bash
pytest tests/ -v
```

All 67 tests run offline using a deterministic `MockEmbeddingModel` (no model downloads needed).

---

## Evaluation Metrics

`hybrid_ai_system.utils` provides standard IR metrics:

```python
from hybrid_ai_system.utils import compute_mrr, compute_recall_at_k, compute_precision_at_k

mrr  = compute_mrr(retrieved_ids, relevant_ids)
r_at_5 = compute_recall_at_k(retrieved_ids, relevant_ids, k=5)
p_at_3 = compute_precision_at_k(retrieved_ids, relevant_ids, k=3)
```
