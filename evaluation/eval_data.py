"""
Labeled evaluation set for retrieval quality.

Each corpus entry is a (doc_id, text) pair; each query maps to the doc_id(s)
that genuinely answer it. The set is built to be *discriminating* rather than
easy, so it reveals where hybrid retrieval helps and where it does not.

It contains three kinds of material:

- **Confusable clusters** — groups of near-identical passages that differ only
  by a rare exact token (an error code, a model number, an API name, a
  citation). Dense embeddings see the whole cluster as similar and often rank
  the wrong member first; lexical BM25 locks onto the exact token. These are
  the cases hybrid fusion is meant to fix.
- **Exact-term queries** targeting those rare tokens.
- **Semantic queries** whose wording differs from the source, where dense
  retrieval is already strong, so the set stays balanced and the aggregate
  number is not cherry-picked toward either method.

Evaluated at a strict top-3 (see ``evaluate_retrieval.py``), where ranking
mistakes actually cost metric points.
"""

from __future__ import annotations

from typing import List, Tuple

CORPUS: List[Tuple[str, str]] = [
    # ── Confusable cluster A: error codes (near-identical wording) ──────────
    ("err_4471", "System diagnostics: error code E-4471 signals a checksum mismatch detected in the payload header during transfer validation."),
    ("err_2203", "System diagnostics: error code E-2203 signals a timeout while waiting for the payload header acknowledgement during transfer validation."),
    ("err_9910", "System diagnostics: error code E-9910 signals a malformed payload header field rejected by the transfer validation stage."),

    # ── Confusable cluster B: product model numbers ────────────────────────
    ("model_x100", "The Aurora X100 sensor module offers a 12-bit ADC, 100 Hz sampling, and a rated operating range of minus 20 to 60 degrees Celsius."),
    ("model_x200", "The Aurora X200 sensor module offers a 16-bit ADC, 400 Hz sampling, and a rated operating range of minus 40 to 85 degrees Celsius."),
    ("model_x300", "The Aurora X300 sensor module offers a 24-bit ADC, 1 kHz sampling, and a rated operating range of minus 55 to 125 degrees Celsius."),

    # ── Confusable cluster C: API endpoints ────────────────────────────────
    ("api_index", "The /index/upload endpoint accepts multipart file uploads and adds their chunks to the retrieval store, returning the number of chunks indexed."),
    ("api_query", "The /query endpoint accepts a question and an optional top_k, runs hybrid retrieval, and returns the answer with the retrieved passages."),
    ("api_stats", "The /stats endpoint reports how many chunks are currently indexed along with the active embedding and generator model names."),

    # ── Confusable cluster D: citations / identifiers ──────────────────────
    ("rfc_7519", "RFC 7519 defines the JSON Web Token (JWT) format, a compact, URL-safe means of representing claims transferred between two parties."),
    ("rfc_6749", "RFC 6749 defines the OAuth 2.0 authorization framework, which lets a third-party application obtain limited access to an HTTP service."),
    ("rfc_8446", "RFC 8446 defines version 1.3 of the Transport Layer Security (TLS) protocol, reducing handshake round trips and removing legacy ciphers."),

    # ── Distinct technical docs (semantic queries target these) ────────────
    ("faiss", "FAISS is a library from Meta AI for efficient similarity search and clustering of dense vectors, with approximate nearest-neighbour indexes for scale."),
    ("bm25", "BM25 is a probabilistic ranking function that scores documents by exact term frequency and inverse document frequency, the backbone of lexical search."),
    ("rrf", "Reciprocal Rank Fusion merges several ranked lists by summing the reciprocals of each item's rank, needing no score calibration between systems."),
    ("rag", "Retrieval-Augmented Generation grounds a language model's answer in passages fetched from an external corpus, reducing hallucination on factual questions."),
    ("chunking", "Chunking splits long documents into overlapping passages sized for the embedding model so retrieval returns coherent, self-contained context."),
    ("transformer", "The Transformer architecture relies entirely on self-attention to relate tokens, dispensing with the recurrence of earlier sequence models."),
    ("amazon_river", "The Amazon carries more water than any other river, discharging into the Atlantic through a vast delta after crossing the South American continent."),
    ("photosynthesis", "Photosynthesis turns carbon dioxide and water into glucose and oxygen using light energy captured by chlorophyll inside plant cells."),
    ("gpu", "A graphics processing unit performs many matrix and vector operations in parallel, which is why it accelerates the training of deep neural networks."),
    ("eiffel", "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, completed in 1889 for the World's Fair."),
]

# (query, [relevant doc_ids])
QUERIES: List[Tuple[str, List[str]]] = [
    # Exact-term queries into the confusable clusters (hybrid should help).
    ("What does error code E-4471 mean?", ["err_4471"]),
    ("meaning of error E-2203", ["err_2203"]),
    ("error E-9910 cause", ["err_9910"]),
    ("sampling rate and ADC of the Aurora X200", ["model_x200"]),
    ("operating temperature range of the Aurora X300", ["model_x300"]),
    ("specs for the Aurora X100 module", ["model_x100"]),
    ("what does the /stats endpoint return", ["api_stats"]),
    ("which endpoint handles multipart file uploads", ["api_index"]),
    ("what is defined in RFC 7519", ["rfc_7519"]),
    ("which RFC specifies OAuth 2.0", ["rfc_6749"]),
    ("RFC 8446 TLS 1.3 handshake", ["rfc_8446"]),

    # Semantic queries where dense retrieval is already strong (balance).
    ("library from Meta for vector similarity search", ["faiss"]),
    ("how to stop a language model from making facts up", ["rag"]),
    ("splitting long documents into overlapping passages", ["chunking"]),
    ("which river discharges the most water into the ocean", ["amazon_river"]),
    ("what hardware speeds up training neural networks", ["gpu"]),
    ("architecture based entirely on self-attention", ["transformer"]),
    ("how plants convert sunlight into chemical energy", ["photosynthesis"]),
]
