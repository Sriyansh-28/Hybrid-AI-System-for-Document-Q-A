"""Centralized configuration dataclasses for the Hybrid AI System."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50
    # Split on these sentence/paragraph boundaries before hard-splitting on char count
    sentence_separators: list = field(
        default_factory=lambda: ["\n\n", "\n", ". ", "! ", "? ", "; "]
    )


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding model."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    max_seq_length: int = 256
    normalize_embeddings: bool = True
    cache_embeddings: bool = True


@dataclass
class RetrievalConfig:
    """Configuration for the hybrid retriever."""

    top_k: int = 5
    # Weight for dense (FAISS) score in hybrid fusion  [0, 1]
    dense_weight: float = 0.6
    # Weight for sparse (BM25) score in hybrid fusion [0, 1]
    sparse_weight: float = 0.4
    # Number of candidates fetched from each retriever before fusion
    dense_candidates: int = 20
    sparse_candidates: int = 20
    # Reciprocal Rank Fusion constant
    rrf_k: int = 60


@dataclass
class GeneratorConfig:
    """Configuration for the answer generator."""

    model_name: str = "google/flan-t5-base"
    max_new_tokens: int = 256
    temperature: float = 0.3
    num_beams: int = 4
    device: str = "cpu"


@dataclass
class FAISSConfig:
    """Configuration for the FAISS vector store."""

    # IVF nlist – number of Voronoi cells; rule-of-thumb: sqrt(N)
    nlist: int = 100
    # How many cells to probe at query time (trade-off speed vs. recall)
    nprobe: int = 10
    embedding_dim: int = 384
    use_gpu: bool = False
    index_path: Optional[str] = None


@dataclass
class SystemConfig:
    """Top-level system configuration."""

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    # Directory that contains documents to index
    docs_dir: Optional[str] = None
    # Where to persist the FAISS index and metadata
    index_dir: Optional[str] = None
