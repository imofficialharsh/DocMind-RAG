"""Application configuration management using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Production settings for DocMind Document Intelligence RAG."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM Provider Configuration
    llm_provider: Literal["groq", "gemini"] = Field(
        default="groq",
        description="LLM provider: 'groq' (default) or 'gemini'.",
    )
    groq_api_key: Optional[str] = Field(
        default=None,
        description="API key for Groq Cloud (configured on backend).",
    )
    groq_model: str = Field(
        default="openai/gpt-oss-120b",
        description="Groq model identifier (120B parameter model).",
    )
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="API key for Google Gemini.",
    )
    gemini_model: str = Field(
        default="gemini-3.5-flash",
        description="Google Gemini model identifier.",
    )

    # Local Embeddings & Reranker
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model for dense embeddings (runs locally).",
    )
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for re-ranking (runs locally).",
    )

    # Storage Paths
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma_db"),
        description="Directory for persistent ChromaDB storage.",
    )
    bm25_persist_dir: Path = Field(
        default=Path("./data/bm25"),
        description="Directory for serialized BM25 index files.",
    )
    cache_dir: Path = Field(
        default=Path("./data/cache"),
        description="Directory for cached parsed PDFs and hashes.",
    )

    # Document Size Limit
    max_file_size_mb: int = Field(
        default=100,
        description="Maximum allowed PDF file size in Megabytes.",
    )

    # Chunking Configuration
    chunk_size: int = Field(
        default=1000,
        description="Target maximum chunk size in characters.",
    )
    chunk_overlap: int = Field(
        default=200,
        description="Chunk overlap in characters.",
    )

    # Retrieval and Re-ranking Configuration
    top_k_dense: int = Field(
        default=30,
        description="Number of candidates retrieved from dense vector search.",
    )
    top_k_bm25: int = Field(
        default=30,
        description="Number of candidates retrieved from BM25 keyword search.",
    )
    rrf_k: int = Field(
        default=60,
        description="Reciprocal Rank Fusion smoothing constant k.",
    )
    top_k_final: int = Field(
        default=10,
        description="Final number of top chunks delivered to LLM after re-ranking.",
    )

    # API Protection & Rate Limiting
    rate_limit_seconds: int = Field(
        default=30,
        description="Minimum cooldown in seconds between consecutive user queries to protect server load and API quota.",
    )

    def ensure_directories(self) -> None:
        """Create necessary persistence and cache directories if they do not exist."""
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_persist_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a singleton cached instance of AppSettings."""
    settings = AppSettings()
    settings.ensure_directories()
    return settings

