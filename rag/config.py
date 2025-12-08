"""
config.py

Centralized configuration for the RAG pipeline.

This module provides dataclasses and utilities for managing pipeline configuration,
including paths, chunking parameters, embedding settings, and ChromaDB options.

Usage:
    from config import PathConfig, ChunkingConfig, EmbeddingConfig, get_default_config

    paths = PathConfig()
    chunking = ChunkingConfig(chunk_size=1500)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Path Configuration
# ============================================================================


@dataclass
class PathConfig:
    """Configuration for all pipeline paths."""

    # Base directories
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    output_dir: Path = field(default_factory=lambda: Path("output"))

    # Input/intermediate directories
    books_dir: Path = field(default_factory=lambda: Path("output/books"))
    clean_dir: Path = field(default_factory=lambda: Path("output/clean_books"))
    manifest_dir: Path = field(default_factory=lambda: Path("output/manifest"))
    chunks_dir: Path = field(default_factory=lambda: Path("output/chunks"))
    embeddings_dir: Path = field(default_factory=lambda: Path("output/embeddings"))
    metadata_dir: Path = field(default_factory=lambda: Path("output/metadata"))

    # ChromaDB
    chroma_dir: Path = field(default_factory=lambda: Path("chroma_db"))

    def __post_init__(self) -> None:
        """Apply environment variable overrides."""
        if os.getenv("OUTPUT_DIR"):
            self.output_dir = Path(os.environ["OUTPUT_DIR"])
            # Update derived paths
            self.books_dir = self.output_dir / "books"
            self.clean_dir = self.output_dir / "clean_books"
            self.manifest_dir = self.output_dir / "manifest"
            self.chunks_dir = self.output_dir / "chunks"
            self.embeddings_dir = self.output_dir / "embeddings"
            self.metadata_dir = self.output_dir / "metadata"

        if os.getenv("CHROMA_DIR"):
            self.chroma_dir = Path(os.environ["CHROMA_DIR"])

    def create_directories(self) -> None:
        """Create all output directories."""
        for path in [
            self.output_dir,
            self.books_dir,
            self.clean_dir,
            self.manifest_dir,
            self.chunks_dir,
            self.embeddings_dir,
            self.metadata_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_file(self) -> Path:
        return self.manifest_dir / "books_manifest.csv"

    @property
    def combined_chunks_file(self) -> Path:
        return self.chunks_dir / "all_chunks.jsonl"

    @property
    def embeddings_parquet(self) -> Path:
        return self.embeddings_dir / "all_chunks_embeddings.parquet"

    @property
    def per_book_embeddings_dir(self) -> Path:
        return self.embeddings_dir / "per_book"


# ============================================================================
# Chunking Configuration
# ============================================================================


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""

    chunk_size: int = 1200
    """Target chunk size in characters."""

    overlap: int = 200
    """Character overlap between consecutive chunks."""

    min_chunk_size: int = 100
    """Minimum chunk size to keep."""

    def __post_init__(self) -> None:
        """Apply environment variable overrides."""
        if os.getenv("CHUNK_SIZE"):
            self.chunk_size = int(os.environ["CHUNK_SIZE"])
        if os.getenv("CHUNK_OVERLAP"):
            self.overlap = int(os.environ["CHUNK_OVERLAP"])

    def validate(self) -> List[str]:
        """Validate configuration, return list of errors."""
        errors = []
        if self.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        if self.overlap < 0:
            errors.append("overlap cannot be negative")
        if self.overlap >= self.chunk_size:
            errors.append("overlap must be less than chunk_size")
        return errors


# ============================================================================
# Embedding Configuration
# ============================================================================


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    """SentenceTransformer model identifier."""

    batch_size: int = 64
    """Number of texts to encode per batch."""

    normalize: bool = True
    """Whether to L2-normalize embeddings."""

    precision: int = 6
    """Decimal places to round embeddings to (0 = no rounding)."""

    embedding_dim: int = 384
    """Expected embedding dimension (for validation)."""

    def __post_init__(self) -> None:
        """Apply environment variable overrides."""
        if os.getenv("EMBEDDING_MODEL"):
            self.model_name = os.environ["EMBEDDING_MODEL"]
        if os.getenv("BATCH_SIZE"):
            self.batch_size = int(os.environ["BATCH_SIZE"])


# ============================================================================
# ChromaDB Configuration
# ============================================================================


@dataclass
class ChromaConfig:
    """Configuration for ChromaDB ingestion."""

    collection_name: str = "arabic_books"
    """Name of the Chroma collection."""

    batch_size: int = 750
    """Number of documents per Chroma add() call."""

    schema_version: str = "1.0"
    """Schema version for metadata."""

    def __post_init__(self) -> None:
        """Apply environment variable overrides."""
        if os.getenv("CHROMA_COLLECTION"):
            self.collection_name = os.environ["CHROMA_COLLECTION"]
        if os.getenv("CHROMA_BATCH_SIZE"):
            self.batch_size = int(os.environ["CHROMA_BATCH_SIZE"])


# ============================================================================
# HuggingFace Configuration
# ============================================================================


@dataclass
class HuggingFaceConfig:
    """Configuration for HuggingFace dataset access."""

    books_dataset: str = "MoMonir/shamela_books"
    """Main books dataset identifier."""

    info_dataset: str = "MoMonir/Shamela_Books_info"
    """Metadata dataset identifier."""

    api_base: str = "https://datasets-server.huggingface.co"
    """HuggingFace API base URL."""


# ============================================================================
# Category and School Mappings
# ============================================================================

CATEGORY_MAPPING: Dict[str, str] = {
    "تفسير": "Tafsir",
    "حديث": "Hadith",
    "فقه": "Fiqh",
    "أصول": "Usul",
    "تاريخ": "Tarikh",
    "سيرة": "Sira",
    "عقيدة": "Aqida",
    "لغة": "Lugha",
    "أدب": "Adab",
}

JURISPRUDENCE_SCHOOLS: Tuple[str, ...] = (
    "Hanafi",
    "Maliki",
    "Shafii",
    "Hanbali",
    "Zahiri",
    "Jafari",
    "Ibadi",
    "General",
)

ERA_BUCKETS: Tuple[str, ...] = (
    "Early",
    "Classical",
    "Middle",
    "Late",
    "Modern",
)


# ============================================================================
# Complete Pipeline Configuration
# ============================================================================


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    paths: PathConfig = field(default_factory=PathConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)

    def validate(self) -> List[str]:
        """Validate all configurations."""
        errors = []
        errors.extend(self.chunking.validate())
        return errors


def get_default_config() -> PipelineConfig:
    """Get the default pipeline configuration with environment overrides."""
    return PipelineConfig()


# ============================================================================
# Utility Functions
# ============================================================================


def get_category_label(arabic_category: str) -> str:
    """Map Arabic category name to English label."""
    for ar, en in CATEGORY_MAPPING.items():
        if ar in arabic_category:
            return en
    return "Other"


def validate_book_id(book_id: str) -> bool:
    """Validate that a book ID is well-formed."""
    if not book_id:
        return False
    try:
        int(book_id)
        return True
    except ValueError:
        return False


def get_era_bucket(death_year: Optional[int]) -> str:
    """Determine era bucket from author death year (Hijri)."""
    if death_year is None:
        return "Unknown"
    if death_year <= 300:
        return "Early"
    elif death_year <= 500:
        return "Classical"
    elif death_year <= 900:
        return "Middle"
    elif death_year <= 1300:
        return "Late"
    else:
        return "Modern"
