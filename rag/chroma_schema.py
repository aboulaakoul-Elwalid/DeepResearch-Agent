"""
Shared constants and helpers describing the Chroma ingestion metadata contract.

These values centralize the schema expectations for Parallax's Shamela book vectors so
that ingestion scripts, QA tooling, and downstream consumers remain in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

CHROMA_DEFAULT_COLLECTION = "arabic_books"
"""Canonical Chroma collection name for Shamela vectors."""

EMBEDDING_DIMENSION = 384
"""Dimension of the MiniLM embeddings produced by `embed_chunks.py`."""

DEFAULT_BATCH_SIZE = 750
"""Reasonable ingestion batch size that balances speed and memory usage."""


@dataclass(frozen=True)
class MetadataField:
    """Describes a single metadata attribute carried alongside each chunk."""

    name: str
    required: bool
    description: str


CHROMA_METADATA_SPECS: Tuple[MetadataField, ...] = (
    MetadataField("book_id", True, "Numeric Shamela identifier for the work."),
    MetadataField("book_title", True, "Canonical Arabic title of the work."),
    MetadataField("author_name", True, "Primary author or compiler."),
    MetadataField("category", True, "High-level genre bucket (Hadith, Tafsir, etc.)."),
    MetadataField("collection_label", True, "Human-friendly grouping label."),
    MetadataField("jurisprudence_school", False, "Fiqh school tag, when applicable."),
    MetadataField("era_bucket", True, "Century or historical era grouping."),
    MetadataField("volume_label", False, "Volume identifier (textual)."),
    MetadataField("chapter_label", False, "Chapter, section, or kitab label."),
    MetadataField("page_reference", False, "Verbatim page citation when available."),
    MetadataField("golden_subset", True, "Boolean flag for curated QA slices."),
    MetadataField("page_start", True, "Start page number (int)."),
    MetadataField("page_end", True, "End page number (int)."),
    MetadataField("chunk_length_chars", True, "Character length of the chunk text."),
    MetadataField("source_path", True, "Relative path to the source chunk JSONL."),
)
CHROMA_METADATA_FIELD_NAMES: Tuple[str, ...] = tuple(
    field.name for field in CHROMA_METADATA_SPECS
)
LEGACY_METADATA_FIELD_NAMES: Tuple[str, ...] = CHROMA_METADATA_FIELD_NAMES
"""Backwards-compatible list of metadata field names."""
CHROMA_METADATA_FIELDS: Tuple[str, ...] = LEGACY_METADATA_FIELD_NAMES
"""Deprecated alias retained for downstream scripts."""

REQUIRED_METADATA_FIELDS = tuple(
    field
    for field in CHROMA_METADATA_FIELD_NAMES
    if any(spec.name == field and spec.required for spec in CHROMA_METADATA_SPECS)
)
OPTIONAL_METADATA_FIELDS = tuple(
    field
    for field in CHROMA_METADATA_FIELD_NAMES
    if any(spec.name == field and not spec.required for spec in CHROMA_METADATA_SPECS)
)


def validate_metadata(
    payload: Dict[str, object], missing_ok: Iterable[str] = ()
) -> Tuple[str, ...]:
    """
    Validate a metadata dict against the schema.

    Args:
        payload: Metadata mapping to inspect.
        missing_ok: Additional field names that may be absent for this validation run.

    Returns:
        A tuple of human-readable error strings. Empty tuple means success.
    """
    errors = []
    allowed_missing = set(OPTIONAL_METADATA_FIELDS) | set(missing_ok)

    for field in CHROMA_METADATA_SPECS:
        value = payload.get(field.name)
        if value in (None, ""):
            if field.required and field.name not in allowed_missing:
                errors.append(f"missing required metadata '{field.name}'")
        elif isinstance(value, str) and not value.strip():
            errors.append(f"empty string for metadata '{field.name}'")

    return tuple(errors)
