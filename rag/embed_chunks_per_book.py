#!/usr/bin/env python3
"""
embed_chunks_per_book.py

Generate per-book Parquet embedding files from per-book chunk JSONL files.
This is required by ingest_chroma.py which expects per-book Parquet files
in the output/embeddings/per_book/ directory.

The output Parquet files contain:
- All metadata fields from the chunks
- The embedding vector
- A source_path field pointing back to the original JSONL

Usage:
    python embed_chunks_per_book.py \
        --chunks-dir output/chunks \
        --output-dir output/embeddings/per_book \
        --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
        --batch-size 64

    # Process specific books only
    python embed_chunks_per_book.py --book-ids 4445 22669

    # Overwrite existing files
    python embed_chunks_per_book.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False

from tqdm import tqdm

LOG = logging.getLogger(__name__)


@dataclass
class BookStats:
    """Statistics for a single book's embedding process."""

    book_id: str
    total_chunks: int = 0
    embedded_chunks: int = 0
    skipped_chunks: int = 0
    embedding_dim: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-book Parquet embedding files."
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("output/chunks"),
        help="Directory containing per-book JSONL chunk files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/embeddings/per_book"),
        help="Directory to write per-book Parquet files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model identifier or local path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of chunks to encode per forward pass.",
    )
    parser.add_argument(
        "--book-ids",
        nargs="+",
        help="Optional list of book IDs to process (default: all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Parquet files.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Apply L2 normalization to embeddings (default: True).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Round embeddings to this many decimal places.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def stream_chunks(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream chunks from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                LOG.warning("Invalid JSON at %s line %d: %s", path, line_num, exc)
                continue


def preprocess_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Clean and validate a chunk record."""
    text = record.get("text", "")
    if not text or not isinstance(text, str):
        return None

    chunk_id = record.get("chunk_id")
    book_id = record.get("book_id")
    if not chunk_id or not book_id:
        return None

    # Normalize fields
    cleaned = dict(record)
    cleaned["text"] = text.strip()
    cleaned["chunk_id"] = str(chunk_id)
    cleaned["book_id"] = str(book_id)

    # Ensure string fields
    for key in (
        "book_title",
        "author_name",
        "category",
        "collection_label",
        "jurisprudence_school",
        "era_bucket",
        "volume_label",
        "chapter_label",
        "page_reference",
    ):
        value = cleaned.get(key)
        cleaned[key] = str(value) if value is not None else ""

    # Ensure boolean
    golden = cleaned.get("golden_subset", True)
    if isinstance(golden, str):
        golden = golden.strip().lower() in ("true", "1", "yes")
    cleaned["golden_subset"] = bool(golden)

    # Ensure integers
    for key in ("page_start", "page_end"):
        value = cleaned.get(key)
        if value is not None:
            try:
                cleaned[key] = int(value)
            except (TypeError, ValueError):
                cleaned[key] = None
        else:
            cleaned[key] = None

    cleaned["chunk_length_chars"] = len(cleaned["text"])

    return cleaned


def build_schema() -> pa.Schema:
    """Build PyArrow schema for output Parquet files."""
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("book_id", pa.string()),
            pa.field("book_title", pa.string()),
            pa.field("author_name", pa.string()),
            pa.field("category", pa.string()),
            pa.field("text", pa.string()),
            pa.field("collection_label", pa.string()),
            pa.field("jurisprudence_school", pa.string()),
            pa.field("era_bucket", pa.string()),
            pa.field("volume_label", pa.string()),
            pa.field("chapter_label", pa.string()),
            pa.field("page_reference", pa.string()),
            pa.field("golden_subset", pa.bool_()),
            pa.field("page_start", pa.int32()),
            pa.field("page_end", pa.int32()),
            pa.field("chunk_length_chars", pa.int32()),
            pa.field("source_path", pa.string()),
            pa.field("embedding_dim", pa.int32()),
            pa.field("embedding", pa.list_(pa.float32())),
        ]
    )


def encode_batch(
    model: Any,
    texts: Sequence[str],
    normalize: bool,
    precision: int,
) -> np.ndarray:
    """Encode a batch of texts into embeddings."""
    embeddings = model.encode(
        list(texts),
        batch_size=len(texts),
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    # Ensure float32
    embeddings = embeddings.astype(np.float32)

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

    if precision > 0:
        embeddings = np.round(embeddings, decimals=precision)

    return embeddings


def process_book(
    jsonl_path: Path,
    output_path: Path,
    model: Any,
    batch_size: int,
    normalize: bool,
    precision: int,
) -> BookStats:
    """Process a single book's chunks and write embeddings to Parquet."""
    book_id = jsonl_path.stem
    stats = BookStats(book_id=book_id)

    # Collect all chunks
    records: List[Dict[str, Any]] = []
    for record in stream_chunks(jsonl_path):
        stats.total_chunks += 1
        cleaned = preprocess_record(record)
        if cleaned is None:
            stats.skipped_chunks += 1
            continue
        records.append(cleaned)

    if not records:
        LOG.warning("No valid chunks in %s", jsonl_path)
        return stats

    # Process in batches
    all_embeddings: List[np.ndarray] = []
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        texts = [r["text"] for r in batch]
        embeddings = encode_batch(model, texts, normalize, precision)
        all_embeddings.append(embeddings)

    # Concatenate all embeddings
    all_embeddings_arr = np.vstack(all_embeddings)
    stats.embedding_dim = all_embeddings_arr.shape[1]
    stats.embedded_chunks = len(records)

    # Build table
    source_path = str(jsonl_path)
    arrays = [
        pa.array([r["chunk_id"] for r in records], pa.string()),
        pa.array([r["book_id"] for r in records], pa.string()),
        pa.array([r.get("book_title", "") for r in records], pa.string()),
        pa.array([r.get("author_name", "") for r in records], pa.string()),
        pa.array([r.get("category", "") for r in records], pa.string()),
        pa.array([r["text"] for r in records], pa.string()),
        pa.array([r.get("collection_label", "") for r in records], pa.string()),
        pa.array([r.get("jurisprudence_school", "") for r in records], pa.string()),
        pa.array([r.get("era_bucket", "") for r in records], pa.string()),
        pa.array([r.get("volume_label", "") for r in records], pa.string()),
        pa.array([r.get("chapter_label", "") for r in records], pa.string()),
        pa.array([r.get("page_reference", "") for r in records], pa.string()),
        pa.array([r.get("golden_subset", True) for r in records], pa.bool_()),
        pa.array([r.get("page_start") for r in records], pa.int32()),
        pa.array([r.get("page_end") for r in records], pa.int32()),
        pa.array([r.get("chunk_length_chars", 0) for r in records], pa.int32()),
        pa.array([source_path] * len(records), pa.string()),
        pa.array([stats.embedding_dim] * len(records), pa.int32()),
        pa.array([emb.tolist() for emb in all_embeddings_arr], pa.list_(pa.float32())),
    ]

    table = pa.Table.from_arrays(arrays, schema=build_schema())

    # Write Parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="zstd")

    return stats


def find_chunk_files(
    chunks_dir: Path,
    book_ids: Optional[List[str]] = None,
) -> List[Path]:
    """Find all JSONL chunk files to process."""
    all_files = sorted(chunks_dir.glob("*.jsonl"))

    # Filter out combined file
    all_files = [f for f in all_files if f.stem != "all_chunks"]

    if book_ids:
        wanted = set(str(bid).strip() for bid in book_ids)
        all_files = [f for f in all_files if f.stem in wanted]
        missing = wanted - {f.stem for f in all_files}
        if missing:
            LOG.warning("Requested book IDs not found: %s", ", ".join(sorted(missing)))

    return all_files


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if SentenceTransformer is None:
        LOG.error(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        )
        raise SystemExit(1)

    if not args.chunks_dir.is_dir():
        LOG.error("Chunks directory not found: %s", args.chunks_dir)
        raise SystemExit(1)

    # Find files to process
    chunk_files = find_chunk_files(args.chunks_dir, args.book_ids)
    if not chunk_files:
        LOG.error("No JSONL chunk files found in %s", args.chunks_dir)
        raise SystemExit(1)

    LOG.info("Found %d chunk files to process", len(chunk_files))

    # Filter already processed
    if not args.overwrite:
        to_process = []
        for f in chunk_files:
            output_path = args.output_dir / f"{f.stem}.parquet"
            if output_path.exists():
                LOG.debug("Skipping %s (already exists)", f.stem)
            else:
                to_process.append(f)
        chunk_files = to_process
        LOG.info("After filtering existing: %d files to process", len(chunk_files))

    if not chunk_files:
        LOG.info("Nothing to process. All books already have embeddings.")
        return

    # Load model
    LOG.info("Loading model: %s", args.model)
    model = SentenceTransformer(args.model)
    LOG.info(
        "Model loaded. Embedding dimension: %d",
        model.get_sentence_embedding_dimension(),
    )

    # Process each book
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_chunks = 0
    total_embedded = 0

    for jsonl_path in tqdm(chunk_files, desc="Processing books"):
        output_path = args.output_dir / f"{jsonl_path.stem}.parquet"

        try:
            stats = process_book(
                jsonl_path=jsonl_path,
                output_path=output_path,
                model=model,
                batch_size=args.batch_size,
                normalize=args.normalize,
                precision=args.precision,
            )
            total_chunks += stats.total_chunks
            total_embedded += stats.embedded_chunks
            LOG.debug(
                "Processed %s: %d chunks, %d embedded, %d skipped",
                stats.book_id,
                stats.total_chunks,
                stats.embedded_chunks,
                stats.skipped_chunks,
            )
        except Exception as exc:
            LOG.exception("Failed to process %s: %s", jsonl_path, exc)
            continue

    LOG.info("=" * 60)
    LOG.info("Embedding complete")
    LOG.info("  Books processed: %d", len(chunk_files))
    LOG.info("  Total chunks: %d", total_chunks)
    LOG.info("  Chunks embedded: %d", total_embedded)
    LOG.info("  Output directory: %s", args.output_dir)


if __name__ == "__main__":
    main()
