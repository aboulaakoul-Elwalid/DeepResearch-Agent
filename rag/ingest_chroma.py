#!/usr/bin/env python
"""
CLI utility for loading per-book Shamela embeddings (Parquet) into a local Chroma DB.

Typical usage::

    python ingest_chroma.py \
        --inputs output/embeddings/per_book \
        --book-ids 4445 22669 \
        --chroma-path ./chroma_db \
        --collection arabic_books

The script streams each Parquet file, hydrates chunk text from its `source_path`
JSONL, validates metadata against `chroma_schema.py`, and batches documents into
Chroma using `collection.add(...)`.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import pyarrow.parquet as pq

from chroma_schema import (
    CHROMA_DEFAULT_COLLECTION,
    CHROMA_METADATA_FIELD_NAMES,
    DEFAULT_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    validate_metadata,
)

LOG = logging.getLogger("ingest_chroma")
PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class ChunkRecord:
    chunk_id: str
    document: str
    metadata: Dict[str, object]
    embedding: List[float]


@dataclass
class IngestionStats:
    parquet_path: Path
    total_chunks: int = 0
    ingested_chunks: int = 0
    batches: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "file": str(self.parquet_path),
            "total_chunks": self.total_chunks,
            "ingested_chunks": self.ingested_chunks,
            "batches": self.batches,
        }


def _coerce_meta_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    return value


class ChunkTextCache:
    """Memoized loader for chunk JSONL files referenced by `source_path`."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._store: Dict[Path, Dict[str, str]] = {}

    def get(self, path_str: Optional[str], chunk_id: str) -> Optional[str]:
        if not path_str:
            return None
        file_path = Path(path_str)
        if not file_path.is_absolute():
            file_path = self._base_dir / file_path
        if file_path not in self._store:
            self._store[file_path] = self._load_file(file_path)
        return self._store[file_path].get(chunk_id)

    def _load_file(self, path: Path) -> Dict[str, str]:
        records: Dict[str, str] = {}
        if not path.exists():
            LOG.error("Chunk text file not found: %s", path)
            return records
        LOG.debug("Loading chunk texts from %s", path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = str(obj.get("chunk_id"))
                text = obj.get("text")
                if cid and isinstance(text, str):
                    records[cid] = text
        LOG.info("Cached %d chunk texts from %s", len(records), path)
        return records


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest per-book Parquet embeddings into a Chroma collection."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["output/embeddings/per_book"],
        help="Parquet files or directories containing per-book embeddings.",
    )
    parser.add_argument(
        "--book-ids",
        nargs="+",
        help="Optional list of numeric book IDs (filenames) to ingest.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of chunks per Chroma add() call (default {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on rows ingested per Parquet file (useful for smoke tests).",
    )
    parser.add_argument(
        "--collection",
        default=CHROMA_DEFAULT_COLLECTION,
        help=f"Chroma collection name (default {CHROMA_DEFAULT_COLLECTION}).",
    )
    parser.add_argument(
        "--chroma-path",
        default=str(PROJECT_ROOT / "chroma_db"),
        help="Filesystem path for Chroma persistent storage.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk files and validate batches without touching Chroma.",
    )
    parser.add_argument(
        "--delete-on-success",
        action="store_true",
        help="Delete each Parquet file after successful ingestion.",
    )
    parser.add_argument(
        "--resume-after",
        help="Skip files until this filename (exclusive) is encountered, then ingest the rest.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging noise (warnings/errors only).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Explicit logging level override.",
    )
    return parser.parse_args(argv)


def configure_logging(args: argparse.Namespace) -> None:
    if args.quiet:
        level = logging.WARNING
    else:
        level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_parquet_paths(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        path = Path(raw)
        if not path.exists():
            LOG.warning("Input path does not exist: %s", path)
            continue
        if path.is_file() and path.suffix == ".parquet":
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(p for p in path.glob("*.parquet") if p.is_file()))
        else:
            LOG.debug("Skipping non-parquet path: %s", path)
    deduped = sorted(dict.fromkeys(paths))
    LOG.info("Discovered %d parquet files", len(deduped))
    return deduped


def filter_parquet_paths(
    paths: Sequence[Path], book_ids: Optional[Sequence[str]]
) -> List[Path]:
    if not book_ids:
        return list(paths)
    wanted = {str(book_id).strip() for book_id in book_ids}
    filtered = [
        path for path in paths if path.stem in wanted and path.suffix == ".parquet"
    ]
    missing = wanted - {path.stem for path in filtered}
    if missing:
        LOG.warning("Requested book IDs not found: %s", ", ".join(sorted(missing)))
    LOG.info("Filtered to %d parquet files based on book IDs", len(filtered))
    return filtered


def iter_chunk_records(
    parquet_path: Path,
    text_cache: ChunkTextCache,
    row_limit: Optional[int],
    reader_batch_size: int,
) -> Iterator[ChunkRecord]:
    pf = pq.ParquetFile(parquet_path)
    yielded = 0
    for batch in pf.iter_batches(batch_size=reader_batch_size):
        columns = batch.to_pydict()
        batch_len = len(columns["chunk_id"])
        for idx in range(batch_len):
            if row_limit is not None and yielded >= row_limit:
                return
            chunk_id_raw = columns["chunk_id"][idx]
            if chunk_id_raw is None:
                raise ValueError(f"Missing chunk_id in {parquet_path}")
            chunk_id = str(chunk_id_raw)

            metadata = {}
            for field in CHROMA_METADATA_FIELD_NAMES:
                values = columns.get(field)
                raw_value = values[idx] if values is not None else None
                metadata[field] = _coerce_meta_value(raw_value)
            book_id_value = metadata.get("book_id")
            if book_id_value is not None:
                metadata["book_id"] = str(book_id_value)
            if metadata.get("book_id") == "117359" and not metadata.get("era_bucket"):
                metadata["era_bucket"] = "Middle Hadith"

            page_start = metadata.get("page_start")
            page_end = metadata.get("page_end")
            if page_start is None and page_end is not None:
                metadata["page_start"] = page_end
            elif page_end is None and page_start is not None:
                metadata["page_end"] = page_start
            elif page_start is None and page_end is None:
                metadata["page_start"] = metadata["page_end"] = -1

            metadata = {
                key: value for key, value in metadata.items() if value is not None
            }
            metadata_errors = validate_metadata(metadata)
            if metadata_errors:
                raise ValueError(
                    f"{parquet_path} chunk {chunk_id} failed metadata validation: {metadata_errors}"
                )

            embedding = columns["embedding"][idx]
            if embedding is None:
                raise ValueError(f"{parquet_path} chunk {chunk_id} missing embedding")
            embedding_values = [float(val) for val in embedding]
            if len(embedding_values) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"{parquet_path} chunk {chunk_id} embedding dim "
                    f"{len(embedding_values)} != expected {EMBEDDING_DIMENSION}"
                )

            source_path = metadata.get("source_path")
            text = text_cache.get(source_path, chunk_id)
            if not text:
                raise ValueError(
                    f"{parquet_path} chunk {chunk_id} missing text in {source_path}"
                )

            yield ChunkRecord(
                chunk_id=chunk_id,
                document=text,
                metadata=metadata,
                embedding=embedding_values,
            )
            yielded += 1


def flush_batch(
    collection,
    batch: List[ChunkRecord],
    dry_run: bool,
) -> int:
    if not batch:
        return 0
    ids = [record.chunk_id for record in batch]
    documents = [record.document for record in batch]
    metadatas = [record.metadata for record in batch]
    embeddings = [record.embedding for record in batch]
    if dry_run:
        LOG.debug("Dry-run: would add %d chunks", len(batch))
        return len(batch)
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return len(batch)


def ingest_file(
    parquet_path: Path,
    collection,
    batch_size: int,
    row_limit: Optional[int],
    dry_run: bool,
    text_cache: ChunkTextCache,
) -> IngestionStats:
    stats = IngestionStats(parquet_path=parquet_path)
    batch: List[ChunkRecord] = []
    reader_batch_size = max(batch_size, 1024)
    LOG.info("Starting %s", parquet_path)
    for record in iter_chunk_records(
        parquet_path=parquet_path,
        text_cache=text_cache,
        row_limit=row_limit,
        reader_batch_size=reader_batch_size,
    ):
        stats.total_chunks += 1
        batch.append(record)
        if len(batch) >= batch_size:
            flushed = flush_batch(collection, batch, dry_run)
            stats.ingested_chunks += flushed
            stats.batches += 1
            batch.clear()
    if batch:
        flushed = flush_batch(collection, batch, dry_run)
        stats.ingested_chunks += flushed
        stats.batches += 1
        batch.clear()
    LOG.info(
        "Finished %s | chunks=%d | ingested=%d | batches=%d",
        parquet_path,
        stats.total_chunks,
        stats.ingested_chunks,
        stats.batches,
    )
    return stats


def maybe_delete_file(path: Path, enabled: bool) -> None:
    if enabled:
        try:
            path.unlink()
            LOG.info("Deleted %s after successful ingestion", path)
        except OSError as exc:
            LOG.error("Failed to delete %s: %s", path, exc)


def build_chroma_collection(args: argparse.Namespace):
    if args.dry_run:
        LOG.info("Dry-run mode: Chroma client will not be instantiated.")
        return None
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=args.chroma_path,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={
            "source": "parallax_shamela",
            "schema_version": "1.0",
        },
    )
    LOG.info(
        "Connected to Chroma collection '%s' at %s",
        args.collection,
        args.chroma_path,
    )
    return collection


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args)

    parquet_paths = resolve_parquet_paths(args.inputs)
    parquet_paths = filter_parquet_paths(parquet_paths, args.book_ids)

    if args.resume_after:
        resume_name = args.resume_after
        if resume_name not in [path.name for path in parquet_paths]:
            LOG.warning(
                "Resume marker %s not found among inputs; processing all files.",
                resume_name,
            )
        else:
            skipped = []
            while parquet_paths and parquet_paths[0].name != resume_name:
                skipped.append(parquet_paths.pop(0))
            if parquet_paths:
                parquet_paths.pop(0)  # drop the resume marker itself
            LOG.info("Resuming after %s (skipped %d files)", resume_name, len(skipped))

    if not parquet_paths:
        LOG.error("No Parquet files to ingest. Check --inputs or --book-ids filters.")
        return 1

    collection = build_chroma_collection(args)
    text_cache = ChunkTextCache(base_dir=PROJECT_ROOT)
    failures: List[Tuple[Path, str]] = []

    for parquet_path in parquet_paths:
        try:
            stats = ingest_file(
                parquet_path=parquet_path,
                collection=collection,
                batch_size=args.batch_size,
                row_limit=args.limit,
                dry_run=args.dry_run,
                text_cache=text_cache,
            )
            LOG.info("Stats: %s", stats.as_dict())
            maybe_delete_file(parquet_path, args.delete_on_success and not args.dry_run)
        except Exception as exc:  # noqa: BLE001 (command-line tool)
            LOG.exception("Failed to ingest %s", parquet_path)
            failures.append((parquet_path, str(exc)))

    if failures:
        LOG.error("Encountered %d failures:", len(failures))
        for path, message in failures:
            LOG.error(" - %s :: %s", path, message)
        return 1
    LOG.info("Completed ingestion of %d files.", len(parquet_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
