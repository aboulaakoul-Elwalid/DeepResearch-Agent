#!/usr/bin/env python3
"""
combine_chunks.py

Utility script to merge all per-book chunk JSONL files into a single manifest-ready
JSONL stream. This is useful when you want to feed the entire corpus into an
embedding job, push it into a vector database, or run QA evaluations without
iterating over dozens of files manually.

Example:
    python combine_chunks.py \
        --input-dir output/chunks \
        --output output/chunks/all_chunks.jsonl

Features:
    - Validates that every chunk record contains the expected fields.
    - Streams data to avoid loading all chunks into memory at once.
    - Emits optional JSON stats (per book/per file) for downstream monitoring.
    - Optionally augments each emitted record with its source chunk file path.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

REQUIRED_FIELDS = (
    "chunk_id",
    "book_id",
    "book_title",
    "author_name",
    "text",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-book chunk JSONL files into a single manifest-ready JSONL."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/chunks"),
        help="Directory containing per-book JSONL chunk files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/chunks/all_chunks.jsonl"),
        help="Destination JSONL file to write the combined chunks.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.jsonl",
        help="Glob pattern used to discover chunk files inside --input-dir.",
    )
    parser.add_argument(
        "--required-fields",
        type=str,
        nargs="*",
        default=list(REQUIRED_FIELDS),
        help="Fields that must exist in each chunk record. Missing fields raise errors.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Optional JSON file to write aggregate chunk statistics.",
    )
    parser.add_argument(
        "--include-source-path",
        action="store_true",
        help="Include a `source_path` field with each emitted chunk record.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def discover_chunk_files(input_dir: Path, pattern: str) -> List[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    files = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(
            f"No chunk files matching '{pattern}' were found in {input_dir}"
        )
    return files


def iter_chunk_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path} line {line_number}: {exc}"
                ) from exc
            yield record


def validate_record(record: dict, required_fields: Sequence[str], source: Path) -> None:
    missing = [field for field in required_fields if field not in record]
    if missing:
        raise ValueError(
            f"Missing required fields {missing} in chunk originating from {source}"
        )


def build_stats_snapshot(
    chunk_files: Sequence[Path],
    output_path: Path,
    total_chunks: int,
    unique_books: set,
    per_book_counts: Counter,
    per_file_counts: Counter,
) -> Dict[str, object]:
    return {
        "output_path": str(output_path),
        "total_chunk_files": len(chunk_files),
        "total_chunks": total_chunks,
        "unique_books": len(unique_books),
        "book_ids": sorted(unique_books),
        "per_book_counts": dict(per_book_counts),
        "per_file_counts": dict(per_file_counts),
    }


def write_stats_file(path: Path, stats: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def as_relative_string(path: Path, base: Optional[Path]) -> str:
    if base is None:
        return str(path)
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def combine_chunks(
    chunk_files: Sequence[Path],
    output_path: Path,
    required_fields: Sequence[str],
    overwrite: bool,
    stats_output: Optional[Path] = None,
    include_source_path: bool = False,
    base_dir: Optional[Path] = None,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file {output_path} already exists. Pass --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_dir_resolved = base_dir.resolve() if base_dir else None

    total_chunks = 0
    unique_books = set()
    per_book_counts = Counter()
    file_chunk_counts = Counter()

    with output_path.open("w", encoding="utf-8") as out_handle:
        for path in chunk_files:
            file_key = as_relative_string(path, base_dir_resolved)
            for record in iter_chunk_records(path):
                validate_record(record, required_fields, path)
                if include_source_path:
                    record_to_write = dict(record)
                    record_to_write["source_path"] = file_key
                else:
                    record_to_write = record
                out_handle.write(json.dumps(record_to_write, ensure_ascii=False))
                out_handle.write("\n")
                total_chunks += 1
                book_id = record.get("book_id")
                if book_id:
                    unique_books.add(book_id)
                    per_book_counts[book_id] += 1
                file_chunk_counts[file_key] += 1

    stats = build_stats_snapshot(
        chunk_files=chunk_files,
        output_path=output_path,
        total_chunks=total_chunks,
        unique_books=unique_books,
        per_book_counts=per_book_counts,
        per_file_counts=file_chunk_counts,
    )

    logging.info(
        "Combined %d chunk files into %s", len(chunk_files), output_path.resolve()
    )
    logging.info("Total chunks written: %d", total_chunks)
    logging.info("Unique books represented: %d", len(unique_books))
    logging.info(
        "Approximate output size: %.2f MB", output_path.stat().st_size / (1024 * 1024)
    )

    if stats_output:
        write_stats_file(stats_output, stats)
        logging.info("Stats summary written to %s", stats_output.resolve())


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    chunk_files = discover_chunk_files(args.input_dir, args.glob)
    logging.info("Discovered %d chunk files under %s", len(chunk_files), args.input_dir)
    combine_chunks(
        chunk_files=chunk_files,
        output_path=args.output,
        required_fields=args.required_fields,
        overwrite=args.overwrite,
        stats_output=args.stats_output,
        include_source_path=args.include_source_path,
        base_dir=args.input_dir,
    )


if __name__ == "__main__":
    main()
