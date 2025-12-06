#!/usr/bin/env python3
"""
create_chunks.py

Produce semantic-friendly JSONL chunks from the cleaned Shamela books. Each chunk
includes overlap for better recall and carries rich metadata sourced from the
manifest so downstream embedding/indexing steps can work directly from the
JSONL files.

Example:
    python create_chunks.py \
        --manifest output/manifest/books_manifest.csv \
        --input-dir output/clean_books \
        --output-dir output/chunks \
        --chunk-size 1200 \
        --overlap 200
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

PAGE_TOKEN_RE = re.compile(r"\[\[PAGE:(\d*)\]\]")


@dataclass
class Paragraph:
    page: Optional[int]
    text: str


@dataclass
class Chunk:
    text: str
    page_start: Optional[int]
    page_end: Optional[int]


def load_manifest(path: Path) -> Dict[str, Dict]:
    df = pd.read_csv(path, dtype=str).fillna("")
    records = {}
    for _, row in df.iterrows():
        book_id = str(row["book_id"]).strip()
        if not book_id:
            continue
        records[book_id] = row.to_dict()
    return records


def find_clean_file(book_id: str, hint: str, input_dir: Path) -> Optional[Path]:
    if hint:
        hinted = Path(hint)
        if hinted.is_file():
            return hinted
        hinted_rel = input_dir / Path(hint)
        if hinted_rel.is_file():
            return hinted_rel

    direct = input_dir / f"{book_id}.txt"
    if direct.is_file():
        return direct

    candidates = sorted(input_dir.glob(f"{book_id}_*.txt"))
    return candidates[0] if candidates else None


def read_paragraphs(text: str) -> List[Paragraph]:
    sections = PAGE_TOKEN_RE.split(text)
    paragraphs: List[Paragraph] = []
    current_page = None

    if sections:
        initial = sections[0].strip()
        if initial:
            paragraphs.extend(_paragraphize(initial, current_page))

    for i in range(1, len(sections), 2):
        page_raw = sections[i]
        body = sections[i + 1] if i + 1 < len(sections) else ""
        current_page = int(page_raw) if page_raw and page_raw.isdigit() else None
        paragraphs.extend(_paragraphize(body, current_page))

    return paragraphs


def _paragraphize(block: str, page: Optional[int]) -> List[Paragraph]:
    out: List[Paragraph] = []
    for para in block.split("\n\n"):
        text = para.strip()
        if text:
            out.append(Paragraph(page=page, text=text))
    return out


def build_chunks(
    paragraphs: Iterable[Paragraph], chunk_size: int, overlap: int
) -> List[Chunk]:
    chunks: List[Chunk] = []
    buffer_parts: List[str] = []
    buffer_len = 0
    pages: List[Optional[int]] = []

    def flush():
        nonlocal buffer_parts, buffer_len, pages
        if not buffer_parts:
            return
        text = "\n\n".join(buffer_parts).strip()
        if not text:
            buffer_parts = []
            buffer_len = 0
            pages = []
            return
        real_pages = [p for p in pages if p is not None]
        chunk = Chunk(
            text=text,
            page_start=min(real_pages) if real_pages else None,
            page_end=max(real_pages) if real_pages else None,
        )
        chunks.append(chunk)

        if overlap > 0:
            tail = text[-overlap:]
            buffer_parts = [tail]
            buffer_len = len(tail)
            pages = real_pages[-5:] if real_pages else []
        else:
            buffer_parts = []
            buffer_len = 0
            pages = []

    for para in paragraphs:
        addition = para.text if not buffer_parts else "\n\n" + para.text
        addition_len = len(addition)
        if buffer_len and buffer_len + addition_len > chunk_size:
            flush()
        buffer_parts.append(para.text if not buffer_parts else "\n\n" + para.text)
        buffer_len = len("\n\n".join(buffer_parts))
        pages.append(para.page)

    flush()
    return chunks


def write_chunks(
    chunks: List[Chunk], metadata: Dict, out_path: Path, book_id: str
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, chunk in enumerate(chunks, start=1):
            record = {
                "chunk_id": f"{book_id}-{idx}",
                "book_id": book_id,
                "book_title": metadata.get("book_title", ""),
                "author_name": metadata.get("author_name", ""),
                "category": metadata.get("category", ""),
                "collection_label": metadata.get("collection_label", ""),
                "jurisprudence_school": metadata.get("jurisprudence_school", ""),
                "era_bucket": metadata.get("era_bucket", ""),
                "golden_subset": bool(
                    str(metadata.get("golden_subset", "True")).lower() == "true"
                ),
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "text": chunk.text.strip(),
            }
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk cleaned Shamela books into JSONL files."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("output/manifest/books_manifest.csv"),
        help="Path to the master manifest CSV.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/clean_books"),
        help="Directory containing cleaned text files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/chunks"),
        help="Directory to store per-book JSONL chunk files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Target chunk size in characters.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Character-level overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N books (useful for smoke tests).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging output."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    manifest = load_manifest(args.manifest)
    processed = 0
    missing = 0

    for book_id, meta in manifest.items():
        if args.limit and processed >= args.limit:
            break

        clean_hint = meta.get("cleaned_file", "")
        clean_path = find_clean_file(book_id, clean_hint, args.input_dir)
        if not clean_path or not clean_path.is_file():
            missing += 1
            logging.warning("Missing cleaned file for %s", book_id)
            continue

        text = clean_path.read_text(encoding="utf-8")
        paragraphs = read_paragraphs(text)
        if not paragraphs:
            logging.warning("No paragraphs found for %s (%s)", book_id, clean_path)
            continue

        chunks = build_chunks(paragraphs, args.chunk_size, args.overlap)
        if not chunks:
            logging.warning("No chunks produced for %s", book_id)
            continue

        out_file = args.output_dir / f"{book_id}.jsonl"
        write_chunks(chunks, meta, out_file, book_id)
        processed += 1
        logging.info(
            "Chunked %s -> %s chunks (%s)",
            book_id,
            len(chunks),
            out_file.relative_to(out_file.parent.parent),
        )

    logging.info(
        "Completed chunking. Books processed: %d | Missing: %d", processed, missing
    )


if __name__ == "__main__":
    main()
