#!/usr/bin/env python3
"""
clean_books.py

Normalize and sanitize the assembled Shamela books so they are ready for
chunking and embeddings. The script performs several cleanup passes:

1. Converts page markers (<!-- Page X -->) into inline tokens (`[[PAGE:X]]`) and collapses redundant metadata
   headers at the top of each file.
2. Normalizes Arabic characters (tatweel removal, unifying alef/hamza forms,
   converting ornate punctuation to ASCII equivalents).
3. Converts footnote callouts written as block quotes into inline bracketed
   notes so downstream chunkers can keep the contextual text.
4. Collapses excessive whitespace and ensures each book ends with a trailing
   newline.

Example:
    python clean_books.py \
        --input-dir output/assembled_books \
        --output-dir output/clean_books \
        --overwrite

The script will only touch files whose output is missing unless --overwrite is
provided.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

# --------------------------------------------------------------------------- #
# Regexes & translation tables
# --------------------------------------------------------------------------- #

PAGE_COMMENT_RE = re.compile(r"<!--\s*Page\s*(\d+)?\s*-->\s*", re.IGNORECASE)
FRONT_MATTER_RE = re.compile(
    r"(?sm)^# .*?(?:^-{3,}\\s*$|^## )",  # everything from H1 down to separator
)
FOOTNOTE_RE = re.compile(r"(?m)^>\\s*\\*\\*Footnote:\\*\\*\\s*(.+)$")

PUNCTUATION_MAP = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "’": "'",
        "‘": "'",
        "–": "-",
        "—": "-",
        "…": "...",
    }
)

ARABIC_NORMALIZATION_MAP = {
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    "ى": "ي",
    "ئ": "ي",
    "ؤ": "و",
    "ٰ": "",
    "ـ": "",  # tatweel
}

ARABIC_TRANSLATION_TABLE = str.maketrans(ARABIC_NORMALIZATION_MAP)

WHITESPACE_RE = re.compile(r"[ \\t]+")
MULTI_NEWLINE_RE = re.compile(r"(\\n\\s*){3,}")

# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #


def normalize_arabic(text: str) -> str:
    """Apply Unicode normalization plus custom Arabic replacements."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(ARABIC_TRANSLATION_TABLE)
    return text


def strip_front_matter(text: str) -> str:
    """Drop the initial metadata block (title, info, horizontal rule)."""
    match = FRONT_MATTER_RE.match(text)
    if match:
        return text[match.end() :].lstrip()
    return text


def collapse_whitespace(text: str) -> str:
    """Condense stray spaces and repeated blank lines."""
    text = WHITESPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\\n\\n", text)
    return text.strip() + "\\n"


def convert_page_markers(text: str) -> str:
    """Replace HTML page comments with [[PAGE:n]] markers."""

    def _replace(match: re.Match) -> str:
        number = (match.group(1) or "").strip()
        return f"[[PAGE:{number}]]\n" if number else "[[PAGE]]\n"

    return PAGE_COMMENT_RE.sub(_replace, text)


def convert_footnotes(text: str) -> str:
    """Turn block-quoted footnotes into inline bracketed notes."""

    def _replace(match: re.Match) -> str:
        content = match.group(1).strip()
        return f"[حاشية: {content}]"

    return FOOTNOTE_RE.sub(_replace, text)


def clean_text(raw_text: str) -> str:
    """Run the full cleaning pipeline on a single book string."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = convert_page_markers(text)
    text = strip_front_matter(text)
    text = text.translate(PUNCTUATION_MAP)
    text = convert_footnotes(text)
    text = normalize_arabic(text)
    text = collapse_whitespace(text)
    return text


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #


def iter_input_files(input_dir: Path, glob: str) -> Iterable[Path]:
    pattern = f"**/{glob}"
    yield from sorted(input_dir.glob(pattern))


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def should_skip(output_file: Path, overwrite: bool) -> bool:
    return output_file.exists() and not overwrite


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean assembled Shamela books for chunking."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/assembled_books"),
        help="Directory containing raw assembled book files (.md).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/clean_books"),
        help="Destination directory for cleaned .txt outputs.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.md",
        help="Glob pattern (relative to input-dir) selecting files to clean.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate outputs even if they already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Clean at most N files (useful for smoke tests).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s %(message)s",
    )

    input_dir = args.input_dir
    output_dir = args.output_dir
    ensure_output_dir(output_dir)

    if not input_dir.is_dir():
        logging.error("Input directory %s does not exist.", input_dir)
        sys.exit(1)

    processed = 0
    skipped = 0
    failed = 0

    for src in iter_input_files(input_dir, args.glob):
        rel = src.relative_to(input_dir)
        dst = output_dir / rel.with_suffix(".txt")
        dst.parent.mkdir(parents=True, exist_ok=True)

        if should_skip(dst, args.overwrite):
            skipped += 1
            logging.debug("Skipping %s (already exists).", dst)
            continue

        try:
            content = src.read_text(encoding="utf-8")
            cleaned = clean_text(content)
            dst.write_text(cleaned, encoding="utf-8")
            processed += 1
            logging.debug("Cleaned %s -> %s", src, dst)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logging.error("Failed to clean %s: %s", src, exc)

        if args.limit and processed >= args.limit:
            break

    logging.info("Processed: %d | Skipped: %d | Failed: %d", processed, skipped, failed)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
