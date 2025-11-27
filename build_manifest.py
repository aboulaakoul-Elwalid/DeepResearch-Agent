#!/usr/bin/env python3
"""
build_manifest.py

Combine Shamela metadata with the assembled / cleaned / chunked book artifacts to
produce a single manifest plus a lightweight markdown summary. The manifest
powers downstream chunking, embedding, and RAG pipelines.

Usage:
    python build_manifest.py \
        --matched-metadata output/metadata/matched_books.json \
        --assembled-index output/assembled_books_index.csv \
        --assembled-dir output/assembled_books \
        --clean-dir output/clean_books \
        --chunks-dir output/chunks \
        --output output/manifest/books_manifest.csv \
        --summary output/stats/manifest_summary.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

CATEGORY_TO_COLLECTION = {
    "التفسير": "Tafsir",
    "علوم القرآن وأصول التفسير": "Tafsir",
    "كتب السنة": "Hadith",
    "شروح الحديث": "Hadith Commentary",
    "العقيدة": "Aqidah",
    "الرقائق والآداب والأذكار": "Spirituality",
    "السيرة النبوية": "Seerah",
    "التاريخ": "Tarikh",
    "التراجم والطبقات": "Biographical",
    "السياسة الشرعية والقضاء": "Fiqh & Governance",
    "أصول الفقه": "Usul al-Fiqh",
    "الفقه الحنفي": "Fiqh (Hanafi)",
    "الفقه الشافعي": "Fiqh (Shafi'i)",
    "الفقه المالكي": "Fiqh (Maliki)",
    "الفقه الحنبلي": "Fiqh (Hanbali)",
}

CATEGORY_TO_SCHOOL = {
    "الفقه الحنفي": "Hanafi",
    "الفقه الشافعي": "Shafi'i",
    "الفقه المالكي": "Maliki",
    "الفقه الحنبلي": "Hanbali",
}

ERA_BUCKETS: List[Tuple[int, str]] = [
    (200, "Early"),
    (400, "Classical"),
    (700, "Middle"),
    (1000, "Late"),
    (1500, "Modern"),
    (10_000, "Contemporary"),
]

DEFAULT_COLLECTION = "General"


@dataclass
class BookRow:
    book_id: str
    book_title: Optional[str]
    author_name: Optional[str]
    author_death_year: Optional[int]
    category: Optional[str]
    collection_label: str
    jurisprudence_school: Optional[str]
    era_bucket: Optional[str]
    golden_subset: bool
    assembled_file: Optional[str]
    assembled_size_bytes: Optional[int]
    total_pages: Optional[int]
    total_volumes: Optional[int]
    cleaned_file: Optional[str]
    cleaned_size_bytes: Optional[int]
    chunks_file: Optional[str]
    chunks_size_bytes: Optional[int]
    publisher: Optional[str]
    edition: Optional[str]
    pages_reported: Optional[int]
    volumes_declared: Optional[str]
    status: str
    tags: List[str]


def maybe_int(value: Optional[str]) -> Optional[int]:
    if value in (None, "", "Null"):
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def infer_collection(category: Optional[str]) -> str:
    if not category:
        return DEFAULT_COLLECTION
    return CATEGORY_TO_COLLECTION.get(category, DEFAULT_COLLECTION)


def infer_school(category: Optional[str]) -> Optional[str]:
    if not category:
        return None
    return CATEGORY_TO_SCHOOL.get(category)


def infer_era(year: Optional[int]) -> Optional[str]:
    if year is None:
        return None
    for ceiling, label in ERA_BUCKETS:
        if year <= ceiling:
            return label
    return ERA_BUCKETS[-1][1]


def compute_size(path_str: Optional[str]) -> Optional[int]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_file():
        return None
    return path.stat().st_size


def load_metadata(path: Path) -> Dict[str, Dict]:
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    records: Dict[str, Dict] = {}
    for payload in raw.values():
        book_id = str(payload.get("id"))
        meta = payload.get("metadata", {})
        records[book_id] = {
            "book_id": book_id,
            "book_title": payload.get("title"),
            "author_name": meta.get("author_name"),
            "author_death_year": maybe_int(meta.get("author_year")),
            "category": meta.get("category"),
            "publisher": meta.get("publisher"),
            "edition": meta.get("edition"),
            "pages_reported": maybe_int(meta.get("pages")),
            "volumes_declared": meta.get("volumes"),
        }
    return records


def load_assembled_index(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing assembled index at {path}")
    df = pd.read_csv(path, dtype=str).fillna("")
    for col in ("total_pages", "total_volumes"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: int(v) if v.isdigit() else None)
    return df


def discover_file(
    directory: Path, book_id: str, suffixes: Tuple[str, ...], hint: Optional[str] = None
) -> Optional[Path]:
    if hint:
        candidate = Path(hint)
        if candidate.is_file():
            return candidate
        candidate = directory / Path(hint)
        if candidate.is_file():
            return candidate

    direct = directory / f"{book_id}"
    if direct.is_file():
        return direct
    for suff in suffixes:
        match = directory / f"{book_id}{suff}"
        if match.is_file():
            return match

    pattern = f"{book_id}_*"
    for suff in suffixes:
        matches = sorted(directory.glob(f"{pattern}{suff}"))
        if matches:
            return matches[0]
    return None


def build_rows(
    metadata: Dict[str, Dict],
    assembled_df: pd.DataFrame,
    project_root: Path,
    assembled_dir: Path,
    clean_dir: Path,
    chunks_dir: Path,
) -> List[BookRow]:
    assembled_lookup = {str(row["book_id"]): row for _, row in assembled_df.iterrows()}
    rows: List[BookRow] = []

    for book_id, meta in metadata.items():
        assembled_info = assembled_lookup.get(book_id, {})
        assembled_hint = assembled_info.get("file_path") or ""
        assembled_path = discover_file(
            assembled_dir,
            book_id,
            suffixes=(".md", ".txt"),
            hint=str(project_root / assembled_hint) if assembled_hint else None,
        )
        assembled_size = compute_size(assembled_path)

        cleaned_path = discover_file(clean_dir, book_id, suffixes=(".txt",), hint=None)
        cleaned_size = compute_size(cleaned_path)

        chunk_path = discover_file(chunks_dir, book_id, suffixes=(".jsonl", ".ndjson"))
        chunk_size = compute_size(chunk_path)

        category = meta.get("category")
        collection_label = infer_collection(category)
        jurisprudence_school = infer_school(category)
        era_bucket = infer_era(meta.get("author_death_year"))

        statuses = []
        if assembled_size:
            statuses.append("assembled")
        if cleaned_size:
            statuses.append("cleaned")
        if chunk_size:
            statuses.append("chunked")
        status = ",".join(statuses) if statuses else "pending"

        tags = [collection_label]
        if jurisprudence_school:
            tags.append(jurisprudence_school)
        if era_bucket:
            tags.append(f"Era:{era_bucket}")

        rows.append(
            BookRow(
                book_id=book_id,
                book_title=meta.get("book_title"),
                author_name=meta.get("author_name"),
                author_death_year=meta.get("author_death_year"),
                category=category,
                collection_label=collection_label,
                jurisprudence_school=jurisprudence_school,
                era_bucket=era_bucket,
                golden_subset=True,
                assembled_file=str(assembled_path) if assembled_path else None,
                assembled_size_bytes=assembled_size,
                total_pages=assembled_info.get("total_pages"),
                total_volumes=assembled_info.get("total_volumes"),
                cleaned_file=str(cleaned_path) if cleaned_path else None,
                cleaned_size_bytes=cleaned_size,
                chunks_file=str(chunk_path) if chunk_path else None,
                chunks_size_bytes=chunk_size,
                publisher=meta.get("publisher"),
                edition=meta.get("edition"),
                pages_reported=meta.get("pages_reported"),
                volumes_declared=meta.get("volumes_declared"),
                status=status,
                tags=tags,
            )
        )
    return rows


def rows_to_dataframe(rows: Iterable[BookRow]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(row) for row in rows])
    df.sort_values("book_title", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def write_manifest(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False, encoding="utf-8")
    elif fmt == "json":
        df.to_json(path, orient="records", force_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def format_counts(series: pd.Series) -> str:
    if series.empty:
        return "- No data"
    total = int(series.sum())
    return "\n".join(
        f"- **{label or 'Unknown'}:** {count} ({(count / total) * 100:.1f}%)"
        for label, count in series.sort_values(ascending=False).items()
    )


def write_summary(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    total_books = len(df)
    assembled = df["status"].str.contains("assembled").sum()
    cleaned = df["status"].str.contains("cleaned").sum()
    chunked = df["status"].str.contains("chunked").sum()
    total_pages = df["total_pages"].dropna().astype(int).sum()
    disk_usage_mb = (
        (
            df["assembled_size_bytes"].dropna().astype(int).sum()
            + df["cleaned_size_bytes"].dropna().astype(int).sum()
        )
        / (1024 * 1024)
        if total_books
        else 0
    )

    content = (
        "# Books Manifest Summary\n\n"
        f"- **Total books:** {total_books}\n"
        f"- **Assembled:** {assembled}\n"
        f"- **Cleaned:** {cleaned}\n"
        f"- **Chunked:** {chunked}\n"
        f"- **Total pages (assembled):** {total_pages:,}\n"
        f"- **Disk usage (assembled+cleaned):** {disk_usage_mb:.2f} MB\n\n"
        "## By Collection\n"
        f"{format_counts(df['collection_label'].value_counts())}\n\n"
        "## By Jurisprudence School\n"
        f"{format_counts(df['jurisprudence_school'].dropna().value_counts())}\n\n"
        "## By Era Bucket\n"
        f"{format_counts(df['era_bucket'].dropna().value_counts())}\n"
    )

    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build master manifest for books.")
    parser.add_argument(
        "--matched-metadata",
        type=Path,
        default=Path("output/metadata/matched_books.json"),
        help="Path to matched_books.json.",
    )
    parser.add_argument(
        "--assembled-index",
        type=Path,
        default=Path("output/assembled_books_index.csv"),
        help="CSV produced by assemble_books.py.",
    )
    parser.add_argument(
        "--assembled-dir",
        type=Path,
        default=Path("output/assembled_books"),
        help="Directory containing assembled markdown files.",
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=Path("output/clean_books"),
        help="Directory containing cleaned text files.",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("output/chunks"),
        help="Directory containing chunk JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/manifest/books_manifest.csv"),
        help="Destination for manifest table.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "json"),
        default="csv",
        help="Manifest output format.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("output/stats/manifest_summary.md"),
        help="Path for markdown summary report.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root used to resolve relative paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata = load_metadata(args.matched_metadata)
    assembled_df = load_assembled_index(args.assembled_index)
    rows = build_rows(
        metadata=metadata,
        assembled_df=assembled_df,
        project_root=args.project_root,
        assembled_dir=args.assembled_dir,
        clean_dir=args.clean_dir,
        chunks_dir=args.chunks_dir,
    )
    df = rows_to_dataframe(rows)

    write_manifest(df, args.output, args.format)
    write_summary(df, args.summary)

    print(f"Manifest entries: {len(df)}")
    print(f"Saved manifest -> {args.output} ({args.format})")
    print(f"Saved summary  -> {args.summary}")


if __name__ == "__main__":
    main()
