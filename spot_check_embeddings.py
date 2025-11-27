#!/usr/bin/env python
"""
Spot-check per-book embedding Parquet files and emit Chroma-ready JSONL samples.

Typical usage
-------------
python spot_check_embeddings.py \
    --inputs output/embeddings/per_book \
    --per-book-limit 3 \
    --jsonl output/qa/chroma_spot_check.jsonl

The script will:
1. Locate all `.parquet` files under the provided paths.
2. Sample `per-book-limit` rows from each file.
3. Fetch the matching chunk text from the `source_path` JSONL (when available).
4. Validate required metadata fields and report any missing values.
5. Optionally write the sampled rows as JSONL records formatted for ingestion
   via `chromadb.Client().add(...)`.

Dependencies: pandas, pyarrow (both already listed in the project metadata).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd
import pyarrow.parquet as pq

DEFAULT_INPUT_DIR = Path("output/embeddings/per_book")
DEFAULT_SUMMARY_PATH = Path("output/qa/chroma_spot_check_summary.json")
CHROMA_METADATA_FIELDS = [
    "book_id",
    "book_title",
    "author_name",
    "category",
    "collection_label",
    "jurisprudence_school",
    "era_bucket",
    "volume_label",
    "chapter_label",
    "page_reference",
    "golden_subset",
    "page_start",
    "page_end",
    "chunk_length_chars",
    "source_path",
]
OPTIONAL_METADATA_FIELDS = {
    "volume_label",
    "chapter_label",
    "page_reference",
    "jurisprudence_school",
}


@dataclass
class SampledChunk:
    parquet_path: Path
    chunk_id: str
    metadata: Dict[str, object]
    embedding: List[float]
    text: Optional[str]
    issues: List[str] = field(default_factory=list)

    def as_chroma_record(self) -> Dict[str, object]:
        return {
            "id": self.chunk_id,
            "text": self.text or "",
            "metadata": self.metadata,
            "embedding": self.embedding,
        }


@dataclass
class ChunkTextCache:
    store: Dict[Path, Dict[str, str]] = field(default_factory=dict)

    def load_required(self, source_path: Optional[str], chunk_ids: Set[str]) -> None:
        if not source_path or not chunk_ids:
            return
        path = Path(source_path)
        if not path.exists():
            return

        cached = self.store.setdefault(path, {})
        remaining = chunk_ids - set(cached.keys())
        if not remaining:
            return

        try:
            handle = path.open("r", encoding="utf-8")
        except OSError:
            return

        try:
            for line in handle:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = obj.get("chunk_id")
                text = obj.get("text")
                if cid is None or text is None:
                    continue
                cid_str = str(cid)
                if cid_str in remaining:
                    cached[cid_str] = text
                    remaining.discard(cid_str)
                    if not remaining:
                        break
        finally:
            handle.close()

    def get_text(self, source_path: Optional[str], chunk_id: str) -> Optional[str]:
        if not source_path or not chunk_id:
            return None
        path = Path(source_path)
        cached = self.store.get(path, {})
        return cached.get(chunk_id)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spot-check embedding Parquet files and emit Chroma-ready samples."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[str(DEFAULT_INPUT_DIR)],
        help="Parquet files or directories to scan. Defaults to output/embeddings/per_book.",
    )
    parser.add_argument(
        "--per-book-limit",
        type=int,
        default=3,
        help="Maximum number of samples to pull from each Parquet file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used when sampling rows.",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        help="Optional path to write sampled rows as Chroma-ready JSONL.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=str(DEFAULT_SUMMARY_PATH),
        help="Path to write the aggregated QA summary report (set to '' to skip).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed stdout summaries (still writes JSONL if requested).",
    )
    return parser.parse_args(argv)


def find_parquet_files(inputs: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for raw in inputs:
        path = Path(raw)
        if not path.exists():
            print(f"[WARN] Path not found: {path}", file=sys.stderr)
            continue
        if path.is_file() and path.suffix == ".parquet":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.glob("*.parquet")))
        else:
            print(f"[WARN] Skipping non-parquet path: {path}", file=sys.stderr)
    return sorted(files)


def sample_rows(parquet_path: Path, limit: int, seed: int) -> pd.DataFrame:
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    if limit >= len(df):
        return df
    random_state = random.Random(seed)
    indices = random_state.sample(range(len(df)), limit)
    return df.iloc[indices].copy()


def validate_metadata(row: Dict[str, object]) -> List[str]:
    issues = []
    for field_name in CHROMA_METADATA_FIELDS:
        if field_name not in row:
            issues.append(f"missing field '{field_name}'")
        elif (
            row[field_name] in (None, "") and field_name not in OPTIONAL_METADATA_FIELDS
        ):
            issues.append(f"empty value for '{field_name}'")
    emb = row.get("embedding")
    if not isinstance(emb, list) or not emb:
        issues.append("embedding missing or empty")
    return issues


def sanitize_metadata(metadata: Dict[str, object]) -> Dict[str, object]:
    def _coerce(value: object) -> Optional[object]:
        if value is None:
            return None
        try:
            if pd.isna(value):  # type: ignore[arg-type]
                return None
        except TypeError:
            pass
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value) if value.is_integer() else value
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value

    return {key: _coerce(val) for key, val in metadata.items()}


def compute_embedding_norm(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return math.sqrt(sum(val * val for val in values))


def collect_samples(
    parquet_files: Sequence[Path], per_book_limit: int, seed: int
) -> List[SampledChunk]:
    samples: List[SampledChunk] = []
    cache = ChunkTextCache()
    for parquet_path in parquet_files:
        df = sample_rows(parquet_path, per_book_limit, seed)
        if df.empty:
            continue

        row_dicts: List[Dict[str, object]] = []
        grouped_sources: Dict[str, Set[str]] = {}

        for _, row in df.iterrows():
            row_data = row.to_dict()
            chunk_value = row_data.get("chunk_id")
            if chunk_value is None:
                continue
            chunk_id = str(chunk_value)
            row_data["chunk_id"] = chunk_id
            row_dicts.append(row_data)
            source_path = row_data.get("source_path")
            if source_path:
                grouped_sources.setdefault(source_path, set()).add(chunk_id)

        for source_path, chunk_ids in grouped_sources.items():
            cache.load_required(source_path, chunk_ids)

        for row_dict in row_dicts:
            chunk_id = row_dict["chunk_id"]
            text = cache.get_text(row_dict.get("source_path"), chunk_id)
            raw_metadata = {
                field: row_dict.get(field)
                for field in CHROMA_METADATA_FIELDS
                if field in row_dict
            }
            metadata = sanitize_metadata(raw_metadata)
            raw_embedding = row_dict.get("embedding")
            if raw_embedding is None:
                embedding_values: List[float] = []
            elif isinstance(raw_embedding, list):
                embedding_values = [float(val) for val in raw_embedding]
            else:
                embedding_values = [float(val) for val in raw_embedding]
            sample = SampledChunk(
                parquet_path=parquet_path,
                chunk_id=chunk_id,
                metadata=metadata,
                embedding=embedding_values,
                text=text,
            )
            sample.issues.extend(
                validate_metadata({**metadata, "embedding": embedding_values})
            )
            if text is None:
                sample.issues.append("chunk text not found (source_path missing?)")
            samples.append(sample)
    return samples


def print_report(samples: Sequence[SampledChunk]) -> None:
    if not samples:
        print("No samples collected.")
        return
    grouped: Dict[Path, List[SampledChunk]] = {}
    for sample in samples:
        grouped.setdefault(sample.parquet_path, []).append(sample)

    for parquet_path, group in grouped.items():
        print("=" * 80)
        print(f"File: {parquet_path} (samples: {len(group)})")
        for sample in group:
            print(f"- chunk_id: {sample.chunk_id}")
            if sample.text:
                preview = sample.text.replace("\n", " ")[:160]
                print(f"  text preview: {preview}...")
            else:
                print("  text preview: [missing]")
            if sample.issues:
                for issue in sample.issues:
                    print(f"  issue: {issue}")
            else:
                print("  issues: none")
            print(
                f"  metadata snippet: {{book_title: {sample.metadata.get('book_title')}, "
                f"collection: {sample.metadata.get('collection_label')}, "
                f"pages: {sample.metadata.get('page_start')}–{sample.metadata.get('page_end')}}}"
            )
    print("=" * 80)
    print(f"Total sampled chunks: {len(samples)}")


def write_jsonl(path: Path, samples: Sequence[SampledChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            record = sample.as_chroma_record()
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_summary(samples: Sequence[SampledChunk]) -> Dict[str, object]:
    generated_at = datetime.now(timezone.utc).isoformat()
    files: Dict[str, Dict[str, object]] = {}
    for sample in samples:
        key = str(sample.parquet_path)
        entry = files.setdefault(
            key,
            {
                "parquet_path": key,
                "sample_count": 0,
                "samples": [],
            },
        )
        entry["sample_count"] += 1
        preview = None
        if sample.text:
            preview = sample.text.replace("\n", " ")[:200]
        metadata_snippet = {
            "book_id": sample.metadata.get("book_id"),
            "book_title": sample.metadata.get("book_title"),
            "collection_label": sample.metadata.get("collection_label"),
            "page_start": sample.metadata.get("page_start"),
            "page_end": sample.metadata.get("page_end"),
        }
        entry["samples"].append(
            {
                "chunk_id": sample.chunk_id,
                "issues": sample.issues,
                "text_preview": preview,
                "metadata": metadata_snippet,
                "text_length_chars": len(sample.text) if sample.text else 0,
                "embedding_dim": len(sample.embedding) if sample.embedding else 0,
                "embedding_norm": compute_embedding_norm(sample.embedding),
            }
        )
    for entry in files.values():
        problem_chunks = [sample for sample in entry["samples"] if sample["issues"]]
        text_lengths = [
            sample.get("text_length_chars", 0) for sample in entry["samples"]
        ]
        embedding_dims = [
            sample.get("embedding_dim")
            for sample in entry["samples"]
            if sample.get("embedding_dim")
        ]
        embedding_norms = [
            sample.get("embedding_norm")
            for sample in entry["samples"]
            if sample.get("embedding_norm") is not None
        ]
        entry["problem_chunk_count"] = len(problem_chunks)
        entry["issue_count"] = sum(len(item["issues"]) for item in problem_chunks)
        entry["total_text_length_chars"] = sum(text_lengths)
        entry["avg_text_length_chars"] = (
            entry["total_text_length_chars"] / len(text_lengths) if text_lengths else 0
        )
        entry["embedding_dim"] = embedding_dims[0] if embedding_dims else None
        entry["avg_embedding_norm"] = (
            sum(embedding_norms) / len(embedding_norms) if embedding_norms else None
        )
        entry["ready_for_ingestion"] = entry["issue_count"] == 0
    return {
        "generated_at": generated_at,
        "total_samples": len(samples),
        "files": list(files.values()),
    }


def write_summary(path: Path, summary: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    parquet_files = find_parquet_files(args.inputs)
    if not parquet_files:
        print("No parquet files found in provided inputs.", file=sys.stderr)
        return 1

    samples = collect_samples(parquet_files, args.per_book_limit, args.seed)

    if not args.quiet:
        print_report(samples)

    if args.summary:
        summary_path = Path(args.summary)
        summary = build_summary(samples)
        write_summary(summary_path, summary)
        if not args.quiet:
            print(f"Wrote QA summary to {summary_path}")

    if args.jsonl:
        output_path = Path(args.jsonl)
        write_jsonl(output_path, samples)
        if not args.quiet:
            print(f"Wrote {len(samples)} Chroma-ready rows to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
