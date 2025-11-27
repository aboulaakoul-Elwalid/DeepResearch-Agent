#!/usr/bin/env python3
"""
embed_chunks.py

Simplified pipeline that reads an aggregated chunk JSONL file, generates
SentenceTransformer embeddings, writes them to Parquet, and produces a JSON
summary with useful statistics.

Usage example:
    python embed_chunks.py \
        --chunks-file output/chunks/all_chunks.jsonl \
        --parquet-file output/embeddings/all_chunks_embeddings.parquet \
        --summary-file output/embeddings/embedding_summary.json \
        --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
        --batch-size 128 \
        --normalize \
        --drop-text
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


@dataclass
class SummaryStats:
    model_name: str
    chunks_file: str
    total_chunks: int = 0
    skipped_chunks: int = 0
    embedding_dim: Optional[int] = None
    mean_norm: Optional[float] = None
    max_norm: Optional[float] = None
    min_norm: Optional[float] = None
    mean_chunk_chars: Optional[float] = None
    total_chars: int = 0

    def update_norms(self, norms: np.ndarray, dim: int) -> None:
        if norms.size == 0:
            return
        norms = norms.astype(np.float32)
        prev_total = self.total_chunks
        batch_size = norms.size
        self.total_chunks += batch_size
        self.embedding_dim = dim

        batch_mean = float(norms.mean())
        batch_max = float(norms.max())
        batch_min = float(norms.min())

        if self.mean_norm is None:
            self.mean_norm = batch_mean
            self.max_norm = batch_max
            self.min_norm = batch_min
        else:
            self.mean_norm = float(
                (self.mean_norm * prev_total + batch_mean * batch_size)
                / self.total_chunks
            )
            self.max_norm = float(max(self.max_norm, batch_max))
            self.min_norm = float(min(self.min_norm, batch_min))

    def record_chunk_lengths(self, lengths: Sequence[int]) -> None:
        if not lengths:
            return
        self.total_chars += sum(lengths)
        if self.total_chunks:
            self.mean_chunk_chars = self.total_chars / self.total_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode aggregated chunk JSONL into Parquet with embeddings."
    )
    parser.add_argument(
        "--chunks-file",
        type=Path,
        default=Path("output/chunks/all_chunks.jsonl"),
        help="Path to the aggregated chunk JSONL file.",
    )
    parser.add_argument(
        "--parquet-file",
        type=Path,
        default=Path("output/embeddings/all_chunks_embeddings.parquet"),
        help="Destination Parquet file (will be overwritten).",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path("output/embeddings/embedding_summary.json"),
        help="Path to write JSON summary statistics.",
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
        help="Number of chunk texts to encode per forward pass.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of chunks to process (0 = process all).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply L2 normalization to embeddings.",
    )
    parser.add_argument(
        "--drop-text",
        action="store_true",
        help="Omit the raw chunk text column in the Parquet output.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Round embeddings to this many decimal places (<=0 keeps float32).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )
    return parser.parse_args()


def stream_chunks(path: Path, limit: int = 0) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            if limit and idx > limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path} line {idx}: {exc}") from exc


def batched(iterable: Iterable[Dict], batch_size: int) -> Iterator[List[Dict]]:
    batch: List[Dict] = []
    for record in iterable:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def preprocess_record(record: Dict) -> Optional[Dict]:
    cleaned = dict(record)

    def _to_str(value: Optional[str]) -> str:
        return "" if value is None else str(value)

    def _to_int(value: object) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    text = _to_str(cleaned.get("text"))
    if not text:
        return None
    cleaned["text"] = text

    for key in (
        "collection_label",
        "jurisprudence_school",
        "era_bucket",
        "volume_label",
        "chapter_label",
        "page_reference",
    ):
        cleaned[key] = _to_str(cleaned.get(key))

    cleaned["volume_label"] = cleaned["volume_label"] or _to_str(cleaned.get("volume"))

    golden_value = cleaned.get("golden_subset", True)
    if isinstance(golden_value, str):
        golden_value = golden_value.strip().lower()
        golden_value = golden_value in {"1", "true", "yes", "y"}
    cleaned["golden_subset"] = bool(golden_value)

    cleaned["page_start"] = _to_int(cleaned.get("page_start"))
    cleaned["page_end"] = _to_int(cleaned.get("page_end"))
    cleaned["chunk_length_chars"] = len(text)

    if not cleaned.get("chunk_id") or not cleaned.get("book_id"):
        return None

    return cleaned


def build_schema(include_text: bool) -> pa.schema:
    fields = [
        pa.field("chunk_id", pa.string()),
        pa.field("book_id", pa.string()),
        pa.field("book_title", pa.string()),
        pa.field("author_name", pa.string()),
        pa.field("category", pa.string()),
    ]
    if include_text:
        fields.append(pa.field("text", pa.string()))
    fields.extend(
        [
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
    return pa.schema(fields)


def encode_embeddings(
    model: SentenceTransformer,
    texts: Sequence[str],
    normalize: bool,
    precision: int,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=len(texts),
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

    if precision > 0:
        embeddings = np.round(embeddings, decimals=precision)

    return embeddings


def create_table(
    batch: Sequence[Dict],
    embeddings: np.ndarray,
    include_text: bool,
    source_path: str,
) -> pa.Table:
    chunk_ids: List[str] = []
    book_ids: List[str] = []
    titles: List[str] = []
    authors: List[str] = []
    categories: List[str] = []
    texts: List[str] = [] if include_text else []
    collection_labels: List[str] = []
    jurisprudence: List[str] = []
    era_buckets: List[str] = []
    volume_labels: List[str] = []
    chapter_labels: List[str] = []
    page_refs: List[str] = []
    golden_subset: List[bool] = []
    page_starts: List[Optional[int]] = []
    page_ends: List[Optional[int]] = []
    chunk_lengths: List[int] = []
    sources: List[str] = []
    embedding_dims: List[int] = []
    vectors: List[List[float]] = []

    for record, vector in zip(batch, embeddings):
        chunk_ids.append(str(record.get("chunk_id", "")))
        book_ids.append(str(record.get("book_id", "")))
        titles.append(record.get("book_title", ""))
        authors.append(record.get("author_name", ""))
        categories.append(record.get("category", ""))
        if include_text:
            texts.append(record.get("text", ""))
        collection_labels.append(record.get("collection_label", ""))
        jurisprudence.append(record.get("jurisprudence_school", ""))
        era_buckets.append(record.get("era_bucket", ""))
        volume_labels.append(record.get("volume_label", ""))
        chapter_labels.append(record.get("chapter_label", ""))
        page_refs.append(record.get("page_reference", ""))
        golden_subset.append(bool(record.get("golden_subset", True)))
        page_starts.append(record.get("page_start"))
        page_ends.append(record.get("page_end"))
        chunk_lengths.append(
            int(record.get("chunk_length_chars", len(record.get("text", ""))))
        )
        sources.append(source_path)
        embedding_dims.append(len(vector))
        vectors.append(vector.tolist())

    arrays: List[pa.Array] = [
        pa.array(chunk_ids, pa.string()),
        pa.array(book_ids, pa.string()),
        pa.array(titles, pa.string()),
        pa.array(authors, pa.string()),
        pa.array(categories, pa.string()),
    ]
    if include_text:
        arrays.append(pa.array(texts, pa.string()))
    arrays.extend(
        [
            pa.array(collection_labels, pa.string()),
            pa.array(jurisprudence, pa.string()),
            pa.array(era_buckets, pa.string()),
            pa.array(volume_labels, pa.string()),
            pa.array(chapter_labels, pa.string()),
            pa.array(page_refs, pa.string()),
            pa.array(golden_subset, pa.bool_()),
            pa.array(page_starts, pa.int32()),
            pa.array(page_ends, pa.int32()),
            pa.array(chunk_lengths, pa.int32()),
            pa.array(sources, pa.string()),
            pa.array(embedding_dims, pa.int32()),
            pa.array(vectors, pa.list_(pa.float32())),
        ]
    )

    return pa.Table.from_arrays(arrays, schema=build_schema(include_text))


def save_summary(path: Path, stats: SummaryStats) -> None:
    ensure_parent(path)
    payload = asdict(stats)
    payload["mean_chunk_chars"] = (
        float(payload["mean_chunk_chars"])
        if payload["mean_chunk_chars"] is not None
        else None
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.chunks_file.is_file():
        raise FileNotFoundError(f"Chunks file not found: {args.chunks_file}")

    ensure_parent(args.parquet_file)
    model = SentenceTransformer(args.model)
    stats = SummaryStats(
        model_name=args.model,
        chunks_file=str(args.chunks_file.resolve()),
    )
    include_text = not args.drop_text
    schema = build_schema(include_text)
    writer = pq.ParquetWriter(
        args.parquet_file,
        schema,
        compression="zstd",
    )

    total = 0
    chunk_iter = stream_chunks(args.chunks_file, limit=args.limit)

    try:
        for batch in tqdm(
            batched(chunk_iter, args.batch_size),
            desc="Embedding batches",
            unit="batch",
        ):
            processed_batch: List[Dict] = []
            for record in batch:
                cleaned = preprocess_record(record)
                if cleaned is None:
                    stats.skipped_chunks += 1
                    continue
                processed_batch.append(cleaned)

            if not processed_batch:
                continue

            texts = [item["text"] for item in processed_batch]
            embeddings = encode_embeddings(
                model=model,
                texts=texts,
                normalize=args.normalize,
                precision=args.precision,
            )
            norms = np.linalg.norm(embeddings, axis=1)
            stats.update_norms(norms, embeddings.shape[1])
            stats.record_chunk_lengths(
                [item["chunk_length_chars"] for item in processed_batch]
            )

            table = create_table(
                batch=processed_batch,
                embeddings=embeddings,
                include_text=include_text,
                source_path=str(args.chunks_file),
            )
            writer.write_table(table)
            total += len(processed_batch)
    finally:
        writer.close()

    logging.info("Embedding complete. Chunks processed: %d", total)
    logging.info("Chunks skipped (invalid/empty): %d", stats.skipped_chunks)
    logging.info("Parquet output written to %s", args.parquet_file.resolve())

    if args.summary_file:
        stats.total_chunks = total
        save_summary(args.summary_file, stats)
        logging.info("Summary written to %s", args.summary_file.resolve())


if __name__ == "__main__":
    main()
