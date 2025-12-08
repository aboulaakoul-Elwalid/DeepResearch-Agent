# RAG Pipeline for Arabic Islamic Texts

This directory contains a complete RAG (Retrieval-Augmented Generation) pipeline for processing Arabic Islamic texts from the Shamela library and ingesting them into ChromaDB for semantic search.

## Index Your Own Documents (Quick Start)

Drop your `.txt`, `.md`, or `.pdf` files into a folder and index them for context-aware chat:

```bash
# 1. Create a docs folder and add your files
mkdir docs
cp your_files/*.md docs/

# 2. Run the ingestion script
python ingest_local.py

# 3. The agent can now search your docs via the local_docs_search tool!
```

### Options

```bash
# Custom input directory
python ingest_local.py --input-dir ./my_notes

# Custom ChromaDB path
python ingest_local.py --chroma-path ./my_chroma

# Clear and re-index
python ingest_local.py --reset

# Verbose output
python ingest_local.py -v
```

### Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Plain text | `.txt` | UTF-8 recommended |
| Markdown | `.md` | Preserves structure |
| PDF | `.pdf` | Requires `pypdf` (`pip install pypdf`) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_DOCS_INPUT_DIR` | `./docs` | Default input directory |
| `LOCAL_DOCS_CHROMA_PATH` | `./chroma_db` | ChromaDB storage path |
| `LOCAL_DOCS_COLLECTION` | `local_docs` | Collection name |

---

## Arabic Books Pipeline (Full)

The pipeline below is for processing the Shamela Arabic Islamic texts library.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
./run_pipeline.sh

# Or run specific stages
./run_pipeline.sh extract chunk embed ingest
```

## Pipeline Overview

```
HuggingFace Dataset
        |
        v
  [1] EXTRACT (extract_books_smart.py / extract_via_api.py)
        |
        v
  [2] ASSEMBLE (assemble_books.py)
        |
        v
  [3] CLEAN (clean_books.py)
        |
        v
  [4] MANIFEST (build_manifest.py)
        |
        v
  [5] CHUNK (create_chunks.py)
        |
        v
  [6] COMBINE (combine_chunks.py)
        |
        v
  [7] EMBED (embed_chunks.py / embed_chunks_per_book.py)
        |
        v
  [8] QA (spot_check_embeddings.py)
        |
        v
  [9] INGEST (ingest_chroma.py)
        |
        v
    ChromaDB
```

## Scripts Reference

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `extract_books_smart.py` | Extract books from HuggingFace dataset | HuggingFace dataset | `output/books/` |
| `extract_via_api.py` | Alternative extraction via HF API | HuggingFace API | `output/books/` |
| `assemble_books.py` | Combine multi-part books | `output/books/` | Assembled text files |
| `clean_books.py` | Normalize and clean text | Raw text files | `output/clean_books/` |
| `build_manifest.py` | Create metadata manifest | Books + metadata | `output/manifest/books_manifest.csv` |
| `create_chunks.py` | Split text into semantic chunks | Clean books + manifest | `output/chunks/*.jsonl` |
| `combine_chunks.py` | Merge per-book chunks | `output/chunks/*.jsonl` | `output/chunks/all_chunks.jsonl` |
| `embed_chunks.py` | Generate embeddings (monolithic) | All chunks JSONL | `output/embeddings/all_chunks_embeddings.parquet` |
| `embed_chunks_per_book.py` | Generate embeddings (per-book) | Per-book JSONL | `output/embeddings/per_book/*.parquet` |
| `spot_check_embeddings.py` | QA validation of embeddings | Parquet files | Validation report |
| `ingest_chroma.py` | Load into ChromaDB | Per-book Parquet | `chroma_db/` |

## Directory Structure

```
rag/
├── run_pipeline.sh          # Pipeline orchestration script
├── config.py                # Centralized configuration
├── chroma_schema.py         # ChromaDB metadata schema
├── requirements.txt         # Python dependencies
├── README.md                # This file
│
├── extract_books_smart.py   # HuggingFace extraction
├── extract_via_api.py       # API-based extraction
├── assemble_books.py        # Book assembly
├── clean_books.py           # Text cleaning
├── build_manifest.py        # Manifest generation
├── create_chunks.py         # Text chunking
├── combine_chunks.py        # Chunk aggregation
├── embed_chunks.py          # Monolithic embedding
├── embed_chunks_per_book.py # Per-book embedding
├── spot_check_embeddings.py # QA validation
├── ingest_chroma.py         # ChromaDB ingestion
│
└── output/                  # Generated outputs (gitignored)
    ├── books/               # Extracted books
    ├── clean_books/         # Cleaned text
    ├── manifest/            # Metadata CSV
    ├── chunks/              # JSONL chunks
    └── embeddings/          # Parquet embeddings
        └── per_book/        # Per-book Parquet files
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OUTPUT_DIR` | `output` | Base output directory |
| `CHUNK_SIZE` | `1200` | Target chunk size (characters) |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `BATCH_SIZE` | `64` | Embedding batch size |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | SentenceTransformer model |
| `CHROMA_DIR` | `chroma_db` | ChromaDB storage path |

### Using config.py

```python
from config import get_default_config

config = get_default_config()
print(config.chunking.chunk_size)  # 1200
print(config.embedding.model_name)  # paraphrase-multilingual-MiniLM-L12-v2
```

## Usage Examples

### Run Complete Pipeline

```bash
./run_pipeline.sh
```

### Run with Custom Parameters

```bash
CHUNK_SIZE=1500 BATCH_SIZE=128 ./run_pipeline.sh
```

### Process Specific Stages

```bash
# Only chunk and embed
./run_pipeline.sh chunk embed

# Skip stages that already have output
./run_pipeline.sh --skip-existing

# Dry run to see what would happen
./run_pipeline.sh --dry-run

# Process only 5 books for testing
./run_pipeline.sh --limit 5
```

### Per-Book Embedding (for ChromaDB)

```bash
# Use per-book mode for ingest_chroma.py compatibility
./run_pipeline.sh --per-book embed ingest
```

### Direct Script Usage

```bash
# Extract specific books
python extract_via_api.py

# Create chunks with custom size
python create_chunks.py --chunk-size 1500 --overlap 300

# Embed specific books
python embed_chunks_per_book.py --book-ids 4445 22669

# Ingest into ChromaDB
python ingest_chroma.py --inputs output/embeddings/per_book --collection arabic_books
```

## Metadata Schema

Each chunk carries the following metadata (see `chroma_schema.py`):

| Field | Required | Description |
|-------|----------|-------------|
| `book_id` | Yes | Numeric Shamela identifier |
| `book_title` | Yes | Arabic title |
| `author_name` | Yes | Primary author |
| `category` | Yes | Genre (Hadith, Tafsir, Fiqh, etc.) |
| `collection_label` | Yes | Human-friendly grouping |
| `jurisprudence_school` | No | Fiqh school tag |
| `era_bucket` | Yes | Historical era |
| `volume_label` | No | Volume identifier |
| `chapter_label` | No | Chapter/section label |
| `page_reference` | No | Page citation |
| `golden_subset` | Yes | QA curation flag |
| `page_start` | Yes | Start page number |
| `page_end` | Yes | End page number |
| `chunk_length_chars` | Yes | Character count |
| `source_path` | Yes | Path to source JSONL |

## Embedding Details

- **Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Dimension**: 384
- **Normalization**: L2-normalized by default
- **Format**: Parquet with zstd compression

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
BATCH_SIZE=32 ./run_pipeline.sh embed
```

### Missing sentence-transformers

```bash
pip install sentence-transformers torch
```

### ChromaDB Connection Issues

Check the ChromaDB path exists and is writable:
```bash
ls -la chroma_db/
```

### Resume After Failure

Use `--skip-existing` to skip completed stages:
```bash
./run_pipeline.sh --skip-existing
```

Or for ingest_chroma.py specifically:
```bash
python ingest_chroma.py --resume-after 12345.parquet
```

## License

See the main project LICENSE file.
