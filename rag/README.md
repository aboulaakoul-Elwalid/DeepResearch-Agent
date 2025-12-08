# RAG Pipeline: From Documents to Searchable Knowledge Base

A complete pipeline for processing documents into a searchable ChromaDB vector database for Retrieval-Augmented Generation (RAG) applications. Originally built for Arabic Islamic texts (Shamela library), but **works with any documents**.

## Overview

This pipeline processes documents through multiple stages:

1. **Extraction** - Get text from your source (PDFs, HuggingFace, plain text)
2. **Chunking** - Split documents into overlapping semantic chunks
3. **Embedding** - Generate vector embeddings using SentenceTransformers
4. **Ingestion** - Load embeddings into ChromaDB for retrieval
5. **Search** - Query via DR-Tulu MCP tools

---

## Bring Your Own Documents (BYOD)

**Want to add your own documents to DR-Tulu?** Follow this simplified path:

### Quick Path: JSONL → ChromaDB (3 Steps)

#### Step 1: Create Chunks JSONL

Your documents need to be in JSONL format:

```json
{"chunk_id": "doc1_001", "book_id": "doc1", "book_title": "My Document", "text": "Your content here..."}
{"chunk_id": "doc1_002", "book_id": "doc1", "book_title": "My Document", "text": "More content..."}
```

**From plain text files:**
```bash
# Simple Python script to create chunks
python -c "
import json
from pathlib import Path

for txt_file in Path('my_docs').glob('*.txt'):
    text = txt_file.read_text()
    # Split by paragraphs
    chunks = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
    for i, chunk in enumerate(chunks):
        print(json.dumps({
            'chunk_id': f'{txt_file.stem}_{i:04d}',
            'book_id': txt_file.stem,
            'book_title': txt_file.stem,
            'text': chunk
        }, ensure_ascii=False))
" > output/chunks/my_docs.jsonl
```

**From PDF files:**
```bash
pip install pdfplumber

python -c "
import pdfplumber
import json
import sys

for pdf_path in sys.argv[1:]:
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ''
            if text.strip():
                print(json.dumps({
                    'chunk_id': f'{pdf_path}_{i}',
                    'book_id': pdf_path.replace('.pdf', ''),
                    'book_title': pdf_path.replace('.pdf', ''),
                    'text': text,
                    'page': i + 1
                }, ensure_ascii=False))
" *.pdf > output/chunks/pdfs.jsonl
```

**From Markdown files:**
```bash
python -c "
import json
from pathlib import Path

for md in Path('.').glob('**/*.md'):
    text = md.read_text()
    chunks = [p for p in text.split('\n\n') if len(p) > 100]
    for i, chunk in enumerate(chunks):
        print(json.dumps({
            'chunk_id': f'{md.stem}_{i}',
            'book_id': md.stem,
            'book_title': md.stem,
            'text': chunk
        }, ensure_ascii=False))
" > output/chunks/docs.jsonl
```

#### Step 2: Generate Embeddings

```bash
python embed_chunks.py \
    --chunks-file output/chunks/my_docs.jsonl \
    --parquet-file output/embeddings/my_docs.parquet \
    --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    --batch-size 128 \
    --normalize
```

#### Step 3: Ingest into ChromaDB

```bash
python ingest_chroma.py \
    --inputs output/embeddings \
    --chroma-path ./chroma_db \
    --collection my_knowledge_base
```

#### Step 4: Configure DR-Tulu

Add to your `.env`:

```bash
ARABIC_BOOKS_CHROMA_PATH=/path/to/your/chroma_db
ARABIC_BOOKS_COLLECTION=my_knowledge_base
```

**Done!** DR-Tulu can now search your documents.

### Recommended Embedding Models

| Model | Languages | Dim | Speed | Quality |
|-------|-----------|-----|-------|---------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 50+ (Arabic, English, etc.) | 384 | Fast | Good |
| `all-MiniLM-L6-v2` | English | 384 | Very Fast | Good |
| `all-mpnet-base-v2` | English | 768 | Medium | Best |
| `intfloat/multilingual-e5-large` | 100+ | 1024 | Slow | Excellent |

### Creating a Custom MCP Tool

To add your knowledge base as a named tool (instead of reusing `arabic_books_search`):

1. Copy the template:
```bash
cp ../DR-Tulu/agent/dr_agent/mcp_backend/local/arabic_books.py \
   ../DR-Tulu/agent/dr_agent/mcp_backend/local/my_docs.py
```

2. Edit `my_docs.py` - change function name and env vars:
```python
def search_my_docs(query: str, n_results: int = 5, ...):
    resolved_chroma_path = os.environ.get("MY_DOCS_CHROMA_PATH", "./chroma_db")
    resolved_collection = os.environ.get("MY_DOCS_COLLECTION", "my_docs")
    # ... rest same as arabic_books.py
```

3. Register in `../DR-Tulu/agent/dr_agent/mcp_backend/main.py`

4. Add to workflow prompt in `unified_tool_calling_deep.yaml`

---

## Shamela Arabic Books Pipeline

The full pipeline below is designed for the Shamela Islamic texts dataset, but the embedding/ingestion scripts work with any JSONL chunks.

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Or install from parent directory
pip install -r ../requirements.txt
```

### Run the Complete Pipeline

```bash
# Run all stages
./run_pipeline.sh all

# Or run with options
./run_pipeline.sh all --verbose --limit 5
```

### Run Individual Stages

```bash
# Extract books from HuggingFace
./run_pipeline.sh extract

# Process specific stages
./run_pipeline.sh clean chunk embed

# Skip stages with existing output
./run_pipeline.sh all --skip-existing
```

## Scripts Reference

### Data Extraction

| Script | Description | When to Use |
|--------|-------------|-------------|
| `extract_books_smart.py` | Memory-efficient streaming extraction | **Recommended** - handles large datasets |
| `extract_books.py` | Basic extraction (loads all into memory) | Small datasets only |
| `extract_via_api.py` | API-based extraction (no download) | When bandwidth is limited |

### Processing Pipeline

| Script | Input | Output |
|--------|-------|--------|
| `assemble_books.py` | Parquet pages + metadata | `output/assembled_books/*.md` |
| `clean_books.py` | Assembled markdown | `output/clean_books/*.txt` |
| `build_manifest.py` | All metadata sources | `output/manifest/books_manifest.csv` |
| `create_chunks.py` | Clean text + manifest | `output/chunks/{book_id}.jsonl` |
| `combine_chunks.py` | Per-book JSONL | `output/chunks/all_chunks.jsonl` |

### Embedding & Ingestion

| Script | Input | Output |
|--------|-------|--------|
| `embed_chunks.py` | Combined JSONL | Single Parquet file |
| `embed_chunks_per_book.py` | Per-book JSONL | Per-book Parquet files |
| `ingest_chroma.py` | Per-book Parquet | ChromaDB collection |
| `spot_check_embeddings.py` | Parquet files | QA report |

### Utilities

| Script | Purpose |
|--------|---------|
| `chroma_schema.py` | Shared schema constants and validation |
| `run_pipeline.sh` | End-to-end orchestration |

## Directory Structure

```
rag/
├── run_pipeline.sh              # Pipeline orchestrator
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── # Extraction scripts
├── extract_books.py
├── extract_books_smart.py
├── extract_via_api.py
│
├── # Processing scripts
├── assemble_books.py
├── clean_books.py
├── build_manifest.py
├── create_chunks.py
├── combine_chunks.py
│
├── # Embedding scripts
├── embed_chunks.py
├── embed_chunks_per_book.py
│
├── # Ingestion & QA
├── ingest_chroma.py
├── spot_check_embeddings.py
├── chroma_schema.py
│
└── output/                      # Generated artifacts
    ├── metadata/
    │   ├── matched_books.json
    │   └── shamela_info.parquet
    ├── assembled_books/         # Complete markdown books
    ├── clean_books/             # Normalized text files
    ├── manifest/
    │   └── books_manifest.csv
    ├── chunks/                  # JSONL chunk files
    │   ├── {book_id}.jsonl
    │   └── all_chunks.jsonl
    ├── embeddings/
    │   ├── per_book/            # Per-book Parquet files
    │   └── embedding_summary.json
    ├── stats/                   # Statistics and reports
    └── qa/                      # QA check results
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OUTPUT_DIR` | `output` | Base output directory |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding model |
| `CHUNK_SIZE` | `1200` | Target chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `BATCH_SIZE` | `64` | Embedding batch size |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB storage path |
| `CHROMA_COLLECTION` | `arabic_books` | ChromaDB collection name |

### Example with Custom Config

```bash
EMBEDDING_MODEL="intfloat/multilingual-e5-large" \
CHUNK_SIZE=1500 \
CHROMA_PATH="/data/vectors/shamela" \
./run_pipeline.sh embed ingest
```

## Detailed Usage

### Extraction

```bash
# Recommended: Memory-efficient streaming
python extract_books_smart.py

# Alternative: API-based (no large downloads)
python extract_via_api.py
```

### Chunking

```bash
# Create chunks with custom size
python create_chunks.py \
    --manifest output/manifest/books_manifest.csv \
    --input-dir output/clean_books \
    --output-dir output/chunks \
    --chunk-size 1500 \
    --overlap 300

# Combine into single file
python combine_chunks.py \
    --input-dir output/chunks \
    --output output/chunks/all_chunks.jsonl \
    --include-source-path
```

### Embedding

```bash
# Per-book embeddings (required for ingest_chroma.py)
python embed_chunks_per_book.py \
    --chunks-dir output/chunks \
    --output-dir output/embeddings/per_book \
    --model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
    --batch-size 128 \
    --normalize

# Single-file embeddings (alternative)
python embed_chunks.py \
    --chunks-file output/chunks/all_chunks.jsonl \
    --parquet-file output/embeddings/all_embeddings.parquet \
    --normalize
```

### ChromaDB Ingestion

```bash
# Ingest all books
python ingest_chroma.py \
    --inputs output/embeddings/per_book \
    --chroma-path ./chroma_db \
    --collection arabic_books

# Ingest specific books
python ingest_chroma.py \
    --inputs output/embeddings/per_book \
    --book-ids 4445 22669 \
    --batch-size 500

# Dry run (validate without writing)
python ingest_chroma.py \
    --inputs output/embeddings/per_book \
    --dry-run
```

### Quality Assurance

```bash
# Check embeddings before ingestion
python spot_check_embeddings.py \
    --inputs output/embeddings/per_book \
    --per-book-limit 5 \
    --summary output/qa/report.json
```

## Metadata Schema

Each chunk in ChromaDB includes the following metadata:

| Field | Required | Description |
|-------|----------|-------------|
| `book_id` | Yes | Shamela book identifier |
| `book_title` | Yes | Arabic book title |
| `author_name` | Yes | Author or compiler |
| `category` | Yes | Genre (Hadith, Tafsir, etc.) |
| `collection_label` | Yes | Human-friendly grouping |
| `jurisprudence_school` | No | Fiqh school (Hanafi, etc.) |
| `era_bucket` | Yes | Historical era |
| `golden_subset` | Yes | Curated QA flag |
| `page_start` | Yes | Start page number |
| `page_end` | Yes | End page number |
| `chunk_length_chars` | Yes | Character count |
| `source_path` | Yes | Path to source JSONL |

## Troubleshooting

### Out of Memory During Extraction

Use the streaming extractor:
```bash
python extract_books_smart.py
```

### Missing Dependencies

```bash
pip install datasets pandas pyarrow tqdm sentence-transformers chromadb torch
```

### ChromaDB Ingestion Fails

Ensure you have per-book Parquet files:
```bash
python embed_chunks_per_book.py --chunks-dir output/chunks --output-dir output/embeddings/per_book
```

### Embedding Model Not Found

Install the model explicitly:
```bash
pip install sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
```

## Pipeline Flow

```
HuggingFace Datasets
        │
        ▼
┌─────────────────┐
│   EXTRACTION    │ extract_books_smart.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    ASSEMBLY     │ assemble_books.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    CLEANING     │ clean_books.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    MANIFEST     │ build_manifest.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    CHUNKING     │ create_chunks.py
└────────┬────────┘   combine_chunks.py
         │
         ▼
┌─────────────────┐
│   EMBEDDING     │ embed_chunks_per_book.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   INGESTION     │ ingest_chroma.py
└────────┬────────┘
         │
         ▼
    ChromaDB
   (Searchable)
```

## License

See the main project LICENSE file.

---

## Testing Your Knowledge Base

### Quick Test with Python

```python
import chromadb

# Connect to your ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("my_knowledge_base")

# Test a query
results = collection.query(
    query_texts=["your test query"],
    n_results=3
)

for i, doc in enumerate(results["documents"][0]):
    print(f"Result {i+1}: {doc[:200]}...")
```

### Test with spot_check_embeddings.py

```bash
python spot_check_embeddings.py \
    --chroma-path ./chroma_db \
    --collection my_knowledge_base \
    --query "test query"
```

### Test with DR-Tulu

Start the gateway and ask a question:

```bash
# Start services
cd .. && make run

# In browser, ask DR-Tulu:
# "Using the arabic_books_search tool, search for: <your query>"
```

## Chunk Format Reference

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | string | Unique identifier |
| `text` | string | Text content |

### Recommended Fields

| Field | Type | Description |
|-------|------|-------------|
| `book_id` | string | Document identifier |
| `book_title` | string | Human-readable title |
| `page_start` | int | Starting page |
| `page_end` | int | Ending page |
| `author` | string | Author name |
| `source_url` | string | Original source |

All metadata is preserved in ChromaDB and returned in search results.
