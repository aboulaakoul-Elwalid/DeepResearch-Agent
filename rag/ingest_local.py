#!/usr/bin/env python3
"""
ingest_local.py

All-in-one script to ingest local documents (.txt, .md, .pdf) into ChromaDB.

Usage:
    python ingest_local.py                          # Index ./docs/ into ChromaDB
    python ingest_local.py --input-dir ./my_docs    # Custom input directory
    python ingest_local.py --reset                  # Clear collection before ingesting
    python ingest_local.py --collection my_docs     # Custom collection name

Environment Variables:
    LOCAL_DOCS_INPUT_DIR     - Default input directory (default: ./docs)
    LOCAL_DOCS_CHROMA_PATH   - ChromaDB persistence path (default: ./chroma_db)
    LOCAL_DOCS_COLLECTION    - Collection name (default: local_docs)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# Default configuration - sensible defaults for OSS
DEFAULT_INPUT_DIR = os.environ.get("LOCAL_DOCS_INPUT_DIR", "./docs")
DEFAULT_CHROMA_PATH = os.environ.get(
    "LOCAL_DOCS_CHROMA_PATH", str(Path(__file__).parent.parent / "chroma_db")
)
DEFAULT_COLLECTION = os.environ.get("LOCAL_DOCS_COLLECTION", "local_docs")

# Chunking settings - balanced for general documents
CHUNK_SIZE = 1200  # characters
CHUNK_OVERLAP = 200  # characters
MIN_CHUNK_SIZE = 100  # minimum to keep

# Embedding model - multilingual, good for mixed content
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def find_documents(input_dir: Path) -> List[Path]:
    """Find all supported documents recursively."""
    supported_extensions = {".txt", ".md", ".pdf"}
    documents = []

    for ext in supported_extensions:
        documents.extend(input_dir.rglob(f"*{ext}"))

    return sorted(documents)


def read_text_file(path: Path) -> str:
    """Read a text or markdown file."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    # Fallback: read with errors ignored
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    """Read a PDF file and extract text."""
    try:
        import pypdf
    except ImportError:
        try:
            import PyPDF2 as pypdf
        except ImportError:
            print(
                f"  [SKIP] {path.name}: Install 'pypdf' for PDF support (pip install pypdf)"
            )
            return ""

    try:
        text_parts = []
        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except Exception as e:
        print(f"  [WARN] Failed to read {path.name}: {e}")
        return ""


def read_document(path: Path) -> Tuple[str, str]:
    """
    Read a document and return (content, file_type).
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        content = read_pdf_file(path)
        file_type = "pdf"
    elif suffix == ".md":
        content = read_text_file(path)
        file_type = "markdown"
    else:  # .txt and others
        content = read_text_file(path)
        file_type = "text"

    return content, file_type


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_size: int = MIN_CHUNK_SIZE,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Uses paragraph boundaries when possible for cleaner splits.
    """
    if not text or len(text.strip()) < min_size:
        return []

    # Normalize whitespace
    text = text.strip()

    # Split on paragraph boundaries first
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph exceeds chunk size
        if len(current_chunk) + len(para) + 2 > chunk_size:
            # Save current chunk if it meets minimum
            if len(current_chunk) >= min_size:
                chunks.append(current_chunk.strip())

            # Start new chunk with overlap from previous
            if current_chunk and overlap > 0:
                # Take last 'overlap' characters as context
                overlap_text = current_chunk[-overlap:].strip()
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = para
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    # Don't forget the last chunk
    if len(current_chunk) >= min_size:
        chunks.append(current_chunk.strip())

    return chunks


def process_documents(
    input_dir: Path,
    verbose: bool = False,
) -> Iterator[Dict]:
    """
    Process all documents and yield chunks with metadata.

    Yields:
        Dict with keys: id, text, metadata
    """
    documents = find_documents(input_dir)

    if not documents:
        print(f"No documents found in {input_dir}")
        return

    print(f"Found {len(documents)} document(s) in {input_dir}")

    total_chunks = 0

    for doc_path in documents:
        rel_path = doc_path.relative_to(input_dir)

        if verbose:
            print(f"  Processing: {rel_path}")

        content, file_type = read_document(doc_path)

        if not content:
            continue

        chunks = chunk_text(content)

        if verbose:
            print(f"    → {len(chunks)} chunks")

        for i, chunk_text_content in enumerate(chunks):
            chunk_id = f"{rel_path}::chunk_{i:04d}"

            yield {
                "id": chunk_id,
                "text": chunk_text_content,
                "metadata": {
                    "source": str(rel_path),
                    "filename": doc_path.name,
                    "file_type": file_type,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            }
            total_chunks += 1

    print(f"Total: {total_chunks} chunks from {len(documents)} documents")


def ingest_to_chroma(
    input_dir: Path,
    chroma_path: Path,
    collection_name: str,
    reset: bool = False,
    verbose: bool = False,
) -> int:
    """
    Ingest documents into ChromaDB.

    Returns:
        Number of chunks ingested
    """
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        print("ERROR: chromadb not installed. Run: pip install chromadb")
        sys.exit(1)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "ERROR: sentence-transformers not installed. Run: pip install sentence-transformers"
        )
        sys.exit(1)

    # Initialize ChromaDB
    print(f"ChromaDB path: {chroma_path}")
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )

    # Handle collection reset
    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except ValueError:
            pass  # Collection doesn't exist

    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Local documents indexed for RAG"},
    )

    existing_count = collection.count()
    if existing_count > 0 and not reset:
        print(f"Collection '{collection_name}' has {existing_count} existing documents")

    # Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Process and ingest
    batch_size = 100
    batch_ids = []
    batch_texts = []
    batch_metadatas = []
    total_ingested = 0

    for chunk in process_documents(input_dir, verbose=verbose):
        batch_ids.append(chunk["id"])
        batch_texts.append(chunk["text"])
        batch_metadatas.append(chunk["metadata"])

        if len(batch_ids) >= batch_size:
            # Embed batch
            embeddings = model.encode(batch_texts, normalize_embeddings=True)

            # Add to collection
            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=embeddings.tolist(),
                metadatas=batch_metadatas,
            )

            total_ingested += len(batch_ids)
            print(f"  Ingested {total_ingested} chunks...")

            # Reset batch
            batch_ids = []
            batch_texts = []
            batch_metadatas = []

    # Handle remaining batch
    if batch_ids:
        embeddings = model.encode(batch_texts, normalize_embeddings=True)
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings.tolist(),
            metadatas=batch_metadatas,
        )
        total_ingested += len(batch_ids)

    final_count = collection.count()
    print(f"\nDone! Collection '{collection_name}' now has {final_count} documents")

    return total_ingested


def main():
    parser = argparse.ArgumentParser(
        description="Ingest local documents into ChromaDB for RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ingest_local.py                          # Index ./docs/
    python ingest_local.py --input-dir ./my_docs    # Custom directory
    python ingest_local.py --reset                  # Clear and re-index
    python ingest_local.py -v                       # Verbose output

Supported formats: .txt, .md, .pdf (requires 'pypdf' package)
        """,
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help=f"Directory containing documents (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--chroma-path",
        "-c",
        type=Path,
        default=Path(DEFAULT_CHROMA_PATH),
        help=f"ChromaDB persistence path (default: {DEFAULT_CHROMA_PATH})",
    )
    parser.add_argument(
        "--collection",
        "-n",
        type=str,
        default=DEFAULT_COLLECTION,
        help=f"Collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--reset",
        "-r",
        action="store_true",
        help="Clear existing collection before ingesting",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        print(f"ERROR: Input directory not found: {args.input_dir}")
        print(f"\nCreate it and add your documents:")
        print(f"  mkdir -p {args.input_dir}")
        print(f"  cp your_docs/*.md {args.input_dir}/")
        sys.exit(1)

    if not args.input_dir.is_dir():
        print(f"ERROR: Not a directory: {args.input_dir}")
        sys.exit(1)

    # Run ingestion
    count = ingest_to_chroma(
        input_dir=args.input_dir,
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        reset=args.reset,
        verbose=args.verbose,
    )

    if count == 0:
        print("\nNo documents were ingested. Check your input directory.")
        sys.exit(1)

    print(f"\nYour documents are now searchable via the 'local_docs_search' tool!")


if __name__ == "__main__":
    main()
