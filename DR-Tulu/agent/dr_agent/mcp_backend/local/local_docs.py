"""
local_docs.py

Search function for locally indexed documents (txt, md, pdf).

This module provides ChromaDB search over the "local_docs" collection,
which is populated by `rag/ingest_local.py`.

Usage:
    from dr_agent.mcp_backend.local.local_docs import search_local_docs

    results = search_local_docs("How do I configure the API?", n_results=5)

Environment Variables:
    LOCAL_DOCS_CHROMA_PATH  - ChromaDB persistence path (default: project/chroma_db)
    LOCAL_DOCS_COLLECTION   - Collection name (default: local_docs)
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_collection(
    chroma_path: str, collection_name: str
):  # pragma: no cover - thin wrapper
    """
    Lazily import chromadb to keep server startup light and return a collection handle.

    We keep this in a helper so that unit tests can patch it easily without importing
    chromadb globally.
    """
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=chroma_path, settings=Settings(anonymized_telemetry=False)
    )
    return client.get_collection(collection_name)


# Default path relative to project root
_DEFAULT_CHROMA_PATH = str(Path(__file__).resolve().parents[5] / "chroma_db")


def search_local_docs(
    query: str,
    n_results: int = 5,
    chroma_path: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search a local ChromaDB collection containing user-indexed documents.

    Args:
        query: Natural language search query.
        n_results: Number of passages to return (default 5, max 10).
        chroma_path: Optional override for Chroma persistence directory.
        collection_name: Optional override for the collection name.

    Environment overrides (if args are omitted):
        - LOCAL_DOCS_CHROMA_PATH
        - LOCAL_DOCS_COLLECTION

    Returns:
        Dictionary with:
            - query: The original query
            - collection: The collection name used
            - chroma_path: The Chroma path used
            - results: List of matching documents, each with:
                - id: Chunk identifier
                - text: The chunk content
                - metadata: Source file info (filename, file_type, chunk_index)
                - distance: Similarity distance (lower = more similar)
            - error: Error message if something went wrong (optional)
    """
    # Resolve configuration from args -> env -> defaults
    resolved_chroma_path = chroma_path or os.environ.get(
        "LOCAL_DOCS_CHROMA_PATH", _DEFAULT_CHROMA_PATH
    )
    resolved_collection = collection_name or os.environ.get(
        "LOCAL_DOCS_COLLECTION", "local_docs"
    )

    effective_k = max(1, min(int(n_results), 10))

    try:
        collection = _load_collection(
            chroma_path=resolved_chroma_path, collection_name=resolved_collection
        )
    except Exception as exc:  # pragma: no cover - defensive surface for server
        return {
            "query": query,
            "collection": resolved_collection,
            "chroma_path": resolved_chroma_path,
            "error": f"Failed to open Chroma collection '{resolved_collection}' at '{resolved_chroma_path}': {exc}",
            "results": [],
        }

    try:
        results = collection.query(
            query_texts=[query],
            n_results=effective_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:  # pragma: no cover - defensive surface for server
        return {
            "query": query,
            "collection": resolved_collection,
            "chroma_path": resolved_chroma_path,
            "error": f"Chroma query failed: {exc}",
            "results": [],
        }

    documents: List[List[str]] = results.get("documents", [[]])
    metadatas: List[List[Dict[str, Any]]] = results.get("metadatas", [[]])
    ids: List[List[str]] = results.get("ids", [[]])
    distances: List[List[float]] = results.get("distances", [[]])

    payload = []
    for idx, doc in enumerate(documents[0] if documents else []):
        metadata = {}
        if metadatas and metadatas[0] and idx < len(metadatas[0]):
            metadata = metadatas[0][idx] or {}

        record_id = None
        if ids and ids[0] and idx < len(ids[0]):
            record_id = ids[0][idx]

        distance = None
        if distances and distances[0] and idx < len(distances[0]):
            distance = distances[0][idx]

        payload.append(
            {
                "id": record_id,
                "text": doc,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return {
        "query": query,
        "collection": resolved_collection,
        "chroma_path": resolved_chroma_path,
        "results": payload,
    }
