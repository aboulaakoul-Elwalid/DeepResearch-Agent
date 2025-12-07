import os
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


def search_arabic_books(
    query: str,
    n_results: int = 5,
    chroma_path: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search a local ChromaDB collection containing Arabic classical texts.

    Args:
        query: Natural language query in Arabic or English.
        n_results: Number of passages to return (default 5, max 10).
        chroma_path: Optional override for Chroma persistence directory.
        collection_name: Optional override for the collection name.

    Environment overrides (if args are omitted):
        - ARABIC_BOOKS_CHROMA_PATH
        - ARABIC_BOOKS_COLLECTION
    """
    # Resolve configuration from args -> env -> defaults
    resolved_chroma_path = chroma_path or os.environ.get(
        "ARABIC_BOOKS_CHROMA_PATH", "/home/elwalid/projects/parallax_project/chroma_db"
    )
    resolved_collection = collection_name or os.environ.get(
        "ARABIC_BOOKS_COLLECTION", "arabic_books"
    )

    effective_k = max(1, min(int(n_results), 10))

    try:
        collection = _load_collection(
            chroma_path=resolved_chroma_path, collection_name=resolved_collection
        )
    except Exception as exc:  # pragma: no cover - defensive surface for server
        return {
            "error": f"Failed to open Chroma collection '{resolved_collection}' at '{resolved_chroma_path}': {exc}",
            "data": [],
        }

    try:
        results = collection.query(
            query_texts=[query],
            n_results=effective_k,
            # Chroma returns ids by default; requesting "ids" raises in newer versions.
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:  # pragma: no cover - defensive surface for server
        return {
            "error": f"Chroma query failed: {exc}",
            "data": [],
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
