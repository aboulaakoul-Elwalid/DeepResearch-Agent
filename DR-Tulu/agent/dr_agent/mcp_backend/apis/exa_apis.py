import os
from typing import Any, Dict, List, Optional

import requests

EXA_BASE_URL = os.environ.get("EXA_API_BASE", "https://api.exa.ai").rstrip("/")


class ExaApiError(Exception):
    """Raised when Exa API returns an error response or missing config."""


def _build_headers() -> Dict[str, str]:
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        raise ExaApiError("Missing EXA_API_KEY environment variable")
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-API-KEY": api_key,
    }


def search_exa(
    query: str,
    num_results: int = 5,
    use_autoprompt: bool = True,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    search_type: str = "neural",
) -> Dict[str, Any]:
    """
    Call Exa's /search endpoint.

    Args:
        query: search query
        num_results: number of results to return (clamped to 1..20)
        use_autoprompt: whether to let Exa refine the query
        include_domains: restrict search to these domains
        exclude_domains: exclude these domains
        search_type: Exa search type (e.g., "neural", "keyword")
    """
    headers = _build_headers()
    k = max(1, min(int(num_results), 20))
    payload: Dict[str, Any] = {
        "query": query,
        "numResults": k,
        "useAutoprompt": bool(use_autoprompt),
        "type": search_type,
    }
    if include_domains:
        payload["includeDomains"] = include_domains
    if exclude_domains:
        payload["excludeDomains"] = exclude_domains

    resp = requests.post(
        f"{EXA_BASE_URL}/search", headers=headers, json=payload, timeout=30
    )
    if resp.status_code >= 400:
        raise ExaApiError(f"Exa search failed ({resp.status_code}): {resp.text}")
    return resp.json()


def search_exa_websets(
    query: str,
    webset_id: Optional[str] = None,
    num_results: int = 5,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Call Exa's websets search endpoint.

    Note: Endpoint path is based on Exa Websets MCP reference. If Exa updates
    the path, set EXA_WEBSETS_PATH env var accordingly.
    """
    headers = _build_headers()
    k = max(1, min(int(num_results), 20))
    payload: Dict[str, Any] = {
        "query": query,
        "numResults": k,
    }
    if webset_id:
        payload["websetId"] = webset_id
    if include_domains:
        payload["includeDomains"] = include_domains
    if exclude_domains:
        payload["excludeDomains"] = exclude_domains

    websets_path = os.environ.get("EXA_WEBSETS_PATH", "/websets/search").lstrip("/")
    resp = requests.post(
        f"{EXA_BASE_URL}/{websets_path}",
        headers=headers,
        json=payload,
        timeout=30,
    )
    if resp.status_code >= 400:
        raise ExaApiError(f"Exa websets search failed ({resp.status_code}): {resp.text}")
    return resp.json()
