"""DuckDuckGo search adapter for local model market/industry claim verification.

Local models (Ollama) have no web access, so we fetch search results via
DuckDuckGo and feed them into the model's reasoning. The model reads the
results and assesses the claim.

No API key needed; free and unlimited.
"""
from __future__ import annotations

from ddgs import DDGS


def search(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo and return results formatted for LLM ingestion.

    Args:
        query: The search query (e.g., "AI market size 2026")
        max_results: Number of results to return (default 5)

    Returns:
        List of dicts with 'title', 'url', 'snippet' keys. Empty list on error.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
    except Exception as e:
        print(f"DuckDuckGo search failed: {e}")
        return []


def format_results(results: list[dict]) -> str:
    """Format search results as a numbered list for the prompt.

    Args:
        results: List of search result dicts from search()

    Returns:
        Formatted string for inclusion in the LLM prompt
    """
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r['title']}")
        lines.append(f"    URL: {r['url']}")
        lines.append(f"    {r['snippet']}")
        lines.append("")
    return "\n".join(lines)
