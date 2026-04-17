"""DuckDuckGo search adapter for local model market/industry claim verification.

Local models (Ollama) have no web access, so we fetch search results via
DuckDuckGo and feed them into the model's reasoning. The model reads the
results and assesses the claim.

No API key needed; free and unlimited.
"""
from __future__ import annotations

import time
from ddgs import DDGS

# DuckDuckGo search timeout (seconds). If the search hangs, move on.
# With 3 parallel workers, a single hung search blocks one thread.
DDGS_TIMEOUT = 15.0
MAX_RETRIES = 2


def search(query: str, max_results: int = 5, verbose: bool = True) -> list[dict]:
    """Search DuckDuckGo and return results formatted for LLM ingestion.

    Args:
        query: The search query (e.g., "AI market size 2026")
        max_results: Number of results to return (default 5)
        verbose: Print progress to stdout (default True)

    Returns:
        List of dicts with 'title', 'url', 'snippet' keys. Empty list on error.
    """
    if verbose:
        print(f"[search] Querying DuckDuckGo: {query[:80]}...")

    for attempt in range(MAX_RETRIES):
        try:
            with DDGS(timeout=DDGS_TIMEOUT) as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if verbose:
                print(f"[search] ✓ Got {len(results)} results for: {query[:80]}")
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in results
            ]
        except TimeoutError as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt  # 1s, 2s backoff
                print(f"[search] ⚠ DuckDuckGo search timed out (attempt {attempt + 1}/{MAX_RETRIES}); retrying in {wait}s...")
                time.sleep(wait)
                continue
            else:
                print(f"[search] ✗ DuckDuckGo search timed out after {MAX_RETRIES} attempts: {query[:80]}")
                return []
        except Exception as e:
            print(f"[search] ✗ Search failed: {type(e).__name__}: {e}")
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
