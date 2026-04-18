"""DuckDuckGo search adapter for local model market/industry claim verification.

Local models (Ollama) have no web access, so we fetch search results via
DuckDuckGo and feed them into the model's reasoning. The model reads the
results and assesses the claim.

No API key needed; free and unlimited.

Timeout strategy
----------------
DDGS(timeout=N) only sets the httpx *connection* timeout. Once a socket is
open, DDG can trickle or stall indefinitely — TimeoutError never fires and the
retry loop never runs. We wrap every search in a daemon thread and call
thread.join(timeout=DDGS_TIMEOUT). If the thread is still alive after the
deadline we abandon it (daemon threads don't block process exit) and return [].
"""
from __future__ import annotations

import threading
import time
from ddgs import DDGS

# Hard wall-clock deadline per search attempt (seconds).
DDGS_TIMEOUT = 12.0
MAX_RETRIES = 2

# DDG handles short keyword queries much better than full verbatim sentences.
# Truncate before sending so we don't waste the timeout on a malformed query.
MAX_QUERY_CHARS = 120


def _ddgs_fetch(query: str, max_results: int, out: list, err: list) -> None:
    """Target for the daemon thread — populates `out` or `err` in-place."""
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        out.extend(
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        )
    except Exception as exc:  # noqa: BLE001
        err.append(exc)


def search(query: str, max_results: int = 5, verbose: bool = True) -> list[dict]:
    """Search DuckDuckGo and return results formatted for LLM ingestion.

    Uses a daemon thread + join(timeout) so the call is guaranteed to return
    within DDGS_TIMEOUT seconds regardless of network behaviour.

    Args:
        query: The search query. Truncated to MAX_QUERY_CHARS automatically.
        max_results: Number of results to return (default 5).
        verbose: Print progress to stdout (default True).

    Returns:
        List of dicts with 'title', 'url', 'snippet' keys. Empty list on error
        or timeout.
    """
    # Truncate long verbatim sentences — DDG returns worse results for them
    # and they're more likely to trigger rate-limiting.
    if len(query) > MAX_QUERY_CHARS:
        query = query[:MAX_QUERY_CHARS]

    if verbose:
        print(f"[search] Querying DuckDuckGo: {query[:80]}...")

    for attempt in range(MAX_RETRIES):
        out: list[dict] = []
        err: list[Exception] = []

        t = threading.Thread(target=_ddgs_fetch, args=(query, max_results, out, err), daemon=True)
        t.start()
        t.join(timeout=DDGS_TIMEOUT)

        if t.is_alive():
            # Thread is stuck on a blocked socket — abandon it.
            if attempt < MAX_RETRIES - 1:
                print(
                    f"[search] ⚠ DuckDuckGo timed out after {DDGS_TIMEOUT:.0f}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES}); retrying..."
                )
                time.sleep(1)
                continue
            else:
                print(
                    f"[search] ✗ DuckDuckGo timed out after {MAX_RETRIES} attempts: {query[:80]}"
                )
                return []

        if err:
            print(f"[search] ✗ Search failed: {type(err[0]).__name__}: {err[0]}")
            return []

        if verbose:
            print(f"[search] ✓ Got {len(out)} results for: {query[:80]}")
        return out

    return []  # unreachable, satisfies type checker


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
