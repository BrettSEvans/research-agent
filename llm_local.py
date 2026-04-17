"""Ollama adapter for local LLM inference.

Wraps Ollama's /api/chat endpoint with `format=<JSON schema>` so the rest of
the pipeline can keep using the same Pydantic output models it uses with
Anthropic's `messages.parse()`.

Local models have no vision (our Ollama targets are text-only) and no server-
side web_search tool. Callers must handle those limitations explicitly —
this module only dispatches structured text calls.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Type, TypeVar

import httpx
from pydantic import BaseModel

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# Local inference on a 9B model is slow on CPU / modest GPUs, but should
# complete within 2 minutes. If it hangs, fail cleanly rather than waiting
# 10 minutes. User can increase OLLAMA_TIMEOUT env var if needed.
OLLAMA_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=float(os.environ.get("OLLAMA_TIMEOUT_READ", "120.0")),
    write=30.0,
    pool=10.0,
)

T = TypeVar("T", bound=BaseModel)


def is_local_model(model: str | None) -> bool:
    """True if the model name refers to an Ollama-served local model.

    Our convention: Anthropic model names start with 'claude-'. Anything else
    (e.g., 'qwen3.5:9b', 'llama3.1:8b') is treated as local.
    """
    if not model:
        return False
    return not model.lower().startswith("claude-")


def extract_pdf_text(pdf_path: str | Path) -> str:
    """Extract per-page text from a PDF for text-only local models.

    Pages are delimited so the extractor can populate the 1-indexed `slide`
    field on each ExtractedClaim.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        parts.append(f"[Slide {i}]\n{text}")
    return "\n\n".join(parts)


def call_structured(
    *,
    model: str,
    system: str,
    user_content: str,
    output_format: Type[T],
    verbose: bool = True,
) -> T:
    """Structured-output chat completion against Ollama.

    Raises RuntimeError with a friendly message if Ollama is unreachable or
    the model is not pulled locally.
    """
    schema = output_format.model_json_schema()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "format": schema,
        "stream": False,
        "options": {"temperature": 0},
    }

    if verbose:
        preview = user_content[:100].replace("\n", " ")
        print(f"[ollama] Querying {model}: {preview}...")

    try:
        r = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
    except httpx.TimeoutException as e:
        raise RuntimeError(
            f"Ollama inference timed out after {OLLAMA_TIMEOUT.read}s on {model}. "
            "Model may be overloaded or stuck. Try restarting: `ollama serve`"
        ) from e
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL}. Is it running? "
            "Start it with `ollama serve` (or install Ollama from https://ollama.com)."
        ) from e

    if r.status_code == 404:
        raise RuntimeError(
            f"Ollama model '{model}' not found locally. Pull it first: "
            f"`ollama pull {model}`."
        )

    r.raise_for_status()
    content = r.json()["message"]["content"]

    if verbose:
        print(f"[ollama] ✓ {model} completed")

    return output_format.model_validate_json(content)
