"""Ollama adapter for local LLM inference.

Wraps Ollama's /api/chat endpoint with `format=<JSON schema>` so the rest of
the pipeline can keep using the same Pydantic output models it uses with
Anthropic's `messages.parse()`.

Local models have no vision (our Ollama targets are text-only) and no server-
side web_search tool. Callers must handle those limitations explicitly —
this module only dispatches structured text calls.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Type, TypeVar

import httpx
from pydantic import BaseModel

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Wall-clock deadline for a streaming inference call.
# With stream=True each chunk has its own read window, so the only way to
# enforce a total time limit is to check elapsed time in the accumulation loop.
# 300s = 5 minutes; enough for full-deck extraction on a 9B model at ~5 tok/s
# generating ~6000 tokens. Override with OLLAMA_TIMEOUT_SECS env var.
OLLAMA_WALL_CLOCK_SECS = float(os.environ.get("OLLAMA_TIMEOUT_SECS", "300.0"))

# Cap output tokens so the model can't loop indefinitely producing JSON.
# DeckExtraction for a 20-claim deck with 4 metrics ≈ 3000–5000 tokens.
# ClaimAssessment is far smaller (~400 tokens). 6000 is a safe upper bound.
OLLAMA_MAX_TOKENS = int(os.environ.get("OLLAMA_MAX_TOKENS", "6000"))

OLLAMA_CONNECT_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)

T = TypeVar("T", bound=BaseModel)


def is_local_model(model: str | None) -> bool:
    """True if the model name refers to an Ollama-served local model.

    Anthropic models start with 'claude-'; Inception models start with 'mercury-'.
    Everything else (e.g. 'qwen3.5:9b', 'llama3.1:8b') is treated as a local
    Ollama model.
    """
    if not model:
        return False
    m = model.lower()
    return not m.startswith("claude-") and not m.startswith("mercury-")


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
    """Structured-output chat completion against Ollama using streaming.

    Streams the response so we can enforce a wall-clock deadline regardless
    of how slowly the model generates tokens. With stream=False the httpx read
    timeout never fires because Ollama continuously sends data during inference;
    the connection only closes when generation finishes — which may take many
    minutes for a complex schema on a 9B model.

    Raises RuntimeError with a friendly message if Ollama is unreachable,
    the model is not pulled, or the wall-clock deadline is exceeded.
    """
    schema = output_format.model_json_schema()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "format": schema,
        "stream": True,   # ← stream so we can enforce a wall-clock timeout
        "options": {
            "temperature": 0,
            "num_predict": OLLAMA_MAX_TOKENS,  # cap output to prevent infinite loops
        },
    }

    if verbose:
        preview = user_content[:100].replace("\n", " ")
        print(f"[ollama] Querying {model}: {preview}...")

    deadline = time.monotonic() + OLLAMA_WALL_CLOCK_SECS
    content_parts: list[str] = []

    try:
        with httpx.stream(
            "POST",
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=OLLAMA_CONNECT_TIMEOUT,
        ) as r:
            if r.status_code == 404:
                raise RuntimeError(
                    f"Ollama model '{model}' not found locally. Pull it first: "
                    f"`ollama pull {model}`."
                )
            r.raise_for_status()

            for line in r.iter_lines():
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"Ollama inference exceeded {OLLAMA_WALL_CLOCK_SECS:.0f}s wall-clock limit "
                        f"on {model}. The deck may be too large for this model, or Ollama is "
                        "overloaded. Try restarting: `ollama serve`"
                    )
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("message", {}).get("content", "")
                if token:
                    content_parts.append(token)
                if chunk.get("done"):
                    break

    except httpx.TimeoutException as e:
        raise RuntimeError(
            f"Ollama connection timed out on {model}. "
            "Is Ollama running? Start it with `ollama serve`."
        ) from e
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL}. Is it running? "
            "Start it with `ollama serve` (or install from https://ollama.com)."
        ) from e

    content = "".join(content_parts)
    if verbose:
        print(f"[ollama] ✓ {model} completed ({len(content)} chars)")

    return output_format.model_validate_json(content)
