"""Inception Labs Mercury adapter.

Uses the OpenAI-compatible chat completions API at https://api.inceptionlabs.ai/v1.
No dedicated SDK needed — we call the REST endpoint directly with httpx, which
is already a project dependency.

Mercury models support structured JSON outputs via response_format=json_schema,
so we can use the same Pydantic output_format pattern as the rest of the pipeline.

Mercury does not have native PDF vision or a server-side web_search tool:
  - PDF extraction: use pypdf (same path as Ollama) via llm_local.extract_pdf_text()
  - Industry claims: fetch DuckDuckGo results via search_api and feed as text

Set INCEPTION_API_KEY in .env to enable. New accounts receive 10M free tokens.
"""
from __future__ import annotations

import json
import os
from typing import Type, TypeVar

import httpx
from pydantic import BaseModel

INCEPTION_BASE_URL = os.environ.get("INCEPTION_BASE_URL", "https://api.inceptionlabs.ai/v1")
INCEPTION_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)

# Known Inception model IDs. All start with "mercury-".
INCEPTION_MODELS = {"mercury-2", "mercury-coder-small"}

T = TypeVar("T", bound=BaseModel)


def is_inception_model(model: str | None) -> bool:
    """True if the model name refers to an Inception Labs Mercury model."""
    if not model:
        return False
    m = model.lower()
    return m in INCEPTION_MODELS or m.startswith("mercury-")


def call_structured(
    *,
    model: str,
    system: str,
    user_content: str,
    output_format: Type[T],
    verbose: bool = True,
    reasoning_effort: str = "medium",
) -> T:
    """Structured-output chat completion against the Inception Labs Mercury API.

    Raises RuntimeError with a friendly message on auth failure, timeout, or
    unknown model — same contract as llm_local.call_structured().
    """
    api_key = os.environ.get("INCEPTION_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "INCEPTION_API_KEY is not set. Get a free key at https://inceptionlabs.ai "
            "and add INCEPTION_API_KEY=<key> to your .env file."
        )

    schema = output_format.model_json_schema()

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.75,
        "reasoning_effort": reasoning_effort,
        # Structured output: strict=false because Pydantic schemas use anyOf for
        # Optional fields, which some providers reject under strict mode.
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": output_format.__name__,
                "schema": schema,
                "strict": False,
            },
        },
    }

    if verbose:
        preview = user_content[:100].replace("\n", " ")
        print(f"[inception] Querying {model} (reasoning={reasoning_effort}): {preview}...")

    try:
        r = httpx.post(
            f"{INCEPTION_BASE_URL}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=INCEPTION_TIMEOUT,
        )
    except httpx.TimeoutException as e:
        raise RuntimeError(
            f"Inception API timed out after {INCEPTION_TIMEOUT.read}s on {model}. "
            "Try again or increase INCEPTION_TIMEOUT_READ."
        ) from e
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"Could not reach Inception API at {INCEPTION_BASE_URL}. "
            "Check your internet connection."
        ) from e

    if r.status_code == 401:
        raise RuntimeError(
            "Inception API key rejected (HTTP 401). "
            "Check INCEPTION_API_KEY in your .env file."
        )
    if r.status_code == 404:
        raise RuntimeError(
            f"Inception model '{model}' not found. "
            "Available models: mercury-2, mercury-coder-small."
        )
    r.raise_for_status()

    content = r.json()["choices"][0]["message"]["content"]

    if verbose:
        print(f"[inception] ✓ {model} completed")

    return output_format.model_validate_json(content)
