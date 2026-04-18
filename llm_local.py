"""Ollama adapter for local LLM inference.

Wraps Ollama's /api/chat endpoint with `format=<JSON schema>` so the rest of
the pipeline can keep using the same Pydantic output models it uses with
Anthropic's `messages.parse()`.

Two model families:
  - Text-only (qwen3.5:9b, llama3.1:8b, …): PDF text extracted with pypdf,
    then sent as a plain user message.
  - Vision (qwen2-vl:7b, llama3.2-vision:11b, …): PDF pages converted to PNG
    images with pdf2image/poppler, sent as base64 image payloads.

Neither family has a server-side web_search tool; callers inject DuckDuckGo
results as text when needed.
"""
from __future__ import annotations

import base64
import io
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
    Everything else (e.g. 'qwen3.5:9b', 'qwen2-vl:7b', 'llama3.1:8b') is
    treated as a local Ollama model — including vision models.
    """
    if not model:
        return False
    m = model.lower()
    return not m.startswith("claude-") and not m.startswith("mercury-")


# Vision-capable Ollama models. These accept base64 image payloads and can
# read image-based PDFs that pypdf cannot extract text from.
_VISION_PREFIXES = ("llama3.2-vision", "qwen2-vl", "gemma4", "minicpm-v", "moondream")

def is_vision_local_model(model: str | None) -> bool:
    """True if this Ollama model supports image inputs (multimodal/vision).

    Must be checked BEFORE is_local_model() in routing logic because vision
    models satisfy both predicates — callers need the more specific path.
    """
    if not model:
        return False
    m = model.lower()
    return any(m.startswith(p) for p in _VISION_PREFIXES)


# Resolution for rendered PDF pages sent to vision models.
# 72 DPI keeps image payloads small (~500×380px per slide) while remaining
# legible for OCR. Raise to 100–120 only if the model misses fine print.
# Override with OLLAMA_VISION_DPI env var.
OLLAMA_VISION_DPI = int(os.environ.get("OLLAMA_VISION_DPI", "72"))

# Starting batch size for vision requests (slides per Ollama call).
# The extractor auto-halves on 500 OOM, so this is the *initial* attempt size.
# Lower values are safer on machines with limited VRAM/RAM.
# Override with OLLAMA_VISION_BATCH env var.
OLLAMA_VISION_BATCH = int(os.environ.get("OLLAMA_VISION_BATCH", "3"))


MIN_EXTRACTABLE_CHARS = 300  # below this, deck is almost certainly image-based


def extract_pdf_text(pdf_path: str | Path) -> str:
    """Extract per-page text from a PDF for text-only local models.

    Pages are delimited so the extractor can populate the 1-indexed `slide`
    field on each ExtractedClaim.

    Raises ValueError if the extracted text is too short to be useful — this
    almost always means the deck is image-based (slides exported as PNGs or
    scanned pages) and cannot be read by text-only models.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        parts.append(f"[Slide {i}]\n{text}")
    full_text = "\n\n".join(parts)

    # Strip slide headers and whitespace to get the real content length
    content_only = "\n".join(
        line for line in full_text.splitlines()
        if not line.startswith("[Slide ")
    ).strip()

    if len(content_only) < MIN_EXTRACTABLE_CHARS:
        raise ValueError(
            f"This PDF appears to be image-based — only {len(content_only)} characters of "
            f"selectable text were found across {len(reader.pages)} slides. "
            "Local models (Ollama) can only read PDFs with selectable text. "
            "Please switch the Extractor model to Claude Haiku or Sonnet, which have "
            "native vision and can read image-based pitch decks."
        )

    return full_text


def pdf_to_base64_images(
    pdf_path: str | Path,
    max_pages: int | None = None,
) -> list[str]:
    """Render PDF pages to PNG and return base64-encoded strings.

    Used by vision Ollama models that accept image payloads instead of text.
    Requires pdf2image and the poppler system library:
        pip install pdf2image
        brew install poppler          # macOS
        apt-get install poppler-utils  # Debian/Ubuntu

    Raises RuntimeError with install instructions if pdf2image is missing.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise RuntimeError(
            "pdf2image is required for vision model extraction but is not installed.\n"
            "  pip install pdf2image\n"
            "  brew install poppler          # macOS\n"
            "  apt-get install poppler-utils  # Linux"
        )

    pages = convert_from_path(str(pdf_path), dpi=OLLAMA_VISION_DPI, fmt="png")
    if max_pages is not None:
        pages = pages[:max_pages]

    result: list[str] = []
    for page in pages:
        buf = io.BytesIO()
        page.save(buf, format="PNG")
        result.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

    return result


def call_structured_vision(
    *,
    model: str,
    system: str,
    user_text: str,
    images_b64: list[str],
    output_format: Type[T],
    verbose: bool = True,
) -> T:
    """Structured-output vision chat completion against Ollama.

    Sends one or more base64 PNG images alongside the user message. Uses the
    same streaming + wall-clock deadline approach as call_structured().

    The `images` field in the Ollama message payload is the standard way to
    pass image data to multimodal models (LLaVA, Qwen2-VL, etc.).
    """
    schema = output_format.model_json_schema()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": user_text,
                "images": images_b64,
            },
        ],
        "format": schema,
        "stream": True,
        "options": {
            "temperature": 0,
            "num_predict": OLLAMA_MAX_TOKENS,
        },
    }

    if verbose:
        print(f"[ollama-vision] Querying {model} with {len(images_b64)} slide(s)...")

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
            if r.status_code == 500:
                raise RuntimeError(
                    f"ollama-vision-500: {model} OOM with {len(images_b64)} image(s) at {OLLAMA_VISION_DPI} DPI"
                )
            r.raise_for_status()

            for line in r.iter_lines():
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"Ollama vision inference exceeded {OLLAMA_WALL_CLOCK_SECS:.0f}s on {model}. "
                        "Try a smaller deck or fewer pages, or restart Ollama: `ollama serve`"
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
            "Start it with `ollama serve`."
        ) from e

    content = "".join(content_parts)
    if verbose:
        print(f"[ollama-vision] ✓ {model} completed ({len(content)} chars)")

    return output_format.model_validate_json(content)


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
