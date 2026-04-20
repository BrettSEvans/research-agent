"""FastAPI web interface for the pitch deck extractor + compliance agent.

Flow:
    1. User uploads a PDF deck.
    2. Extractor returns a structured DeckExtraction (displayed for review).
    3. User clicks "Run Compliance Check" — compliance agent runs against SEC
       filings using the deck context as clarifying metadata only.
    4. Compliance report is displayed, with INSUFFICIENT_EVIDENCE shown
       explicitly when the report cannot be completed.

Run:
    uvicorn web:app --reload
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import tempfile
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Annotated

import anthropic
import httpx
from dotenv import load_dotenv
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates

from agent import iter_compliance_report, run_compliance_report, SAVED_REPORTS_DIR
from deck_context import DeckContext
from extractor import extract_from_pdf
from version import ANALYZER_VERSION, EXTRACTOR_VERSION

load_dotenv()

BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"
CONTEXT_DIR = BASE / "deck_contexts"
SAVED_DIR = BASE / "saved_extractions"
UPLOADS.mkdir(exist_ok=True)
CONTEXT_DIR.mkdir(exist_ok=True)
SAVED_DIR.mkdir(exist_ok=True)

app = FastAPI(title="VC Pitch Deck + Compliance")
templates = Jinja2Templates(directory=str(BASE / "templates"))


# ───────────────────────── auth configuration ─────────────────────────────

# Shared-password Basic Auth (fallback / API clients)
_BASIC_USER = os.environ.get("BASIC_AUTH_USER", "")
_BASIC_PASS = os.environ.get("BASIC_AUTH_PASSWORD", "")

# Google OAuth2
GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
# Public URL of this deployment — used to build the OAuth callback URL.
# Example: https://myapp.railway.app   (no trailing slash)
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
# Optional comma-separated list of allowed email domains, e.g. "acme.com,partner.io"
_ALLOWED_DOMAINS = {
    d.strip().lower()
    for d in os.environ.get("ALLOWED_EMAIL_DOMAINS", "").split(",")
    if d.strip()
}

# Session cookie — HMAC-signed JSON, no extra dependencies
SECRET_KEY       = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
SESSION_COOKIE   = "vc_session"
SESSION_MAX_AGE  = 86400 * 30   # 30 days
_SECURE_COOKIES  = BASE_URL.startswith("https://")

GOOGLE_AUTH_URL     = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL    = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


# ── Session helpers ────────────────────────────────────────────────────────

def _sign_session(payload: dict) -> str:
    """Return a compact, HMAC-signed token encoding *payload* + expiry."""
    body = {**payload, "exp": int(time.time()) + SESSION_MAX_AGE}
    data = base64.urlsafe_b64encode(
        json.dumps(body, separators=(",", ":")).encode()
    ).rstrip(b"=").decode()
    sig = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
    return f"{data}.{sig}"


def _load_session(token: str) -> dict | None:
    """Verify and decode a session token; returns None if invalid / expired."""
    try:
        data, sig = token.rsplit(".", 1)
        expected = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        # Restore padding before decoding
        payload = json.loads(base64.urlsafe_b64decode(data + "=="))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


def _check_basic_auth(request: Request) -> bool:
    """Return True when a valid Basic Auth header is present."""
    if not (_BASIC_USER and _BASIC_PASS):
        return False
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(auth[6:]).decode("utf-8")
        u, p = decoded.split(":", 1)
        return u == _BASIC_USER and p == _BASIC_PASS
    except Exception:
        return False


# ── Auth middleware ────────────────────────────────────────────────────────

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Unified auth gate: Google session cookie OR HTTP Basic Auth.

    Rules:
    - If neither Google SSO nor Basic Auth is configured → open access.
    - /auth/* routes are always allowed (login, callback, logout).
    - Static assets (fonts, icons) are always allowed.
    - Valid Google session cookie → allow.
    - Valid Basic Auth header → allow (useful for API / curl access).
    - Otherwise: browser → redirect to /auth/login; API → 401.
    """
    google_enabled = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)
    auth_required  = google_enabled or bool(_BASIC_USER and _BASIC_PASS)

    if not auth_required:
        return await call_next(request)

    path = request.url.path
    # Auth routes and health-check are always public
    if path.startswith("/auth") or path in ("/health", "/favicon.ico"):
        return await call_next(request)

    # 1. Google session cookie
    if google_enabled:
        token = request.cookies.get(SESSION_COOKIE, "")
        if token and _load_session(token):
            return await call_next(request)

    # 2. HTTP Basic Auth header (API clients / curl)
    if _check_basic_auth(request):
        return await call_next(request)

    # 3. Not authenticated
    accept = request.headers.get("Accept", "")
    if google_enabled and "text/html" in accept:
        return RedirectResponse(url="/auth/login", status_code=302)

    return Response(
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="VC Compliance"'},
        content="Unauthorized",
    )


def get_api_key(request: Request) -> str | None:
    """Read Anthropic API key from X-Anthropic-API-Key header, falling back to env var."""
    return request.headers.get("X-Anthropic-API-Key") or os.environ.get("ANTHROPIC_API_KEY")


def _sanitize_session(raw: str) -> str:
    safe = "".join(c for c in raw if c.isalnum() or c == "-")[:64]
    return safe or "default"


def get_session_dirs(request: Request) -> tuple[Path, Path]:
    """Return (session_uploads_dir, session_context_dir) for the current session.

    The X-Session-ID header value is sanitised to alphanumeric + hyphens and
    used as a subdirectory under the global UPLOADS / CONTEXT_DIR roots so
    that each browser session is fully isolated.
    """
    sid = _sanitize_session(request.headers.get("X-Session-ID", ""))
    uploads = UPLOADS / sid
    context = CONTEXT_DIR / sid
    uploads.mkdir(parents=True, exist_ok=True)
    context.mkdir(parents=True, exist_ok=True)
    return uploads, context


# ───────────────────────── Google SSO routes ──────────────────────────────

@app.get("/health")
async def health():
    """Public health check endpoint — always returns 200 (used by Railway/Render)."""
    return JSONResponse({"status": "ok"})


@app.get("/auth/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request=request, name="login.html", context={
        "google_enabled": bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET),
    })


@app.get("/auth/google")
async def google_login(request: Request):
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
        raise HTTPException(status_code=503, detail="Google SSO is not configured.")
    state = secrets.token_urlsafe(20)
    params = {
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  f"{BASE_URL}/auth/google/callback",
        "response_type": "code",
        "scope":         "openid email profile",
        "state":         state,
        "access_type":   "online",
        "prompt":        "select_account",
    }
    url = GOOGLE_AUTH_URL + "?" + urllib.parse.urlencode(params)
    resp = RedirectResponse(url=url, status_code=302)
    resp.set_cookie("oauth_state", state, max_age=600, httponly=True,
                    samesite="lax", secure=_SECURE_COOKIES)
    return resp


@app.get("/auth/google/callback")
async def google_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    if error:
        return RedirectResponse(url=f"/auth/login?error={urllib.parse.quote(error)}", status_code=302)

    stored_state = request.cookies.get("oauth_state", "")
    if not stored_state or not hmac.compare_digest(stored_state, state):
        return RedirectResponse(url="/auth/login?error=state_mismatch", status_code=302)

    try:
        async with httpx.AsyncClient() as client:
            tok = await client.post(GOOGLE_TOKEN_URL, data={
                "client_id":     GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "code":          code,
                "grant_type":    "authorization_code",
                "redirect_uri":  f"{BASE_URL}/auth/google/callback",
            }, headers={"Accept": "application/json"})
            tok.raise_for_status()
            access_token = tok.json().get("access_token", "")

            info = await client.get(GOOGLE_USERINFO_URL,
                                    headers={"Authorization": f"Bearer {access_token}"})
            info.raise_for_status()
            userinfo = info.json()
    except Exception as exc:
        return RedirectResponse(
            url=f"/auth/login?error={urllib.parse.quote(str(exc))}", status_code=302
        )

    email: str = userinfo.get("email", "")
    if _ALLOWED_DOMAINS and email.split("@")[-1].lower() not in _ALLOWED_DOMAINS:
        return RedirectResponse(url="/auth/login?error=domain_not_allowed", status_code=302)

    session_payload = {
        "email":   email,
        "name":    userinfo.get("name", email),
        "picture": userinfo.get("picture", ""),
    }
    token = _sign_session(session_payload)
    resp  = RedirectResponse(url="/", status_code=302)
    resp.set_cookie(SESSION_COOKIE, token, max_age=SESSION_MAX_AGE,
                    httponly=True, samesite="lax", secure=_SECURE_COOKIES)
    resp.delete_cookie("oauth_state")
    return resp


@app.get("/auth/logout")
async def logout():
    resp = RedirectResponse(url="/auth/login", status_code=302)
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@app.get("/auth/me")
async def auth_me(request: Request):
    """Return the current user's info (or null) — polled by the frontend."""
    token = request.cookies.get(SESSION_COOKIE, "")
    if token:
        session = _load_session(token)
        if session:
            return JSONResponse({
                "email":   session.get("email"),
                "name":    session.get("name"),
                "picture": session.get("picture"),
            })
    return JSONResponse(None)


# ───────────────────────── main app routes ────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


ALLOWED_MODELS = {
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    # Inception Labs Mercury models
    "mercury-2",
    "mercury-coder-small",
    # Local (Ollama) text-only models
    "qwen3.5:9b",
    # Local (Ollama) vision models — can read image-based PDFs
    "llama3.2-vision:11b",
    "gemma4:latest",
    "gemma4:26b",
}

# Only real Anthropic models are valid for /extract/complete (which uses the
# Anthropic SDK directly; local model names would cause a silent 500).
CLOUD_MODELS = {"claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-6"}


@app.post("/extract")
async def extract(
    request: Request,
    file: UploadFile = File(...),
    extractor_model: Annotated[str | None, Form()] = None,
):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if extractor_model and extractor_model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {extractor_model}")

    session_uploads, _ = get_session_dirs(request)
    content = await file.read()
    token = uuid.uuid4().hex[:12]
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf", dir=str(session_uploads), prefix=f"{token}_"
    ) as tmp:
        tmp.write(content)
        pdf_path = Path(tmp.name)

    try:
        client = anthropic.Anthropic(api_key=get_api_key(request))
        from extractor import extract_basics_and_infer_stage
        extraction = extract_basics_and_infer_stage(pdf_path, client=client, model=extractor_model)
    except Exception as exc:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc

    # Do not save DeckContext or unlink the PDF yet; we need it for /extract/deep
    return JSONResponse(
        {
            "context_id": token,
            "extraction": extraction.model_dump(),
            "extractor_version": EXTRACTOR_VERSION,
        }
    )

@app.post("/extract/deep")
async def extract_deep(
    request: Request,
    context_id: Annotated[str, Form()],
    extractor_model: Annotated[str | None, Form()] = None,
    startup_stage: Annotated[str | None, Form()] = None,
    modules: Annotated[str | None, Form()] = None,
):
    session_uploads, session_ctx = get_session_dirs(request)
    pdf_paths = list(session_uploads.glob(f"{context_id}_*.pdf"))
    context_path = session_ctx / f"deck_{context_id}.json"

    # No PDF in uploads — this context was loaded from a saved extraction.
    # The context file already exists, so skip re-extraction and return it as-is.
    if not pdf_paths:
        if context_path.exists():
            extraction_data = json.loads(context_path.read_text())
            return JSONResponse(
                {
                    "context_id": context_id,
                    "context_path": str(context_path),
                    "extraction": extraction_data,
                    "extractor_version": EXTRACTOR_VERSION,
                    "skipped": True,
                    "skip_reason": "Deep extraction skipped: loaded from saved extraction (no PDF available). Using existing extraction.",
                }
            )
        raise HTTPException(
            status_code=404,
            detail="PDF not found for this context and no saved extraction exists. Please upload the PDF again.",
        )

    pdf_path = pdf_paths[0]
    requested_metrics = modules.split(",") if modules else None
    failed_path = session_ctx / f"deck_{context_id}_failed.json"

    try:
        client = anthropic.Anthropic(api_key=get_api_key(request))
        extraction, failed_pages = extract_from_pdf(
            pdf_path,
            client=client,
            model=extractor_model,
            requested_metrics=requested_metrics,
        )
    except Exception as exc:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Deep extraction failed: {exc}") from exc

    # Save full context
    context = DeckContext(extraction)
    context.save(context_path)

    if failed_pages:
        # Keep the PDF so /extract/complete can send failed pages to a cloud model
        failed_path.write_text(json.dumps({"failed_pages": failed_pages}))
    else:
        # No failures — clean up the PDF and any stale sidecar
        pdf_path.unlink(missing_ok=True)
        failed_path.unlink(missing_ok=True)

    return JSONResponse(
        {
            "context_id": context_id,
            "context_path": str(context_path),
            "extraction": extraction.model_dump(),
            "extractor_version": EXTRACTOR_VERSION,
            "failed_pages": failed_pages,
        }
    )


@app.post("/extract/complete")
async def extract_complete(
    request: Request,
    context_id: Annotated[str, Form()],
    cloud_model: Annotated[str, Form()] = "claude-haiku-4-5",
    modules: Annotated[str | None, Form()] = None,
):
    """Complete a partial vision extraction by sending failed pages to a cloud model.

    When a local vision model (e.g. llama3.2-vision) cannot process certain slides
    due to memory limits, those page numbers are recorded in a sidecar file. This
    endpoint reads that sidecar, sends the failed pages to a Claude model, merges
    the results into the existing DeckExtraction, and saves the merged context.
    """
    session_uploads, session_ctx = get_session_dirs(request)
    context_path = session_ctx / f"deck_{context_id}.json"
    failed_path = session_ctx / f"deck_{context_id}_failed.json"

    if not context_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown context_id: {context_id}")
    if not failed_path.exists():
        raise HTTPException(status_code=400, detail="No failed-page sidecar found for this context.")

    pdf_paths = list(session_uploads.glob(f"{context_id}_*.pdf"))
    if not pdf_paths:
        raise HTTPException(
            status_code=404,
            detail="PDF not found — it may have already been deleted. Re-upload the deck and try again.",
        )

    if cloud_model not in CLOUD_MODELS:
        raise HTTPException(status_code=400, detail=f"Model must be a cloud model for completion: {cloud_model}")

    pdf_path = pdf_paths[0]
    failed_data = json.loads(failed_path.read_text())
    failed_pages: list[int] = failed_data.get("failed_pages", [])

    if not failed_pages:
        failed_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Failed-page sidecar is empty — nothing to complete.")

    requested_metrics = modules.split(",") if modules else None

    try:
        client = anthropic.Anthropic(api_key=get_api_key(request))
        from extractor import extract_failed_pages_with_claude, DeckExtraction

        # Extract the failed pages with the cloud model
        cloud_extraction = extract_failed_pages_with_claude(
            pdf_path=pdf_path,
            page_indices=failed_pages,
            client=client,
            model=cloud_model,
            requested_metrics=requested_metrics,
        )

        # Load and merge with the existing partial extraction
        existing_context = DeckContext.load(context_path)
        ex = existing_context.extraction

        merged_claims = ex.claims + cloud_extraction.claims

        seen_metrics: set[str] = {m.metric_name for m in ex.key_metrics}
        merged_metrics = list(ex.key_metrics)
        for m in cloud_extraction.key_metrics:
            if m.metric_name not in seen_metrics:
                seen_metrics.add(m.metric_name)
                merged_metrics.append(m)

        pages_str = ", ".join(str(p) for p in sorted(failed_pages))
        merged_notes = "\n\n".join(filter(None, [
            ex.extraction_notes,
            f"[Cloud completion · slides {pages_str} via {cloud_model}] {cloud_extraction.extraction_notes}",
        ]))

        merged_extraction = DeckExtraction(
            company=ex.company,
            claims=merged_claims,
            fiscal_year_end=ex.fiscal_year_end or cloud_extraction.fiscal_year_end,
            currency=ex.currency or cloud_extraction.currency,
            stage_assessment=ex.stage_assessment or cloud_extraction.stage_assessment,
            key_metrics=merged_metrics,
            extraction_notes=merged_notes,
        )

        # Persist merged context
        merged_context = DeckContext(merged_extraction)
        merged_context.save(context_path)

        # Clean up sidecar and PDF
        failed_path.unlink(missing_ok=True)
        pdf_path.unlink(missing_ok=True)

        return JSONResponse(
            {
                "context_id": context_id,
                "extraction": merged_extraction.model_dump(),
                "extractor_version": EXTRACTOR_VERSION,
                "completed_pages": failed_pages,
            }
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cloud completion failed: {exc}") from exc


@app.post("/verify/stream")
async def verify_stream(
    request: Request,
    context_id: Annotated[str, Form()],
    forms: Annotated[str, Form()] = "10-K,10-Q,S-1,8-K",
    filings_limit: Annotated[int, Form()] = 3,
    top_k: Annotated[int, Form()] = 5,
    analyzer_model: Annotated[str | None, Form()] = None,
    extractor_model: Annotated[str | None, Form()] = None,
    startup_stage: Annotated[str | None, Form()] = None,
    modules: Annotated[str | None, Form()] = None,
):
    """Server-Sent Events endpoint: emits one claim_result event per claim
    as soon as analysis finishes, so the browser can render incrementally."""
    if analyzer_model and analyzer_model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {analyzer_model}")
    _, session_ctx = get_session_dirs(request)
    context_path = session_ctx / f"deck_{context_id}.json"
    if not context_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown context_id: {context_id}")

    deck = DeckContext.load(context_path)
    claims = deck.claims_for_verification()

    from sec import lookup_cik
    cik = None
    key = deck.company_lookup_key()
    if key:
        cik = key.zfill(10) if (key.isdigit() and len(key) <= 10) else lookup_cik(key)

    api_key = get_api_key(request)

    def event_stream():
        try:
            for event in iter_compliance_report(
                claims=claims,
                cik=cik,
                deck=deck,
                forms=[f.strip() for f in forms.split(",") if f.strip()],
                filings_limit=filings_limit,
                top_k=top_k,
                verbose=True,
                analyzer_model=analyzer_model,
                extractor_model=extractor_model,
                startup_stage=startup_stage,
                modules=modules.split(",") if modules else None,
                api_key=api_key,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'event': 'error', 'data': {'message': str(exc)}})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if behind a proxy
        },
    )


# ───────────────────────── saved extractions ──────────────────────────────

@app.get("/saved-extractions")
async def list_saved_extractions():
    """Return all saved extractions, newest first, with per-category claim stats."""
    items = []
    for p in sorted(SAVED_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text())
            extraction = data.get("extraction", {})

            # Tally claims by category
            claims = extraction.get("claims", [])
            by_category: dict[str, int] = {}
            for cl in claims:
                cat = cl.get("category", "other")
                by_category[cat] = by_category.get(cat, 0) + 1

            # Key metrics names
            metrics = [m.get("metric_name", "") for m in extraction.get("key_metrics", []) if m.get("metric_name")]

            items.append({
                "save_id": p.stem,
                "company_name": data.get("meta", {}).get("company_name", "Unknown"),
                "original_filename": data.get("meta", {}).get("original_filename", ""),
                "extractor_model": data.get("meta", {}).get("extractor_model", ""),
                "extractor_version": data.get("meta", {}).get("extractor_version", ""),
                "saved_at": data.get("meta", {}).get("saved_at", ""),
                # Stats for the info popover
                "claims_count": len(claims),
                "claims_by_category": by_category,
                "key_metrics": metrics,
                "stage": extraction.get("stage_assessment", {}).get("stage") if extraction.get("stage_assessment") else None,
            })
        except Exception:
            continue
    return JSONResponse(items)


@app.post("/saved-extractions")
async def save_extraction(
    request: Request,
    context_id: Annotated[str, Form()],
    original_filename: Annotated[str, Form()] = "",
    extractor_model: Annotated[str, Form()] = "",
):
    """Persist a session extraction to the saved library."""
    _, session_ctx = get_session_dirs(request)
    context_path = session_ctx / f"deck_{context_id}.json"
    if not context_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown context_id: {context_id}")

    extraction_data = json.loads(context_path.read_text())
    company_name = extraction_data.get("company", {}).get("name", "Unknown")
    save_id = uuid.uuid4().hex[:12]
    saved = {
        "meta": {
            "save_id": save_id,
            "company_name": company_name,
            "original_filename": original_filename,
            "extractor_model": extractor_model,
            "extractor_version": EXTRACTOR_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
        "extraction": extraction_data,
    }
    (SAVED_DIR / f"{save_id}.json").write_text(json.dumps(saved, indent=2))
    return JSONResponse({"save_id": save_id, "company_name": company_name})


@app.post("/saved-extractions/{save_id}/load")
async def load_saved_extraction(request: Request, save_id: str):
    """Load a saved extraction back into a fresh session context."""
    saved_path = SAVED_DIR / f"{save_id}.json"
    if not saved_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown save_id: {save_id}")

    data = json.loads(saved_path.read_text())
    extraction_data = data["extraction"]

    # Create a fresh session context so /verify/stream works normally
    _, session_ctx = get_session_dirs(request)
    token = uuid.uuid4().hex[:12]
    context_path = session_ctx / f"deck_{token}.json"
    context_path.write_text(json.dumps(extraction_data, indent=2))

    return JSONResponse({
        "context_id": token,
        "extraction": extraction_data,
        "meta": data.get("meta", {}),
    })


@app.delete("/saved-extractions/{save_id}")
async def delete_saved_extraction(save_id: str):
    saved_path = SAVED_DIR / f"{save_id}.json"
    if not saved_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown save_id: {save_id}")
    saved_path.unlink()
    return JSONResponse({"deleted": save_id})


@app.post("/verify")
async def verify(
    request: Request,
    context_id: Annotated[str, Form()],
    forms: Annotated[str, Form()] = "10-K,10-Q,S-1,8-K",
    filings_limit: Annotated[int, Form()] = 3,
    top_k: Annotated[int, Form()] = 5,
    analyzer_model: Annotated[str | None, Form()] = None,
    startup_stage: Annotated[str | None, Form()] = None,
    modules: Annotated[str | None, Form()] = None,
):
    if analyzer_model and analyzer_model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {analyzer_model}")
    _, session_ctx = get_session_dirs(request)
    context_path = session_ctx / f"deck_{context_id}.json"
    if not context_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown context_id: {context_id}")

    deck = DeckContext.load(context_path)
    claims = deck.claims_for_verification()

    # Resolve CIK using the deck's identity. Deliberately do not fall back to
    # other sources — compliance must be transparent about what it can and
    # can't identify.
    from sec import lookup_cik
    cik = None
    key = deck.company_lookup_key()
    if key:
        if key.isdigit() and len(key) <= 10:
            cik = key.zfill(10)
        else:
            cik = lookup_cik(key)

    report = run_compliance_report(
        claims=claims,
        cik=cik,
        deck=deck,
        forms=[f.strip() for f in forms.split(",") if f.strip()],
        filings_limit=filings_limit,
        top_k=top_k,
        verbose=True,
        analyzer_model=analyzer_model,
        startup_stage=startup_stage,
        modules=modules.split(",") if modules else None,
        api_key=get_api_key(request),
    )
    return report

# ───────────────────────── saved reports ──────────────────────────────

@app.get("/reports")
async def list_reports():
    """Return all saved compliance reports, newest first, with per-verdict counts."""
    items = []
    if not SAVED_REPORTS_DIR.exists():
        return JSONResponse(items)

    for p in sorted(SAVED_REPORTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text())
            results = data.get("results", [])

            # Per-verdict counts (actual verdict names from analyzer.py)
            consistent   = sum(1 for r in results if r.get("verdict") == "CONSISTENT")
            contradicts  = sum(1 for r in results if r.get("verdict") == "CONTRADICTS")
            unsupported  = sum(1 for r in results if r.get("verdict") == "UNSUPPORTED")
            insufficient = sum(1 for r in results if r.get("verdict") == "INSUFFICIENT_EVIDENCE")
            fls_flags    = sum(1 for r in results if r.get("verdict") == "CONTRADICTS" and r.get("forward_looking"))

            # Top flagged findings for dashboard preview (up to 3)
            top_flags = [
                {"claim": r.get("claim", ""), "verdict": r.get("verdict", ""), "forward_looking": r.get("forward_looking", False)}
                for r in results if r.get("verdict") == "CONTRADICTS"
            ][:3]

            items.append({
                "report_id": data.get("report_id", p.stem.replace("report_", "")),
                "company_name": data.get("company_name", "Unknown"),
                "generated_at": data.get("generated_at", ""),
                "assumed_industry": data.get("assumed_industry", ""),
                "cik": data.get("cik", ""),
                # Models + versions
                "extractor_model": data.get("extractor_model", ""),
                "extractor_version": data.get("extractor_version", ""),
                "analyzer_model": data.get("analyzer_model", ""),
                "analyzer_version": data.get("analyzer_version", ""),
                # Verdict breakdown
                "claims_analyzed": data.get("claims_analyzed", 0),
                "consistent": consistent,
                "contradicts": contradicts,
                "unsupported": unsupported,
                "insufficient": insufficient,
                "fls_flags": fls_flags,
                "top_flags": top_flags,
            })
        except Exception:
            continue
    return JSONResponse(items)

@app.get("/reports/{report_id}")
async def get_report(report_id: str):
    """Retrieve a specific saved compliance report."""
    report_path = SAVED_REPORTS_DIR / f"report_{report_id}.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown report_id: {report_id}")
    return JSONResponse(json.loads(report_path.read_text()))

@app.delete("/reports/{report_id}")
async def delete_report(report_id: str):
    """Delete a saved compliance report."""
    report_path = SAVED_REPORTS_DIR / f"report_{report_id}.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown report_id: {report_id}")
    report_path.unlink()
    return JSONResponse({"deleted": report_id})

