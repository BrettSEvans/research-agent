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

This module now only wires the app together: middleware, static/template
config, startup tasks, and `include_router` calls. All route handlers live in
the `routers/` package, grouped by feature area.
"""
from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from auth import load_session
# Legacy bootstrap helper (kept for backwards compat)
from auth import ensure_default_organization_and_user  # noqa: F401
from db import get_db, init_db, seed_ca_regulatory_sources, seed_eu_regulatory_sources

from routers import admin as admin_router
from routers import auth as auth_router
from routers import pages as pages_router
from routers import reports as reports_router
from routers import saved as saved_router
from routers import shared as shared_router
from routers import slides as slides_router
from routers import upload as upload_router
from routers import verify as verify_router
from routers._deps import (
    GOOGLE_CLIENT_ID,
    SESSION_COOKIE,
    _BASIC_PASS,
    _BASIC_USER,
    _check_basic_auth,
)

load_dotenv()

app = FastAPI(title="VC Pitch Deck + Compliance")

# Serve extracted CSS/JS for templates/index.html
app.mount("/static", StaticFiles(directory="static"), name="static")

# Allow saasless.ai to call the /admin/whitelist API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://saasless.ai",
        "https://www.saasless.ai",
        "https://lovable.app",
        "https://preview.lovable.app",
    ],
    allow_origin_regex=r"https://.*\.lovable\.app",
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["X-Admin-Key", "Content-Type"],
    allow_credentials=False,
)

# Initialise DB tables and run column migrations on startup
init_db()

# Seed EU regulatory sources (idempotent, runs only on first startup)
_seed_db = next(get_db())
try:
    seed_eu_regulatory_sources(_seed_db)
finally:
    _seed_db.close()

# Seed CA regulatory sources (idempotent, runs only on first startup)
_seed_ca_db = next(get_db())
try:
    seed_ca_regulatory_sources(_seed_ca_db)
finally:
    _seed_ca_db.close()

# Bootstrap default org/user from env vars so Basic Auth + legacy data still works
_default_db = next(get_db())
try:
    ensure_default_organization_and_user(_default_db)
finally:
    _default_db.close()

# Start background scheduler for daily regulatory updates
from scheduler import start_scheduler, stop_scheduler

_scheduler = start_scheduler()


@app.on_event("shutdown")
def shutdown_event():
    """Shutdown scheduler on FastAPI shutdown."""
    stop_scheduler()


# ── Middleware ─────────────────────────────────────────────────────────────

@app.middleware("http")
async def frame_options_middleware(request: Request, call_next):
    """Allow this app to be embedded in an iframe on saasless.ai."""
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = "frame-ancestors 'self' https://saasless.ai"
    return response


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
    google_enabled = bool(GOOGLE_CLIENT_ID)
    auth_required  = google_enabled or bool(_BASIC_USER and _BASIC_PASS)

    if not auth_required:
        return await call_next(request)

    path = request.url.path
    if path.startswith("/auth") or path.startswith("/shared") or path.startswith("/admin/whitelist") or path.startswith("/static") or path in ("/health", "/favicon.ico"):
        return await call_next(request)

    if google_enabled:
        token = request.cookies.get(SESSION_COOKIE, "")
        if token and load_session(token):
            return await call_next(request)

    if _check_basic_auth(request):
        return await call_next(request)

    accept = request.headers.get("Accept", "")
    if google_enabled and "text/html" in accept:
        return RedirectResponse(url="/auth/login", status_code=302)

    return Response(
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="VC Compliance"'},
        content="Unauthorized",
    )


# ── Routers ────────────────────────────────────────────────────────────────
# Included in the same sequence the original routes were declared in web.py.
app.include_router(pages_router.router)      # /health, /notifications, /config, /
app.include_router(admin_router.router)      # /debug/whitelist-check, /admin/whitelist*
app.include_router(auth_router.router)       # /auth/login, /auth/google/one-tap, /auth/logout, /auth/me
app.include_router(slides_router.router)     # /auth/google/slides-*, /extract-from-google-slides
app.include_router(upload_router.router)     # /extract, /extract/deep, /extract/complete
app.include_router(verify_router.router)     # /verify, /verify/stream
app.include_router(saved_router.router)      # /saved-extractions*
app.include_router(reports_router.router)    # /reports*
app.include_router(shared_router.router)     # /shared/*
