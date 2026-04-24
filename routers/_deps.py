"""Shared dependencies and globals for router modules.

Extracted from web.py during the APIRouter refactor. These are plain module
globals (config, paths, constants) plus a handful of helper functions used by
multiple routers.
"""
from __future__ import annotations

import hmac
import os
import secrets
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.templating import Jinja2Templates

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
UPLOADS = BASE / "uploads"
CONTEXT_DIR = BASE / "deck_contexts"
SAVED_DIR = BASE / "saved_extractions"
UPLOADS.mkdir(exist_ok=True)
CONTEXT_DIR.mkdir(exist_ok=True)
SAVED_DIR.mkdir(exist_ok=True)

# ── Templates ──────────────────────────────────────────────────────────────
templates = Jinja2Templates(directory=str(BASE / "templates"))

# ── Auth config ────────────────────────────────────────────────────────────
_BASIC_USER = os.environ.get("BASIC_AUTH_USER", "")
_BASIC_PASS = os.environ.get("BASIC_AUTH_PASSWORD", "")

GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")

_ALLOWED_DOMAINS = {
    d.strip().lower()
    for d in os.environ.get("ALLOWED_EMAIL_DOMAINS", "").split(",")
    if d.strip()
}
_ALLOWED_EMAILS = {
    e.strip().lower()
    for e in os.environ.get("ALLOWED_EMAILS", "brettevanssf@gmail.com").split(",")
    if e.strip()
}

ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")

SECRET_KEY      = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
SESSION_COOKIE  = "vc_session"
SESSION_MAX_AGE = 86400 * 30  # 30 days
_SECURE_COOKIES = BASE_URL.startswith("https://")

GOOGLE_AUTH_URL     = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL    = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

GOOGLE_SLIDES_REDIRECT_URI = BASE_URL + "/auth/google/slides-callback"
GOOGLE_SLIDES_SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
]

# ── Allowed models ─────────────────────────────────────────────────────────
ALLOWED_MODELS = {
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
}
CLOUD_MODELS = ALLOWED_MODELS


# ── Helpers ────────────────────────────────────────────────────────────────
def get_api_key(request: Request) -> str | None:
    """Return the server's Anthropic API key from env. Ignores request headers."""
    return os.environ.get("ANTHROPIC_API_KEY")


def _sanitize_session(raw: str) -> str:
    safe = "".join(c for c in raw if c.isalnum() or c == "-")[:64]
    return safe or "default"


def get_session_dirs(request: Request) -> tuple[Path, Path]:
    """Return (session_uploads_dir, session_context_dir) for the current session."""
    sid = _sanitize_session(request.headers.get("X-Session-ID", ""))
    uploads = UPLOADS / sid
    context = CONTEXT_DIR / sid
    uploads.mkdir(parents=True, exist_ok=True)
    context.mkdir(parents=True, exist_ok=True)
    return uploads, context


def _check_basic_auth(request: Request) -> bool:
    """Return True if the request carries valid HTTP Basic Auth credentials."""
    if not (_BASIC_USER and _BASIC_PASS):
        return False
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Basic "):
        return False
    import base64
    try:
        decoded = base64.b64decode(auth[6:]).decode("utf-8")
        user, _, password = decoded.partition(":")
        return user == _BASIC_USER and password == _BASIC_PASS
    except Exception:
        return False


def _require_admin_key(request: Request) -> None:
    """Raise 403 if the request doesn't carry a valid admin API key."""
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Admin API key not configured on server.")
    key = request.headers.get("X-Admin-Key", "")
    if not hmac.compare_digest(key, ADMIN_API_KEY):
        raise HTTPException(status_code=403, detail="Forbidden")
