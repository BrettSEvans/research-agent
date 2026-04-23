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

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from agent import iter_compliance_report, run_compliance_report, SAVED_REPORTS_DIR
from auth import create_api_key, ensure_default_organization_and_user
from db import get_db, init_db, seed_eu_regulatory_sources, seed_ca_regulatory_sources
from deck_context import DeckContext
from extractor import extract_from_pdf
from models import Organization, Report, SavedExtraction, User, Whitelist
from version import ANALYZER_VERSION, EXTRACTOR_VERSION
from conversion import detect_file_type, convert_to_pdf, fetch_google_slides_pdf

# Google OAuth2 imports
try:
    from google_auth_oauthlib.flow import Flow
    from google.oauth2.credentials import Credentials
    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False

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

# Allow saasless.ai to call the /admin/whitelist API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://saasless.ai", "https://www.saasless.ai"],
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


# Graceful shutdown
@app.on_event("shutdown")
def shutdown_event():
    """Shutdown scheduler on FastAPI shutdown."""
    stop_scheduler()


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
# Optional comma-separated list of specific emails always allowed, regardless of domain.
# e.g. ALLOWED_EMAILS=admin@gmail.com,partner@otherdomain.com
_ALLOWED_EMAILS = {
    e.strip().lower()
    for e in os.environ.get("ALLOWED_EMAILS", "brettevanssf@gmail.com").split(",")
    if e.strip()
}

# Admin API key — used by saasless.ai to manage the PitchPerfect whitelist.
# Set ADMIN_API_KEY env var to a strong random secret.
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")

# Session cookie — HMAC-signed JSON, no extra dependencies
SECRET_KEY       = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
SESSION_COOKIE   = "vc_session"
SESSION_MAX_AGE  = 86400 * 30   # 30 days
_SECURE_COOKIES  = BASE_URL.startswith("https://")

GOOGLE_AUTH_URL     = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL    = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

# Google Slides OAuth2 configuration (subset of Google Drive access)
GOOGLE_SLIDES_REDIRECT_URI = BASE_URL + "/auth/google/slides-callback"
GOOGLE_SLIDES_SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",  # Read-only access to files
]


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


def _is_email_allowed(email: str, db: Session) -> bool:
    """
    Check if an email passes the whitelist (env vars OR DB entries).

    Returns True if:
    - No restrictions are configured (open access)
    - Email matches an entry in _ALLOWED_EMAILS
    - Email domain matches an entry in _ALLOWED_DOMAINS
    - Email or domain is in the Whitelist table
    """
    email = email.lower()
    domain = email.split("@")[-1]

    # Check env var lists
    if email in _ALLOWED_EMAILS:
        return True
    if _ALLOWED_DOMAINS and domain in _ALLOWED_DOMAINS:
        return True

    # Check DB whitelist
    db_email_hit = db.query(Whitelist).filter_by(value=email, type="email").first()
    db_domain_hit = db.query(Whitelist).filter_by(value=domain, type="domain").first()
    if db_email_hit or db_domain_hit:
        return True

    # If any restriction exists, deny by default
    any_restriction = bool(_ALLOWED_EMAILS or _ALLOWED_DOMAINS or db.query(Whitelist).first())
    if any_restriction:
        return False

    # No restrictions configured — allow all
    return True


# ── Auth middleware ────────────────────────────────────────────────────────

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
    # Auth routes and health-check are always public
    if path.startswith("/auth") or path.startswith("/shared") or path.startswith("/admin/whitelist") or path in ("/health", "/favicon.ico"):
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
    """Return the server's Anthropic API key from env. Ignores request headers.

    All Anthropic usage is billed to the operator's account — users no longer
    supply their own key. The `request` param is kept for signature compatibility
    with all existing callers.
    """
    return os.environ.get("ANTHROPIC_API_KEY")


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


# ───────────────────────── User DB helpers ────────────────────────────────

def _find_or_create_user_org(db: Session, email: str, display_name: str = "", picture: str = "") -> tuple[User, Organization]:
    """Return (user, org) for the given Google-authenticated email.

    Organisation is derived from the email domain (e.g. firm.com).
    Both org and user are created on first login; subsequent logins are no-ops.
    """
    domain = email.split("@")[-1].lower()

    org = db.query(Organization).filter_by(name=domain).first()
    if not org:
        org = Organization(name=domain)
        db.add(org)
        db.flush()  # get org.id before creating user

    user = db.query(User).filter_by(email=email).first()
    if not user:
        user = User(
            email=email,
            display_name=display_name or email,
            hashed_password="oauth",   # placeholder — Google users never use password auth
            api_key=create_api_key(),
            organization_id=org.id,
        )
        db.add(user)

    else:
        # Keep display_name / picture fresh on each login
        if display_name:
            user.display_name = display_name

    db.commit()
    db.refresh(user)
    db.refresh(org)
    return user, org


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """FastAPI dependency: return the authenticated User ORM object.

    Checks Google session cookie first, then falls back to HTTP Basic Auth.
    Raises HTTP 401 if neither is valid.
    """
    # 1. Google session cookie (has user_id after the auth bridge)
    token = request.cookies.get(SESSION_COOKIE, "")
    if token:
        session = _load_session(token)
        if session:
            user_id = session.get("user_id")
            if user_id:
                user = db.get(User, user_id)
                if user:
                    return user
            # Legacy cookie (email only, no user_id) — look up by email
            email = session.get("email", "")
            if email:
                user = db.query(User).filter_by(email=email).first()
                if user:
                    return user
                # First login with new auth bridge — create records now
                user, _ = _find_or_create_user_org(
                    db, email,
                    display_name=session.get("name", ""),
                    picture=session.get("picture", ""),
                )
                return user

    # 2. HTTP Basic Auth — map to the default org/user
    if _check_basic_auth(request):
        default_email = os.environ.get("DEFAULT_ADMIN_EMAIL", "")
        if default_email:
            user = db.query(User).filter_by(email=default_email).first()
            if user:
                return user
        # Fallback: first user in DB
        user = db.query(User).first()
        if user:
            return user

    raise HTTPException(status_code=401, detail="Not authenticated")


# ───────────────────────── Google SSO routes ──────────────────────────────

@app.get("/health")
async def health():
    """Public health check endpoint — always returns 200 (used by Railway/Render)."""
    return JSONResponse({"status": "ok"})


@app.get("/auth/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request=request, name="login.html", context={
        "google_enabled":   bool(GOOGLE_CLIENT_ID),
        "google_client_id": GOOGLE_CLIENT_ID,
    })


@app.post("/auth/google/one-tap")
async def google_one_tap(request: Request, db: Session = Depends(get_db)):
    """Verify a Google One Tap credential (signed JWT) and create a session.

    Called by the login page JS after the user approves the One Tap prompt.
    Works inside cross-origin iframes on saasless.ai because no redirect is
    needed — the credential comes back via a JS callback, not a URL parameter.
    """
    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests

    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=503, detail="Google SSO is not configured on this server.")

    body = await request.form()
    credential = body.get("credential", "")
    if not credential:
        raise HTTPException(status_code=400, detail="Missing credential")

    try:
        idinfo = google_id_token.verify_oauth2_token(
            credential, google_requests.Request(), GOOGLE_CLIENT_ID
        )
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")

    email   = idinfo.get("email", "").lower()
    name    = idinfo.get("name", email)
    picture = idinfo.get("picture", "")

    # Whitelist check — env-var lists OR DB entries; open if nothing configured.
    if not _is_email_allowed(email, db):
        raise HTTPException(status_code=403, detail="Your email is not on the access list. Contact the administrator.")

    user, org = _find_or_create_user_org(db, email, display_name=name, picture=picture)
    session_payload = {
        "email":   email,
        "name":    name,
        "picture": picture,
        "user_id": user.id,
        "org_id":  org.id,
    }
    token = _sign_session(session_payload)
    resp  = JSONResponse({"ok": True})
    # SameSite=None is required so the cookie is sent in cross-origin iframe
    # requests. Secure=True is mandatory when SameSite=None (Railway = HTTPS).
    resp.set_cookie(SESSION_COOKIE, token, max_age=SESSION_MAX_AGE,
                    httponly=True, samesite="none", secure=True)
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


# ─────────────────────── Notification API ────────────────────────────────────
# User notifications for regulatory updates. Requires valid session.

@app.get("/notifications")
async def get_notifications(request: Request, db: Session = Depends(get_db)):
    """
    Fetch undismissed notifications for the current user.

    Requires: Valid session cookie
    Returns: [{"id", "module", "source_name", "title", "body", "created_at"}]
    """
    token = request.cookies.get(SESSION_COOKIE, "")
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    session = _load_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    from models import Notification

    notifications = (
        db.query(Notification)
        .filter_by(user_id=user_id, dismissed_at=None)
        .order_by(Notification.created_at.desc())
        .all()
    )

    return JSONResponse([
        {
            "id": n.id,
            "module": n.module,
            "source_name": n.source_name,
            "title": n.title,
            "body": n.body,
            "created_at": n.created_at.isoformat(),
        }
        for n in notifications
    ])


@app.post("/notifications/{notification_id}/dismiss")
async def dismiss_notification(
    notification_id: int, request: Request, db: Session = Depends(get_db)
):
    """
    Dismiss a notification (mark as dismissed).

    Requires: Valid session cookie + ownership (user_id matches)
    Returns: {"ok": true}
    """
    token = request.cookies.get(SESSION_COOKIE, "")
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    session = _load_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    from models import Notification
    from datetime import datetime, timezone

    notification = db.query(Notification).filter_by(id=notification_id).first()
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")

    if notification.user_id != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    notification.dismissed_at = datetime.now(timezone.utc)
    db.commit()

    return JSONResponse({"ok": True})


# ─────────────────────── Admin whitelist API ────────────────────────────────
# Protected by X-Admin-Key header. Called by saasless.ai to manage who
# can log in via Google SSO. No Google session required — uses its own key.

def _require_admin_key(request: Request) -> None:
    """Raise 403 if the request doesn't carry a valid admin API key."""
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Admin API key not configured on server.")
    key = request.headers.get("X-Admin-Key", "")
    if not hmac.compare_digest(key, ADMIN_API_KEY):
        raise HTTPException(status_code=403, detail="Forbidden")


@app.get("/admin/whitelist")
async def whitelist_list(request: Request, db: Session = Depends(get_db)):
    """Return all whitelisted emails and domains."""
    _require_admin_key(request)
    entries = db.query(Whitelist).order_by(Whitelist.created_at.desc()).all()
    return JSONResponse([
        {
            "id":         e.id,
            "value":      e.value,
            "type":       e.type,
            "added_by":   e.added_by,
            "created_at": e.created_at.isoformat(),
        }
        for e in entries
    ])


class WhitelistAddRequest(BaseModel):
    value: str   # email address or domain (e.g. "user@example.com" or "example.com")
    added_by: str | None = None


@app.post("/admin/whitelist")
async def whitelist_add(body: WhitelistAddRequest, request: Request, db: Session = Depends(get_db)):
    """Add an email or domain to the whitelist."""
    _require_admin_key(request)
    value = body.value.strip().lower()
    if not value:
        raise HTTPException(status_code=400, detail="value is required")
    entry_type = "email" if "@" in value else "domain"
    existing = db.query(Whitelist).filter_by(value=value).first()
    if existing:
        return JSONResponse({"id": existing.id, "value": existing.value, "type": existing.type}, status_code=200)
    entry = Whitelist(value=value, type=entry_type, added_by=body.added_by)
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return JSONResponse({"id": entry.id, "value": entry.value, "type": entry.type}, status_code=201)


@app.delete("/admin/whitelist/{entry_id}")
async def whitelist_delete(entry_id: int, request: Request, db: Session = Depends(get_db)):
    """Remove an entry from the whitelist by ID."""
    _require_admin_key(request)
    entry = db.get(Whitelist, entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    db.delete(entry)
    db.commit()
    return JSONResponse({"ok": True})


@app.get("/config")
async def get_config(request: Request):
    """Return feature flags and model configuration for the frontend.

    The frontend uses this to:
    - Determine which UI cards to show (models, API key)
    - Override hardcoded models when ENABLE_CLIENT_MODELS=true
    - Bind configuration to the current session

    This endpoint is unauthenticated so the frontend can load config before login.
    It only returns feature flags, not sensitive data.
    """
    enable_client_models = os.environ.get("ENABLE_CLIENT_MODELS", "false").lower() == "true"

    # Determine models based on flag
    if enable_client_models:
        extractor_model = "claude-opus-4-6"
        analyzer_model = "claude-opus-4-6"
    else:
        # Default to current hardcoded values in frontend
        extractor_model = "claude-sonnet-4-6"
        analyzer_model = "claude-sonnet-4-6"

    return JSONResponse({
        "enable_client_models": enable_client_models,
        "extractor_model": extractor_model,
        "analyzer_model": analyzer_model,
        "allowed_models": list(ALLOWED_MODELS),
    })


# ───────────────────────── Google Slides OAuth2 ──────────────────────────────

@app.get("/auth/google/slides-auth")
async def google_slides_auth(request: Request):
    """Initiate Google OAuth2 flow for Google Slides access."""
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
        raise HTTPException(
            status_code=400,
            detail="Google OAuth2 credentials not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET."
        )

    try:
        flow = Flow.from_client_config(
            {
                "installed": {
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uris": [GOOGLE_SLIDES_REDIRECT_URI],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            },
            scopes=GOOGLE_SLIDES_SCOPES,
            redirect_uri=GOOGLE_SLIDES_REDIRECT_URI,
        )
        auth_url, state = flow.authorization_url(prompt="consent", access_type="offline")

        # Store state in a cookie for validation on callback
        resp = RedirectResponse(url=auth_url, status_code=302)
        resp.set_cookie("google_slides_state", state, max_age=3600, httponly=True, samesite="lax")
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OAuth flow init failed: {str(e)}")


@app.get("/auth/google/slides-callback")
async def google_slides_callback(request: Request, code: str = None, state: str = None):
    """OAuth2 callback after user grants Google Slides access."""
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")

    # Validate state cookie
    stored_state = request.cookies.get("google_slides_state")
    if not stored_state or stored_state != state:
        raise HTTPException(status_code=400, detail="State mismatch — possible CSRF attack")

    try:
        flow = Flow.from_client_config(
            {
                "installed": {
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uris": [GOOGLE_SLIDES_REDIRECT_URI],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            },
            scopes=GOOGLE_SLIDES_SCOPES,
            redirect_uri=GOOGLE_SLIDES_REDIRECT_URI,
        )
        flow.fetch_token(code=code)
        creds = flow.credentials

        # Redirect back to dashboard with token in URL
        # In production, you'd want to pass this securely (encrypted token, session, etc.)
        resp = RedirectResponse(
            url=f"/dashboard?google_slides_token={creds.token}",
            status_code=302
        )
        resp.delete_cookie("google_slides_state")
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token exchange failed: {str(e)}")


@app.post("/extract-from-google-slides")
async def extract_google_slides(
    request: Request,
    presentation_id: Annotated[str, Form()],
    access_token: Annotated[str, Form()],
    extractor_model: Annotated[str | None, Form()] = None,
):
    """Download a Google Slides presentation as PDF and run extraction."""
    if extractor_model and extractor_model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {extractor_model}")

    session_uploads, _ = get_session_dirs(request)
    token = uuid.uuid4().hex[:12]
    pdf_path = session_uploads / f"slides-{token}.pdf"

    try:
        # Download Google Slides as PDF using OAuth2 token
        await fetch_google_slides_pdf(access_token, presentation_id, pdf_path)
    except ValueError as e:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to download Google Slides: {str(e)}")

    # Perform extraction on the downloaded PDF
    try:
        client = anthropic.Anthropic(api_key=get_api_key(request))
        from extractor import extract_basics_and_infer_stage
        extraction = extract_basics_and_infer_stage(pdf_path, client=client, model=extractor_model)
    except Exception as exc:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc

    return JSONResponse(
        {
            "context_id": token,
            "extraction": extraction.model_dump(),
            "extractor_version": EXTRACTOR_VERSION,
        }
    )


# ───────────────────────── main app routes ────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


ALLOWED_MODELS = {
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
}

# All allowed models are Anthropic-cloud; local/Inception paths removed along
# with the BYOK-era model dropdown.
CLOUD_MODELS = ALLOWED_MODELS


@app.post("/extract")
async def extract(
    request: Request,
    file: UploadFile = File(...),
    extractor_model: Annotated[str | None, Form()] = None,
):
    # Detect and validate file type
    try:
        file_type = detect_file_type(file.filename or "")
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported: PDF, PowerPoint (.ppt/.pptx), Word (.doc/.docx), Excel (.xlsx). {str(e)}"
        )

    if extractor_model and extractor_model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {extractor_model}")

    session_uploads, _ = get_session_dirs(request)
    content = await file.read()
    token = uuid.uuid4().hex[:12]

    # Save file temporarily for conversion
    temp_suffix = f".{file_type}" if file_type != "pdf" else ".pdf"
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=temp_suffix, dir=str(session_uploads), prefix=f"{token}_"
    ) as tmp:
        tmp.write(content)
        temp_path = Path(tmp.name)

    # Convert to PDF if needed
    try:
        if file_type != "pdf":
            pdf_path = await convert_to_pdf(temp_path, file_type)
        else:
            # Already PDF, just rename to standard naming
            pdf_path = session_uploads / f"{token}_{uuid.uuid4().hex[:8]}.pdf"
            temp_path.rename(pdf_path)
    except ValueError as e:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"File conversion failed: {str(e)}")

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
async def list_saved_extractions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return all saved extractions for the current org, newest first."""
    rows = (
        db.query(SavedExtraction)
        .filter_by(organization_id=current_user.organization_id)
        .order_by(SavedExtraction.saved_at.desc())
        .all()
    )
    items = []
    for row in rows:
        try:
            extraction = json.loads(row.extraction_json) if row.extraction_json else {}
            meta = json.loads(row.meta_json) if row.meta_json else {}

            claims = extraction.get("claims", [])
            by_category: dict[str, int] = {}
            for cl in claims:
                cat = cl.get("category", "other")
                by_category[cat] = by_category.get(cat, 0) + 1

            metrics = [m.get("metric_name", "") for m in extraction.get("key_metrics", []) if m.get("metric_name")]

            items.append({
                "save_id":          row.save_id,
                "company_name":     row.company_name or "Unknown",
                "original_filename": row.original_filename or "",
                "extractor_model":  row.extractor_model or "",
                "extractor_version": row.extractor_version or "",
                "saved_at":         row.saved_at.isoformat() if row.saved_at else "",
                "claims_count":     len(claims),
                "claims_by_category": by_category,
                "key_metrics":      metrics,
                "stage":            extraction.get("stage_assessment", {}).get("stage") if extraction.get("stage_assessment") else None,
                "is_public":        row.is_public,
                "owner_email":      meta.get("owner_email", ""),
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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Persist a session extraction to the saved library."""
    _, session_ctx = get_session_dirs(request)
    context_path = session_ctx / f"deck_{context_id}.json"
    if not context_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown context_id: {context_id}")

    extraction_data = json.loads(context_path.read_text())
    company_name = extraction_data.get("company", {}).get("name", "Unknown")
    save_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc)
    meta = {
        "save_id": save_id,
        "company_name": company_name,
        "original_filename": original_filename,
        "extractor_model": extractor_model,
        "extractor_version": EXTRACTOR_VERSION,
        "saved_at": now.isoformat(),
        "owner_email": current_user.email,
    }
    saved_file = {
        "meta": meta,
        "extraction": extraction_data,
    }
    # Write to disk (backup) and to DB (source of truth for queries)
    (SAVED_DIR / f"{save_id}.json").write_text(json.dumps(saved_file, indent=2))
    db.add(SavedExtraction(
        save_id=save_id,
        owner_id=current_user.id,
        organization_id=current_user.organization_id,
        company_name=company_name,
        original_filename=original_filename,
        extractor_model=extractor_model,
        extractor_version=EXTRACTOR_VERSION,
        meta_json=json.dumps(meta),
        extraction_json=json.dumps(extraction_data),
        saved_at=now,
    ))
    db.commit()
    return JSONResponse({"save_id": save_id, "company_name": company_name})


@app.post("/saved-extractions/{save_id}/load")
async def load_saved_extraction(
    request: Request,
    save_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Load a saved extraction back into a fresh session context."""
    row = db.query(SavedExtraction).filter_by(
        save_id=save_id,
        organization_id=current_user.organization_id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Unknown save_id: {save_id}")

    extraction_data = json.loads(row.extraction_json)
    meta = json.loads(row.meta_json) if row.meta_json else {}

    # Create a fresh session context so /verify/stream works normally
    _, session_ctx = get_session_dirs(request)
    token = uuid.uuid4().hex[:12]
    context_path = session_ctx / f"deck_{token}.json"
    context_path.write_text(json.dumps(extraction_data, indent=2))

    return JSONResponse({
        "context_id": token,
        "extraction": extraction_data,
        "meta": meta,
    })


@app.post("/saved-extractions/{save_id}/share")
async def share_saved_extraction(
    save_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate a public read-only share link for a saved extraction."""
    row = db.query(SavedExtraction).filter_by(
        save_id=save_id,
        organization_id=current_user.organization_id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Unknown save_id: {save_id}")

    if not row.share_token:
        row.share_token = secrets.token_urlsafe(24)
    row.is_public = True
    db.commit()
    return JSONResponse({"share_url": f"{BASE_URL}/shared/extraction/{row.share_token}"})


@app.delete("/saved-extractions/{save_id}")
async def delete_saved_extraction(
    save_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    row = db.query(SavedExtraction).filter_by(
        save_id=save_id,
        organization_id=current_user.organization_id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Unknown save_id: {save_id}")
    saved_path = SAVED_DIR / f"{save_id}.json"
    if saved_path.exists():
        saved_path.unlink()
    db.delete(row)
    db.commit()
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
async def list_reports(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return all compliance reports for the current org, newest first."""
    rows = (
        db.query(Report)
        .filter_by(organization_id=current_user.organization_id)
        .order_by(Report.generated_at.desc())
        .all()
    )
    items = []
    for row in rows:
        try:
            data = json.loads(row.report_json)
            results = data.get("results", [])

            consistent   = sum(1 for r in results if r.get("verdict") == "CONSISTENT")
            contradicts  = sum(1 for r in results if r.get("verdict") == "CONTRADICTS")
            unsupported  = sum(1 for r in results if r.get("verdict") == "UNSUPPORTED")
            insufficient = sum(1 for r in results if r.get("verdict") == "INSUFFICIENT_EVIDENCE")
            fls_flags    = sum(1 for r in results if r.get("verdict") == "CONTRADICTS" and r.get("forward_looking"))

            top_flags = [
                {"claim": r.get("claim", ""), "verdict": r.get("verdict", ""), "forward_looking": r.get("forward_looking", False)}
                for r in results if r.get("verdict") == "CONTRADICTS"
            ][:3]

            items.append({
                "report_id":        row.report_id,
                "company_name":     row.company_name or data.get("company_name", "Unknown"),
                "generated_at":     row.generated_at.isoformat() if row.generated_at else data.get("generated_at", ""),
                "assumed_industry": data.get("assumed_industry", ""),
                "cik":              row.cik or data.get("cik", ""),
                "extractor_model":  row.extractor_model or data.get("extractor_model", ""),
                "extractor_version": data.get("extractor_version", ""),
                "analyzer_model":   row.analyzer_model or data.get("analyzer_model", ""),
                "analyzer_version": data.get("analyzer_version", ""),
                "claims_analyzed":  data.get("claims_analyzed", 0),
                "consistent":       consistent,
                "contradicts":      contradicts,
                "unsupported":      unsupported,
                "insufficient":     insufficient,
                "fls_flags":        fls_flags,
                "top_flags":        top_flags,
                "is_public":        row.is_public,
                "owner_email":      data.get("owner_email", ""),
            })
        except Exception:
            continue
    return JSONResponse(items)


@app.get("/reports/{report_id}")
async def get_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve a specific saved compliance report (org-gated)."""
    row = db.query(Report).filter_by(
        report_id=report_id,
        organization_id=current_user.organization_id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Unknown report_id: {report_id}")
    return JSONResponse(json.loads(row.report_json))


@app.post("/reports/{report_id}/share")
async def share_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate a public read-only share link for a compliance report."""
    row = db.query(Report).filter_by(
        report_id=report_id,
        organization_id=current_user.organization_id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Unknown report_id: {report_id}")

    if not row.share_token:
        row.share_token = secrets.token_urlsafe(24)
    row.is_public = True
    db.commit()
    return JSONResponse({"share_url": f"{BASE_URL}/shared/report/{row.share_token}"})


@app.delete("/reports/{report_id}")
async def delete_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a saved compliance report (org-gated)."""
    row = db.query(Report).filter_by(
        report_id=report_id,
        organization_id=current_user.organization_id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Unknown report_id: {report_id}")
    report_path = SAVED_REPORTS_DIR / f"report_{report_id}.json"
    if report_path.exists():
        report_path.unlink()
    db.delete(row)
    db.commit()
    return JSONResponse({"deleted": report_id})


# ───────────────────────── public share routes ────────────────────────────

@app.get("/shared/report/{share_token}")
async def shared_report(share_token: str, db: Session = Depends(get_db)):
    """Public read-only report view — no authentication required."""
    row = db.query(Report).filter_by(share_token=share_token, is_public=True).first()
    if not row:
        raise HTTPException(status_code=404, detail="Share link not found or no longer active.")
    return JSONResponse(json.loads(row.report_json))


@app.get("/shared/extraction/{share_token}")
async def shared_extraction(share_token: str, db: Session = Depends(get_db)):
    """Public read-only extraction view — no authentication required."""
    row = db.query(SavedExtraction).filter_by(share_token=share_token, is_public=True).first()
    if not row:
        raise HTTPException(status_code=404, detail="Share link not found or no longer active.")
    extraction = json.loads(row.extraction_json)
    meta = json.loads(row.meta_json) if row.meta_json else {}
    return JSONResponse({"extraction": extraction, "meta": meta})

