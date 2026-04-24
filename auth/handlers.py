"""Authentication handlers: session management, whitelist checking, user/org creation."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from auth.core import create_api_key
from models import Organization, User, Whitelist

logger = logging.getLogger(__name__)

# Session configuration (imported from web.py)
SESSION_MAX_AGE = 86400 * 30  # 30 days
SESSION_COOKIE = "vc_session"
SECRET_KEY = os.environ.get("SECRET_KEY") or "dev_secret_key_change_in_production"

# Whitelist configuration (env var lists)
_ALLOWED_EMAILS = {
    e.strip().lower()
    for e in os.environ.get("ALLOWED_EMAILS", "").split(",")
    if e.strip()
}
_ALLOWED_DOMAINS = {
    d.strip().lower()
    for d in os.environ.get("ALLOWED_EMAIL_DOMAINS", "").split(",")
    if d.strip()
}


def sign_session(payload: dict) -> str:
    """Return a compact, HMAC-signed token encoding *payload* + expiry.

    Format: Base64-encoded JSON + HMAC-SHA256 signature
    Expiry: SESSION_MAX_AGE seconds from creation

    Args:
        payload: Dictionary of session data (e.g., {"user_id": 1, "email": "..."})

    Returns:
        Signed session token (format: {base64_json}.{signature})
    """
    body = {**payload, "exp": int(time.time()) + SESSION_MAX_AGE}
    data = base64.urlsafe_b64encode(
        json.dumps(body, separators=(",", ":")).encode()
    ).rstrip(b"=").decode()
    sig = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
    return f"{data}.{sig}"


def load_session(token: str) -> Optional[dict]:
    """Verify and decode a session token; returns None if invalid or expired.

    Validates HMAC signature and checks expiry timestamp.

    Args:
        token: Session token (format: {base64_json}.{signature})

    Returns:
        Decoded session payload dictionary if valid, None otherwise
    """
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


def is_email_allowed(email: str, db: Session) -> bool:
    """Check if an email passes the whitelist (env vars OR DB entries).

    Three-tier system:
    1. Env vars (ALLOWED_EMAILS, ALLOWED_EMAIL_DOMAINS) — always allows
    2. DB Whitelist table (email or domain entries)
    3. Open access if no restrictions configured

    Args:
        email: Email address to check (normalized to lowercase)
        db: SQLAlchemy session

    Returns:
        True if email is allowed, False if explicitly denied by whitelist
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


def find_or_create_user_org(
    db: Session,
    email: str,
    display_name: str = "",
    picture: str = "",
) -> Tuple[User, Organization]:
    """Return (user, org) for the given Google-authenticated email.

    Organisation is derived from the email domain (e.g. firm.com).
    Both org and user are created on first login; subsequent logins are no-ops.
    User gets a random API key on creation.

    Args:
        db: SQLAlchemy session
        email: Email address (will be normalized to lowercase)
        display_name: User's display name (optional, defaults to email)
        picture: User's profile picture URL (currently unused but available for future)

    Returns:
        Tuple of (User, Organization) objects
    """
    email = email.lower()
    domain = email.split("@")[-1].lower()

    # Get or create org by domain
    org = db.query(Organization).filter_by(name=domain).first()
    if not org:
        org = Organization(name=domain)
        db.add(org)
        db.flush()  # get org.id before creating user

    # Get or create user
    user = db.query(User).filter_by(email=email).first()
    if not user:
        user = User(
            email=email,
            display_name=display_name or email,
            hashed_password="oauth",  # placeholder — Google users never use password auth
            api_key=create_api_key(),
            organization_id=org.id,
        )
        db.add(user)
    else:
        # Keep display_name fresh on each login
        if display_name:
            user.display_name = display_name

    db.commit()
    db.refresh(user)
    db.refresh(org)
    return user, org
