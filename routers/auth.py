"""Authentication routes: login page, Google One Tap, logout, /auth/me."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from auth import (
    find_or_create_user_org,
    is_email_allowed,
    load_session,
    sign_session,
)
from db import get_db

from ._deps import (
    GOOGLE_CLIENT_ID,
    SESSION_COOKIE,
    SESSION_MAX_AGE,
    templates,
)

router = APIRouter()


@router.get("/auth/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request=request, name="login.html", context={
        "google_enabled":   bool(GOOGLE_CLIENT_ID),
        "google_client_id": GOOGLE_CLIENT_ID,
    })


@router.post("/auth/google/one-tap")
async def google_one_tap(request: Request, db: Session = Depends(get_db)):
    """Verify a Google One Tap credential (signed JWT) and create a session."""
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

    if not is_email_allowed(email, db):
        raise HTTPException(status_code=403, detail="Your email is not on the access list. Contact the administrator.")

    user, org = find_or_create_user_org(db, email, display_name=name, picture=picture)
    session_payload = {
        "email":   email,
        "name":    name,
        "picture": picture,
        "user_id": user.id,
        "org_id":  org.id,
    }
    token = sign_session(session_payload)
    resp  = JSONResponse({"ok": True})
    resp.set_cookie(SESSION_COOKIE, token, max_age=SESSION_MAX_AGE,
                    httponly=True, samesite="none", secure=True)
    return resp


@router.get("/auth/logout")
async def logout():
    resp = RedirectResponse(url="/auth/login", status_code=302)
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@router.get("/auth/me")
async def auth_me(request: Request):
    """Return the current user's info (or null) — polled by the frontend."""
    token = request.cookies.get(SESSION_COOKIE, "")
    if token:
        session = load_session(token)
        if session:
            return JSONResponse({
                "email":   session.get("email"),
                "name":    session.get("name"),
                "picture": session.get("picture"),
            })
    return JSONResponse(None)
