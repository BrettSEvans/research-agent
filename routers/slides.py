"""Google Slides OAuth2 + extraction routes."""
from __future__ import annotations

import uuid
from typing import Annotated

import anthropic
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from conversion import fetch_google_slides_pdf
from version import EXTRACTOR_VERSION

from ._deps import (
    ALLOWED_MODELS,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_SLIDES_REDIRECT_URI,
    GOOGLE_SLIDES_SCOPES,
    get_api_key,
    get_session_dirs,
)

try:
    from google_auth_oauthlib.flow import Flow
    from google.oauth2.credentials import Credentials  # noqa: F401
    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False

router = APIRouter()


@router.get("/auth/google/slides-auth")
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

        resp = RedirectResponse(url=auth_url, status_code=302)
        resp.set_cookie("google_slides_state", state, max_age=3600, httponly=True, samesite="lax")
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OAuth flow init failed: {str(e)}")


@router.get("/auth/google/slides-callback")
async def google_slides_callback(request: Request, code: str = None, state: str = None):
    """OAuth2 callback after user grants Google Slides access."""
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")

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

        resp = RedirectResponse(
            url=f"/dashboard?google_slides_token={creds.token}",
            status_code=302
        )
        resp.delete_cookie("google_slides_state")
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token exchange failed: {str(e)}")


@router.post("/extract-from-google-slides")
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
        await fetch_google_slides_pdf(access_token, presentation_id, pdf_path)
    except ValueError as e:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to download Google Slides: {str(e)}")

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
