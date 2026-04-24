"""Top-level page renders plus miscellaneous user-facing APIs.

Contains:
- GET  /            — index page (HTML)
- GET  /health      — public health check
- GET  /config      — feature flags for the frontend
- GET  /notifications, POST /notifications/{id}/dismiss — user notifications
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from auth import load_session
from db import get_db

from ._deps import ALLOWED_MODELS, SESSION_COOKIE, templates

router = APIRouter()


@router.get("/health")
async def health():
    """Public health check endpoint — always returns 200 (used by Railway/Render)."""
    return JSONResponse({"status": "ok"})


@router.get("/notifications")
async def get_notifications(request: Request, db: Session = Depends(get_db)):
    """Fetch undismissed notifications for the current user."""
    token = request.cookies.get(SESSION_COOKIE, "")
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    session = load_session(token)
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


@router.post("/notifications/{notification_id}/dismiss")
async def dismiss_notification(
    notification_id: int, request: Request, db: Session = Depends(get_db)
):
    """Dismiss a notification (mark as dismissed)."""
    token = request.cookies.get(SESSION_COOKIE, "")
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    session = load_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    from models import Notification

    notification = db.query(Notification).filter_by(id=notification_id).first()
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")

    if notification.user_id != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    notification.dismissed_at = datetime.now(timezone.utc)
    db.commit()

    return JSONResponse({"ok": True})


@router.get("/config")
async def get_config(request: Request):
    """Return feature flags and model configuration for the frontend."""
    enable_client_models = os.environ.get("ENABLE_CLIENT_MODELS", "false").lower() == "true"

    if enable_client_models:
        extractor_model = "claude-opus-4-6"
        analyzer_model = "claude-opus-4-6"
    else:
        extractor_model = "claude-sonnet-4-6"
        analyzer_model = "claude-sonnet-4-6"

    return JSONResponse({
        "enable_client_models": enable_client_models,
        "extractor_model": extractor_model,
        "analyzer_model": analyzer_model,
        "allowed_models": list(ALLOWED_MODELS),
    })


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")
