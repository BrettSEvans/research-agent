"""Saved extraction library endpoints."""
from __future__ import annotations

import json
import secrets
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from auth import get_current_user
from db import get_db
from models import SavedExtraction, User
from version import EXTRACTOR_VERSION

from ._deps import BASE_URL, SAVED_DIR, get_session_dirs

router = APIRouter()


@router.get("/saved-extractions")
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


@router.post("/saved-extractions")
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


@router.post("/saved-extractions/{save_id}/load")
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

    _, session_ctx = get_session_dirs(request)
    token = uuid.uuid4().hex[:12]
    context_path = session_ctx / f"deck_{token}.json"
    context_path.write_text(json.dumps(extraction_data, indent=2))

    return JSONResponse({
        "context_id": token,
        "extraction": extraction_data,
        "meta": meta,
    })


@router.post("/saved-extractions/{save_id}/share")
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


@router.delete("/saved-extractions/{save_id}")
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
