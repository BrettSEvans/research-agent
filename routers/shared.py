"""Public share-link endpoints — no authentication required."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from db import get_db
from models import Report, SavedExtraction

router = APIRouter()


@router.get("/shared/report/{share_token}")
async def shared_report(share_token: str, db: Session = Depends(get_db)):
    """Public read-only report view — no authentication required."""
    row = db.query(Report).filter_by(share_token=share_token, is_public=True).first()
    if not row:
        raise HTTPException(status_code=404, detail="Share link not found or no longer active.")
    return JSONResponse(json.loads(row.report_json))


@router.get("/shared/extraction/{share_token}")
async def shared_extraction(share_token: str, db: Session = Depends(get_db)):
    """Public read-only extraction view — no authentication required."""
    row = db.query(SavedExtraction).filter_by(share_token=share_token, is_public=True).first()
    if not row:
        raise HTTPException(status_code=404, detail="Share link not found or no longer active.")
    extraction = json.loads(row.extraction_json)
    meta = json.loads(row.meta_json) if row.meta_json else {}
    return JSONResponse({"extraction": extraction, "meta": meta})
