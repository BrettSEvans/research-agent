"""Compliance report list / retrieval / share / delete endpoints."""
from __future__ import annotations

import json
import secrets

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from agent import SAVED_REPORTS_DIR
from auth import get_current_user
from db import get_db
from models import Report, User

from ._deps import BASE_URL

router = APIRouter()


@router.get("/reports")
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


@router.get("/reports/{report_id}")
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


@router.post("/reports/{report_id}/share")
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


@router.delete("/reports/{report_id}")
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
