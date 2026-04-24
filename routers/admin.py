"""Admin whitelist management + debug endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import is_email_allowed
from db import get_db
from models import Whitelist

from ._deps import _require_admin_key

router = APIRouter()


@router.get("/debug/whitelist-check")
async def debug_whitelist_check(email: str, db: Session = Depends(get_db)):
    """Temporary debug: test if an email passes the whitelist check."""
    from models import Whitelist as W
    email_lower = email.lower()
    domain = email_lower.split("@")[-1]
    db_email = db.query(W).filter_by(value=email_lower, type="email").first()
    db_domain = db.query(W).filter_by(value=domain, type="domain").first()
    all_entries = db.query(W).all()
    result = is_email_allowed(email_lower, db)
    return JSONResponse({
        "email": email_lower,
        "allowed": result,
        "db_email_hit": db_email.value if db_email else None,
        "db_domain_hit": db_domain.value if db_domain else None,
        "total_whitelist_entries": len(all_entries),
        "all_values": [e.value for e in all_entries],
    })


@router.get("/admin/whitelist")
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
    value: str
    added_by: str | None = None


@router.post("/admin/whitelist")
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


@router.delete("/admin/whitelist/{entry_id}")
async def whitelist_delete(entry_id: int, request: Request, db: Session = Depends(get_db)):
    """Remove an entry from the whitelist by ID."""
    _require_admin_key(request)
    entry = db.get(Whitelist, entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    db.delete(entry)
    db.commit()
    return JSONResponse({"ok": True})
