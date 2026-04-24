"""Compliance verification routes: /verify and /verify/stream."""
from __future__ import annotations

import json
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from agent import iter_compliance_report, run_compliance_report
from auth import get_current_user
from db import get_db
from deck_context import DeckContext
from models import Report, User

from ._deps import (
    ALLOWED_MODELS,
    get_api_key,
    get_session_dirs,
)

router = APIRouter()


@router.post("/verify/stream")
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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """SSE endpoint: emits one claim_result event per claim."""
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
                if event.get("event") == "done":
                    try:
                        report_data = event["data"]["report"]
                        report_id = report_data.get("report_id") or str(uuid.uuid4())
                        report_data["owner_email"] = current_user.email
                        row = Report(
                            report_id=report_id,
                            owner_id=current_user.id,
                            organization_id=current_user.organization_id,
                            company_name=report_data.get("company_name"),
                            cik=report_data.get("cik"),
                            extractor_model=report_data.get("extractor_model"),
                            analyzer_model=report_data.get("analyzer_model"),
                            report_json=json.dumps(report_data),
                        )
                        db.add(row)
                        db.commit()
                        event["data"]["report"]["report_id"] = report_id
                    except Exception as save_exc:
                        print(f"[verify/stream] Warning: failed to save report to DB: {save_exc}")
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'event': 'error', 'data': {'message': str(exc)}})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/verify")
async def verify(
    request: Request,
    context_id: Annotated[str, Form()],
    forms: Annotated[str, Form()] = "10-K,10-Q,S-1,8-K",
    filings_limit: Annotated[int, Form()] = 3,
    top_k: Annotated[int, Form()] = 5,
    analyzer_model: Annotated[str | None, Form()] = None,
    startup_stage: Annotated[str | None, Form()] = None,
    modules: Annotated[str | None, Form()] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
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

    try:
        report_data = report if isinstance(report, dict) else json.loads(report.body)
        report_id = report_data.get("report_id") or str(uuid.uuid4())
        report_data["owner_email"] = current_user.email
        row = Report(
            report_id=report_id,
            owner_id=current_user.id,
            organization_id=current_user.organization_id,
            company_name=report_data.get("company_name"),
            cik=report_data.get("cik"),
            extractor_model=report_data.get("extractor_model"),
            analyzer_model=report_data.get("analyzer_model"),
            report_json=json.dumps(report_data),
        )
        db.add(row)
        db.commit()
    except Exception as save_exc:
        print(f"[verify] Warning: failed to save report to DB: {save_exc}")

    return report
