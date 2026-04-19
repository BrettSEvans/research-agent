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

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import anthropic
from dotenv import load_dotenv

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.templating import Jinja2Templates

from agent import iter_compliance_report, run_compliance_report, SAVED_REPORTS_DIR
from auth import (
    authenticate_user,
    create_user,
    ensure_default_organization_and_user,
    get_user_by_api_key,
)
from db import SessionLocal, init_db, migrate_existing_data
from deck_context import DeckContext
from extractor import DeckExtraction as DeckExtractionSchema
from extractor import extract_failed_pages_with_claude, extract_from_pdf
from models import (
    DeckContext as DeckContextModel,
    Project as ProjectModel,
    Report as ReportModel,
    SavedExtraction as SavedExtractionModel,
    Upload as UploadModel,
)
from version import ANALYZER_VERSION, EXTRACTOR_VERSION

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

auth_scheme = HTTPBearer()
PUBLIC_PATHS = {
    "/",
    "/auth/login",
    "/openapi.json",
    "/docs",
    "/redoc",
    "/docs/oauth2-redirect",
}


def get_default_project_id(request: Request) -> int:
    user = request.state.user
    db = request.state.db
    default_name = os.environ.get("DEFAULT_PROJECT_NAME", "default")
    project = (
        db.query(ProjectModel)
        .filter(ProjectModel.name == default_name, ProjectModel.organization_id == user.organization_id)
        .first()
    )
    if not project:
        project = ProjectModel(
            name=default_name,
            organization_id=user.organization_id,
            owner_id=user.id,
        )
        db.add(project)
        db.commit()
        db.refresh(project)
    return project.id


def require_context_ownership(request: Request, context_id: str) -> DeckContextModel:
    db = request.state.db
    context = (
        db.query(DeckContextModel)
        .filter(DeckContextModel.context_id == context_id)
        .filter(DeckContextModel.organization_id == request.state.user.organization_id)
        .first()
    )
    if not context:
        raise HTTPException(status_code=404, detail="Unknown context_id or unauthorized")
    return context


def require_saved_extraction(request: Request, save_id: str) -> SavedExtractionModel:
    db = request.state.db
    saved = (
        db.query(SavedExtractionModel)
        .filter(SavedExtractionModel.save_id == save_id)
        .filter(SavedExtractionModel.organization_id == request.state.user.organization_id)
        .first()
    )
    if not saved:
        raise HTTPException(status_code=404, detail="Unknown save_id or unauthorized")
    return saved


def require_report(request: Request, report_id: str) -> ReportModel:
    db = request.state.db
    report = (
        db.query(ReportModel)
        .filter(ReportModel.report_id == report_id)
        .filter(ReportModel.organization_id == request.state.user.organization_id)
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Unknown report_id or unauthorized")
    return report


@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    request.state.db = SessionLocal()
    request.state.user = None
    try:
        response = await call_next(request)
        request.state.db.commit()
        return response
    except Exception:
        request.state.db.rollback()
        raise
    finally:
        request.state.db.close()


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    if path in PUBLIC_PATHS or request.method == "OPTIONS":
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(
            {"detail": "Missing Authorization header. Use Bearer token."},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    token = auth_header.split(" ", 1)[1].strip()
    user = get_user_by_api_key(request.state.db, token)
    if not user:
        return JSONResponse(
            {"detail": "Invalid or expired API token."},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    request.state.user = user
    return await call_next(request)


@app.on_event("startup")
def startup():
    init_db()
    db = SessionLocal()
    try:
        organization, user, project = ensure_default_organization_and_user(db)
    finally:
        db.close()
    migrate_existing_data(
        default_user_email=os.environ.get("DEFAULT_ADMIN_EMAIL", "brettevanssf@gmail.com"),
        default_org_name=os.environ.get("DEFAULT_ORG_NAME", "brettevanssf"),
        default_project_name=os.environ.get("DEFAULT_PROJECT_NAME", "default"),
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/auth/login")
async def auth_login(
    request: Request,
    email: Annotated[str, Form()],
    password: Annotated[str, Form()],
):
    user = authenticate_user(request.state.db, email, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return JSONResponse(
        {
            "email": user.email,
            "display_name": user.display_name,
            "organization": user.organization.name,
            "api_key": user.api_key,
        }
    )


@app.post("/auth/register")
async def auth_register(
    request: Request,
    email: Annotated[str, Form()],
    password: Annotated[str, Form()],
    display_name: Annotated[str | None, Form()] = None,
):
    current_user = request.state.user
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    user = create_user(
        request.state.db,
        email=email,
        password=password,
        organization=current_user.organization,
        display_name=display_name,
    )
    return JSONResponse(
        {
            "email": user.email,
            "display_name": user.display_name,
            "organization": user.organization.name,
            "api_key": user.api_key,
        }
    )


@app.get("/users/me")
async def get_current_user(request: Request):
    user = request.state.user
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return JSONResponse(
        {
            "email": user.email,
            "display_name": user.display_name,
            "organization": user.organization.name,
            "is_admin": user.is_admin,
            "created_at": user.created_at.isoformat(),
        }
    )


@app.get("/projects")
async def list_projects(request: Request):
    user = request.state.user
    projects = (
        request.state.db.query(ProjectModel)
        .filter(ProjectModel.organization_id == user.organization_id)
        .order_by(ProjectModel.created_at.desc())
        .all()
    )
    return JSONResponse(
        [
            {
                "project_id": project.id,
                "name": project.name,
                "description": project.description,
                "owner": project.owner.display_name if project.owner else None,
                "created_at": project.created_at.isoformat(),
            }
            for project in projects
        ]
    )


@app.post("/projects")
async def create_project(
    request: Request,
    name: Annotated[str, Form()],
    description: Annotated[str | None, Form()] = None,
):
    user = request.state.user
    existing = (
        request.state.db.query(ProjectModel)
        .filter(ProjectModel.organization_id == user.organization_id, ProjectModel.name == name)
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="Project already exists")
    project = ProjectModel(
        name=name,
        description=description,
        organization_id=user.organization_id,
        owner_id=user.id,
    )
    request.state.db.add(project)
    request.state.db.commit()
    request.state.db.refresh(project)
    return JSONResponse(
        {
            "project_id": project.id,
            "name": project.name,
            "description": project.description,
            "owner": user.display_name,
            "created_at": project.created_at.isoformat(),
        }
    )


ALLOWED_MODELS = {
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    # Inception Labs Mercury models
    "mercury-2",
    "mercury-coder-small",
    # Local (Ollama) text-only models
    "qwen3.5:9b",
    # Local (Ollama) vision models — can read image-based PDFs
    "llama3.2-vision:11b",
    "gemma4:latest",
    "gemma4:26b",
}

# Cloud-only models valid for /extract/complete. Local models must not be used
# here because extract_failed_pages_with_claude() always calls the Anthropic SDK.
CLOUD_MODELS = {"claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-6"}


@app.post("/extract")
async def extract(
    request: Request,
    file: UploadFile = File(...),
    extractor_model: Annotated[str | None, Form()] = None,
):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if extractor_model and extractor_model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {extractor_model}")

    user = request.state.user
    project_id = get_default_project_id(request)

    content = await file.read()
    token = uuid.uuid4().hex[:12]
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf", dir=str(UPLOADS), prefix=f"{token}_"
    ) as tmp:
        tmp.write(content)
        pdf_path = Path(tmp.name)

    upload = UploadModel(
        upload_token=token,
        owner_id=user.id,
        organization_id=user.organization_id,
        project_id=project_id,
        original_filename=file.filename,
        stored_path=str(pdf_path),
    )
    request.state.db.add(upload)
    request.state.db.commit()

    deck_context = DeckContextModel(
        context_id=token,
        owner_id=user.id,
        organization_id=user.organization_id,
        project_id=project_id,
        original_filename=file.filename,
        extractor_model=extractor_model,
        extractor_version=EXTRACTOR_VERSION,
        status="uploaded",
    )
    request.state.db.add(deck_context)
    request.state.db.commit()

    try:
        client = anthropic.Anthropic()
        from extractor import extract_basics_and_infer_stage

        extraction = extract_basics_and_infer_stage(pdf_path, client=client, model=extractor_model)
    except Exception as exc:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc

    # Do not unlink the PDF yet; keep the upload until deep extraction completes.
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
    deck_record = require_context_ownership(request, context_id)
    pdf_paths = list(UPLOADS.glob(f"{context_id}_*.pdf"))
    context_path = CONTEXT_DIR / f"deck_{context_id}.json"

    # No PDF in uploads — this context was loaded from a saved extraction.
    # The context file already exists, so skip re-extraction and return it as-is.
    if not pdf_paths:
        if context_path.exists() or deck_record.extraction_json:
            if not context_path.exists() and deck_record.extraction_json:
                context_path.write_text(deck_record.extraction_json)
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
    failed_path = CONTEXT_DIR / f"deck_{context_id}_failed.json"

    try:
        client = anthropic.Anthropic()
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

    deck_record.context_path = str(context_path)
    deck_record.extraction_json = json.dumps(extraction.model_dump())
    deck_record.status = "saved"
    request.state.db.add(deck_record)
    request.state.db.commit()

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
    deck_record = require_context_ownership(request, context_id)
    context_path = CONTEXT_DIR / f"deck_{context_id}.json"
    failed_path = CONTEXT_DIR / f"deck_{context_id}_failed.json"

    if not context_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown context_id: {context_id}")
    if not failed_path.exists():
        raise HTTPException(status_code=400, detail="No failed-page sidecar found for this context.")

    pdf_paths = list(UPLOADS.glob(f"{context_id}_*.pdf"))
    if not pdf_paths:
        raise HTTPException(
            status_code=404,
            detail="PDF not found — it may have already been deleted. Re-upload the deck and try again.",
        )

    if cloud_model not in CLOUD_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Cloud completion requires a Claude model. Got: {cloud_model!r}. "
                   f"Valid options: {sorted(CLOUD_MODELS)}",
        )

    pdf_path = pdf_paths[0]
    failed_data = json.loads(failed_path.read_text())
    failed_pages: list[int] = failed_data.get("failed_pages", [])

    if not failed_pages:
        failed_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Failed-page sidecar is empty — nothing to complete.")

    requested_metrics = modules.split(",") if modules else None

    try:
        client = anthropic.Anthropic()

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

        merged_extraction = DeckExtractionSchema(
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
        deck_record.context_path = str(context_path)
        deck_record.extraction_json = json.dumps(merged_extraction.model_dump())
        deck_record.status = "saved"
        request.state.db.add(deck_record)
        request.state.db.commit()

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
    require_context_ownership(request, context_id)
    context_path = CONTEXT_DIR / f"deck_{context_id}.json"
    if not context_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown context_id: {context_id}")

    deck = DeckContext.load(context_path)
    claims = deck.claims_for_verification()

    from sec import lookup_cik
    cik = None
    key = deck.company_lookup_key()
    if key:
        cik = key.zfill(10) if (key.isdigit() and len(key) <= 10) else lookup_cik(key)

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
                db=request.state.db,
                owner_id=request.state.user.id,
                organization_id=request.state.user.organization_id,
                project_id=deck_record.project_id,
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
async def list_saved_extractions(request: Request):
    """Return all saved extractions, newest first, with per-category claim stats."""
    items = []
    rows = (
        request.state.db.query(SavedExtractionModel)
        .filter(SavedExtractionModel.organization_id == request.state.user.organization_id)
        .order_by(SavedExtractionModel.saved_at.desc())
        .all()
    )
    for row in rows:
        try:
            extraction = json.loads(row.extraction_json)
        except Exception:
            extraction = {}
        claims = extraction.get("claims", [])
        by_category: dict[str, int] = {}
        for cl in claims:
            cat = cl.get("category", "other")
            by_category[cat] = by_category.get(cat, 0) + 1
        metrics = [m.get("metric_name", "") for m in extraction.get("key_metrics", []) if m.get("metric_name")]
        items.append(
            {
                "save_id": row.save_id,
                "company_name": row.company_name or "Unknown",
                "original_filename": row.original_filename or "",
                "extractor_model": row.extractor_model or "",
                "extractor_version": row.extractor_version or "",
                "saved_at": row.saved_at.isoformat(),
                "claims_count": len(claims),
                "claims_by_category": by_category,
                "key_metrics": metrics,
                "stage": extraction.get("stage_assessment", {}).get("stage") if extraction.get("stage_assessment") else None,
                "project_id": row.project_id,
            }
        )
    return JSONResponse(items)


@app.post("/saved-extractions")
async def save_extraction(
    request: Request,
    context_id: Annotated[str, Form()],
    original_filename: Annotated[str, Form()] = "",
    extractor_model: Annotated[str, Form()] = "",
    project_id: Annotated[int | None, Form()] = None,
):
    """Persist a session extraction to the saved library."""
    deck_record = require_context_ownership(request, context_id)
    context_path = CONTEXT_DIR / f"deck_{context_id}.json"
    if not context_path.exists() and not deck_record.extraction_json:
        raise HTTPException(status_code=404, detail=f"Unknown context_id: {context_id}")

    if not context_path.exists() and deck_record.extraction_json:
        context_path.write_text(deck_record.extraction_json)

    extraction_data = json.loads(context_path.read_text())
    company_name = extraction_data.get("company", {}).get("name", "Unknown")
    save_id = uuid.uuid4().hex[:12]
    saved = {
        "meta": {
            "save_id": save_id,
            "company_name": company_name,
            "original_filename": original_filename,
            "extractor_model": extractor_model,
            "extractor_version": EXTRACTOR_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
        "extraction": extraction_data,
    }
    (SAVED_DIR / f"{save_id}.json").write_text(json.dumps(saved, indent=2))

    row = SavedExtractionModel(
        save_id=save_id,
        owner_id=request.state.user.id,
        organization_id=request.state.user.organization_id,
        project_id=project_id or deck_record.project_id,
        company_name=company_name,
        original_filename=original_filename,
        extractor_model=extractor_model,
        extractor_version=EXTRACTOR_VERSION,
        meta_json=json.dumps(saved["meta"]),
        extraction_json=json.dumps(extraction_data),
    )
    request.state.db.add(row)
    request.state.db.commit()
    return JSONResponse({"save_id": save_id, "company_name": company_name})


@app.post("/saved-extractions/{save_id}/load")
async def load_saved_extraction(request: Request, save_id: str):
    """Load a saved extraction back into a fresh session context."""
    saved = require_saved_extraction(request, save_id)
    extraction_data = json.loads(saved.extraction_json)

    token = uuid.uuid4().hex[:12]
    context_path = CONTEXT_DIR / f"deck_{token}.json"
    context_path.write_text(json.dumps(extraction_data, indent=2))

    new_context = DeckContextModel(
        context_id=token,
        owner_id=request.state.user.id,
        organization_id=request.state.user.organization_id,
        project_id=saved.project_id,
        original_filename=saved.original_filename,
        extractor_model=saved.extractor_model,
        extractor_version=saved.extractor_version,
        context_path=str(context_path),
        extraction_json=saved.extraction_json,
        status="saved",
    )
    request.state.db.add(new_context)
    request.state.db.commit()

    return JSONResponse({
        "context_id": token,
        "extraction": extraction_data,
        "meta": json.loads(saved.meta_json or "{}"),
    })


@app.delete("/saved-extractions/{save_id}")
async def delete_saved_extraction(request: Request, save_id: str):
    saved = require_saved_extraction(request, save_id)
    saved_path = SAVED_DIR / f"{save_id}.json"
    if saved_path.exists():
        saved_path.unlink()
    request.state.db.delete(saved)
    request.state.db.commit()
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
    require_context_ownership(request, context_id)
    context_path = CONTEXT_DIR / f"deck_{context_id}.json"
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
        db=request.state.db,
        owner_id=request.state.user.id,
        organization_id=request.state.user.organization_id,
        project_id=require_context_ownership(request, context_id).project_id,
    )
    return report

# ───────────────────────── saved reports ──────────────────────────────

@app.get("/reports")
async def list_reports(request: Request):
    """Return all saved compliance reports, newest first, with per-verdict counts."""
    items = []
    rows = (
        request.state.db.query(ReportModel)
        .filter(ReportModel.organization_id == request.state.user.organization_id)
        .order_by(ReportModel.generated_at.desc())
        .all()
    )
    for row in rows:
        try:
            data = json.loads(row.report_json)
            results = data.get("results", [])
        except Exception:
            data = {}
            results = []

        consistent = sum(1 for r in results if r.get("verdict") == "CONSISTENT")
        contradicts = sum(1 for r in results if r.get("verdict") == "CONTRADICTS")
        unsupported = sum(1 for r in results if r.get("verdict") == "UNSUPPORTED")
        insufficient = sum(1 for r in results if r.get("verdict") == "INSUFFICIENT_EVIDENCE")
        fls_flags = sum(1 for r in results if r.get("verdict") == "CONTRADICTS" and r.get("forward_looking"))
        top_flags = [
            {"claim": r.get("claim", ""), "verdict": r.get("verdict", ""), "forward_looking": r.get("forward_looking", False)}
            for r in results if r.get("verdict") == "CONTRADICTS"
        ][:3]

        items.append({
            "report_id": row.report_id,
            "company_name": row.company_name or "Unknown",
            "generated_at": row.generated_at.isoformat(),
            "assumed_industry": data.get("assumed_industry", ""),
            "cik": row.cik or "",
            "extractor_model": row.extractor_model or "",
            "extractor_version": data.get("extractor_version", ""),
            "analyzer_model": row.analyzer_model or "",
            "analyzer_version": data.get("analyzer_version", ""),
            "claims_analyzed": data.get("claims_analyzed", 0),
            "consistent": consistent,
            "contradicts": contradicts,
            "unsupported": unsupported,
            "insufficient": insufficient,
            "fls_flags": fls_flags,
            "top_flags": top_flags,
            "project_id": row.project_id,
        })
    return JSONResponse(items)

@app.get("/reports/{report_id}")
async def get_report(request: Request, report_id: str):
    """Retrieve a specific saved compliance report."""
    report = require_report(request, report_id)
    try:
        return JSONResponse(json.loads(report.report_json))
    except Exception:
        raise HTTPException(status_code=500, detail="Could not parse saved report data.")

@app.delete("/reports/{report_id}")
async def delete_report(request: Request, report_id: str):
    """Delete a saved compliance report."""
    report = require_report(request, report_id)
    request.state.db.delete(report)
    request.state.db.commit()
    return JSONResponse({"deleted": report_id})

