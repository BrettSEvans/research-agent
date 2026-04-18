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
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

import anthropic
from dotenv import load_dotenv
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from agent import iter_compliance_report, run_compliance_report, SAVED_REPORTS_DIR
from deck_context import DeckContext
from extractor import extract_from_pdf
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


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


ALLOWED_MODELS = {
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    # Inception Labs Mercury models
    "mercury-2",
    "mercury-coder-small",
    # Local (Ollama) models
    "qwen3.5:9b",
}


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    extractor_model: Annotated[str | None, Form()] = None,
):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if extractor_model and extractor_model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {extractor_model}")

    content = await file.read()
    token = uuid.uuid4().hex[:12]
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf", dir=str(UPLOADS), prefix=f"{token}_"
    ) as tmp:
        tmp.write(content)
        pdf_path = Path(tmp.name)

    try:
        client = anthropic.Anthropic()
        from extractor import extract_basics_and_infer_stage
        extraction = extract_basics_and_infer_stage(pdf_path, client=client, model=extractor_model)
    except Exception as exc:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc

    # Do not save DeckContext or unlink the PDF yet; we need it for /extract/deep
    return JSONResponse(
        {
            "context_id": token,
            "extraction": extraction.model_dump(),
            "extractor_version": EXTRACTOR_VERSION,
        }
    )

@app.post("/extract/deep")
async def extract_deep(
    context_id: Annotated[str, Form()],
    extractor_model: Annotated[str | None, Form()] = None,
    startup_stage: Annotated[str | None, Form()] = None,
    modules: Annotated[str | None, Form()] = None,
):
    pdf_paths = list(UPLOADS.glob(f"{context_id}_*.pdf"))
    context_path = CONTEXT_DIR / f"deck_{context_id}.json"

    # No PDF in uploads — this context was loaded from a saved extraction.
    # The context file already exists, so skip re-extraction and return it as-is.
    if not pdf_paths:
        if context_path.exists():
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

    try:
        client = anthropic.Anthropic()
        extraction = extract_from_pdf(
            pdf_path,
            client=client,
            model=extractor_model,
            requested_metrics=requested_metrics,
        )
    except Exception as exc:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Deep extraction failed: {exc}") from exc

    # Save full context and clean up the uploaded PDF
    context = DeckContext(extraction)
    context.save(context_path)
    pdf_path.unlink(missing_ok=True)

    return JSONResponse(
        {
            "context_id": context_id,
            "context_path": str(context_path),
            "extraction": extraction.model_dump(),
            "extractor_version": EXTRACTOR_VERSION,
        }
    )


@app.post("/verify/stream")
def verify_stream(
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
async def list_saved_extractions():
    """Return all saved extractions, newest first."""
    items = []
    for p in sorted(SAVED_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text())
            items.append({
                "save_id": p.stem,
                "company_name": data.get("meta", {}).get("company_name", "Unknown"),
                "original_filename": data.get("meta", {}).get("original_filename", ""),
                "extractor_model": data.get("meta", {}).get("extractor_model", ""),
                "extractor_version": data.get("meta", {}).get("extractor_version", ""),
                "saved_at": data.get("meta", {}).get("saved_at", ""),
            })
        except Exception:
            continue
    return JSONResponse(items)


@app.post("/saved-extractions")
async def save_extraction(
    context_id: Annotated[str, Form()],
    original_filename: Annotated[str, Form()] = "",
    extractor_model: Annotated[str, Form()] = "",
):
    """Persist a session extraction to the saved library."""
    context_path = CONTEXT_DIR / f"deck_{context_id}.json"
    if not context_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown context_id: {context_id}")

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
    return JSONResponse({"save_id": save_id, "company_name": company_name})


@app.post("/saved-extractions/{save_id}/load")
async def load_saved_extraction(save_id: str):
    """Load a saved extraction back into a fresh session context."""
    saved_path = SAVED_DIR / f"{save_id}.json"
    if not saved_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown save_id: {save_id}")

    data = json.loads(saved_path.read_text())
    extraction_data = data["extraction"]

    # Create a fresh session context so /verify/stream works normally
    token = uuid.uuid4().hex[:12]
    context_path = CONTEXT_DIR / f"deck_{token}.json"
    context_path.write_text(json.dumps(extraction_data, indent=2))

    return JSONResponse({
        "context_id": token,
        "extraction": extraction_data,
        "meta": data.get("meta", {}),
    })


@app.delete("/saved-extractions/{save_id}")
async def delete_saved_extraction(save_id: str):
    saved_path = SAVED_DIR / f"{save_id}.json"
    if not saved_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown save_id: {save_id}")
    saved_path.unlink()
    return JSONResponse({"deleted": save_id})


@app.post("/verify")
async def verify(
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
    )
    return report

# ───────────────────────── saved reports ──────────────────────────────

@app.get("/reports")
async def list_reports():
    """Return all saved compliance reports, newest first."""
    items = []
    if not SAVED_REPORTS_DIR.exists():
        return JSONResponse(items)

    for p in sorted(SAVED_REPORTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text())
            
            # Compute compliance metric: Verified / (Verified + Contradicts + Flagged)
            # Ignoring INSUFFICIENT_EVIDENCE to get a clearer picture of actual checks.
            verified = sum(1 for r in data.get("results", []) if r.get("verdict") == "VERIFIED")
            flagged = sum(1 for r in data.get("results", []) if r.get("verdict") in ("CONTRADICTS", "FLAGGED"))
            total_checked = verified + flagged
            
            if total_checked > 0:
                metric_pct = round((verified / total_checked) * 100)
                metric_str = f"{metric_pct}% ({verified}/{total_checked} claims)"
            else:
                metric_str = "N/A"

            items.append({
                "report_id": data.get("report_id", p.stem.replace("report_", "")),
                "company_name": data.get("company_name", "Unknown"),
                "generated_at": data.get("generated_at", ""),
                "analyzer_model": data.get("analyzer_model", ""),
                "compliance_metric": metric_str,
                "claims_analyzed": data.get("claims_analyzed", 0),
            })
        except Exception:
            continue
    return JSONResponse(items)

@app.get("/reports/{report_id}")
async def get_report(report_id: str):
    """Retrieve a specific saved compliance report."""
    report_path = SAVED_REPORTS_DIR / f"report_{report_id}.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown report_id: {report_id}")
    return JSONResponse(json.loads(report_path.read_text()))

@app.delete("/reports/{report_id}")
async def delete_report(report_id: str):
    """Delete a saved compliance report."""
    report_path = SAVED_REPORTS_DIR / f"report_{report_id}.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown report_id: {report_id}")
    report_path.unlink()
    return JSONResponse({"deleted": report_id})

