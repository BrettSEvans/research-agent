"""Upload + extraction routes: /extract, /extract/deep, /extract/complete."""
from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

import anthropic
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from conversion import convert_to_pdf, detect_file_type
from deck_context import DeckContext
from extractor import extract_from_pdf
from version import EXTRACTOR_VERSION

from ._deps import (
    ALLOWED_MODELS,
    CLOUD_MODELS,
    get_api_key,
    get_session_dirs,
)

router = APIRouter()


@router.post("/extract")
async def extract(
    request: Request,
    file: UploadFile = File(...),
    extractor_model: Annotated[str | None, Form()] = None,
):
    try:
        file_type = detect_file_type(file.filename or "")
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported: PDF, PowerPoint (.ppt/.pptx), Word (.doc/.docx), Excel (.xlsx). {str(e)}"
        )

    if extractor_model and extractor_model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {extractor_model}")

    session_uploads, _ = get_session_dirs(request)
    content = await file.read()
    token = uuid.uuid4().hex[:12]

    temp_suffix = f".{file_type}" if file_type != "pdf" else ".pdf"
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=temp_suffix, dir=str(session_uploads), prefix=f"{token}_"
    ) as tmp:
        tmp.write(content)
        temp_path = Path(tmp.name)

    try:
        if file_type != "pdf":
            pdf_path = await convert_to_pdf(temp_path, file_type)
        else:
            pdf_path = session_uploads / f"{token}_{uuid.uuid4().hex[:8]}.pdf"
            temp_path.rename(pdf_path)
    except ValueError as e:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"File conversion failed: {str(e)}")

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


@router.post("/extract/deep")
async def extract_deep(
    request: Request,
    context_id: Annotated[str, Form()],
    extractor_model: Annotated[str | None, Form()] = None,
    startup_stage: Annotated[str | None, Form()] = None,
    modules: Annotated[str | None, Form()] = None,
):
    session_uploads, session_ctx = get_session_dirs(request)
    pdf_paths = list(session_uploads.glob(f"{context_id}_*.pdf"))
    context_path = session_ctx / f"deck_{context_id}.json"

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
    failed_path = session_ctx / f"deck_{context_id}_failed.json"

    try:
        client = anthropic.Anthropic(api_key=get_api_key(request))
        extraction, failed_pages = extract_from_pdf(
            pdf_path,
            client=client,
            model=extractor_model,
            requested_metrics=requested_metrics,
        )
    except Exception as exc:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Deep extraction failed: {exc}") from exc

    context = DeckContext(extraction)
    context.save(context_path)

    if failed_pages:
        failed_path.write_text(json.dumps({"failed_pages": failed_pages}))
    else:
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


@router.post("/extract/complete")
async def extract_complete(
    request: Request,
    context_id: Annotated[str, Form()],
    cloud_model: Annotated[str, Form()] = "claude-haiku-4-5",
    modules: Annotated[str | None, Form()] = None,
):
    """Complete a partial vision extraction by sending failed pages to a cloud model."""
    session_uploads, session_ctx = get_session_dirs(request)
    context_path = session_ctx / f"deck_{context_id}.json"
    failed_path = session_ctx / f"deck_{context_id}_failed.json"

    if not context_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown context_id: {context_id}")
    if not failed_path.exists():
        raise HTTPException(status_code=400, detail="No failed-page sidecar found for this context.")

    pdf_paths = list(session_uploads.glob(f"{context_id}_*.pdf"))
    if not pdf_paths:
        raise HTTPException(
            status_code=404,
            detail="PDF not found — it may have already been deleted. Re-upload the deck and try again.",
        )

    if cloud_model not in CLOUD_MODELS:
        raise HTTPException(status_code=400, detail=f"Model must be a cloud model for completion: {cloud_model}")

    pdf_path = pdf_paths[0]
    failed_data = json.loads(failed_path.read_text())
    failed_pages: list[int] = failed_data.get("failed_pages", [])

    if not failed_pages:
        failed_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Failed-page sidecar is empty — nothing to complete.")

    requested_metrics = modules.split(",") if modules else None

    try:
        client = anthropic.Anthropic(api_key=get_api_key(request))
        from extractor import extract_failed_pages_with_claude, DeckExtraction

        cloud_extraction = extract_failed_pages_with_claude(
            pdf_path=pdf_path,
            page_indices=failed_pages,
            client=client,
            model=cloud_model,
            requested_metrics=requested_metrics,
        )

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

        merged_extraction = DeckExtraction(
            company=ex.company,
            claims=merged_claims,
            fiscal_year_end=ex.fiscal_year_end or cloud_extraction.fiscal_year_end,
            currency=ex.currency or cloud_extraction.currency,
            stage_assessment=ex.stage_assessment or cloud_extraction.stage_assessment,
            key_metrics=merged_metrics,
            extraction_notes=merged_notes,
        )

        merged_context = DeckContext(merged_extraction)
        merged_context.save(context_path)

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
