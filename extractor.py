"""Pitch deck extractor.

Parses a PDF pitch deck using Claude's native PDF support and returns
structured information intended as input to the compliance agent.

Design constraints (explicit, not optional):
- EXTRACT ONLY. The extractor does not analyze, compare, or judge claims.
- VERBATIM. Claims are captured as exact quotes from the deck where possible.
- NO FABRICATION. If a field is not present in the deck, it is null and the
  absence is recorded in `extraction_notes`.
- MINIMAL KNOWLEDGE OF THE CONSUMER. The extractor knows the shape of data the
  compliance agent needs (company identity, falsifiable claims, fiscal context)
  but does not know how claims will be verified.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

import llm_local
import llm_inception
from stage import StageAssessment, FundingStage

# Default: Haiku 4.5. Extraction is a structured, mechanical task — cheap model
# is the right fit. Override with EXTRACTOR_MODEL env var if needed.
DEFAULT_MODEL = "claude-haiku-4-5"


def _resolve_model() -> str:
    return os.environ.get("EXTRACTOR_MODEL", DEFAULT_MODEL)


def _thinking_kwargs(model: str, budget_tokens: int = 3000) -> dict:
    """Return extended-thinking kwargs for extraction.

    Extraction is a structured, mechanical task — it does not benefit from
    deep reasoning. Adaptive thinking routinely burns 10k+ tokens on a full
    deck and is the primary cause of extraction hangs on Series A/B (4 metrics
    → model reasons extensively before writing output).

    Cap at 3000 tokens: enough to resolve ambiguous slides but fast enough to
    return promptly. Haiku and non-Claude models don't support thinking at all.
    """
    if "haiku" in model or llm_local.is_local_model(model) or llm_inception.is_inception_model(model):
        return {}
    return {"thinking": {"type": "enabled", "budget_tokens": budget_tokens}}


class CompanyIdentity(BaseModel):
    name: str = Field(description="Official company name exactly as stated in the deck.")
    ticker: str | None = Field(
        default=None,
        description="Public ticker if explicitly stated in the deck. Null otherwise. NEVER guess.",
    )
    cik: str | None = Field(
        default=None,
        description="SEC CIK if explicitly stated. Null otherwise. NEVER guess.",
    )
    website: str | None = Field(default=None, description="Website URL if shown in the deck.")
    description: str | None = Field(
        default=None,
        description="One-sentence company description, verbatim from the deck if available.",
    )
    industry: str | None = Field(
        default=None,
        description=(
            "Industry/sector the company operates in (e.g., 'B2B SaaS for legal tech', "
            "'fintech lending', 'biotech oncology'). Prefer terminology used in the deck. "
            "If no industry is stated directly, infer conservatively from the product "
            "description and market claims. Null only if the deck provides no basis."
        ),
    )
    founders: list[str] = Field(
        default_factory=list,
        description="Names of the founders."
    )


class ExtractedClaim(BaseModel):
    text: str = Field(
        description="Clear restatement of the claim as a single falsifiable sentence."
    )
    verbatim: str = Field(
        description="Exact quote from the deck supporting this claim. Required."
    )
    slide: int = Field(description="1-indexed slide/page number where the claim appears.")
    category: Literal[
        "financial", "market", "product", "team", "traction", "projection", "regulatory", "other"
    ]
    likely_forward_looking: bool = Field(
        description=(
            "True if the claim is a projection, expectation, plan, or target. "
            "False for historical facts."
        )
    )


class ExtractedMetric(BaseModel):
    metric_name: str = Field(description="e.g., 'ARR', 'CAC', 'LTV', 'NRR', 'EBITDA', 'Waitlist Signups', 'Founder Domain Expertise'")
    value: str = Field(description="The extracted value, e.g., '$5M', '$100', '120%', 'Ex-Googler'")

class DeckExtraction(BaseModel):
    company: CompanyIdentity
    claims: list[ExtractedClaim] = Field(
        description=(
            "Only specific, falsifiable claims. Skip vague marketing language. "
            "Prioritize numbers, timelines, market sizes, capabilities, and regulatory statements."
        )
    )
    fiscal_year_end: str | None = Field(
        default=None,
        description="E.g., 'December 31' — only if stated or clearly implied by dated figures.",
    )
    currency: str | None = Field(
        default=None,
        description="Primary currency for financial figures (e.g., 'USD') if inferable from the deck.",
    )
    # --- Stage Inference ---
    stage_assessment: StageAssessment | None = Field(default=None, description="Inferred stage based on the deck.")

    # --- Stage-Specific Extracted Fields ---
    key_metrics: list[ExtractedMetric] = Field(
        default_factory=list, 
        description="Extract key startup metrics based on the inferred stage (e.g. ARR, CAC, LTV, NRR, Gross Margin, EBITDA, Rule of 40, Exit Strategy, Early Validation Signals)."
    )

    extraction_notes: str = Field(
        description=(
            "Honest commentary on what was and was NOT found. Call out ambiguity, "
            "missing fields (e.g., 'no ticker stated — company appears to be private'), "
            "and any areas where the deck is vague. This is read by the compliance agent."
        )
    )


SYSTEM_PROMPT = """You extract structured information from startup pitch decks for a venture capital compliance pipeline.

ABSOLUTE RULES:
1. EXTRACT ONLY. You do not analyze, judge, or compare claims. You are not the compliance agent.
2. VERBATIM. Every claim must include an exact quote (`verbatim` field) from the deck.
3. NO FABRICATION. If a field is not in the deck, it is null. NEVER guess tickers, CIKs, or numbers.
4. FALSIFIABLE ONLY. A claim must be specific and checkable. Skip vague statements like "we're disrupting the industry" — include "$12M ARR as of Q2 2025".
5. FORWARD-LOOKING FLAG. True only for projections/expectations/targets (e.g., "will reach $100M by 2027"). False for stated historical facts.
6. TRANSPARENCY. In extraction_notes, list every notable field you could NOT extract and any ambiguity you encountered. The downstream compliance agent depends on this to decide when to flag INSUFFICIENT_EVIDENCE.

Prioritize these claim categories because the compliance agent cross-references them against SEC filings:
- financial: revenue, ARR, margins, burn, runway (state the period)
- market: TAM/SAM/SOM, growth rates, segment sizing
- traction: customer counts, retention, LTV/CAC
- projection: future revenue, growth, unit economics, timelines
- regulatory: compliance status, licenses, pending filings
- product / team: only if they include specific verifiable claims

INDUSTRY FIELD:
The compliance agent uses the `industry` field when the company has no SEC CIK
(early-stage startups). With only a company name and no filings, it needs an
industry label to scope web-based verification of market/TAM claims. Fill this
field whenever the deck gives you enough basis — even if industry is not stated
verbatim, conservative inference from the product description and market
claims is appropriate (e.g., a deck about "AI-powered legal contract review"
→ industry "B2B SaaS for legal tech"). Leave null only when the deck provides
truly no basis.

You will also infer the startup's funding stage (Pre-Seed, Seed, Series A, Series B, Series C+) 
and extract specific metrics if present (e.g. LTV, CAC, NRR, EBITDA, exit strategy). 
Make sure you include the `stage_assessment` based on raise amounts, ARR, and traction metrics."""
class BasicExtraction(BaseModel):
    company: CompanyIdentity
    stage_assessment: StageAssessment | None = Field(default=None, description="Inferred stage based on the deck.")


def extract_basics_and_infer_stage(
    pdf_path: str | Path,
    client: anthropic.Anthropic | None = None,
    model: str | None = None,
) -> BasicExtraction:
    """Pass 1: Extract only the company identity and infer the funding stage."""
    model = model or _resolve_model()

    # Vision-capable local models: render first 3 pages as images.
    # Company name, website, and stage signals are almost always on slides 1–3.
    # Check this BEFORE is_local_model() — vision models satisfy both predicates.
    if llm_local.is_vision_local_model(model):
        images = llm_local.pdf_to_base64_images(pdf_path, max_pages=3)
        basic_system = (
            "You extract company identity information and infer the funding stage "
            "from startup pitch deck slides.\n\n"
            "RULES:\n"
            "- Extract the company NAME exactly as written — never paraphrase or guess.\n"
            "- TICKER and CIK: only if explicitly stated. NEVER guess.\n"
            "- WEBSITE: extract the URL if it appears anywhere.\n"
            "- INDUSTRY: infer conservatively from the product and market claims if not stated.\n"
            "- FOUNDERS: list all founder names shown on team slides.\n"
            "- STAGE: infer from raise amount, ARR, prior rounds, and traction signals.\n"
            "- Do NOT extract detailed claims — that happens in the deep extraction pass."
        )
        return _vision_call_with_retry(
            model=model,
            system=basic_system,
            user_text="Extract the company identity fields and infer the funding stage from these slides.",
            images_b64=images,
            output_format=BasicExtraction,
            label="basic extraction",
        )

    # Text-only local (Ollama) and Inception models — extract text with pypdf first.
    if llm_local.is_local_model(model) or llm_inception.is_inception_model(model):
        page_text = llm_local.extract_pdf_text(pdf_path)
        caller = llm_inception.call_structured if llm_inception.is_inception_model(model) else llm_local.call_structured
        return caller(
            model=model,
            system="Extract the basic company identity and infer the startup's funding stage from the deck.",
            user_content=f"Raw text of the pitch deck:\n\n{page_text}",
            output_format=BasicExtraction,
        )

    client = client or anthropic.Anthropic()
    pdf_bytes = Path(pdf_path).read_bytes()
    b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    basic_system = (
        "You extract company identity information and infer the funding stage from a startup pitch deck.\n\n"
        "RULES:\n"
        "- Extract the company NAME exactly as written — never paraphrase or guess.\n"
        "- TICKER and CIK: only if explicitly stated. NEVER guess. Leave null for private companies.\n"
        "- WEBSITE: extract the URL if it appears anywhere in the deck.\n"
        "- DESCRIPTION: one sentence verbatim from the deck if available, else a conservative summary.\n"
        "- INDUSTRY: required. Even if not stated directly, infer conservatively from the product and market "
        "claims (e.g. 'AI-powered contract review' → 'B2B SaaS for legal tech'). Leave null only if the deck "
        "gives truly no basis.\n"
        "- FOUNDERS: list all founder names shown on team slides or bios.\n"
        "- STAGE: infer from raise amount, ARR, prior rounds, and traction signals. "
        "Set confidence low if signals are ambiguous.\n"
        "- Do NOT extract detailed claims — that happens in the deep extraction pass."
    )

    response = client.messages.parse(
        model=model,
        max_tokens=4000,
        system=basic_system,
        **_thinking_kwargs(model),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extract the company identity fields and infer the funding stage.",
                    },
                ],
            }
        ],
        output_format=BasicExtraction,
    )
    return response.parsed_output


def _vision_call_with_retry(
    *,
    model: str,
    system: str,
    user_text: str,
    images_b64: list[str],
    output_format,
    label: str = "",
) -> object:
    """Call call_structured_vision, halving the image list on each 500 OOM.

    If the model OOMs on N images it retries with N//2, N//4, … down to 1.
    A 500 on a single image is a hard failure (model/hardware issue).
    """
    imgs = images_b64
    while imgs:
        try:
            return llm_local.call_structured_vision(
                model=model,
                system=system,
                user_text=user_text,
                images_b64=imgs,
                output_format=output_format,
            )
        except RuntimeError as exc:
            if "ollama-vision-500" not in str(exc) or len(imgs) <= 1:
                raise
            half = len(imgs) // 2
            print(
                f"[ollama-vision] ⚠ OOM{' on ' + label if label else ''} "
                f"with {len(imgs)} images → retrying with {half}"
            )
            imgs = imgs[:half]
    raise RuntimeError("Vision extraction failed: no images to send")


def _merge_deck_extractions(
    batches: list[DeckExtraction],
    batch_size: int,
) -> DeckExtraction:
    """Merge partial DeckExtraction results from batched vision inference.

    Company identity and stage come from the first batch (usually the cover /
    intro slides). Claims and key metrics are concatenated across all batches;
    extraction notes are joined with a per-batch header.
    """
    if len(batches) == 1:
        return batches[0]

    all_claims = []
    for batch_idx, ex in enumerate(batches):
        all_claims.extend(ex.claims)

    seen_metric_names: set[str] = set()
    all_metrics: list[ExtractedMetric] = []
    for ex in batches:
        for m in ex.key_metrics:
            if m.metric_name not in seen_metric_names:
                seen_metric_names.add(m.metric_name)
                all_metrics.append(m)

    notes_parts = []
    for i, ex in enumerate(batches):
        if ex.extraction_notes:
            # Derive header from actual claim slide numbers rather than batch_size
            # math, which is wrong for the final (short) batch.
            slides = sorted({cl.slide for cl in ex.claims}) if ex.claims else []
            if slides:
                header = f"[Slides {slides[0]}–{slides[-1]}]"
            else:
                header = f"[Batch {i + 1}]"
            notes_parts.append(f"{header} {ex.extraction_notes}")
    combined_notes = (
        "\n\n".join(notes_parts) if notes_parts
        else "Batched extraction — no per-batch notes available."
    )

    return DeckExtraction(
        company=batches[0].company,
        claims=all_claims,
        fiscal_year_end=next((e.fiscal_year_end for e in batches if e.fiscal_year_end), None),
        currency=next((e.currency for e in batches if e.currency), None),
        stage_assessment=next((e.stage_assessment for e in batches if e.stage_assessment), None),
        key_metrics=all_metrics,
        extraction_notes=combined_notes,
    )


def extract_from_pdf(
    pdf_path: str | Path,
    client: anthropic.Anthropic | None = None,
    model: str | None = None,
    requested_metrics: list[str] | None = None,
) -> tuple[DeckExtraction, list[int]]:
    """Extract structured deck information from a PDF file.

    Returns a ``(DeckExtraction, failed_pages)`` tuple. ``failed_pages`` is a
    list of 1-indexed slide numbers that the local vision model could not
    process (OOM on a single-image batch). These pages should be sent to a
    cloud model via ``extract_failed_pages_with_claude()``.

    Routes to Ollama for local (qwen/llama/...) models — which are text-only,
    so page text is extracted with pypdf first. Claude models receive the
    native PDF document block. Vision-capable local models receive rendered
    page images, batched to avoid Ollama OOM errors.
    """
    model = model or _resolve_model()

    metrics_str = "No specific metrics requested."
    if requested_metrics:
        metrics_str = "Please extract the following specific metrics if they exist in the deck: " + ", ".join(requested_metrics)

    # --- Vision-capable local models: render pages as images, batch to avoid OOM ---
    # Check BEFORE is_local_model() — vision models satisfy both predicates.
    if llm_local.is_vision_local_model(model):
        all_images = llm_local.pdf_to_base64_images(pdf_path)
        total = len(all_images)
        initial_batch = llm_local.OLLAMA_VISION_BATCH

        # Queue of (start_idx, images_slice) — auto-halved on 500 OOM.
        # start_idx is 0-based; slide numbers in prompts are 1-based.
        queue: list[tuple[int, list[str]]] = [
            (i, all_images[i : i + initial_batch])
            for i in range(0, total, initial_batch)
        ]
        print(f"[ollama-vision] {total} slide(s) → {len(queue)} batch(es) of ≤{initial_batch}")

        results: list[DeckExtraction] = []
        failed_pages: list[int] = []

        while queue:
            start_idx, batch_imgs = queue.pop(0)
            start_slide = start_idx + 1
            end_slide = start_idx + len(batch_imgs)

            user_text = (
                f"These are slides {start_slide}–{end_slide} of a {total}-slide "
                "startup pitch deck. Extract structured pitch deck information. "
                f"For each claim, set `slide` to the absolute slide number "
                f"({start_slide}–{end_slide}). Every claim must include a `verbatim` "
                f"field with the exact quote from the deck. {metrics_str}"
            )
            try:
                # Call directly — the queue itself handles halving on OOM.
                # Using _vision_call_with_retry here would double-halve, wasting
                # Ollama calls on pages that will ultimately land in failed_pages.
                result = llm_local.call_structured_vision(
                    model=model,
                    system=SYSTEM_PROMPT,
                    user_text=user_text,
                    images_b64=batch_imgs,
                    output_format=DeckExtraction,
                )
                results.append(result)
            except RuntimeError as exc:
                if "ollama-vision-500" not in str(exc):
                    raise
                if len(batch_imgs) <= 1:
                    # Single page OOM — skip gracefully; will be completed by cloud model
                    failed_pages.append(start_slide)
                    print(
                        f"[ollama-vision] ⚠ Slide {start_slide} OOM on single page "
                        "— skipped for cloud LLM fallback"
                    )
                else:
                    # Multi-page OOM — split and re-queue as halves
                    mid = len(batch_imgs) // 2
                    queue.insert(0, (start_idx + mid, batch_imgs[mid:]))
                    queue.insert(0, (start_idx, batch_imgs[:mid]))

        if not results:
            raise RuntimeError(
                f"Vision extraction failed: all {total} pages exceeded local model capacity. "
                "Please use a Claude model (Haiku or Sonnet) which can read image-based PDFs natively."
            )

        if failed_pages:
            pages_str = ", ".join(str(p) for p in failed_pages)
            print(
                f"[ollama-vision] ✓ Extraction complete — {len(failed_pages)} page(s) "
                f"skipped for cloud fallback: slides {pages_str}"
            )

        return _merge_deck_extractions(results, initial_batch), failed_pages

    # --- Text-only local (Ollama) and Inception: pre-extract text with pypdf ---
    if llm_local.is_local_model(model) or llm_inception.is_inception_model(model):
        page_text = llm_local.extract_pdf_text(pdf_path)
        user_content = (
            "The following is the raw text of a startup pitch deck, split by "
            "slide. Extract structured pitch deck information. For each claim, "
            "set `slide` to the 1-indexed slide number where it appears. "
            "Every claim must include a `verbatim` field with the exact quote "
            f"from the deck. {metrics_str}\n\n"
            f"{page_text}"
        )
        caller = llm_inception.call_structured if llm_inception.is_inception_model(model) else llm_local.call_structured
        return caller(
            model=model,
            system=SYSTEM_PROMPT,
            user_content=user_content,
            output_format=DeckExtraction,
        ), []

    # --- Anthropic path: native PDF support ---
    client = client or anthropic.Anthropic()
    pdf_bytes = Path(pdf_path).read_bytes()
    b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    response = client.messages.parse(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        **_thinking_kwargs(model),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract the structured pitch deck information. "
                            "Include every falsifiable claim with its slide number and verbatim quote. "
                            f"{metrics_str}"
                        ),
                    },
                ],
            }
        ],
        output_format=DeckExtraction,
    )
    return response.parsed_output, []


def extract_failed_pages_with_claude(
    pdf_path: str | Path,
    page_indices: list[int],
    client: anthropic.Anthropic | None = None,
    model: str = "claude-haiku-4-5",
    requested_metrics: list[str] | None = None,
) -> DeckExtraction:
    """Extract claims from specific pages using Claude's native PDF support.

    Used to complete a partial vision extraction where some pages could not be
    processed by the local model (OOM). Claude receives the full PDF but is
    instructed to extract claims only from the listed pages.

    Args:
        pdf_path: Path to the PDF file (must still be on disk).
        page_indices: 1-indexed slide numbers to extract from.
        client: Anthropic client (created if None).
        model: Claude model to use (Haiku recommended for cost).
        requested_metrics: Optional list of specific metric names to extract.

    Returns:
        DeckExtraction with claims scoped to the requested pages.
    """
    client = client or anthropic.Anthropic()
    pdf_bytes = Path(pdf_path).read_bytes()
    b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    metrics_str = "No specific metrics requested."
    if requested_metrics:
        metrics_str = (
            "Please extract the following specific metrics if present: "
            + ", ".join(requested_metrics)
        )

    pages_str = ", ".join(str(p) for p in sorted(page_indices))

    response = client.messages.parse(
        model=model,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        **_thinking_kwargs(model),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Extract structured pitch deck information from pages {pages_str} ONLY. "
                            "These specific pages could not be processed by the local vision model "
                            "and need cloud-based extraction. "
                            "For each claim, include the correct absolute slide number and a verbatim quote. "
                            f"{metrics_str}"
                        ),
                    },
                ],
            }
        ],
        output_format=DeckExtraction,
    )
    return response.parsed_output
