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
from stage import StageAssessment, FundingStage

# Default: Haiku 4.5. Extraction is a structured, mechanical task — cheap model
# is the right fit. Override with EXTRACTOR_MODEL env var if needed.
DEFAULT_MODEL = "claude-haiku-4-5"


def _resolve_model() -> str:
    return os.environ.get("EXTRACTOR_MODEL", DEFAULT_MODEL)


def _thinking_kwargs(model: str) -> dict:
    """Adaptive thinking is supported on Opus 4.6 and Sonnet 4.6, not Haiku 4.5."""
    if "haiku" in model:
        return {}
    return {"thinking": {"type": "adaptive"}}


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

    if llm_local.is_local_model(model):
        page_text = llm_local.extract_pdf_text(pdf_path)
        return llm_local.call_structured(
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


def extract_from_pdf(
    pdf_path: str | Path,
    client: anthropic.Anthropic | None = None,
    model: str | None = None,
    requested_metrics: list[str] | None = None,
) -> DeckExtraction:
    """Extract structured deck information from a PDF file.

    Routes to Ollama for local (qwen/llama/...) models — which are text-only,
    so page text is extracted with pypdf first. Claude models receive the
    native PDF document block.
    """
    model = model or _resolve_model()
    
    metrics_str = "No specific metrics requested."
    if requested_metrics:
        metrics_str = "Please extract the following specific metrics if they exist in the deck: " + ", ".join(requested_metrics)

    # --- Local model path: no vision; use pypdf to pre-extract text ---
    if llm_local.is_local_model(model):
        page_text = llm_local.extract_pdf_text(pdf_path)
        user_content = (
            "The following is the raw text of a startup pitch deck, split by "
            "slide. Extract structured pitch deck information. For each claim, "
            "set `slide` to the 1-indexed slide number where it appears. "
            "Every claim must include a `verbatim` field with the exact quote "
            f"from the deck. {metrics_str}\n\n"
            f"{page_text}"
        )
        return llm_local.call_structured(
            model=model,
            system=SYSTEM_PROMPT,
            user_content=user_content,
            output_format=DeckExtraction,
        )

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
    return response.parsed_output
