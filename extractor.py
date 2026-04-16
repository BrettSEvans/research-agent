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
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

MODEL = "claude-opus-4-6"


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
- product / team: only if they include specific verifiable claims"""


def extract_from_pdf(
    pdf_path: str | Path,
    client: anthropic.Anthropic | None = None,
) -> DeckExtraction:
    """Extract structured deck information from a PDF file."""
    client = client or anthropic.Anthropic()
    pdf_bytes = Path(pdf_path).read_bytes()
    b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
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
                            "Include every falsifiable claim with its slide number and verbatim quote."
                        ),
                    },
                ],
            }
        ],
        output_format=DeckExtraction,
    )
    return response.parsed_output
