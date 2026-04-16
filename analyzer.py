"""Claude-powered claim analysis.

Given a claim (typically a forward-looking statement from a pitch deck, data
room, or investor memo) and the top-k most semantically similar passages from
a company's SEC filings, Claude classifies the relationship as CONTRADICTS,
UNSUPPORTED, CONSISTENT, or INSUFFICIENT_EVIDENCE and identifies whether the
claim is forward-looking.

Optional deck context is passed through as *clarifying* metadata only. SEC
filings remain the sole source of truth for verification.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
import anthropic

from retriever import Hit

MODEL = "claude-opus-4-6"

SYSTEM_PROMPT = """You are a securities-compliance analyst assisting a venture capital firm.

You receive:
1. A CLAIM a startup has made to investors (often forward-looking — projections, expected revenue, market size, timelines).
2. PASSAGES retrieved from that company's SEC filings via dense vector search.
3. Optionally, DECK CONTEXT — metadata from the pitch deck that made the claim. Use this ONLY to disambiguate details (e.g., which fiscal year the claim refers to, whether the company is public). DO NOT use the deck to verify the claim. The deck is what is being checked, not the evidence.

Verdicts:
- CONTRADICTS: a filing passage materially conflicts with the claim.
- UNSUPPORTED: the claim is specific and verifiable, but filings don't address it.
- CONSISTENT: a filing passage explicitly corroborates the claim.
- INSUFFICIENT_EVIDENCE: the retrieved passages don't contain enough information to decide, OR the claim itself is too vague to evaluate against filings.

Also flag whether the claim is a "forward-looking statement" as defined by the Private Securities Litigation Reform Act of 1995 — projections, expectations, plans, or anticipated performance. These are the highest-risk claims.

ABSOLUTE RULES:
- NEVER fabricate evidence or citations. If retrieved passages do not contain the information needed, return INSUFFICIENT_EVIDENCE and say so in the explanation.
- NEVER use the deck context as evidence for or against the claim — it is only for disambiguation.
- ALWAYS cite passage numbers (e.g., "P2") you relied on. `cited_passages` may be empty for UNSUPPORTED or INSUFFICIENT_EVIDENCE.
- If the explanation requires data you do not have, state explicitly what additional filing or disclosure would be needed."""


class ClaimAssessment(BaseModel):
    verdict: str = Field(
        description="One of: CONTRADICTS, UNSUPPORTED, CONSISTENT, INSUFFICIENT_EVIDENCE"
    )
    forward_looking: bool = Field(
        description="Is the claim a forward-looking statement (projection, expectation, plan)?"
    )
    severity: str = Field(description="One of: HIGH, MEDIUM, LOW, NONE")
    explanation: str = Field(
        description=(
            "Brief reasoning (2-4 sentences) citing passage numbers like P1, P2. "
            "For INSUFFICIENT_EVIDENCE, explicitly state what additional information "
            "(e.g., which filing type, which disclosure) would be needed to decide."
        )
    )
    cited_passages: list[int] = Field(
        description="1-indexed passage numbers the assessment relies on. May be empty."
    )
    missing_information: str | None = Field(
        default=None,
        description=(
            "When the verdict is UNSUPPORTED or INSUFFICIENT_EVIDENCE, describe the "
            "specific missing information (e.g., 'no disclosure of Q3 2026 projected "
            "Services revenue in reviewed 10-Q filings'). Null for CONTRADICTS/CONSISTENT."
        ),
    )


def _format_passages(hits: list[Hit]) -> str:
    lines = []
    for i, hit in enumerate(hits, start=1):
        f = hit.passage.filing
        header = f"[P{i}] {f.form} filed {f.filing_date} (accession {f.accession}) — score={hit.score:.3f}"
        lines.append(f"{header}\n{hit.passage.text}")
    return "\n\n---\n\n".join(lines)


def analyze_claim(
    client: anthropic.Anthropic,
    claim: str,
    hits: list[Hit],
    deck_context: str | None = None,
) -> ClaimAssessment:
    sections = [f"CLAIM:\n{claim}"]
    if deck_context:
        sections.append(f"DECK CONTEXT (clarifying metadata only):\n{deck_context}")
    sections.append(
        "PASSAGES FROM SEC FILINGS (ranked by semantic similarity):\n\n"
        f"{_format_passages(hits)}"
    )
    sections.append("Assess the claim against these passages.")
    user_content = "\n\n".join(sections)

    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        output_format=ClaimAssessment,
    )
    return response.parsed_output
