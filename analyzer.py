"""Claude-powered claim analysis.

Given a claim (typically a forward-looking statement from a pitch deck, data
room, or investor memo) and the top-k most semantically similar passages from
a company's SEC filings, Claude classifies the relationship as CONTRADICTS,
UNSUPPORTED, CONSISTENT, or INSUFFICIENT_EVIDENCE and identifies whether the
claim is forward-looking.
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

Your job: determine whether the claim is contradicted, unsupported, or consistent
with the company's regulatory filings. Be rigorous and skeptical. A claim is
CONTRADICTS only if a filing passage materially conflicts with it. It is
UNSUPPORTED if filings don't address it. CONSISTENT requires explicit corroboration.

Also flag whether the claim is a "forward-looking statement" as defined by the
Private Securities Litigation Reform Act of 1995 — projections, expectations,
plans, or anticipated performance. These are the highest-risk claims.

Always cite passage numbers (e.g., "P2") you relied on. Do not fabricate."""


class ClaimAssessment(BaseModel):
    verdict: str = Field(
        description="One of: CONTRADICTS, UNSUPPORTED, CONSISTENT, INSUFFICIENT_EVIDENCE"
    )
    forward_looking: bool = Field(
        description="Is the claim a forward-looking statement (projection, expectation, plan)?"
    )
    severity: str = Field(description="One of: HIGH, MEDIUM, LOW, NONE")
    explanation: str = Field(
        description="Brief reasoning (2-4 sentences) citing passage numbers like P1, P2."
    )
    cited_passages: list[int] = Field(
        description="1-indexed passage numbers the assessment relies on."
    )


def _format_passages(hits: list[Hit]) -> str:
    lines = []
    for i, hit in enumerate(hits, start=1):
        f = hit.passage.filing
        header = f"[P{i}] {f.form} filed {f.filing_date} (accession {f.accession}) — score={hit.score:.3f}"
        lines.append(f"{header}\n{hit.passage.text}")
    return "\n\n---\n\n".join(lines)


def analyze_claim(
    client: anthropic.Anthropic, claim: str, hits: list[Hit]
) -> ClaimAssessment:
    user_content = (
        f"CLAIM:\n{claim}\n\n"
        f"PASSAGES FROM SEC FILINGS (ranked by semantic similarity):\n\n"
        f"{_format_passages(hits)}\n\n"
        "Assess the claim against these passages."
    )

    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        output_format=ClaimAssessment,
    )
    return response.parsed_output
