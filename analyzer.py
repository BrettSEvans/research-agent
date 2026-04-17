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

import os

from pydantic import BaseModel, Field
import anthropic

import llm_local
from retriever import Hit

# Default: Sonnet 4.6. Analysis is a reasoning task and benefits from adaptive
# thinking. Override with ANALYZER_MODEL env var.
DEFAULT_MODEL = "claude-sonnet-4-6"


def _resolve_model() -> str:
    return os.environ.get("ANALYZER_MODEL", DEFAULT_MODEL)


def _thinking_kwargs(model: str) -> dict:
    if "haiku" in model:
        return {}
    return {"thinking": {"type": "adaptive"}}

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


INDUSTRY_SYSTEM_PROMPT = """You are a market research analyst assisting a venture capital firm.

You receive a CLAIM about an industry, market size, TAM/SAM/SOM, growth rate, or competitive landscape — NOT a claim about a specific company's financial performance.

Use the web_search tool to find authoritative sources (industry reports, government data, reputable publications like Statista, IBISWorld, Gartner, McKinsey, or recent public filings from comparable companies) and assess whether the claim is supported.

Verdicts:
- CONTRADICTS: web sources materially conflict with the claim.
- UNSUPPORTED: the claim is specific and checkable, but no authoritative web sources corroborate it.
- CONSISTENT: web sources explicitly corroborate the claim (include specific figures/dates in the explanation).
- INSUFFICIENT_EVIDENCE: web search did not return enough information to decide, OR the claim is not actually an industry/market claim and requires company-specific data you do not have.

ABSOLUTE RULES:
- NEVER fabricate statistics, figures, or sources. Every cited number/claim in `explanation` must come from a specific web search result with its URL or publication name named inline.
- If the claim is actually company-specific (revenue, projections, traction for *this* company) and not industry-level, return INSUFFICIENT_EVIDENCE and explain that SEC filings or company disclosures would be required.
- In `missing_information`, name the additional primary source that would be needed (if any).
- `cited_passages` must be empty — you are not working with numbered passages. Use `explanation` to reference URLs/publications inline."""


def analyze_industry_claim(
    client: anthropic.Anthropic,
    claim: str,
    *,
    company_name: str,
    industry: str | None,
    model: str | None = None,
) -> ClaimAssessment:
    """Verify an industry/market claim using Claude's web_search tool.

    Used when the target company has no resolvable SEC CIK (early-stage
    startups). Grounded in real URLs from web search — no fabrication.
    """
    model = model or _resolve_model()

    # Local models have no server-side web_search tool. Refuse cleanly rather
    # than fabricate — no URLs, no statistics we cannot cite.
    if llm_local.is_local_model(model):
        return ClaimAssessment(
            verdict="INSUFFICIENT_EVIDENCE",
            forward_looking=False,
            severity="NONE",
            explanation=(
                f"Local model '{model}' has no web-search capability, so this "
                "industry/TAM claim cannot be verified against authoritative "
                "online sources without fabrication. Re-run this claim with a "
                "Claude model (which has the web_search tool) or supply an "
                "industry report directly."
            ),
            cited_passages=[],
            missing_information=(
                "Authoritative industry/market research (Statista, IBISWorld, "
                "Gartner, government data, or analyst reports) — or re-run "
                "with a Claude model that supports web_search."
            ),
        )

    context_lines = [f"Company: {company_name}"]
    if industry:
        context_lines.append(f"Assumed industry: {industry}")
    else:
        context_lines.append(
            "Assumed industry: NOT STATED — infer conservatively from the claim itself."
        )
    context = "\n".join(context_lines)

    user_content = (
        f"{context}\n\n"
        f"CLAIM:\n{claim}\n\n"
        "Use web search to find authoritative data supporting or contradicting this "
        "claim, then assess. Cite URLs or publication names inline in your explanation."
    )

    response = client.messages.parse(
        model=model,
        max_tokens=16000,
        system=INDUSTRY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        tools=[{"type": "web_search_20260209", "name": "web_search"}],
        output_format=ClaimAssessment,
        **_thinking_kwargs(model),
    )
    return response.parsed_output


def analyze_claim(
    client: anthropic.Anthropic,
    claim: str,
    hits: list[Hit],
    deck_context: str | None = None,
    model: str | None = None,
) -> ClaimAssessment:
    model = model or _resolve_model()
    sections = [f"CLAIM:\n{claim}"]
    if deck_context:
        sections.append(f"DECK CONTEXT (clarifying metadata only):\n{deck_context}")
    sections.append(
        "PASSAGES FROM SEC FILINGS (ranked by semantic similarity):\n\n"
        f"{_format_passages(hits)}"
    )
    sections.append("Assess the claim against these passages.")
    user_content = "\n\n".join(sections)

    # Local Ollama path — same prompt, same Pydantic schema, different backend.
    if llm_local.is_local_model(model):
        return llm_local.call_structured(
            model=model,
            system=SYSTEM_PROMPT,
            user_content=user_content,
            output_format=ClaimAssessment,
        )

    response = client.messages.parse(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        **_thinking_kwargs(model),
        messages=[{"role": "user", "content": user_content}],
        output_format=ClaimAssessment,
    )
    return response.parsed_output
