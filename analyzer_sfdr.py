"""EU SFDR/CSRD Compliance Analyzer

Assesses pitch deck claims against EU sustainable finance regulations:
- SFDR (EU 2019/2088) — Sustainable Finance Disclosure Regulation
- CSRD (EU 2022/2464) — Corporate Sustainability Reporting Directive
- EU AI Act (EU 2024/1689) — Artificial Intelligence Act
- ESRS (EU 2023/2772) — European Sustainability Reporting Standards

Implements the AnalyzerModule protocol with ClaimAssessment output.
"""

from __future__ import annotations

import anthropic
from pydantic import BaseModel, Field

from analyzer_protocol import ClaimAssessment, AnalyzerModule
from retriever import Hit
import llm_local
import llm_inception
from extractor import EsgMetrics


# Default model
DEFAULT_MODEL = "claude-sonnet-4-20250514"


def _resolve_model() -> str:
    import os
    return os.environ.get("ANALYZER_MODEL", DEFAULT_MODEL)


def _thinking_kwargs(model: str, budget_tokens: int = 4000) -> dict:
    """Return extended-thinking kwargs if supported by model."""
    if "haiku" in model or llm_local.is_local_model(model) or llm_inception.is_inception_model(model):
        return {}
    return {"thinking": {"type": "enabled", "budget_tokens": budget_tokens}}


def _call_anthropic_with_retry(fn, *args, **kwargs):
    """Call Anthropic API with Tenacity retry on rate limits."""
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception(lambda e: isinstance(e, anthropic.RateLimitError)),
        reraise=True
    )
    def _call():
        return fn(*args, **kwargs)

    return _call()


def _format_passages(hits: list[Hit]) -> str:
    """Format retrieval results for the prompt."""
    if not hits:
        return "[No relevant EU regulatory passages found]"

    passages = []
    for i, hit in enumerate(hits, 1):
        source_info = "EU Regulatory Text"
        if hit.passage.filing:
            source_info = f"{hit.passage.filing.title} (score={hit.score:.3f})"
        passages.append(f"[P{i}] {source_info}\n{hit.passage.text}")

    return "\n\n---\n\n".join(passages)


SYSTEM_PROMPT_SFDR = """You are a Venture Capital Compliance Auditor specializing in EU sustainable finance law.

You audit pitch deck claims against:
- SFDR (EU 2019/2088) — Sustainable Finance Disclosure Regulation
- CSRD (EU 2022/2464) — Corporate Sustainability Reporting Directive
- EU AI Act (EU 2024/1689) — Artificial Intelligence Act
- ESRS (EU 2023/2772) — European Sustainability Reporting Standards

Task: Given a claim from a startup pitch deck and the most relevant passages from EU regulatory text,
classify the claim as:
  CONSISTENT          — claim is supported by and aligns with regulatory requirements
  CONTRADICTS         — claim contradicts a specific regulatory requirement
  UNSUPPORTED         — claim makes a regulatory assertion but lacks required evidence
  INSUFFICIENT_EVIDENCE — regulatory relevance cannot be determined from available data
  CRITICAL_ABSENT     — required data (e.g. Scope 1/2/3 emissions) is entirely missing
  GREENWASHING_RISK   — qualitative ESG claim without quantifiable, audited data

Output structured JSON matching ClaimAssessment schema with:
- verdict: one of the above
- severity: HIGH | MEDIUM | LOW | NONE
- explanation: 2-4 sentences with specific regulatory citations
- cited_passages: 1-indexed passage numbers [P1, P2, ...]; may be empty
- red_flags: list of critical compliance issues
- warnings: list of non-critical issues requiring clarification
- verified: list of confirmed compliant data points
- action_items: specific questions to ask founder

ZERO HALLUCINATION: if data is absent, state DATA_ABSENT. Do not infer or assume.
This is an AI compliance audit, not formal legal advice.
Forward-looking statements (projections, targets) should be flagged as uncertain."""

SYSTEM_PROMPT_ESG_COMPLETENESS = """You are a compliance auditor reviewing ESG (Environmental, Social, Governance) data.

Check founder disclosures against SFDR and CSRD requirements:

REQUIRED per SFDR Article 4 and CSRD Directive:
- Scope 1 GHG emissions (tCO2e with year)
- Scope 2 emissions (tCO2e with year)
- Scope 3 emissions (tCO2e with year, if material)
- Third-party sustainability audit (named auditor, verification standard)
- Board composition with diversity metrics (% women, % underrepresented groups)

ADDITIONAL per EU AI Act (if AI in regulated sectors):
- AI use in Health, Finance, HR, Infrastructure requires transparency per Annex III
- High-risk classification requires formal risk assessment documentation

For each MISSING required field, return CRITICAL_ABSENT verdict with:
- Red flag citing the specific SFDR/CSRD article
- Action item: specific founder question to obtain the data

For each PRESENT field, return CONSISTENT verdict with:
- Verified list entry confirming the disclosure

Examples:
- Missing Scope 1 emissions → verdict=CRITICAL_ABSENT, severity=HIGH, action_item="Request quantified Scope 1 emissions for past 2 years with measurement methodology"
- Missing board diversity data → verdict=CRITICAL_ABSENT, severity=MEDIUM, action_item="Request board composition with gender/demographic breakdown per CSRD requirements"
- AI in healthcare without transparency statement → verdict=CRITICAL_ABSENT, severity=HIGH, red_flag="EU AI Act Annex III: High-risk AI in healthcare requires documented risk assessment"

Be specific: cite article numbers, name the missing disclosure, provide exact questions."""


def analyze_claim(
    client: anthropic.Anthropic,
    claim: str,
    hits: list[Hit],
    deck_context: str | None = None,
    model: str | None = None,
    esg_metrics: EsgMetrics | None = None,
) -> ClaimAssessment:
    """Analyze a claim against EU regulatory standards.

    Args:
        client: Anthropic API client
        claim: The claim text to analyze
        hits: Retrieved passages from EU regulatory KB
        deck_context: Optional metadata (fiscal year, public/private status, etc.)
        model: Model to use (defaults to ANALYZER_MODEL env var or Claude Sonnet)
        esg_metrics: Optional ESG metrics extracted from deck context

    Returns:
        ClaimAssessment with EU jurisdiction verdicts
    """
    model = model or _resolve_model()

    # Format passages for the prompt
    formatted_passages = _format_passages(hits)

    # Build user message
    user_message = f"Claim: {claim}\n\n"
    if deck_context:
        user_message += f"Deck context: {deck_context}\n\n"
    if esg_metrics:
        user_message += f"ESG Context:\n{_format_esg_context(esg_metrics)}\n\n"
    user_message += f"Relevant EU regulatory passages:\n\n{formatted_passages}"

    # Route by model type
    if llm_local.is_vision_local_model(model):
        # Local vision models don't support structured output with complex schemas
        raise NotImplementedError("EU analyzer not supported for local vision models")

    if llm_inception.is_inception_model(model):
        response = llm_inception.call_structured(
            model=model,
            system=SYSTEM_PROMPT_SFDR,
            user_content=user_message,
            output_format=ClaimAssessment,
        )
    elif llm_local.is_local_model(model):
        response = llm_local.call_structured(
            model=model,
            system=SYSTEM_PROMPT_SFDR,
            user_content=user_message,
            output_format=ClaimAssessment,
        )
    else:
        # Claude API with native support
        response = _call_anthropic_with_retry(
            client.messages.parse,
            model=model,
            max_tokens=8000,
            system=SYSTEM_PROMPT_SFDR,
            **_thinking_kwargs(model, budget_tokens=4000),
            messages=[{"role": "user", "content": user_message}],
            response_model=ClaimAssessment,
        )

    # Ensure jurisdiction field is set
    response.jurisdiction = "eu_sfdr_csrd"
    return response


def analyze_esg_completeness(
    client: anthropic.Anthropic,
    esg_metrics: EsgMetrics | None,
    model: str | None = None,
) -> list[ClaimAssessment]:
    """Check for required-but-absent ESG fields.

    Runs independently of claims to flag missing disclosures.

    Args:
        client: Anthropic API client
        esg_metrics: Extracted ESG metrics (may be None)
        model: Model to use

    Returns:
        List of ClaimAssessment verdicts, one per required field category
    """
    model = model or _resolve_model()
    results = []

    if not esg_metrics:
        # No ESG data extracted at all
        results.append(ClaimAssessment(
            verdict="CRITICAL_ABSENT",
            severity="HIGH",
            forward_looking=False,
            explanation="No ESG metrics extracted from the deck. SFDR Art. 4 and CSRD require disclosure of PAI (Principal Adverse Impacts) including GHG emissions, board diversity, and governance metrics.",
            cited_passages=[],
            jurisdiction="eu_sfdr_csrd",
            red_flags=["No ESG data provided (SFDR Art. 4 PAI disclosure missing)"],
            action_items=["Request comprehensive ESG disclosure including Scope 1/2/3 emissions, board composition, third-party audit confirmation"],
        ))
        return results

    # Check each required field
    required_checks = [
        ("scope_1_emissions", "Scope 1 GHG Emissions", "SFDR Art. 4, CSRD Annex I"),
        ("scope_2_emissions", "Scope 2 GHG Emissions", "SFDR Art. 4, CSRD Annex I"),
        ("scope_3_emissions", "Scope 3 GHG Emissions (if material)", "SFDR Art. 4, CSRD Annex I"),
        ("has_third_party_audit", "Third-Party Audit", "CSRD §5(1) verification requirement"),
        ("board_diversity_pct", "Board Diversity", "CSRD §5(1), Directive 2022/2464"),
    ]

    for field_name, label, citation in required_checks:
        value = getattr(esg_metrics, field_name, None)

        if value is None or (isinstance(value, bool) and not value):
            # Missing required field
            results.append(ClaimAssessment(
                verdict="CRITICAL_ABSENT",
                severity="HIGH",
                forward_looking=False,
                explanation=f"{label} data is absent or unconfirmed. {citation} requires companies to disclose this metric before investor intake.",
                cited_passages=[],
                jurisdiction="eu_sfdr_csrd",
                red_flags=[f"Missing {label} disclosure ({citation})"],
                action_items=[f"Request {label} data with verification methodology per CSRD §5(1)"],
            ))
        else:
            # Field is present
            results.append(ClaimAssessment(
                verdict="CONSISTENT",
                severity="NONE",
                forward_looking=False,
                explanation=f"{label} data is present in the deck.",
                cited_passages=[],
                jurisdiction="eu_sfdr_csrd",
                verified=[f"{label}: {value}"],
            ))

    # Check for greenwashing risk: qualitative claims without audit
    if (esg_metrics.scope_1_emissions or esg_metrics.scope_2_emissions or esg_metrics.scope_3_emissions) \
       and not esg_metrics.has_third_party_audit:
        results.append(ClaimAssessment(
            verdict="GREENWASHING_RISK",
            severity="MEDIUM",
            forward_looking=False,
            explanation="Deck claims GHG emissions reductions but lacks third-party audit verification. CSRD §5(1) requires independent verification of PAI metrics.",
            cited_passages=[],
            jurisdiction="eu_sfdr_csrd",
            red_flags=["Greenwashing risk: GHG claims without audit (CSRD §5(1))"],
            action_items=["Request third-party audit report (Bureau Veritas, DNV, TUV, or equivalent) covering the stated emission figures"],
        ))

    # Check for AI risk without transparency
    if esg_metrics.ai_risk_sector and not esg_metrics.ai_transparency_statement:
        results.append(ClaimAssessment(
            verdict="CRITICAL_ABSENT",
            severity="HIGH",
            forward_looking=False,
            explanation="Deck indicates AI use in regulated sector (Health/Finance/HR/Infra) but lacks required transparency. EU AI Act Annex III classifies such use as high-risk and requires documented risk assessment.",
            cited_passages=[],
            jurisdiction="eu_sfdr_csrd",
            red_flags=["EU AI Act Annex III: AI in regulated sector requires transparency documentation"],
            action_items=["Request AI risk assessment and transparency statement per EU AI Act §3"],
        ))

    return results


def _format_esg_context(esg_metrics: EsgMetrics) -> str:
    """Format ESG metrics for inclusion in the analysis prompt."""
    lines = []
    if esg_metrics.scope_1_emissions:
        lines.append(f"- Scope 1 Emissions: {esg_metrics.scope_1_emissions}")
    if esg_metrics.scope_2_emissions:
        lines.append(f"- Scope 2 Emissions: {esg_metrics.scope_2_emissions}")
    if esg_metrics.scope_3_emissions:
        lines.append(f"- Scope 3 Emissions: {esg_metrics.scope_3_emissions}")
    if esg_metrics.has_third_party_audit:
        lines.append(f"- Third-Party Audit: {esg_metrics.audit_body or 'Confirmed'}")
    if esg_metrics.board_diversity_pct:
        lines.append(f"- Board Diversity: {esg_metrics.board_diversity_pct}")
    if esg_metrics.ai_risk_sector:
        lines.append(f"- AI in Regulated Sector: Yes")
    if esg_metrics.ai_transparency_statement:
        lines.append(f"- AI Transparency: {esg_metrics.ai_transparency_statement}")
    if esg_metrics.sfdr_article_claim:
        lines.append(f"- SFDR Classification: {esg_metrics.sfdr_article_claim}")

    return "\n".join(lines) if lines else "No ESG metrics provided"
