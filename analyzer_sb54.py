"""California SB 54 Compliance Analyzer

Assesses pitch deck founder demographics against California SB 54:
- CA SB 54 (Nonprofit Integrity Act and board diversity reporting)
- Checks for founder demographic disclosure completeness
- Flags missing diversity data and underrepresented founder statistics

Implements the AnalyzerModule protocol with ClaimAssessment output.
"""

from __future__ import annotations

import anthropic
from pydantic import BaseModel, Field

from analyzer_protocol import ClaimAssessment, AnalyzerModule
from retriever import Hit
import llm_local
import llm_inception
from extractor import FounderDemographics


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


SYSTEM_PROMPT_CA_SB54 = """You are a California Compliance Auditor specializing in SB 54 founder diversity requirements.

You audit pitch deck disclosures against:
- CA SB 54 — Nonprofit and for-profit board/founder diversity reporting
- CA diversity investment disclosure standards
- VC transparency requirements for founder demographic data

Task: Given founder demographic data from a startup pitch deck and SB 54 requirements,
classify the disclosure completeness as:
  CONSISTENT          — founder demographics are fully disclosed with diversity metrics
  INSUFFICIENT_EVIDENCE — founder data present but diversity metrics incomplete
  CRITICAL_ABSENT     — required demographic data is entirely missing
  UNSUPPORTED         — claims of diversity exist but lack supporting documentation
  DATA_QUALITY_ISSUE  — demographic data conflicts or appears inconsistent

Output structured JSON matching ClaimAssessment schema with:
- verdict: one of the above
- severity: HIGH | MEDIUM | LOW | NONE
- explanation: 2-4 sentences with specific SB 54 citations
- red_flags: list of critical disclosure gaps
- warnings: list of data quality issues
- verified: list of confirmed demographic data points
- action_items: specific questions to ask founder about missing disclosures

ZERO HALLUCINATION: if diversity data is absent, state MISSING. Do not infer or assume.
This is an AI compliance audit, not formal legal advice.
Focus on whether founder demographics were disclosed, not on evaluating whether diversity is "sufficient"."""


def analyze_demographic_completeness(
    client: anthropic.Anthropic,
    founder_demographics: FounderDemographics | None,
    model: str | None = None,
) -> list[ClaimAssessment]:
    """Check for required-but-absent founder demographic fields per SB 54.

    Runs independently to flag missing demographic disclosures.

    Args:
        client: Anthropic API client
        founder_demographics: Extracted founder demographics (may be None)
        model: Model to use

    Returns:
        List of ClaimAssessment verdicts, one per required field category
    """
    model = model or _resolve_model()
    results = []

    if not founder_demographics:
        # No demographic data extracted at all
        results.append(ClaimAssessment(
            verdict="CRITICAL_ABSENT",
            severity="HIGH",
            forward_looking=False,
            explanation="No founder demographic data extracted from the deck. CA SB 54 encourages disclosure of founder diversity metrics including gender, race/ethnicity, and background.",
            cited_passages=[],
            jurisdiction="ca_sb54",
            red_flags=["No founder demographic disclosure provided (CA SB 54 transparency expected)"],
            action_items=["Request founder demographic disclosure including: founder count, gender breakdown, race/ethnicity breakdown, educational background, prior startup experience"],
        ))
        return results

    # Check each required field
    required_checks = [
        ("founder_count", "Founder Count", "SB 54 transparency requirement"),
        ("gender_diversity", "Gender Diversity Breakdown", "SB 54 demographic disclosure"),
        ("race_ethnicity_data", "Race/Ethnicity Disclosure", "SB 54 diversity reporting"),
        ("educational_background", "Educational Background", "Founder credential disclosure"),
        ("prior_startup_experience", "Prior Startup Experience", "Track record disclosure"),
    ]

    for field_name, label, citation in required_checks:
        value = getattr(founder_demographics, field_name, None)

        if value is None or (isinstance(value, (list, bool)) and not value):
            # Missing required field
            results.append(ClaimAssessment(
                verdict="CRITICAL_ABSENT",
                severity="MEDIUM",
                forward_looking=False,
                explanation=f"{label} data is absent or unconfirmed. {citation} expects companies to disclose founder background information.",
                cited_passages=[],
                jurisdiction="ca_sb54",
                red_flags=[f"Missing {label} disclosure ({citation})"],
                action_items=[f"Request {label} data with specific founder details per {citation}"],
            ))
        else:
            # Field is present
            results.append(ClaimAssessment(
                verdict="CONSISTENT",
                severity="NONE",
                forward_looking=False,
                explanation=f"{label} data is present in the deck.",
                cited_passages=[],
                jurisdiction="ca_sb54",
                verified=[f"{label}: {value}"],
            ))

    # Check for diversity disclosure completeness
    has_gender = founder_demographics.gender_diversity or founder_demographics.women_founder_pct is not None
    has_race = founder_demographics.race_ethnicity_data or founder_demographics.underrepresented_minority_pct is not None
    has_background = founder_demographics.industry_expertise or founder_demographics.educational_background

    if not (has_gender and has_race and has_background):
        results.append(ClaimAssessment(
            verdict="INSUFFICIENT_EVIDENCE",
            severity="MEDIUM",
            forward_looking=False,
            explanation="Founder demographic disclosure is incomplete. A comprehensive disclosure includes gender, race/ethnicity, and professional background.",
            cited_passages=[],
            jurisdiction="ca_sb54",
            red_flags=["Incomplete founder demographic disclosure (missing multiple data categories)"],
            action_items=["Request comprehensive founder demographic profile covering all of: gender breakdown, race/ethnicity breakdown, educational institutions, prior experience"],
        ))

    return results
