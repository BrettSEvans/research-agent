"""Greenwashing and ESG Claims Risk Detector

Advanced pattern detection for suspicious ESG claims and potential greenwashing.
Analyzes claim ratios, missing evidence, and inconsistencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
from extractor import EsgMetrics


@dataclass
class GreenwashingRisk:
    """Greenwashing risk assessment for ESG claims."""
    risk_level: str  # CRITICAL, HIGH, MEDIUM, LOW, NONE
    risk_score: float  # 0-100, higher is more risky
    detected_patterns: List[str]
    missing_evidence: List[str]
    red_flags: List[str]
    recommendation: str


def detect_greenwashing_risk(esg_metrics: EsgMetrics | None) -> GreenwashingRisk:
    """Detect greenwashing and ESG claims risk.

    Analyzes patterns in ESG disclosure for signs of:
    - Vague claims without quantification
    - Missing supporting evidence
    - Inconsistent disclosures
    - Suspicious ratios
    - Scope creep without substance

    Args:
        esg_metrics: Extracted ESG metrics from deck

    Returns:
        GreenwashingRisk assessment with patterns and recommendations
    """
    if not esg_metrics:
        return GreenwashingRisk(
            risk_level="MEDIUM",
            risk_score=60.0,
            detected_patterns=["No ESG data disclosed"],
            missing_evidence=["All ESG metrics absent"],
            red_flags=["Complete lack of ESG disclosure suggests potential greenwashing"],
            recommendation="Request comprehensive ESG disclosure including quantified emissions, audit verification, and board diversity metrics.",
        )

    patterns = []
    missing_evidence = []
    red_flags = []
    risk_score = 0.0

    # Pattern 1: Claims without quantification
    if esg_metrics.sfdr_article_claim and not esg_metrics.scope_1_emissions and not esg_metrics.scope_2_emissions:
        patterns.append("SFDR article claim without quantified emissions")
        missing_evidence.append("GHG emissions data")
        red_flags.append("Article 8/9 sustainability claim lacks quantified environmental metrics")
        risk_score += 25

    # Pattern 2: Selective disclosure (some scopes missing)
    scope_count = sum(1 for s in [esg_metrics.scope_1_emissions, esg_metrics.scope_2_emissions, esg_metrics.scope_3_emissions] if s)
    if scope_count == 1:
        patterns.append("Only one emissions scope disclosed (suspicious selective reporting)")
        missing_evidence.append("Scope 2 and 3 emissions data")
        red_flags.append("Selective Scope disclosure may hide material emissions")
        risk_score += 15

    # Pattern 3: Missing audit for ESG claims
    has_emissions = bool(esg_metrics.scope_1_emissions or esg_metrics.scope_2_emissions or esg_metrics.scope_3_emissions)
    if has_emissions and not esg_metrics.has_third_party_audit:
        patterns.append("Emissions claims without third-party audit")
        missing_evidence.append("Third-party audit verification")
        red_flags.append("Unaudited ESG claims increase greenwashing risk")
        risk_score += 30

    # Pattern 4: Missing board diversity with ESG claims
    if esg_metrics.sfdr_article_claim and not esg_metrics.board_diversity_pct:
        patterns.append("SFDR article claim without board diversity disclosure")
        missing_evidence.append("Board composition and diversity metrics")
        red_flags.append("Board diversity is required ESRS disclosure; absence undermines credibility")
        risk_score += 20

    # Pattern 5: AI in regulated sector without transparency
    if esg_metrics.ai_risk_sector and not esg_metrics.ai_transparency_statement:
        patterns.append("AI use in regulated sector without transparency statement")
        missing_evidence.append("AI risk assessment and transparency documentation")
        red_flags.append("EU AI Act Annex III: High-risk AI requires formal risk assessment")
        risk_score += 25

    # Pattern 6: Supply chain claims without audit
    if esg_metrics.supply_chain_disclosure and not esg_metrics.has_third_party_audit:
        patterns.append("Supply chain sustainability claim without audit")
        missing_evidence.append("Supply chain audit or verification")
        red_flags.append("Unverified supply chain claims present material ESG risk")
        risk_score += 15

    # Pattern 7: No diversity data despite being asked
    if not esg_metrics.board_diversity_pct:
        patterns.append("No board diversity disclosure")
        missing_evidence.append("Board composition and diversity metrics")
        red_flags.append("Absence of board diversity disclosure suggests potential governance risk")
        risk_score += 10

    # Determine overall risk level
    if risk_score >= 80:
        risk_level = "CRITICAL"
    elif risk_score >= 60:
        risk_level = "HIGH"
    elif risk_score >= 40:
        risk_level = "MEDIUM"
    elif risk_score >= 20:
        risk_level = "LOW"
    else:
        risk_level = "NONE"

    # Generate recommendation
    recommendation = _generate_greenwashing_recommendation(patterns, risk_level, risk_score)

    return GreenwashingRisk(
        risk_level=risk_level,
        risk_score=round(risk_score, 1),
        detected_patterns=patterns,
        missing_evidence=missing_evidence,
        red_flags=red_flags,
        recommendation=recommendation,
    )


def _generate_greenwashing_recommendation(
    patterns: List[str],
    risk_level: str,
    risk_score: float,
) -> str:
    """Generate greenwashing risk recommendation."""
    if not patterns:
        return "✓ No obvious greenwashing patterns detected. ESG disclosures appear authentic."

    if risk_level == "CRITICAL":
        return f"🚨 CRITICAL GREENWASHING RISK ({risk_score:.0f}/100): {len(patterns)} suspicious pattern(s) detected. Request detailed audit, quantified metrics, and third-party verification before proceeding."

    if risk_level == "HIGH":
        return f"⚠️  HIGH GREENWASHING RISK ({risk_score:.0f}/100): {len(patterns)} pattern(s) suggest incomplete ESG disclosure. Require independent audit and comprehensive data before investment."

    if risk_level == "MEDIUM":
        return f"⚠️  MEDIUM GREENWASHING RISK ({risk_score:.0f}/100): {len(patterns)} pattern(s) identified. Verify claims with third-party audit; request missing data."

    if risk_level == "LOW":
        return f"ℹ️  LOW GREENWASHING RISK ({risk_score:.0f}/100): Minor gaps in ESG disclosure. Request clarification on {len(patterns)} item(s)."

    return "✓ ESG claims appear well-supported and substantiated."


def compare_esg_metrics(
    claimed: EsgMetrics,
    verified: EsgMetrics | None = None,
) -> dict:
    """Compare claimed vs. verified ESG metrics to detect inconsistencies.

    Args:
        claimed: ESG metrics from pitch deck
        verified: ESG metrics from independent sources (optional)

    Returns:
        Comparison results with inconsistencies flagged
    """
    if not verified:
        return {
            "has_verification": False,
            "inconsistencies": [],
            "recommendation": "Request independent verification of all ESG metrics."
        }

    inconsistencies = []

    # Compare emissions figures
    if claimed.scope_1_emissions and verified.scope_1_emissions:
        if claimed.scope_1_emissions != verified.scope_1_emissions:
            inconsistencies.append("Scope 1 emissions figures differ between pitch and audit")

    # Check audit status consistency
    if claimed.has_third_party_audit != verified.has_third_party_audit:
        inconsistencies.append("Audit status inconsistency between claimed and verified")

    # Check diversity metrics consistency
    if claimed.board_diversity_pct and verified.board_diversity_pct:
        if claimed.board_diversity_pct != verified.board_diversity_pct:
            inconsistencies.append("Board diversity percentage inconsistent")

    return {
        "has_verification": True,
        "inconsistencies": inconsistencies,
        "recommendation": "Discrepancies found. Clarify with company and audit provider." if inconsistencies else "✓ Verified metrics match claimed metrics."
    }
