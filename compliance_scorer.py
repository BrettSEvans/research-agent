"""Compliance Scoring Engine

Calculates overall compliance scores and risk assessments across jurisdictions.
Aggregates individual claim assessments into actionable compliance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
from analyzer_protocol import ClaimAssessment


@dataclass
class ComplianceScore:
    """Overall compliance assessment score."""
    jurisdiction: str
    overall_score: float  # 0-100, higher is better
    risk_level: str  # CRITICAL, HIGH, MEDIUM, LOW, NONE
    passing_claims: int
    flagged_claims: int
    missing_data_claims: int
    contradiction_count: int
    severity_breakdown: Dict[str, int]
    recommendation: str


def calculate_compliance_score(
    jurisdiction: str,
    assessments: List[ClaimAssessment],
) -> ComplianceScore:
    """Calculate overall compliance score for a jurisdiction.

    Args:
        jurisdiction: Jurisdiction identifier (sec, eu_sfdr_csrd, ca_sb54)
        assessments: List of claim assessments for this jurisdiction

    Returns:
        ComplianceScore with detailed breakdown
    """
    if not assessments:
        return ComplianceScore(
            jurisdiction=jurisdiction,
            overall_score=100.0,
            risk_level="NONE",
            passing_claims=0,
            flagged_claims=0,
            missing_data_claims=0,
            contradiction_count=0,
            severity_breakdown={"NONE": 0},
            recommendation="No claims to assess.",
        )

    # Verdict scoring
    verdict_scores = {
        "CONSISTENT": 100,
        "UNSUPPORTED": 50,
        "INSUFFICIENT_EVIDENCE": 25,
        "CONTRADICTS": 0,
        "CRITICAL_ABSENT": 10,
        "GREENWASHING_RISK": 20,
        "DATA_QUALITY_ISSUE": 30,
    }

    # Severity weighting
    severity_weights = {
        "NONE": 1.0,
        "LOW": 0.8,
        "MEDIUM": 0.5,
        "HIGH": 0.2,
    }

    scores = []
    severity_counts = {"NONE": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0}
    contradiction_count = 0

    for assessment in assessments:
        # Get base score for verdict
        base_score = verdict_scores.get(assessment.verdict, 50)

        # Apply severity penalty
        severity = assessment.severity or "NONE"
        severity_weight = severity_weights.get(severity, 1.0)
        weighted_score = base_score * severity_weight

        scores.append(weighted_score)
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

        if assessment.verdict == "CONTRADICTS" and assessment.forward_looking:
            contradiction_count += 1

    # Calculate overall score (average of all weighted scores)
    overall_score = sum(scores) / len(scores) if scores else 100.0

    # Determine risk level based on overall score and verdict distribution
    passing = sum(1 for a in assessments if a.verdict == "CONSISTENT")
    flagged = sum(1 for a in assessments if a.verdict == "CONTRADICTS")
    missing_data = sum(1 for a in assessments if a.verdict in ["CRITICAL_ABSENT", "INSUFFICIENT_EVIDENCE"])

    if overall_score >= 80:
        risk_level = "NONE"
    elif overall_score >= 60:
        risk_level = "LOW"
    elif overall_score >= 30:
        risk_level = "MEDIUM"
    elif overall_score >= 15:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    # Adjust risk level based on critical contradictions
    if contradiction_count >= 3:
        risk_level = "CRITICAL"
    elif contradiction_count >= 2 and risk_level != "CRITICAL":
        risk_level = "HIGH"

    # Adjust risk level based on missing critical data
    if missing_data >= len(assessments) * 0.5:  # > 50% missing
        if risk_level == "CRITICAL":
            risk_level = "HIGH"
        elif risk_level not in ["HIGH"]:
            risk_level = "MEDIUM"

    # Generate recommendation
    recommendation = _generate_recommendation(
        jurisdiction, overall_score, risk_level, passing, flagged, missing_data
    )

    return ComplianceScore(
        jurisdiction=jurisdiction,
        overall_score=round(overall_score, 1),
        risk_level=risk_level,
        passing_claims=passing,
        flagged_claims=flagged,
        missing_data_claims=missing_data,
        contradiction_count=contradiction_count,
        severity_breakdown=severity_counts,
        recommendation=recommendation,
    )


def _generate_recommendation(
    jurisdiction: str,
    score: float,
    risk_level: str,
    passing: int,
    flagged: int,
    missing: int,
) -> str:
    """Generate actionable recommendation based on compliance score."""
    if risk_level == "NONE":
        return f"✓ {jurisdiction.upper()} compliance is strong ({score:.1f}/100)."

    if risk_level == "CRITICAL":
        action_items = []
        if flagged > 0:
            action_items.append(f"address {flagged} contradictions")
        if missing > 0:
            action_items.append(f"provide missing data for {missing} items")
        return f"⚠️  CRITICAL: {jurisdiction.upper()} requires immediate action. {', '.join(action_items)}."

    if risk_level == "HIGH":
        return f"⚠️  HIGH RISK: {jurisdiction.upper()} compliance score is {score:.1f}/100. Review all flagged claims and missing data."

    if risk_level == "MEDIUM":
        return f"⚠️  MEDIUM RISK: {jurisdiction.upper()} has {missing} items with missing data. Address before next funding round."

    return f"ℹ️  LOW RISK: {jurisdiction.upper()} compliance acceptable but review {missing} gaps."


def aggregate_scores(
    scores: List[ComplianceScore],
) -> Dict:
    """Aggregate compliance scores across all jurisdictions.

    Args:
        scores: List of ComplianceScore objects from different jurisdictions

    Returns:
        Aggregated score summary
    """
    if not scores:
        return {
            "overall_risk": "NONE",
            "jurisdictions": [],
            "critical_issues": 0,
            "primary_recommendation": "No assessments performed.",
        }

    # Determine overall risk level (worst case across jurisdictions)
    risk_hierarchy = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NONE": 4}
    worst_risk = min(scores, key=lambda s: risk_hierarchy.get(s.risk_level, 5)).risk_level

    # Count critical issues
    critical_issues = sum(1 for s in scores if s.risk_level == "CRITICAL")
    high_risk_issues = sum(1 for s in scores if s.risk_level == "HIGH")

    # Overall recommendation
    if critical_issues > 0:
        primary_rec = f"🚨 {critical_issues} jurisdiction(s) require critical attention before proceeding."
    elif high_risk_issues > 0:
        primary_rec = f"⚠️  {high_risk_issues} jurisdiction(s) have high-risk issues. Remediation recommended."
    else:
        primary_rec = "✓ Overall compliance posture acceptable. Continue monitoring."

    return {
        "overall_risk": worst_risk,
        "critical_issues": critical_issues,
        "high_risk_issues": high_risk_issues,
        "jurisdictions": [
            {
                "jurisdiction": s.jurisdiction,
                "score": s.overall_score,
                "risk": s.risk_level,
                "recommendation": s.recommendation,
            }
            for s in scores
        ],
        "primary_recommendation": primary_rec,
    }
