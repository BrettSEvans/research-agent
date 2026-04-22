"""Predictive Risk Modeling and Early Warning System

Forecasts future compliance risks, identifies early warning indicators,
and estimates time to remediation using historical patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class WarningLevel(Enum):
    """Early warning indicator severity."""
    CRITICAL = "critical"  # Risk will likely become critical within 7 days
    HIGH = "high"  # Risk will likely become critical within 30 days
    MEDIUM = "medium"  # Risk may become critical within 90 days
    LOW = "low"  # Minor risk, no immediate action needed


@dataclass
class EarlyWarningIndicator:
    """Early warning signal of compliance deterioration."""
    indicator_type: str  # e.g., "greenwashing_pattern", "audit_gap", "missing_disclosure"
    warning_level: WarningLevel
    current_value: float
    threshold: float
    days_to_threshold: int
    affected_jurisdiction: str
    recommended_action: str


@dataclass
class PredictiveRiskModel:
    """Predictive risk assessment result."""
    jurisdiction: str
    current_risk: str
    predicted_risk_30_days: str
    predicted_risk_90_days: str
    early_warnings: List[EarlyWarningIndicator]
    critical_path_items: List[str]
    estimated_remediation_days: int
    confidence: float
    risk_trajectory: str  # ascending, stable, descending


def predict_future_risk(
    current_score: float,
    current_risk_level: str,
    flagged_issues: int,
    remediation_velocity: float,  # Issues resolved per day
    jurisdiction: str,
) -> PredictiveRiskModel:
    """Predict future compliance risk based on current state and velocity.

    Args:
        current_score: Current compliance score (0-100)
        current_risk_level: Current risk level (CRITICAL, HIGH, MEDIUM, LOW, NONE)
        flagged_issues: Number of flagged issues
        remediation_velocity: Issues resolved per day (negative = issues accumulating)
        jurisdiction: Target jurisdiction

    Returns:
        PredictiveRiskModel with 30/90-day forecast and early warnings
    """
    # Simulate 30-day and 90-day projections
    score_30_day = current_score + (remediation_velocity * 30 * 2)  # 2 points per issue fixed
    score_30_day = max(0, min(100, score_30_day))

    score_90_day = current_score + (remediation_velocity * 90 * 2)
    score_90_day = max(0, min(100, score_90_day))

    # Map scores to risk levels
    def score_to_risk(score):
        if score >= 80:
            return "NONE"
        elif score >= 60:
            return "LOW"
        elif score >= 40:
            return "MEDIUM"
        elif score >= 20:
            return "HIGH"
        else:
            return "CRITICAL"

    risk_30_day = score_to_risk(score_30_day)
    risk_90_day = score_to_risk(score_90_day)

    # Determine trajectory
    if score_90_day > current_score + 10:
        trajectory = "descending"  # Improving
    elif score_90_day < current_score - 10:
        trajectory = "ascending"  # Worsening
    else:
        trajectory = "stable"

    # Identify early warning indicators
    warnings = _identify_early_warnings(
        current_score, flagged_issues, remediation_velocity, jurisdiction
    )

    # Calculate critical path (items that block progress)
    critical_path = _identify_critical_path(flagged_issues, jurisdiction)

    # Estimate remediation time
    if remediation_velocity > 0:
        estimated_days = max(1, int(flagged_issues / remediation_velocity))
    else:
        estimated_days = 999  # No progress being made

    return PredictiveRiskModel(
        jurisdiction=jurisdiction,
        current_risk=current_risk_level,
        predicted_risk_30_days=risk_30_day,
        predicted_risk_90_days=risk_90_day,
        early_warnings=warnings,
        critical_path_items=critical_path,
        estimated_remediation_days=estimated_days,
        confidence=min(1.0, abs(remediation_velocity) / 0.1),  # Max confidence at 0.1 issues/day
        risk_trajectory=trajectory,
    )


def _identify_early_warnings(
    score: float,
    flagged_issues: int,
    velocity: float,
    jurisdiction: str,
) -> List[EarlyWarningIndicator]:
    """Identify early warning indicators of compliance deterioration."""
    warnings = []

    # Warning 1: Stalled remediation
    if flagged_issues > 5 and velocity <= 0:
        warnings.append(EarlyWarningIndicator(
            indicator_type="stalled_remediation",
            warning_level=WarningLevel.CRITICAL,
            current_value=flagged_issues,
            threshold=5,
            days_to_threshold=7,
            affected_jurisdiction=jurisdiction,
            recommended_action="Immediately escalate remediation efforts or risk critical status within days.",
        ))

    # Warning 2: Rapid score decline
    if velocity < -0.5:  # More than 0.5 points/day declining
        days_to_critical = (20 - score) / abs(velocity)
        if days_to_critical < 30:
            warnings.append(EarlyWarningIndicator(
                indicator_type="rapid_decline",
                warning_level=WarningLevel.HIGH if days_to_critical > 7 else WarningLevel.CRITICAL,
                current_value=score,
                threshold=20,
                days_to_threshold=int(days_to_critical),
                affected_jurisdiction=jurisdiction,
                recommended_action=f"Address newly surfaced issues immediately; critical status in ~{int(days_to_critical)} days.",
            ))

    # Warning 3: High issue count
    if flagged_issues > 10:
        warnings.append(EarlyWarningIndicator(
            indicator_type="high_issue_count",
            warning_level=WarningLevel.MEDIUM if flagged_issues < 20 else WarningLevel.HIGH,
            current_value=flagged_issues,
            threshold=10,
            days_to_threshold=30,
            affected_jurisdiction=jurisdiction,
            recommended_action="Prioritize critical issues; allocate more resources to remediation.",
        ))

    # Warning 4: Compliance below 40 (medium-risk zone)
    if 20 < score < 40:
        warnings.append(EarlyWarningIndicator(
            indicator_type="medium_risk_zone",
            warning_level=WarningLevel.MEDIUM,
            current_value=score,
            threshold=40,
            days_to_threshold=int((40 - score) / max(velocity, 0.1)),
            affected_jurisdiction=jurisdiction,
            recommended_action="Accelerate remediation to exit medium-risk zone before critical status.",
        ))

    return warnings


def _identify_critical_path(flagged_issues: int, jurisdiction: str) -> List[str]:
    """Identify critical path items (blocking dependencies)."""
    items = []

    if jurisdiction == "eu_sfdr_csrd":
        if flagged_issues > 0:
            items.append("Obtain third-party ESG audit (blocks all emissions claims)")
            items.append("Document Scope 1 & 2 emissions (prerequisite for SFDR claims)")
            items.append("Board diversity disclosure (required ESRS)")

    elif jurisdiction == "ca_sb54":
        if flagged_issues > 0:
            items.append("Founder demographic disclosure (baseline requirement)")
            items.append("Board composition documentation (blocks SB 164 compliance)")

    elif jurisdiction == "sec":
        if flagged_issues > 0:
            items.append("File corrective 8-K or 10-Q (blocks other filings)")
            items.append("Update MD&A with accurate information (required for next 10-K)")

    return items[:3]  # Top 3 blocking items


def estimate_remediation_timeline(
    jurisdiction: str,
    flagged_issues: int,
    issue_complexity: str,  # simple, moderate, complex
) -> Dict:
    """Estimate timeline to remediate all flagged issues.

    Args:
        jurisdiction: Target jurisdiction
        flagged_issues: Number of flagged issues
        issue_complexity: Complexity level affecting resolution time

    Returns:
        Remediation timeline estimate
    """
    # Complexity multipliers (days per issue)
    complexity_multipliers = {
        "simple": 1,        # 1 day per issue
        "moderate": 3,      # 3 days per issue
        "complex": 7,       # 7 days per issue
    }

    multiplier = complexity_multipliers.get(issue_complexity, 3)

    # Jurisdiction-specific overhead
    jurisdiction_overhead = {
        "eu_sfdr_csrd": 14,  # Need audit, takes longer
        "ca_sb54": 7,        # Demographic collection
        "sec": 10,           # SEC filing process
    }

    overhead = jurisdiction_overhead.get(jurisdiction, 5)

    # Calculate total timeline
    total_days = (flagged_issues * multiplier) + overhead

    # Break into phases
    phases = [
        {
            "phase": "Assessment",
            "duration_days": int(overhead * 0.4),
            "description": "Review issues and plan remediation",
        },
        {
            "phase": "Remediation",
            "duration_days": int(flagged_issues * multiplier),
            "description": "Address flagged issues",
        },
        {
            "phase": "Verification",
            "duration_days": int(overhead * 0.6),
            "description": "Verify fixes and obtain attestation",
        },
    ]

    return {
        "jurisdiction": jurisdiction,
        "flagged_issues": flagged_issues,
        "issue_complexity": issue_complexity,
        "estimated_total_days": total_days,
        "optimistic_days": int(total_days * 0.7),
        "pessimistic_days": int(total_days * 1.3),
        "phases": phases,
        "critical_path": _identify_critical_path(flagged_issues, jurisdiction),
        "recommendation": (
            f"Plan for {total_days} days to remediate all {flagged_issues} issues. "
            f"Optimistic: {int(total_days * 0.7)} days. Pessimistic: {int(total_days * 1.3)} days."
        ),
    }


def risk_mitigation_priorities(
    warnings: List[EarlyWarningIndicator],
    critical_path: List[str],
) -> Dict:
    """Prioritize risk mitigation actions.

    Args:
        warnings: Early warning indicators
        critical_path: Critical path items

    Returns:
        Prioritized action list
    """
    # Assign priority scores
    actions = []

    for warning in warnings:
        if warning.warning_level == WarningLevel.CRITICAL:
            priority = 1
        elif warning.warning_level == WarningLevel.HIGH:
            priority = 2
        elif warning.warning_level == WarningLevel.MEDIUM:
            priority = 3
        else:
            priority = 4

        actions.append({
            "priority": priority,
            "action": warning.recommended_action,
            "days_to_escalate": warning.days_to_threshold,
            "type": "early_warning",
        })

    # Add critical path items (highest priority)
    for i, item in enumerate(critical_path):
        actions.append({
            "priority": 0.5 + (i * 0.1),  # Even higher priority
            "action": f"Address critical blocker: {item}",
            "days_to_escalate": 7,
            "type": "critical_path",
        })

    # Sort by priority
    actions.sort(key=lambda a: a["priority"])

    return {
        "total_actions": len(actions),
        "prioritized_actions": actions,
        "recommendation": f"Execute {len(actions)} actions in priority order. Critical blockers (priority {actions[0]['priority'] if actions else 'N/A'}) must complete first.",
    }
