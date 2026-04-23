"""Compliance Trend Analysis and Forecasting

Tracks historical compliance scores, calculates trends, and forecasts future
compliance based on remediation patterns and historical velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from enum import Enum


class TrendDirection(Enum):
    """Compliance trend direction."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


@dataclass
class ComplianceDataPoint:
    """Single compliance score measurement."""
    timestamp: datetime
    jurisdiction: str
    score: float
    risk_level: str
    flagged_claims: int
    missing_data_claims: int


@dataclass
class TrendAnalysis:
    """Compliance trend analysis result."""
    jurisdiction: str
    data_points: List[ComplianceDataPoint]
    trend_direction: TrendDirection
    trend_score: float  # -1.0 to 1.0 (declining to improving)
    velocity: float  # Score change per day
    estimated_critical_date: Optional[datetime]  # When risk becomes CRITICAL
    estimated_resolution_date: Optional[datetime]  # When compliance improves to NONE
    confidence: float  # 0-1, based on data points
    recommendation: str


def analyze_compliance_trend(
    data_points: List[ComplianceDataPoint],
    jurisdiction: str,
) -> TrendAnalysis:
    """Analyze compliance trend over time.

    Args:
        data_points: Historical compliance measurements
        jurisdiction: Target jurisdiction

    Returns:
        TrendAnalysis with trend direction, velocity, and forecasts
    """
    filtered_points = [p for p in data_points if p.jurisdiction == jurisdiction]

    if not filtered_points:
        return TrendAnalysis(
            jurisdiction=jurisdiction,
            data_points=[],
            trend_direction=TrendDirection.STABLE,
            trend_score=0.0,
            velocity=0.0,
            estimated_critical_date=None,
            estimated_resolution_date=None,
            confidence=0.0,
            recommendation="No historical data available.",
        )

    # Sort by timestamp
    sorted_points = sorted(filtered_points, key=lambda p: p.timestamp)

    # Calculate trend direction and velocity
    if len(sorted_points) >= 2:
        first = sorted_points[0]
        last = sorted_points[-1]
        score_change = last.score - first.score
        days_elapsed = (last.timestamp - first.timestamp).days
        velocity = score_change / max(days_elapsed, 1)
        trend_direction = (
            TrendDirection.IMPROVING if velocity > 0.5
            else TrendDirection.DECLINING if velocity < -0.5
            else TrendDirection.STABLE
        )
        trend_score = min(1.0, max(-1.0, velocity / 50))  # Normalize to [-1, 1]
        confidence = min(1.0, len(sorted_points) / 5)  # Max confidence at 5+ points
    else:
        velocity = 0.0
        trend_direction = TrendDirection.STABLE
        trend_score = 0.0
        confidence = 0.0

    # Forecast critical date
    estimated_critical_date = None
    if velocity < -0.5 and sorted_points[-1].score > 20:
        days_to_critical = (20 - sorted_points[-1].score) / velocity
        estimated_critical_date = sorted_points[-1].timestamp + timedelta(days=days_to_critical)

    # Forecast resolution date
    estimated_resolution_date = None
    if velocity > 0.5 and sorted_points[-1].score < 80:
        days_to_resolution = (80 - sorted_points[-1].score) / velocity
        estimated_resolution_date = sorted_points[-1].timestamp + timedelta(days=days_to_resolution)

    # Generate recommendation
    current_score = sorted_points[-1].score if sorted_points else 50
    recommendation = _generate_trend_recommendation(
        trend_direction, velocity, current_score, confidence
    )

    return TrendAnalysis(
        jurisdiction=jurisdiction,
        data_points=sorted_points,
        trend_direction=trend_direction,
        trend_score=round(trend_score, 3),
        velocity=round(velocity, 2),
        estimated_critical_date=estimated_critical_date,
        estimated_resolution_date=estimated_resolution_date,
        confidence=round(confidence, 2),
        recommendation=recommendation,
    )


def _generate_trend_recommendation(
    direction: TrendDirection,
    velocity: float,
    current_score: float,
    confidence: float,
) -> str:
    """Generate trend-based recommendation."""
    if direction == TrendDirection.IMPROVING:
        if velocity > 5:
            return f"✓ Strong improvement trend (+{velocity:.1f} pts/day). Maintain current remediation pace."
        else:
            return f"✓ Gradual improvement (+{velocity:.1f} pts/day). Continue efforts."

    if direction == TrendDirection.DECLINING:
        if velocity < -5:
            return f"⚠️  CRITICAL DECLINE ({velocity:.1f} pts/day). Immediate action required to prevent critical status."
        else:
            return f"⚠️  Declining trend ({velocity:.1f} pts/day). Address new issues before compliance erodes further."

    return f"ℹ️  Score is stable. Monitor for changes and address any flagged issues."


def forecast_compliance_score(
    data_points: List[ComplianceDataPoint],
    jurisdiction: str,
    days_ahead: int = 30,
) -> Dict:
    """Forecast future compliance score based on historical trend.

    Args:
        data_points: Historical compliance measurements
        jurisdiction: Target jurisdiction
        days_ahead: Number of days to forecast (default 30)

    Returns:
        Forecast with predicted score, confidence interval, and scenarios
    """
    trend = analyze_compliance_trend(data_points, jurisdiction)

    if not trend.data_points:
        return {
            "jurisdiction": jurisdiction,
            "forecast_days": days_ahead,
            "predicted_score": None,
            "confidence": 0.0,
            "recommendation": "Insufficient historical data for forecasting.",
        }

    # Base prediction: current score + (velocity × days)
    current_score = trend.data_points[-1].score
    predicted_score = current_score + (trend.velocity * days_ahead)
    predicted_score = max(0, min(100, predicted_score))  # Clamp to [0, 100]

    # Confidence intervals (wider for lower confidence)
    std_error = (1 - trend.confidence) * 15  # Max error: 15 points
    lower_bound = max(0, predicted_score - std_error)
    upper_bound = min(100, predicted_score + std_error)

    # Scenarios
    optimistic = min(100, predicted_score + std_error)
    pessimistic = max(0, predicted_score - std_error)

    return {
        "jurisdiction": jurisdiction,
        "forecast_days": days_ahead,
        "current_score": round(current_score, 1),
        "predicted_score": round(predicted_score, 1),
        "lower_bound": round(lower_bound, 1),
        "upper_bound": round(upper_bound, 1),
        "optimistic_scenario": round(optimistic, 1),
        "pessimistic_scenario": round(pessimistic, 1),
        "confidence": round(trend.confidence, 2),
        "trend": trend.trend_direction.value,
        "velocity": trend.velocity,
    }


def calculate_remediation_velocity(
    assessments_before: List[Dict],
    assessments_after: List[Dict],
    days_elapsed: int,
) -> Dict:
    """Calculate how quickly compliance issues are being remediated.

    Args:
        assessments_before: Previous assessment results
        assessments_after: Current assessment results
        days_elapsed: Days between assessments

    Returns:
        Remediation velocity metrics
    """
    if days_elapsed < 1:
        return {
            "velocity": 0.0,
            "issues_resolved": 0,
            "new_issues": 0,
            "remediation_rate": 0.0,
        }

    # Count critical/high-severity issues
    def count_issues(assessments):
        return sum(1 for a in assessments if a.get("severity") in ["HIGH", "CRITICAL"])

    before_count = count_issues(assessments_before)
    after_count = count_issues(assessments_after)

    issues_resolved = max(0, before_count - after_count)
    new_issues = max(0, after_count - before_count)
    remediation_rate = issues_resolved / max(before_count, 1)

    return {
        "velocity": round(issues_resolved / days_elapsed, 2),  # Issues/day
        "issues_resolved": issues_resolved,
        "new_issues": new_issues,
        "remediation_rate": round(remediation_rate, 2),
        "days_to_zero_critical": (
            days_elapsed / remediation_rate if remediation_rate > 0 else float('inf')
        ),
    }
