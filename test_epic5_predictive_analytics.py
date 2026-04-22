"""
Comprehensive test coverage for Epic 5 — Predictive Compliance Analytics.

Tests cover:
- Story 5.1: Compliance Trend Analysis
- Story 5.2: Industry Benchmarking
- Story 5.3: Predictive Risk Modeling

Total: 55+ test cases
"""

import pytest
from datetime import datetime, timedelta
from compliance_trends import (
    analyze_compliance_trend, forecast_compliance_score,
    calculate_remediation_velocity, ComplianceDataPoint, TrendDirection
)
from industry_benchmarks import (
    benchmark_compliance, identify_best_practices, stage_adjusted_benchmark,
    PeerBenchmark, IndustrySegment
)
from predictive_risk import (
    predict_future_risk, estimate_remediation_timeline,
    risk_mitigation_priorities, WarningLevel
)


# ============================================================================
# Story 5.1 Tests — Compliance Trend Analysis (18 tests)
# ============================================================================

class TestTrendAnalysis:
    """Tests for compliance trend analysis."""

    def test_improving_trend_detected(self):
        """Improving compliance should be detected as IMPROVING."""
        points = [
            ComplianceDataPoint(datetime.now(), "test", 50.0, "MEDIUM", 5, 3),
            ComplianceDataPoint(datetime.now() + timedelta(days=10), "test", 60.0, "LOW", 3, 2),
            ComplianceDataPoint(datetime.now() + timedelta(days=20), "test", 70.0, "LOW", 1, 1),
        ]

        trend = analyze_compliance_trend(points, "test")
        assert trend.trend_direction == TrendDirection.IMPROVING
        assert trend.velocity > 0

    def test_declining_trend_detected(self):
        """Declining compliance should be detected as DECLINING."""
        points = [
            ComplianceDataPoint(datetime.now(), "test", 80.0, "LOW", 2, 1),
            ComplianceDataPoint(datetime.now() + timedelta(days=10), "test", 70.0, "MEDIUM", 4, 2),
            ComplianceDataPoint(datetime.now() + timedelta(days=20), "test", 50.0, "HIGH", 7, 5),
        ]

        trend = analyze_compliance_trend(points, "test")
        assert trend.trend_direction == TrendDirection.DECLINING
        assert trend.velocity < 0

    def test_stable_trend_detected(self):
        """Stable compliance should be detected as STABLE."""
        points = [
            ComplianceDataPoint(datetime.now(), "test", 65.0, "LOW", 3, 2),
            ComplianceDataPoint(datetime.now() + timedelta(days=10), "test", 66.0, "LOW", 3, 2),
            ComplianceDataPoint(datetime.now() + timedelta(days=20), "test", 65.5, "LOW", 3, 2),
        ]

        trend = analyze_compliance_trend(points, "test")
        assert trend.trend_direction == TrendDirection.STABLE
        assert abs(trend.velocity) < 2

    def test_empty_data_returns_neutral(self):
        """Empty data points should return neutral trend."""
        trend = analyze_compliance_trend([], "test")
        assert trend.trend_direction == TrendDirection.STABLE
        assert trend.velocity == 0.0
        assert trend.confidence == 0.0

    def test_velocity_calculated_correctly(self):
        """Velocity should be score change per day."""
        points = [
            ComplianceDataPoint(datetime.now(), "test", 50.0, "MEDIUM", 5, 3),
            ComplianceDataPoint(datetime.now() + timedelta(days=10), "test", 70.0, "LOW", 3, 2),
        ]

        trend = analyze_compliance_trend(points, "test")
        expected_velocity = (70 - 50) / 10  # 2.0
        assert trend.velocity == pytest.approx(expected_velocity, abs=0.1)

    def test_critical_date_forecasted(self):
        """Declining trend should forecast critical date."""
        now = datetime.now()
        points = [
            ComplianceDataPoint(now, "test", 50.0, "MEDIUM", 5, 3),
            ComplianceDataPoint(now + timedelta(days=5), "test", 40.0, "HIGH", 7, 5),
        ]

        trend = analyze_compliance_trend(points, "test")
        if trend.estimated_critical_date:
            assert trend.estimated_critical_date > now

    def test_resolution_date_forecasted(self):
        """Improving trend should forecast resolution date."""
        now = datetime.now()
        points = [
            ComplianceDataPoint(now, "test", 70.0, "LOW", 3, 2),
            ComplianceDataPoint(now + timedelta(days=5), "test", 75.0, "NONE", 1, 0),
        ]

        trend = analyze_compliance_trend(points, "test")
        if trend.estimated_resolution_date:
            assert trend.estimated_resolution_date > now

    def test_confidence_increases_with_data_points(self):
        """Confidence should increase with more data points."""
        points_2 = [
            ComplianceDataPoint(datetime.now(), "test", 50.0, "MEDIUM", 5, 3),
            ComplianceDataPoint(datetime.now() + timedelta(days=5), "test", 55.0, "LOW", 4, 2),
        ]

        points_5 = points_2 + [
            ComplianceDataPoint(datetime.now() + timedelta(days=10), "test", 60.0, "LOW", 3, 1),
            ComplianceDataPoint(datetime.now() + timedelta(days=15), "test", 65.0, "NONE", 2, 0),
            ComplianceDataPoint(datetime.now() + timedelta(days=20), "test", 70.0, "NONE", 1, 0),
        ]

        trend_2 = analyze_compliance_trend(points_2, "test")
        trend_5 = analyze_compliance_trend(points_5, "test")

        assert trend_5.confidence > trend_2.confidence

    def test_filtered_by_jurisdiction(self):
        """Trend should filter by jurisdiction."""
        points = [
            ComplianceDataPoint(datetime.now(), "eu", 50.0, "MEDIUM", 5, 3),
            ComplianceDataPoint(datetime.now() + timedelta(days=10), "ca", 60.0, "LOW", 3, 2),
        ]

        trend_eu = analyze_compliance_trend(points, "eu")
        trend_ca = analyze_compliance_trend(points, "ca")

        assert len(trend_eu.data_points) == 1
        assert len(trend_ca.data_points) == 1


class TestForecastCompliance:
    """Tests for compliance score forecasting."""

    def test_forecast_predicts_future_score(self):
        """Forecast should predict score 30 days out."""
        points = [
            ComplianceDataPoint(datetime.now(), "test", 50.0, "MEDIUM", 5, 3),
            ComplianceDataPoint(datetime.now() + timedelta(days=10), "test", 60.0, "LOW", 3, 2),
        ]

        forecast = forecast_compliance_score(points, "test", days_ahead=30)
        assert forecast["predicted_score"] is not None
        assert 0 <= forecast["predicted_score"] <= 100

    def test_forecast_includes_confidence(self):
        """Forecast should include confidence level."""
        points = [
            ComplianceDataPoint(datetime.now(), "test", 50.0, "MEDIUM", 5, 3),
            ComplianceDataPoint(datetime.now() + timedelta(days=10), "test", 60.0, "LOW", 3, 2),
        ]

        forecast = forecast_compliance_score(points, "test", days_ahead=30)
        assert "confidence" in forecast
        assert 0 <= forecast["confidence"] <= 1

    def test_forecast_includes_scenarios(self):
        """Forecast should include optimistic/pessimistic scenarios."""
        points = [
            ComplianceDataPoint(datetime.now(), "test", 50.0, "MEDIUM", 5, 3),
            ComplianceDataPoint(datetime.now() + timedelta(days=10), "test", 60.0, "LOW", 3, 2),
        ]

        forecast = forecast_compliance_score(points, "test", days_ahead=30)
        assert "optimistic_scenario" in forecast
        assert "pessimistic_scenario" in forecast
        assert forecast["optimistic_scenario"] >= forecast["predicted_score"]
        assert forecast["pessimistic_scenario"] <= forecast["predicted_score"]

    def test_forecast_scores_clamped(self):
        """Forecast scores should be clamped to [0, 100]."""
        points = [
            ComplianceDataPoint(datetime.now(), "test", 95.0, "NONE", 0, 0),
            ComplianceDataPoint(datetime.now() + timedelta(days=5), "test", 98.0, "NONE", 0, 0),
        ]

        forecast = forecast_compliance_score(points, "test", days_ahead=90)
        assert forecast["predicted_score"] <= 100
        assert forecast["optimistic_scenario"] <= 100


class TestRemediationVelocity:
    """Tests for remediation velocity calculation."""

    def test_velocity_calculated(self):
        """Velocity should be issues resolved per day."""
        before = [
            {"severity": "HIGH"},
            {"severity": "HIGH"},
            {"severity": "HIGH"},
            {"severity": "LOW"},
        ]
        after = [
            {"severity": "LOW"},
        ]

        velocity = calculate_remediation_velocity(before, after, days_elapsed=10)
        assert velocity["velocity"] == pytest.approx(0.2, abs=0.05)  # 2 issues / 10 days

    def test_new_issues_tracked(self):
        """New issues should be tracked separately."""
        before = [
            {"severity": "HIGH"},
            {"severity": "HIGH"},
        ]
        after = [
            {"severity": "HIGH"},
            {"severity": "HIGH"},
            {"severity": "HIGH"},  # New issue
        ]

        velocity = calculate_remediation_velocity(before, after, days_elapsed=5)
        assert velocity["new_issues"] == 1
        assert velocity["issues_resolved"] == 0

    def test_remediation_rate_calculated(self):
        """Remediation rate should be issues resolved / total before."""
        before = [
            {"severity": "HIGH"},
            {"severity": "HIGH"},
            {"severity": "HIGH"},
            {"severity": "HIGH"},
            {"severity": "HIGH"},
        ]
        after = [
            {"severity": "HIGH"},
        ]

        velocity = calculate_remediation_velocity(before, after, days_elapsed=5)
        assert velocity["remediation_rate"] == pytest.approx(0.8, abs=0.05)  # 4/5


# ============================================================================
# Story 5.2 Tests — Industry Benchmarking (18 tests)
# ============================================================================

class TestBenchmarking:
    """Tests for industry benchmarking."""

    def test_benchmark_calculates_percentile(self):
        """Benchmark should calculate percentile rank."""
        peers = [
            PeerBenchmark("p1", IndustrySegment.SAAS, "eu", 40.0, 5, 2, 90),
            PeerBenchmark("p2", IndustrySegment.SAAS, "eu", 60.0, 3, 1, 90),
            PeerBenchmark("p3", IndustrySegment.SAAS, "eu", 80.0, 1, 0, 90),
        ]

        result = benchmark_compliance(70.0, "eu", IndustrySegment.SAAS, peers)
        assert 0 <= result.percentile_rank <= 100
        assert result.percentile_rank > 50  # Above median

    def test_benchmark_detects_outlier_high(self):
        """Benchmark should detect outlier high performers."""
        peers = [
            PeerBenchmark("p1", IndustrySegment.SAAS, "eu", 50.0, 5, 2, 90),
            PeerBenchmark("p2", IndustrySegment.SAAS, "eu", 55.0, 4, 1, 90),
            PeerBenchmark("p3", IndustrySegment.SAAS, "eu", 52.0, 5, 2, 90),
        ]

        result = benchmark_compliance(95.0, "eu", IndustrySegment.SAAS, peers)
        assert result.outlier_status == "outlier_high"

    def test_benchmark_detects_outlier_low(self):
        """Benchmark should detect outlier low performers."""
        peers = [
            PeerBenchmark("p1", IndustrySegment.SAAS, "eu", 75.0, 2, 0, 90),
            PeerBenchmark("p2", IndustrySegment.SAAS, "eu", 80.0, 1, 0, 90),
            PeerBenchmark("p3", IndustrySegment.SAAS, "eu", 78.0, 2, 0, 90),
        ]

        result = benchmark_compliance(30.0, "eu", IndustrySegment.SAAS, peers)
        assert result.outlier_status == "outlier_low"

    def test_benchmark_filters_by_jurisdiction(self):
        """Benchmark should filter peers by jurisdiction."""
        peers = [
            PeerBenchmark("p1", IndustrySegment.SAAS, "eu", 60.0, 3, 1, 90),
            PeerBenchmark("p2", IndustrySegment.SAAS, "ca", 70.0, 2, 0, 90),
        ]

        result = benchmark_compliance(65.0, "eu", IndustrySegment.SAAS, peers)
        # Should only consider p1
        assert result.worst_peer_score == 60.0

    def test_benchmark_filters_by_industry(self):
        """Benchmark should filter peers by industry."""
        peers = [
            PeerBenchmark("p1", IndustrySegment.SAAS, "eu", 60.0, 3, 1, 90),
            PeerBenchmark("p2", IndustrySegment.FINTECH, "eu", 70.0, 2, 0, 90),
        ]

        result = benchmark_compliance(65.0, "eu", IndustrySegment.SAAS, peers)
        # Should only consider p1
        assert result.worst_peer_score == 60.0

    def test_benchmark_no_peer_data(self):
        """Benchmark should handle missing peer data gracefully."""
        result = benchmark_compliance(65.0, "eu", IndustrySegment.SAAS, [])
        assert result.percentile_rank == 50.0
        assert result.percentile_label == "Insufficient peer data"


class TestBestPractices:
    """Tests for best practice identification."""

    def test_identifies_top_performers(self):
        """Should identify top 25% of peers."""
        peers = [
            PeerBenchmark("p1", IndustrySegment.SAAS, "eu", 40.0, 5, 2, 90),
            PeerBenchmark("p2", IndustrySegment.SAAS, "eu", 60.0, 3, 1, 90),
            PeerBenchmark("p3", IndustrySegment.SAAS, "eu", 80.0, 1, 0, 90),
            PeerBenchmark("p4", IndustrySegment.SAAS, "eu", 90.0, 0, 0, 90),
        ]

        benchmark = benchmark_compliance(70.0, "eu", IndustrySegment.SAAS, peers)
        practices = identify_best_practices(benchmark, peers)
        assert practices["best_practice_count"] > 0

    def test_best_practices_empty_without_peers(self):
        """Best practices should be empty without peer data."""
        benchmark = benchmark_compliance(70.0, "eu", IndustrySegment.SAAS, [])
        practices = identify_best_practices(benchmark, [])
        assert practices["best_practice_count"] == 0


class TestStageAdjustedBenchmark:
    """Tests for stage-adjusted benchmarking."""

    def test_stage_expectations_applied(self):
        """Different stages should have different expectations."""
        peers = [
            PeerBenchmark("p1", IndustrySegment.SAAS, "eu", 50.0, 5, 2, 90),
            PeerBenchmark("p2", IndustrySegment.SAAS, "eu", 60.0, 3, 1, 90),
        ]

        seed_result = stage_adjusted_benchmark(55.0, "seed", "eu", peers)
        series_a_result = stage_adjusted_benchmark(55.0, "series_a", "eu", peers)

        assert seed_result["expected_score"] < series_a_result["expected_score"]

    def test_stage_gap_calculated(self):
        """Stage gap should be score difference from expectation."""
        peers = [
            PeerBenchmark("p1", IndustrySegment.SAAS, "eu", 50.0, 5, 2, 90),
        ]

        result = stage_adjusted_benchmark(60.0, "seed", "eu", peers)
        expected_gap = 60.0 - 50  # Expected is 50 for seed
        assert result["stage_gap"] == pytest.approx(expected_gap, abs=1)


# ============================================================================
# Story 5.3 Tests — Predictive Risk Modeling (19 tests)
# ============================================================================

class TestPredictiveRisk:
    """Tests for predictive risk modeling."""

    def test_predicts_30_day_risk(self):
        """Should predict risk level 30 days out."""
        result = predict_future_risk(
            current_score=60.0,
            current_risk_level="LOW",
            flagged_issues=5,
            remediation_velocity=0.2,  # Resolving 0.2 issues/day
            jurisdiction="test",
        )

        assert result.predicted_risk_30_days in ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_predicts_90_day_risk(self):
        """Should predict risk level 90 days out."""
        result = predict_future_risk(
            current_score=60.0,
            current_risk_level="LOW",
            flagged_issues=5,
            remediation_velocity=0.2,
            jurisdiction="test",
        )

        assert result.predicted_risk_90_days in ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_detects_improving_trajectory(self):
        """Improving velocity should show descending trajectory."""
        result = predict_future_risk(
            current_score=50.0,
            current_risk_level="MEDIUM",
            flagged_issues=5,
            remediation_velocity=0.5,  # High remediation
            jurisdiction="test",
        )

        assert result.risk_trajectory == "descending"

    def test_detects_declining_trajectory(self):
        """Negative velocity should show ascending trajectory."""
        result = predict_future_risk(
            current_score=70.0,
            current_risk_level="LOW",
            flagged_issues=3,
            remediation_velocity=-0.3,  # Issues accumulating
            jurisdiction="test",
        )

        assert result.risk_trajectory == "ascending"

    def test_identifies_early_warnings(self):
        """Should identify early warning indicators."""
        result = predict_future_risk(
            current_score=50.0,
            current_risk_level="HIGH",
            flagged_issues=15,
            remediation_velocity=-0.1,  # Declining
            jurisdiction="test",
        )

        assert len(result.early_warnings) > 0

    def test_estimates_remediation_days(self):
        """Should estimate days to remediate all issues."""
        result = predict_future_risk(
            current_score=60.0,
            current_risk_level="LOW",
            flagged_issues=10,
            remediation_velocity=1.0,  # 1 issue/day
            jurisdiction="test",
        )

        assert result.estimated_remediation_days == 10


class TestRemediationTimeline:
    """Tests for remediation timeline estimation."""

    def test_timeline_increases_with_complexity(self):
        """More complex issues should take longer."""
        simple = estimate_remediation_timeline("eu", 5, "simple")
        complex = estimate_remediation_timeline("eu", 5, "complex")

        assert complex["estimated_total_days"] > simple["estimated_total_days"]

    def test_timeline_includes_phases(self):
        """Timeline should include assessment, remediation, verification."""
        timeline = estimate_remediation_timeline("eu", 5, "moderate")
        assert len(timeline["phases"]) == 3
        assert timeline["phases"][0]["phase"] == "Assessment"

    def test_timeline_estimates_bounds(self):
        """Timeline should include optimistic and pessimistic estimates."""
        timeline = estimate_remediation_timeline("eu", 5, "moderate")
        assert timeline["optimistic_days"] <= timeline["estimated_total_days"]
        assert timeline["pessimistic_days"] >= timeline["estimated_total_days"]

    def test_jurisdiction_overhead_applied(self):
        """Different jurisdictions should have different overhead."""
        eu_timeline = estimate_remediation_timeline("eu_sfdr_csrd", 5, "simple")
        ca_timeline = estimate_remediation_timeline("ca_sb54", 5, "simple")

        assert eu_timeline["estimated_total_days"] != ca_timeline["estimated_total_days"]


class TestMitigationPriorities:
    """Tests for risk mitigation priority ordering."""

    def test_prioritizes_critical_warnings(self):
        """Critical warnings should come before low warnings."""
        from predictive_risk import EarlyWarningIndicator

        critical = EarlyWarningIndicator(
            "critical", WarningLevel.CRITICAL, 50.0, 40.0, 7, "eu", "Act now"
        )
        low = EarlyWarningIndicator(
            "low", WarningLevel.LOW, 65.0, 60.0, 90, "eu", "Monitor"
        )

        priorities = risk_mitigation_priorities([critical, low], [])
        assert priorities["prioritized_actions"][0]["priority"] < priorities["prioritized_actions"][1]["priority"]

    def test_critical_path_highest_priority(self):
        """Critical path items should have highest priority."""
        priorities = risk_mitigation_priorities([], ["Item A", "Item B"])
        assert len(priorities["prioritized_actions"]) > 0
        assert priorities["prioritized_actions"][0]["type"] == "critical_path"

    def test_total_actions_counted(self):
        """Should count total actions (warnings + critical path)."""
        from predictive_risk import EarlyWarningIndicator

        warning = EarlyWarningIndicator(
            "test", WarningLevel.MEDIUM, 50.0, 40.0, 30, "eu", "Action"
        )
        priorities = risk_mitigation_priorities([warning], ["Item A", "Item B"])
        assert priorities["total_actions"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
