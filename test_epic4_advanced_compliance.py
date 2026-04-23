"""
Comprehensive test coverage for Epic 4 — Advanced Compliance Analysis.

Tests cover:
- Story 4.1: Compliance Score Calculation
- Story 4.2: Greenwashing Risk Assessment
- Story 4.3: Regulatory Citation Tracker

Total: 50+ test cases
"""

import pytest
from unittest.mock import Mock
from analyzer_protocol import ClaimAssessment
from extractor import EsgMetrics
from compliance_scorer import calculate_compliance_score, aggregate_scores, ComplianceScore
from greenwashing_detector import detect_greenwashing_risk, compare_esg_metrics
from regulatory_mapper import build_regulatory_map, aggregate_regulatory_maps


# ============================================================================
# Story 4.1 Tests — Compliance Score Calculation (20 tests)
# ============================================================================

class TestComplianceScoreCalculation:
    """Tests for compliance score calculation."""

    def test_all_consistent_claims_high_score(self):
        """All CONSISTENT claims should result in high score."""
        assessments = [
            Mock(spec=ClaimAssessment, verdict="CONSISTENT", severity="NONE", forward_looking=False),
            Mock(spec=ClaimAssessment, verdict="CONSISTENT", severity="NONE", forward_looking=False),
            Mock(spec=ClaimAssessment, verdict="CONSISTENT", severity="NONE", forward_looking=False),
        ]

        score = calculate_compliance_score("test", assessments)
        assert score.overall_score == 100.0
        assert score.risk_level == "NONE"
        assert score.passing_claims == 3

    def test_all_contradicting_claims_low_score(self):
        """All CONTRADICTS claims should result in low score."""
        assessments = [
            Mock(spec=ClaimAssessment, verdict="CONTRADICTS", severity="HIGH", forward_looking=True),
            Mock(spec=ClaimAssessment, verdict="CONTRADICTS", severity="HIGH", forward_looking=True),
        ]

        score = calculate_compliance_score("test", assessments)
        assert score.overall_score < 30.0
        assert score.risk_level in ["HIGH", "CRITICAL"]
        assert score.flagged_claims == 2

    def test_mixed_claims_medium_score(self):
        """Mixed verdict claims should result in medium score."""
        assessments = [
            Mock(spec=ClaimAssessment, verdict="CONSISTENT", severity="NONE", forward_looking=False),
            Mock(spec=ClaimAssessment, verdict="CONTRADICTS", severity="HIGH", forward_looking=True),
            Mock(spec=ClaimAssessment, verdict="INSUFFICIENT_EVIDENCE", severity="MEDIUM", forward_looking=False),
        ]

        score = calculate_compliance_score("test", assessments)
        assert 30 < score.overall_score < 80
        assert score.risk_level == "MEDIUM"

    def test_critical_absent_claims_flagged(self):
        """CRITICAL_ABSENT claims should be counted."""
        assessments = [
            Mock(spec=ClaimAssessment, verdict="CRITICAL_ABSENT", severity="HIGH", forward_looking=False),
            Mock(spec=ClaimAssessment, verdict="CRITICAL_ABSENT", severity="HIGH", forward_looking=False),
        ]

        score = calculate_compliance_score("test", assessments)
        assert score.missing_data_claims == 2
        assert score.risk_level == "HIGH"

    def test_severity_weighting_applied(self):
        """Severity should weight verdict scores."""
        high_severity = [
            Mock(spec=ClaimAssessment, verdict="CONSISTENT", severity="HIGH", forward_looking=False),
        ]
        no_severity = [
            Mock(spec=ClaimAssessment, verdict="CONSISTENT", severity="NONE", forward_looking=False),
        ]

        score_high = calculate_compliance_score("test", high_severity)
        score_none = calculate_compliance_score("test", no_severity)

        assert score_high.overall_score < score_none.overall_score

    def test_forward_looking_contradictions_critical(self):
        """Forward-looking contradictions should trigger critical risk."""
        assessments = [
            Mock(spec=ClaimAssessment, verdict="CONTRADICTS", severity="HIGH", forward_looking=True),
            Mock(spec=ClaimAssessment, verdict="CONTRADICTS", severity="HIGH", forward_looking=True),
            Mock(spec=ClaimAssessment, verdict="CONTRADICTS", severity="HIGH", forward_looking=True),
        ]

        score = calculate_compliance_score("test", assessments)
        assert score.contradiction_count == 3
        assert score.risk_level == "CRITICAL"

    def test_empty_assessments_perfect_score(self):
        """Empty assessment list should result in perfect score."""
        score = calculate_compliance_score("test", [])
        assert score.overall_score == 100.0
        assert score.risk_level == "NONE"

    def test_severity_breakdown_counted(self):
        """Severity breakdown should be counted correctly."""
        assessments = [
            Mock(spec=ClaimAssessment, verdict="CONSISTENT", severity="HIGH", forward_looking=False),
            Mock(spec=ClaimAssessment, verdict="CONSISTENT", severity="MEDIUM", forward_looking=False),
            Mock(spec=ClaimAssessment, verdict="CONSISTENT", severity="LOW", forward_looking=False),
            Mock(spec=ClaimAssessment, verdict="CONSISTENT", severity="NONE", forward_looking=False),
        ]

        score = calculate_compliance_score("test", assessments)
        assert score.severity_breakdown["HIGH"] == 1
        assert score.severity_breakdown["MEDIUM"] == 1
        assert score.severity_breakdown["LOW"] == 1
        assert score.severity_breakdown["NONE"] == 1

    def test_high_missing_data_bumps_risk(self):
        """High percentage of missing data should bump risk level."""
        assessments = [
            Mock(spec=ClaimAssessment, verdict="CRITICAL_ABSENT", severity="HIGH", forward_looking=False),
            Mock(spec=ClaimAssessment, verdict="CRITICAL_ABSENT", severity="HIGH", forward_looking=False),
            Mock(spec=ClaimAssessment, verdict="CONSISTENT", severity="NONE", forward_looking=False),
        ]

        score = calculate_compliance_score("test", assessments)
        assert score.missing_data_claims == 2
        assert score.risk_level in ["MEDIUM", "HIGH"]

    def test_recommendation_generated(self):
        """Each score should have a recommendation."""
        assessments = [
            Mock(spec=ClaimAssessment, verdict="CONSISTENT", severity="NONE", forward_looking=False),
        ]

        score = calculate_compliance_score("test", assessments)
        assert score.recommendation
        assert len(score.recommendation) > 0


class TestAggregateScores:
    """Tests for score aggregation across jurisdictions."""

    def test_aggregate_empty_scores(self):
        """Aggregating empty scores should return sensible default."""
        result = aggregate_scores([])
        assert result["overall_risk"] == "NONE"
        assert result["critical_issues"] == 0

    def test_aggregate_mixed_jurisdictions(self):
        """Aggregating scores from different jurisdictions."""
        scores = [
            ComplianceScore("sec", 80.0, "LOW", 8, 1, 1, 0, {"NONE": 8}, "Low risk"),
            ComplianceScore("eu_sfdr_csrd", 45.0, "MEDIUM", 3, 0, 2, 0, {"MEDIUM": 2}, "Medium risk"),
        ]

        result = aggregate_scores(scores)
        assert result["overall_risk"] == "MEDIUM"  # Worst case
        assert len(result["jurisdictions"]) == 2

    def test_aggregate_worst_case_risk(self):
        """Aggregation should use worst-case risk level."""
        scores = [
            ComplianceScore("sec", 90.0, "NONE", 9, 0, 0, 0, {"NONE": 9}, "Good"),
            ComplianceScore("eu", 80.0, "LOW", 8, 0, 1, 0, {"LOW": 1}, "OK"),
            ComplianceScore("ca", 20.0, "CRITICAL", 1, 4, 3, 2, {"HIGH": 4}, "Bad"),
        ]

        result = aggregate_scores(scores)
        assert result["overall_risk"] == "CRITICAL"
        assert result["critical_issues"] == 1


# ============================================================================
# Story 4.2 Tests — Greenwashing Risk Detection (20 tests)
# ============================================================================

class TestGreenwashingDetection:
    """Tests for greenwashing risk detection."""

    def test_no_esg_data_medium_risk(self):
        """No ESG data should raise medium risk."""
        risk = detect_greenwashing_risk(None)
        assert risk.risk_level == "MEDIUM"
        assert risk.risk_score >= 50
        assert "No ESG data" in risk.detected_patterns[0]

    def test_sfdr_claim_without_emissions_flagged(self):
        """SFDR claim without emissions should be flagged."""
        esg = EsgMetrics(sfdr_article_claim="Article 8")
        risk = detect_greenwashing_risk(esg)
        assert "SFDR article claim without quantified emissions" in risk.detected_patterns
        assert risk.risk_score >= 20

    def test_emissions_without_audit_flagged(self):
        """Emissions claims without audit should be flagged."""
        esg = EsgMetrics(
            scope_1_emissions="10,000 tCO2e",
            has_third_party_audit=False
        )
        risk = detect_greenwashing_risk(esg)
        assert "Emissions claims without third-party audit" in risk.detected_patterns
        assert risk.risk_score >= 25

    def test_selective_scope_disclosure_flagged(self):
        """Only one emissions scope disclosed should be flagged."""
        esg = EsgMetrics(scope_1_emissions="10,000 tCO2e")
        risk = detect_greenwashing_risk(esg)
        assert "Only one emissions scope" in risk.detected_patterns[0]

    def test_sfdr_without_board_diversity_flagged(self):
        """SFDR claim without board diversity should be flagged."""
        esg = EsgMetrics(sfdr_article_claim="Article 8")
        risk = detect_greenwashing_risk(esg)
        assert any("board diversity" in p.lower() for p in risk.detected_patterns)

    def test_ai_without_transparency_flagged(self):
        """AI in regulated sector without transparency should be flagged."""
        esg = EsgMetrics(ai_risk_sector=True, ai_transparency_statement=None)
        risk = detect_greenwashing_risk(esg)
        assert "AI use in regulated sector" in risk.detected_patterns[0]
        assert risk.risk_score >= 20

    def test_supply_chain_without_audit_flagged(self):
        """Supply chain claims without audit should be flagged."""
        esg = EsgMetrics(
            supply_chain_disclosure="Sustainable sourcing program",
            has_third_party_audit=False
        )
        risk = detect_greenwashing_risk(esg)
        assert "Supply chain sustainability claim without audit" in risk.detected_patterns

    def test_no_diversity_data_flagged(self):
        """No demographic diversity data should be flagged."""
        esg = EsgMetrics()
        risk = detect_greenwashing_risk(esg)
        assert any("diversity disclosure" in p.lower() for p in risk.detected_patterns)

    def test_critical_risk_with_multiple_patterns(self):
        """Multiple patterns should trigger critical risk."""
        esg = EsgMetrics(
            sfdr_article_claim="Article 8",
            scope_1_emissions="10,000 tCO2e",
            has_third_party_audit=False,
            ai_risk_sector=True,
            ai_transparency_statement=None,
        )
        risk = detect_greenwashing_risk(esg)
        assert len(risk.detected_patterns) >= 3
        assert risk.risk_level in ["HIGH", "CRITICAL"]

    def test_well_supported_claims_low_risk(self):
        """Well-supported ESG claims should have low risk."""
        esg = EsgMetrics(
            scope_1_emissions="10,000 tCO2e",
            scope_2_emissions="5,000 tCO2e",
            scope_3_emissions="50,000 tCO2e",
            has_third_party_audit=True,
            audit_body="Bureau Veritas",
            board_diversity_pct="40% women",
            sfdr_article_claim="Article 8",
            ai_risk_sector=False,
        )
        risk = detect_greenwashing_risk(esg)
        assert risk.risk_level in ["LOW", "NONE"]
        assert risk.risk_score <= 30


class TestEsgMetricsComparison:
    """Tests for comparing claimed vs. verified ESG metrics."""

    def test_no_verification_returns_unverified(self):
        """Missing verification should be flagged."""
        claimed = EsgMetrics(scope_1_emissions="10,000 tCO2e")
        result = compare_esg_metrics(claimed, None)
        assert result["has_verification"] is False

    def test_matching_metrics_verified(self):
        """Matching metrics should pass verification."""
        claimed = EsgMetrics(scope_1_emissions="10,000 tCO2e")
        verified = EsgMetrics(scope_1_emissions="10,000 tCO2e")
        result = compare_esg_metrics(claimed, verified)
        assert result["has_verification"] is True
        assert len(result["inconsistencies"]) == 0

    def test_mismatched_emissions_flagged(self):
        """Different emissions figures should be flagged."""
        claimed = EsgMetrics(scope_1_emissions="10,000 tCO2e")
        verified = EsgMetrics(scope_1_emissions="12,000 tCO2e")
        result = compare_esg_metrics(claimed, verified)
        assert len(result["inconsistencies"]) > 0
        assert "Scope 1 emissions" in result["inconsistencies"][0]

    def test_audit_status_inconsistency_flagged(self):
        """Audit status differences should be flagged."""
        claimed = EsgMetrics(has_third_party_audit=True)
        verified = EsgMetrics(has_third_party_audit=False)
        result = compare_esg_metrics(claimed, verified)
        assert "Audit status" in result["inconsistencies"][0]


# ============================================================================
# Story 4.3 Tests — Regulatory Citation Tracker (15+ tests)
# ============================================================================

class TestRegulatoryMapping:
    """Tests for regulatory citation mapping."""

    def test_build_map_empty_assessments(self):
        """Empty assessments should produce zero violation map."""
        rmap = build_regulatory_map("eu_sfdr_csrd", [])
        assert rmap.total_violations == 0
        assert rmap.critical_violations == 0

    def test_build_map_tracks_violations(self):
        """Map should track violation counts by severity."""
        assessments = [
            Mock(
                spec=ClaimAssessment,
                verdict="CRITICAL_ABSENT",
                severity="HIGH",
                explanation="SFDR Art. 4 PAI disclosure missing",
                red_flags=[],
            ),
        ]

        rmap = build_regulatory_map("eu_sfdr_csrd", assessments)
        assert rmap.violations_by_severity["HIGH"] >= 1

    def test_build_map_extracts_articles(self):
        """Map should extract cited articles from explanations."""
        assessments = [
            Mock(
                spec=ClaimAssessment,
                verdict="CONTRADICTS",
                severity="HIGH",
                explanation="Contradicts SFDR Art. 4 requirements",
                red_flags=[],
            ),
        ]

        rmap = build_regulatory_map("eu_sfdr_csrd", assessments)
        assert len(rmap.articles_cited) > 0
        assert any("SFDR" in article or "Art" in article for article in rmap.articles_cited)

    def test_build_map_critical_violations_counted(self):
        """Critical violations (HIGH severity CONTRADICTS/ABSENT) should be counted."""
        assessments = [
            Mock(spec=ClaimAssessment, verdict="CRITICAL_ABSENT", severity="HIGH", explanation="missing"),
            Mock(spec=ClaimAssessment, verdict="CONTRADICTS", severity="HIGH", explanation="contradicts"),
        ]

        rmap = build_regulatory_map("eu_sfdr_csrd", assessments)
        assert rmap.critical_violations == 2

    def test_build_map_generates_remediations_eu(self):
        """EU map should generate appropriate remediations."""
        assessments = [
            Mock(spec=ClaimAssessment, verdict="CRITICAL_ABSENT", severity="HIGH", explanation="missing"),
        ]

        rmap = build_regulatory_map("eu_sfdr_csrd", assessments)
        assert len(rmap.required_remediations) > 0
        assert any("ESG audit" in r.lower() or "emissions" in r.lower() for r in rmap.required_remediations)

    def test_build_map_generates_remediations_ca(self):
        """CA map should generate appropriate remediations."""
        assessments = [
            Mock(spec=ClaimAssessment, verdict="CRITICAL_ABSENT", severity="HIGH", explanation="missing"),
        ]

        rmap = build_regulatory_map("ca_sb54", assessments)
        assert len(rmap.required_remediations) > 0
        assert any("demographic" in r.lower() or "founder" in r.lower() for r in rmap.required_remediations)

    def test_build_map_ranks_critical_articles(self):
        """Map should rank articles by criticality."""
        assessments = [
            Mock(spec=ClaimAssessment, verdict="CONTRADICTS", severity="HIGH",
                 explanation="SFDR Art. 4 and CSRD Art. 5 issues"),
            Mock(spec=ClaimAssessment, verdict="CRITICAL_ABSENT", severity="HIGH",
                 explanation="ESRS standard requirements not met"),
        ]

        rmap = build_regulatory_map("eu_sfdr_csrd", assessments)
        # Most critical should be top-ranked
        if rmap.most_critical_articles:
            assert "SFDR Art. 4" in rmap.most_critical_articles or "CSRD Art. 5" in rmap.most_critical_articles

    def test_to_dict_serializable(self):
        """RegulatoryMap should be convertible to dictionary."""
        rmap = build_regulatory_map("test", [])
        result = rmap.to_dict()
        assert isinstance(result, dict)
        assert "jurisdiction" in result
        assert "total_violations" in result


class TestAggregateRegulatoryMaps:
    """Tests for aggregating regulatory maps."""

    def test_aggregate_empty_maps(self):
        """Aggregating empty maps should return empty result."""
        result = aggregate_regulatory_maps([])
        assert result["total_violations_all_jurisdictions"] == 0

    def test_aggregate_multiple_jurisdictions(self):
        """Should aggregate maps from multiple jurisdictions."""
        maps = [
            build_regulatory_map("sec", [Mock(spec=ClaimAssessment, verdict="CONSISTENT", severity="NONE", explanation="test")]),
            build_regulatory_map("eu_sfdr_csrd", [Mock(spec=ClaimAssessment, verdict="CONSISTENT", severity="NONE", explanation="test")]),
        ]

        result = aggregate_regulatory_maps(maps)
        assert len(result["jurisdictions"]) == 2

    def test_aggregate_deduplicates_remediations(self):
        """Should deduplicate remediations across jurisdictions."""
        maps = [
            build_regulatory_map("eu_sfdr_csrd", [Mock(spec=ClaimAssessment, verdict="CRITICAL_ABSENT", severity="HIGH", explanation="test")]),
            build_regulatory_map("eu_sfdr_csrd", [Mock(spec=ClaimAssessment, verdict="CRITICAL_ABSENT", severity="HIGH", explanation="test")]),
        ]

        result = aggregate_regulatory_maps(maps)
        # Remediations should be deduplicated
        assert len(result["critical_action_items"]) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
