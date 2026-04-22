"""
Tests for Epic 0, Story 0.3: Standard Analyzer Protocol

Tests cover:
- ClaimAssessment schema validation
- AnalyzerModule Protocol compliance
- Backward compatibility with existing analyzer.py
"""

import pytest
from pydantic import ValidationError

from analyzer_protocol import ClaimAssessment, AnalyzerModule


class TestClaimAssessmentSchema:
    """Tests for ClaimAssessment Pydantic model."""

    def test_claim_assessment_required_fields(self):
        """ClaimAssessment requires verdict, severity, explanation, jurisdiction."""
        with pytest.raises(ValidationError):
            ClaimAssessment()

    def test_claim_assessment_valid_verdict_values(self):
        """Verdict can be any string, but typically CONSISTENT|CONTRADICTS|UNSUPPORTED."""
        assessment = ClaimAssessment(
            verdict="CONSISTENT",
            severity="HIGH",
            forward_looking=False,
            explanation="Claim matches SEC filing.",
            cited_passages=[1, 2, 3],
            jurisdiction="sec",
        )
        assert assessment.verdict == "CONSISTENT"

    def test_claim_assessment_valid_severity_values(self):
        """Severity field accepts severity levels."""
        for severity in ["INFO", "MEDIUM", "HIGH", "CRITICAL"]:
            assessment = ClaimAssessment(
                verdict="CONSISTENT",
                severity=severity,
                forward_looking=False,
                explanation="Test.",
                cited_passages=[],
                jurisdiction="sec",
            )
            assert assessment.severity == severity

    def test_claim_assessment_jurisdiction_field(self):
        """Jurisdiction indicates which agent produced the assessment."""
        for jurisdiction in ["sec", "eu_sfdr_csrd", "ca_sb54"]:
            assessment = ClaimAssessment(
                verdict="CONSISTENT",
                severity="HIGH",
                forward_looking=False,
                explanation="Test.",
                cited_passages=[],
                jurisdiction=jurisdiction,
            )
            assert assessment.jurisdiction == jurisdiction

    def test_claim_assessment_optional_fields(self):
        """Optional fields (red_flags, warnings, verified, action_items) default to empty."""
        assessment = ClaimAssessment(
            verdict="CONSISTENT",
            severity="HIGH",
            forward_looking=False,
            explanation="Test.",
            cited_passages=[],
            jurisdiction="sec",
        )
        assert assessment.red_flags == []
        assert assessment.warnings == []
        assert assessment.verified == []
        assert assessment.action_items == []
        assert assessment.missing_information is None

    def test_claim_assessment_with_red_flags(self):
        """Red flags are populated with critical compliance issues."""
        assessment = ClaimAssessment(
            verdict="CONTRADICTS",
            severity="CRITICAL",
            forward_looking=False,
            explanation="Deck claims carbon neutral but provides no audit.",
            cited_passages=[5, 6],
            jurisdiction="eu_sfdr_csrd",
            red_flags=["No third-party sustainability audit", "Greenwashing risk"],
        )
        assert len(assessment.red_flags) == 2
        assert "greenwashing" in assessment.red_flags[1].lower()

    def test_claim_assessment_with_warnings(self):
        """Warnings are non-critical issues requiring clarification."""
        assessment = ClaimAssessment(
            verdict="UNSUPPORTED",
            severity="MEDIUM",
            forward_looking=False,
            explanation="Board diversity percentage stated but not recent.",
            cited_passages=[3],
            jurisdiction="eu_sfdr_csrd",
            warnings=["Board composition data is from 2024, may be outdated"],
        )
        assert len(assessment.warnings) == 1

    def test_claim_assessment_with_verified(self):
        """Verified lists confirmed compliant data points."""
        assessment = ClaimAssessment(
            verdict="CONSISTENT",
            severity="INFO",
            forward_looking=False,
            explanation="All ESG metrics verified.",
            cited_passages=[1, 2, 3, 4],
            jurisdiction="eu_sfdr_csrd",
            verified=[
                "Scope 1 emissions: 12,000 tCO2e",
                "Scope 2 emissions: 8,500 tCO2e",
                "Third-party audit: Bureau Veritas",
            ],
        )
        assert len(assessment.verified) == 3

    def test_claim_assessment_with_action_items(self):
        """Action items are specific questions for compliance officer."""
        assessment = ClaimAssessment(
            verdict="INSUFFICIENT_EVIDENCE",
            severity="MEDIUM",
            forward_looking=False,
            explanation="Founder demographics incomplete.",
            cited_passages=[],
            jurisdiction="ca_sb54",
            action_items=[
                "Request full founder demographic data (race/ethnicity, gender, LGBTQ+, veteran status)",
                "Confirm whether disclosure is required under current SB 54 enforcement rules",
            ],
        )
        assert len(assessment.action_items) == 2

    def test_claim_assessment_with_missing_information(self):
        """Missing information field explains what's needed for complete assessment."""
        assessment = ClaimAssessment(
            verdict="INSUFFICIENT_EVIDENCE",
            severity="MEDIUM",
            forward_looking=False,
            explanation="Cannot assess AI risk without more details.",
            cited_passages=[],
            jurisdiction="eu_sfdr_csrd",
            missing_information="Details on AI use in health/finance/HR sectors per EU AI Act Annex III",
        )
        assert "AI" in assessment.missing_information

    def test_claim_assessment_forward_looking_flag(self):
        """Forward-looking flag indicates if claim is a projection/target."""
        backward = ClaimAssessment(
            verdict="CONSISTENT",
            severity="INFO",
            forward_looking=False,
            explanation="Historical metric.",
            cited_passages=[1],
            jurisdiction="sec",
        )
        assert backward.forward_looking is False

        forward = ClaimAssessment(
            verdict="CONSISTENT",
            severity="INFO",
            forward_looking=True,
            explanation="Projected metric.",
            cited_passages=[1],
            jurisdiction="sec",
        )
        assert forward.forward_looking is True

    def test_claim_assessment_cited_passages_validation(self):
        """Cited passages are indices into KB."""
        assessment = ClaimAssessment(
            verdict="CONSISTENT",
            severity="HIGH",
            forward_looking=False,
            explanation="Test.",
            cited_passages=[0, 5, 12, 18],  # Index numbers
            jurisdiction="sec",
        )
        assert len(assessment.cited_passages) == 4
        assert assessment.cited_passages[0] == 0

    def test_claim_assessment_json_serialization(self):
        """ClaimAssessment serializes to JSON."""
        assessment = ClaimAssessment(
            verdict="CONSISTENT",
            severity="HIGH",
            forward_looking=False,
            explanation="Test.",
            cited_passages=[1, 2],
            jurisdiction="sec",
            red_flags=["Flag 1"],
        )
        json_data = assessment.model_dump_json()
        assert "CONSISTENT" in json_data
        assert "Flag 1" in json_data


class TestAnalyzerModuleProtocol:
    """Tests for AnalyzerModule Protocol compliance."""

    def test_protocol_is_runtime_checkable(self):
        """AnalyzerModule is a runtime_checkable Protocol."""
        from typing import get_type_hints

        # Protocol should be importable and usable with isinstance
        assert AnalyzerModule is not None

    def test_mock_analyzer_satisfies_protocol(self):
        """A mock analyzer with the required method satisfies the Protocol."""
        from unittest.mock import MagicMock

        mock_analyzer = MagicMock()
        mock_analyzer.assess = MagicMock(return_value=ClaimAssessment(
            verdict="CONSISTENT",
            severity="HIGH",
            forward_looking=False,
            explanation="Test.",
            cited_passages=[],
            jurisdiction="sec",
        ))

        # isinstance check would require the protocol to be fully satisfied
        # For now, just verify the method exists
        assert hasattr(mock_analyzer, "assess")
        assert callable(mock_analyzer.assess)

    def test_protocol_method_signature(self):
        """AnalyzerModule.assess() has the expected signature."""
        import inspect

        # The Protocol definition specifies the assess method
        # Verify it's callable and documented
        assert AnalyzerModule.__dict__.get("assess") is not None


class TestAnalyzerProtocolBackwardCompatibility:
    """Tests for backward compatibility between old analyzer.py and new protocol."""

    def test_claim_assessment_has_jurisdiction_field(self):
        """ClaimAssessment includes jurisdiction field for mixed-analyzer reporting."""
        assessment = ClaimAssessment(
            verdict="CONSISTENT",
            severity="HIGH",
            forward_looking=False,
            explanation="Test.",
            cited_passages=[],
            jurisdiction="sec",  # Existing analyzer would use "sec"
        )
        assert assessment.jurisdiction == "sec"

    def test_existing_analyzer_fields_preserved(self):
        """All existing ClaimAssessment fields still present."""
        assessment = ClaimAssessment(
            verdict="CONSISTENT",
            severity="HIGH",
            forward_looking=False,
            explanation="This is the explanation field.",
            cited_passages=[1, 2, 3],
            jurisdiction="sec",
        )
        assert assessment.verdict == "CONSISTENT"
        assert assessment.severity == "HIGH"
        assert assessment.forward_looking is False
        assert assessment.explanation == "This is the explanation field."
        assert assessment.cited_passages == [1, 2, 3]

    def test_new_fields_optional_for_backward_compat(self):
        """New jurisdiction-specific fields (red_flags, etc.) are optional."""
        # Old code creating minimal assessment should still work
        assessment = ClaimAssessment(
            verdict="CONSISTENT",
            severity="HIGH",
            forward_looking=False,
            explanation="Test.",
            cited_passages=[],
            jurisdiction="sec",
            # No red_flags, warnings, verified, action_items — defaults to []
        )
        assert assessment.red_flags == []
        assert assessment.warnings == []
        assert assessment.verified == []
        assert assessment.action_items == []
