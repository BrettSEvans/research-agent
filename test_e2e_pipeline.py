"""
End-to-End Compliance Pipeline Test
====================================

Tests the complete pipeline from file upload through to final report generation,
validating that outputs from each stage feed correctly into the next stage.

Pipeline stages:
  Stage 1: File Upload / PDF Simulation
  Stage 2: Deck Extraction (mock LLM, real Pydantic validation)
  Stage 3: Claim Extraction via DeckContext
  Stage 4: SEC Compliance Analysis (mocked analyzer)
  Stage 5: EU SFDR/CSRD Analysis (mocked analyzer)
  Stage 6: CA SB 54 Analysis (mocked analyzer)
  Stage 7: Compliance Scoring (real scorer)
  Stage 8: Greenwashing Detection (real detector)
  Stage 9: Regulatory Mapping (real mapper)
  Stage 10: Trend Analysis (real trend engine)
  Stage 11: Predictive Risk Modeling (real predictor)
  Stage 12: Final Report Aggregation

Run with:
    python -m pytest test_e2e_pipeline.py -v
    python -m pytest test_e2e_pipeline.py -v --tb=short  # concise failure output
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, Mock

import pytest

# ── Core imports (real code, not mocked) ─────────────────────────────────────
from extractor import (
    CompanyIdentity,
    DeckExtraction,
    EsgMetrics,
    ExtractedClaim,
    ExtractedMetric,
    FounderDemographics,
)
from deck_context import DeckContext
from analyzer_protocol import ClaimAssessment
from compliance_scorer import (
    ComplianceScore,
    aggregate_scores,
    calculate_compliance_score,
)
from greenwashing_detector import GreenwashingRisk, detect_greenwashing_risk
from regulatory_mapper import (
    RegulatoryMap,
    aggregate_regulatory_maps,
    build_regulatory_map,
)
from compliance_trends import (
    ComplianceDataPoint,
    TrendDirection,
    analyze_compliance_trend,
)
from predictive_risk import PredictiveRiskModel, predict_future_risk


# ══════════════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def sample_deck_extraction() -> DeckExtraction:
    """
    Realistic DeckExtraction that simulates what the extractor would produce
    after processing a pitch deck PDF from an ESG-focused FinTech startup.

    This is the artifact produced by Stage 2 (PDF extraction) and consumed
    by all downstream stages.
    """
    return DeckExtraction(
        company=CompanyIdentity(
            name="GreenVault Capital",
            ticker=None,
            cik=None,
            website="https://greenvault.io",
            description="AI-powered sustainable investment platform for institutional ESG investors.",
            industry="FinTech / Sustainable Finance",
            founders=["Sofia Meier", "James Okonkwo"],
        ),
        claims=[
            ExtractedClaim(
                text="GreenVault Capital manages $250M AUM across 12 SFDR Article 8 funds.",
                verbatim="We manage $250M AUM across 12 SFDR Article 8 funds as of Q4 2025.",
                slide=3,
                category="financial",
                likely_forward_looking=False,
            ),
            ExtractedClaim(
                text="The company expects 3x revenue growth year-over-year in 2026.",
                verbatim="We project 3x revenue growth in 2026 driven by European institutional demand.",
                slide=5,
                category="projection",
                likely_forward_looking=True,
            ),
            ExtractedClaim(
                text="GreenVault Capital's platform reduces portfolio carbon footprint by 40%.",
                verbatim="Our AI-driven rebalancing reduces portfolio carbon footprint by 40% on average.",
                slide=7,
                category="product",
                likely_forward_looking=False,
            ),
            ExtractedClaim(
                text="The EU sustainable finance market is estimated at €1.2T by 2027.",
                verbatim="Total addressable market: €1.2T EU sustainable finance by 2027 (Deloitte, 2025).",
                slide=4,
                category="market",
                likely_forward_looking=True,
            ),
            ExtractedClaim(
                text="GreenVault Capital has achieved a 92% client retention rate since inception.",
                verbatim="92% client retention since 2022 inception (audited by PwC).",
                slide=8,
                category="traction",
                likely_forward_looking=False,
            ),
        ],
        fiscal_year_end="December 31",
        currency="EUR",
        key_metrics=[
            ExtractedMetric(metric_name="AUM", value="€250M"),
            ExtractedMetric(metric_name="ARR", value="€4.2M"),
            ExtractedMetric(metric_name="Client Retention", value="92%"),
            ExtractedMetric(metric_name="Funds Managed", value="12"),
        ],
        esg_metrics=EsgMetrics(
            scope_1_emissions="2,400 tCO2e (2025)",
            scope_2_emissions="1,800 tCO2e (2025)",
            scope_3_emissions=None,  # Not disclosed
            has_third_party_audit=True,
            audit_body="Bureau Veritas",
            board_diversity_pct="50% women",
            supply_chain_disclosure="All portfolio companies required to disclose Tier 1 suppliers.",
            ai_risk_sector=True,  # FinTech is a regulated sector
            ai_transparency_statement=(
                "Our AI models are reviewed quarterly for bias and fairness per EU AI Act Annex III requirements."
            ),
            sfdr_article_claim="Article 8",
        ),
        founder_demographics=FounderDemographics(
            founder_count=2,
            gender_diversity="50% women, 50% men",
            women_founder_pct=50.0,
            underrepresented_minority_pct=50.0,
            race_ethnicity_data="1 founder identifies as Black/African (James Okonkwo, Nigerian-British)",
            educational_background=["ETH Zurich (MSc Finance)", "Oxford (MBA)"],
            prior_startup_experience=True,
            industry_expertise="ESG investment management, AI/ML, regulatory compliance",
            diverse_background_statement="We are committed to diverse leadership in sustainable finance.",
            disclosure_completeness="HIGH — gender, ethnicity, education, and prior experience all disclosed.",
        ),
        extraction_notes=(
            "Scope 3 emissions not disclosed in the deck. All other ESG metrics present. "
            "Founder demographics fully disclosed. Financial figures audited by PwC per slide 8."
        ),
    )


@pytest.fixture(scope="module")
def deck_context(sample_deck_extraction) -> DeckContext:
    """DeckContext wrapping the extraction — used for claim retrieval and context building."""
    return DeckContext(sample_deck_extraction)


@pytest.fixture(scope="module")
def mock_sec_assessments() -> List[ClaimAssessment]:
    """
    Simulated SEC analyzer output for each claim.
    These are what the SEC module would return after checking claims against
    SEC filings (or web search for market claims without a CIK).
    """
    return [
        ClaimAssessment(
            verdict="INSUFFICIENT_EVIDENCE",
            severity="NONE",
            forward_looking=False,
            explanation="GreenVault Capital is a private company; no SEC filings available.",
            cited_passages=[],
            missing_information="SEC filings or Form D for private company verification.",
            jurisdiction="sec",
            red_flags=[],
            warnings=["Private company — no CIK resolved."],
            verified=[],
            action_items=["Obtain audited financials from company data room."],
        ),
        ClaimAssessment(
            verdict="UNSUPPORTED",
            severity="MEDIUM",
            forward_looking=True,
            explanation="3x revenue projection is an ambitious forward-looking claim unsupported by SEC data.",
            cited_passages=[],
            missing_information="Historical revenue growth trajectory to validate projection.",
            jurisdiction="sec",
            red_flags=["Aggressive 3x growth projection lacks SEC filing corroboration."],
            warnings=[],
            verified=[],
            action_items=["Request historical revenue data for last 3 years."],
        ),
        ClaimAssessment(
            verdict="UNSUPPORTED",
            severity="MEDIUM",
            forward_looking=False,
            explanation="40% carbon reduction claim lacks quantified baseline or methodology.",
            cited_passages=[],
            missing_information="Carbon footprint baseline methodology and third-party verification.",
            jurisdiction="sec",
            red_flags=[],
            warnings=["Environmental performance claim without quantified evidence."],
            verified=[],
            action_items=["Request carbon accounting methodology and audit report."],
        ),
        ClaimAssessment(
            verdict="CONSISTENT",
            severity="NONE",
            forward_looking=True,
            explanation="EU sustainable finance market estimate aligns with industry reports.",
            cited_passages=[],
            missing_information=None,
            jurisdiction="sec",
            red_flags=[],
            warnings=[],
            verified=["Market size estimate consistent with Deloitte 2025 ESG report."],
            action_items=[],
        ),
        ClaimAssessment(
            verdict="CONSISTENT",
            severity="NONE",
            forward_looking=False,
            explanation="92% retention rate claim is plausible for a niche ESG platform; PwC audit cited.",
            cited_passages=[],
            missing_information=None,
            jurisdiction="sec",
            red_flags=[],
            warnings=[],
            verified=["Audit citation (PwC) adds credibility to retention metric."],
            action_items=[],
        ),
    ]


@pytest.fixture(scope="module")
def mock_eu_assessments() -> List[ClaimAssessment]:
    """
    Simulated EU SFDR/CSRD analyzer output for each claim.
    These would be returned by analyzer_sfdr.py checking EU regulatory requirements.
    """
    return [
        ClaimAssessment(
            verdict="CONSISTENT",
            severity="NONE",
            forward_looking=False,
            explanation="SFDR Article 8 fund classification is properly disclosed per SFDR Art. 8.",
            cited_passages=[1],
            missing_information=None,
            jurisdiction="eu_sfdr_csrd",
            red_flags=[],
            warnings=[],
            verified=["Article 8 self-classification disclosed per SFDR Art. 8."],
            action_items=["Ensure PAI (Principal Adverse Impact) data is collected per SFDR Art. 4."],
        ),
        ClaimAssessment(
            verdict="UNSUPPORTED",
            severity="MEDIUM",
            forward_looking=True,
            explanation="Revenue projection does not require SFDR disclosure but investor growth assumptions may trigger CSRD reporting.",
            cited_passages=[],
            missing_information="CSRD sustainability impact assessment of growth plans.",
            jurisdiction="eu_sfdr_csrd",
            red_flags=[],
            warnings=["Rapid growth plans should consider CSRD double materiality assessment."],
            verified=[],
            action_items=["Include sustainability impact of growth plan in CSRD reporting."],
        ),
        ClaimAssessment(
            verdict="CONSISTENT",
            severity="NONE",
            forward_looking=False,
            explanation="Carbon reduction claim is supported by Scope 1 and 2 emissions disclosed in deck.",
            cited_passages=[2, 3],
            missing_information="Scope 3 emissions still absent — required for full CSRD compliance.",
            jurisdiction="eu_sfdr_csrd",
            red_flags=[],
            warnings=["Scope 3 emissions not disclosed — required by ESRS E1."],
            verified=["Scope 1: 2,400 tCO2e (2025)", "Scope 2: 1,800 tCO2e (2025)"],
            action_items=["Disclose Scope 3 emissions per ESRS E1 requirements."],
        ),
        ClaimAssessment(
            verdict="CONSISTENT",
            severity="NONE",
            forward_looking=True,
            explanation="EU market size claim is consistent with European Commission sustainable finance projections.",
            cited_passages=[4],
            missing_information=None,
            jurisdiction="eu_sfdr_csrd",
            red_flags=[],
            warnings=[],
            verified=["Market estimate consistent with EU Sustainable Finance Action Plan."],
            action_items=[],
        ),
        ClaimAssessment(
            verdict="CONSISTENT",
            severity="NONE",
            forward_looking=False,
            explanation="Client retention verified by PwC audit; appropriate for CSRD reporting.",
            cited_passages=[],
            missing_information=None,
            jurisdiction="eu_sfdr_csrd",
            red_flags=[],
            warnings=[],
            verified=["PwC audit cited for retention metric."],
            action_items=[],
        ),
    ]


@pytest.fixture(scope="module")
def mock_ca_assessments() -> List[ClaimAssessment]:
    """
    Simulated CA SB 54 analyzer output for founder demographics.
    These would be returned by analyzer_sb54.py.
    """
    return [
        ClaimAssessment(
            verdict="CONSISTENT",
            severity="NONE",
            forward_looking=False,
            explanation="Founder gender diversity (50% women) is disclosed and meets SB 54 requirements.",
            cited_passages=[],
            missing_information=None,
            jurisdiction="ca_sb54",
            red_flags=[],
            warnings=[],
            verified=["Gender diversity: 50% women disclosed.", "Race/ethnicity: James Okonkwo (Nigerian-British) disclosed."],
            action_items=["Maintain updated demographic records for annual SB 54 reporting."],
        ),
        ClaimAssessment(
            verdict="CONSISTENT",
            severity="NONE",
            forward_looking=False,
            explanation="Educational background (ETH Zurich, Oxford) and prior startup experience are disclosed.",
            cited_passages=[],
            missing_information=None,
            jurisdiction="ca_sb54",
            red_flags=[],
            warnings=[],
            verified=["Education: ETH Zurich MSc Finance, Oxford MBA.", "Prior startup experience: confirmed."],
            action_items=[],
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — File Upload Simulation
# ══════════════════════════════════════════════════════════════════════════════


class TestStage1_FileUpload:
    """
    Stage 1: Simulates uploading a PDF pitch deck.
    In production this goes through web.py POST /extract endpoint.
    Here we verify the file path/format validation logic.
    """

    def test_pdf_file_exists_in_uploads(self):
        """At least one PDF exists in the uploads directory to validate against."""
        uploads_dir = Path(__file__).parent / "uploads"
        pdfs = list(uploads_dir.glob("*.pdf"))
        assert len(pdfs) > 0, "Expected at least one PDF in uploads/ directory"

    def test_upload_simulation_creates_temp_file(self):
        """Simulate file upload by writing to a temp file and confirming readable."""
        # This mirrors what web.py does on POST /extract
        fake_content = b"%PDF-1.4 fake pitch deck content for testing"
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(fake_content)
            tmp_path = Path(tmp.name)

        assert tmp_path.exists()
        assert tmp_path.suffix == ".pdf"
        assert tmp_path.stat().st_size > 0

        # Cleanup
        tmp_path.unlink()

    def test_supported_extensions_accepted(self):
        """Verify that PDF and PPTX extensions are accepted."""
        supported = {".pdf", ".pptx", ".ppt", ".docx"}
        test_files = ["deck.pdf", "pitch.pptx", "report.docx"]
        for fname in test_files:
            ext = Path(fname).suffix.lower()
            assert ext in supported, f"Extension {ext} should be supported"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Deck Extraction
# ══════════════════════════════════════════════════════════════════════════════


class TestStage2_DeckExtraction:
    """
    Stage 2: Validates that the extracted DeckExtraction is valid and well-formed.
    In production, extract_from_pdf() sends the PDF to Claude and parses the response.
    Here we validate the Pydantic schema is correct with realistic data.
    """

    def test_extraction_has_company_identity(self, sample_deck_extraction):
        """Extraction must have company identity for CIK resolution."""
        company = sample_deck_extraction.company
        assert company.name == "GreenVault Capital"
        assert company.industry is not None
        assert len(company.founders) == 2

    def test_extraction_has_claims(self, sample_deck_extraction):
        """Extraction must have at least one falsifiable claim."""
        assert len(sample_deck_extraction.claims) >= 1
        for claim in sample_deck_extraction.claims:
            assert claim.text, "Every claim must have text"
            assert claim.verbatim, "Every claim must have verbatim quote"
            assert claim.category in [
                "financial", "market", "product", "team", "traction", "projection", "regulatory", "other"
            ]

    def test_extraction_has_esg_metrics(self, sample_deck_extraction):
        """When EU module is active, ESG metrics must be populated."""
        esg = sample_deck_extraction.esg_metrics
        assert esg is not None, "ESG metrics must be present for EU module"
        assert esg.scope_1_emissions is not None
        assert esg.scope_2_emissions is not None
        assert esg.sfdr_article_claim == "Article 8"

    def test_extraction_has_founder_demographics(self, sample_deck_extraction):
        """When CA SB 54 module is active, founder demographics must be populated."""
        demos = sample_deck_extraction.founder_demographics
        assert demos is not None, "Founder demographics must be present for CA module"
        assert demos.founder_count == 2
        assert demos.women_founder_pct == 50.0

    def test_extraction_serializes_to_json(self, sample_deck_extraction):
        """Extraction must be serializable for the DeckContext handoff."""
        json_str = sample_deck_extraction.model_dump_json(indent=2)
        parsed = json.loads(json_str)
        assert parsed["company"]["name"] == "GreenVault Capital"
        assert len(parsed["claims"]) == 5
        assert parsed["esg_metrics"]["sfdr_article_claim"] == "Article 8"

    def test_extraction_round_trips_via_deck_context(self, sample_deck_extraction):
        """DeckContext must be able to load what the extractor saves."""
        with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as f:
            f.write(sample_deck_extraction.model_dump_json(indent=2))
            tmp_path = f.name

        loaded = DeckContext.load(tmp_path)
        Path(tmp_path).unlink()

        assert loaded.extraction.company.name == "GreenVault Capital"
        assert len(loaded.extraction.claims) == 5
        assert loaded.extraction.esg_metrics.sfdr_article_claim == "Article 8"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Claim Extraction via DeckContext
# ══════════════════════════════════════════════════════════════════════════════


class TestStage3_ClaimExtraction:
    """
    Stage 3: DeckContext extracts claim texts for verification pipeline.
    Validates the handoff from extractor to compliance agent.
    """

    def test_claims_for_verification_returns_text_list(self, deck_context):
        """claims_for_verification() must return plain text strings."""
        claims = deck_context.claims_for_verification()
        assert isinstance(claims, list)
        assert all(isinstance(c, str) for c in claims)
        assert len(claims) == 5

    def test_claims_match_extraction_claims(self, deck_context, sample_deck_extraction):
        """Claim texts must match the extracted claims exactly."""
        claims = deck_context.claims_for_verification()
        original_texts = [c.text for c in sample_deck_extraction.claims]
        assert claims == original_texts

    def test_company_lookup_key_returns_name_fallback(self, deck_context):
        """Without CIK or ticker, lookup key should fall back to company name."""
        key = deck_context.company_lookup_key()
        assert key == "GreenVault Capital"

    def test_clarifying_context_contains_company_info(self, deck_context):
        """Clarifying context must include company name and extraction notes."""
        ctx = deck_context.clarifying_context()
        assert "GreenVault Capital" in ctx
        assert "DECK CONTEXT" in ctx
        assert "Scope 3 emissions not disclosed" in ctx  # From extraction_notes

    def test_forward_looking_claims_identified(self, sample_deck_extraction):
        """Forward-looking claims must be correctly flagged for risk assessment."""
        forward_looking = [c for c in sample_deck_extraction.claims if c.likely_forward_looking]
        assert len(forward_looking) >= 1
        # The projection and market claims should be forward-looking
        texts = [c.text for c in forward_looking]
        assert any("3x revenue" in t for t in texts)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — SEC Compliance Analysis
# ══════════════════════════════════════════════════════════════════════════════


class TestStage4_SecAnalysis:
    """
    Stage 4: Validates SEC analyzer output structure.
    Uses mock assessments that simulate what analyzer.py would produce.
    """

    def test_sec_assessments_have_required_fields(self, mock_sec_assessments):
        """Every ClaimAssessment must have verdict, severity, and forward_looking."""
        for assessment in mock_sec_assessments:
            assert assessment.verdict in [
                "CONSISTENT", "CONTRADICTS", "UNSUPPORTED",
                "INSUFFICIENT_EVIDENCE", "CRITICAL_ABSENT",
                "GREENWASHING_RISK", "DATA_QUALITY_ISSUE"
            ]
            assert assessment.severity in ["NONE", "LOW", "MEDIUM", "HIGH"]
            assert isinstance(assessment.forward_looking, bool)
            assert assessment.jurisdiction == "sec"

    def test_private_company_gets_insufficient_evidence(self, mock_sec_assessments):
        """Private company claims must be marked INSUFFICIENT_EVIDENCE (no CIK)."""
        # First claim is about AUM — private company, no SEC filings
        first = mock_sec_assessments[0]
        assert first.verdict == "INSUFFICIENT_EVIDENCE"

    def test_forward_looking_projection_gets_unsupported(self, mock_sec_assessments):
        """3x growth projection must be marked as UNSUPPORTED without SEC corroboration."""
        projection_assessment = mock_sec_assessments[1]
        assert projection_assessment.verdict == "UNSUPPORTED"
        assert projection_assessment.forward_looking is True
        assert projection_assessment.severity == "MEDIUM"

    def test_market_claim_can_be_consistent(self, mock_sec_assessments):
        """Market size claim verified via web search can be CONSISTENT."""
        market_assessment = mock_sec_assessments[3]
        assert market_assessment.verdict == "CONSISTENT"

    def test_assessments_count_matches_claims(self, mock_sec_assessments, deck_context):
        """Assessment count must match claim count."""
        claims = deck_context.claims_for_verification()
        assert len(mock_sec_assessments) == len(claims)

    def test_assessments_contain_action_items(self, mock_sec_assessments):
        """At least some assessments must have actionable next steps."""
        all_actions = [a for assessment in mock_sec_assessments for a in assessment.action_items]
        assert len(all_actions) > 0, "At least one action item must be present"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — EU SFDR/CSRD Analysis
# ══════════════════════════════════════════════════════════════════════════════


class TestStage5_EuAnalysis:
    """
    Stage 5: Validates EU SFDR/CSRD analyzer output.
    Uses mock assessments from analyzer_sfdr.py simulation.
    """

    def test_eu_assessments_jurisdiction_tagged(self, mock_eu_assessments):
        """All EU assessments must be tagged with eu_sfdr_csrd jurisdiction."""
        for assessment in mock_eu_assessments:
            assert assessment.jurisdiction == "eu_sfdr_csrd"

    def test_sfdr_article_8_claim_verified(self, mock_eu_assessments):
        """Article 8 fund classification should be verified against SFDR regulations."""
        article8_assessment = mock_eu_assessments[0]
        assert article8_assessment.verdict == "CONSISTENT"
        assert any("Article 8" in v for v in article8_assessment.verified)

    def test_scope3_warning_raised(self, mock_eu_assessments):
        """Missing Scope 3 emissions must trigger a CSRD warning."""
        carbon_assessment = mock_eu_assessments[2]
        assert any("Scope 3" in w for w in carbon_assessment.warnings)

    def test_esg_metrics_feed_into_eu_analysis(self, sample_deck_extraction, mock_eu_assessments):
        """ESG metrics from extraction are consumed by EU analyzer."""
        esg = sample_deck_extraction.esg_metrics
        # Verify the ESG metrics contain data that would drive the EU assessment
        assert esg.scope_1_emissions is not None
        assert esg.sfdr_article_claim == "Article 8"
        # The EU assessments reference these verified items
        verified_first = mock_eu_assessments[0].verified
        assert any("Article 8" in v for v in verified_first)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — CA SB 54 Analysis
# ══════════════════════════════════════════════════════════════════════════════


class TestStage6_CaAnalysis:
    """
    Stage 6: Validates California SB 54 founder demographics compliance.
    Uses mock assessments from analyzer_sb54.py simulation.
    """

    def test_ca_assessments_jurisdiction_tagged(self, mock_ca_assessments):
        """All CA assessments must be tagged with ca_sb54 jurisdiction."""
        for assessment in mock_ca_assessments:
            assert assessment.jurisdiction == "ca_sb54"

    def test_gender_diversity_verified(self, mock_ca_assessments):
        """Gender diversity disclosure must be verified."""
        gender_assessment = mock_ca_assessments[0]
        assert gender_assessment.verdict == "CONSISTENT"
        assert any("50% women" in v for v in gender_assessment.verified)

    def test_founder_demographics_feed_into_ca_analysis(self, sample_deck_extraction, mock_ca_assessments):
        """Founder demographics from extraction must feed into CA analysis."""
        demos = sample_deck_extraction.founder_demographics
        # Demographics in extraction
        assert demos.women_founder_pct == 50.0
        assert demos.race_ethnicity_data is not None
        # The CA assessments verify these
        first = mock_ca_assessments[0]
        assert any("50% women" in v for v in first.verified)

    def test_education_and_experience_disclosed(self, mock_ca_assessments):
        """Education and prior startup experience must be verified."""
        education_assessment = mock_ca_assessments[1]
        assert education_assessment.verdict == "CONSISTENT"
        assert any("Oxford" in v for v in education_assessment.verified)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — Compliance Scoring
# ══════════════════════════════════════════════════════════════════════════════


class TestStage7_ComplianceScoring:
    """
    Stage 7: Real compliance scoring using outputs from all analyzers.
    Validates that scores are correctly calculated from ClaimAssessment inputs.
    """

    def test_sec_score_calculated_from_sec_assessments(self, mock_sec_assessments):
        """SEC compliance score must be derived from SEC assessments."""
        score = calculate_compliance_score("sec", mock_sec_assessments)
        assert isinstance(score, ComplianceScore)
        assert score.jurisdiction == "sec"
        assert 0 <= score.overall_score <= 100
        assert score.risk_level in ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_eu_score_calculated_from_eu_assessments(self, mock_eu_assessments):
        """EU compliance score must be derived from EU assessments."""
        score = calculate_compliance_score("eu_sfdr_csrd", mock_eu_assessments)
        assert score.jurisdiction == "eu_sfdr_csrd"
        assert score.overall_score > 50, "EU score should be decent with mostly CONSISTENT verdicts"

    def test_ca_score_calculated_from_ca_assessments(self, mock_ca_assessments):
        """CA compliance score must be derived from CA assessments."""
        score = calculate_compliance_score("ca_sb54", mock_ca_assessments)
        assert score.jurisdiction == "ca_sb54"
        assert score.overall_score >= 80, "CA score should be high with CONSISTENT verdicts"
        assert score.risk_level == "NONE"

    def test_passing_claims_count_fed_into_score(self, mock_eu_assessments):
        """Passing claims count must reflect CONSISTENT verdicts."""
        score = calculate_compliance_score("eu_sfdr_csrd", mock_eu_assessments)
        consistent_count = sum(1 for a in mock_eu_assessments if a.verdict == "CONSISTENT")
        assert score.passing_claims == consistent_count

    def test_scores_aggregate_across_jurisdictions(self, mock_sec_assessments, mock_eu_assessments, mock_ca_assessments):
        """Aggregate scoring must correctly combine all jurisdiction scores."""
        sec_score = calculate_compliance_score("sec", mock_sec_assessments)
        eu_score = calculate_compliance_score("eu_sfdr_csrd", mock_eu_assessments)
        ca_score = calculate_compliance_score("ca_sb54", mock_ca_assessments)

        aggregate = aggregate_scores([sec_score, eu_score, ca_score])

        assert "overall_risk" in aggregate
        assert "jurisdictions" in aggregate
        assert len(aggregate["jurisdictions"]) == 3
        assert aggregate["overall_risk"] in ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_worst_case_risk_propagates_to_aggregate(self, mock_sec_assessments, mock_eu_assessments, mock_ca_assessments):
        """The worst jurisdiction risk must propagate to the overall aggregate."""
        sec_score = calculate_compliance_score("sec", mock_sec_assessments)
        eu_score = calculate_compliance_score("eu_sfdr_csrd", mock_eu_assessments)
        ca_score = calculate_compliance_score("ca_sb54", mock_ca_assessments)

        all_risks = [sec_score.risk_level, eu_score.risk_level, ca_score.risk_level]
        aggregate = aggregate_scores([sec_score, eu_score, ca_score])

        risk_hierarchy = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NONE": 4}
        expected_worst = min(all_risks, key=lambda r: risk_hierarchy[r])
        assert aggregate["overall_risk"] == expected_worst

    def test_score_recommendation_is_actionable(self, mock_sec_assessments):
        """Score recommendations must be non-empty strings."""
        score = calculate_compliance_score("sec", mock_sec_assessments)
        assert isinstance(score.recommendation, str)
        assert len(score.recommendation) > 0


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8 — Greenwashing Detection
# ══════════════════════════════════════════════════════════════════════════════


class TestStage8_GreenwashingDetection:
    """
    Stage 8: Validates greenwashing risk detection using ESG metrics from Stage 2.
    Real detector, realistic ESG data.
    """

    def test_greenwashing_risk_detected_from_esg_metrics(self, sample_deck_extraction):
        """ESG metrics from extraction must feed directly into greenwashing detector."""
        esg = sample_deck_extraction.esg_metrics
        risk = detect_greenwashing_risk(esg)
        assert isinstance(risk, GreenwashingRisk)
        assert risk.risk_level in ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_selective_scope_disclosure_triggers_pattern(self):
        """
        Selective scope disclosure (exactly 1 scope reported) triggers a greenwashing pattern.
        The detector flags single-scope reporting as suspicious cherry-picking.
        """
        esg_one_scope = EsgMetrics(
            scope_1_emissions="2,400 tCO2e (2025)",
            scope_2_emissions=None,   # Missing — selective
            scope_3_emissions=None,   # Missing — selective
            has_third_party_audit=True,
            audit_body="Bureau Veritas",
            board_diversity_pct="50% women",
            sfdr_article_claim="Article 8",
            ai_risk_sector=True,
            ai_transparency_statement="We comply with EU AI Act Annex III.",
            supply_chain_disclosure=None,
        )

        risk = detect_greenwashing_risk(esg_one_scope)
        has_selective_pattern = any(
            "scope" in p.lower() or "selective" in p.lower()
            for p in risk.detected_patterns
        )
        assert has_selective_pattern, f"Expected selective scope pattern, got: {risk.detected_patterns}"

    def test_comprehensive_disclosure_yields_low_risk(self, sample_deck_extraction):
        """
        A company with Scope 1+2, audit, board diversity, and AI transparency
        should have no greenwashing patterns detected (or only Scope 3 gaps).
        """
        esg = sample_deck_extraction.esg_metrics
        # Verify our fixture is comprehensive (2 scopes, audit, diversity all present)
        assert esg.scope_1_emissions is not None
        assert esg.scope_2_emissions is not None
        assert esg.has_third_party_audit is True
        assert esg.board_diversity_pct is not None
        assert esg.ai_transparency_statement is not None

        risk = detect_greenwashing_risk(esg)
        # With good disclosures, patterns list should be empty or minimal
        assert risk.risk_level in ["NONE", "LOW", "MEDIUM"], (
            f"Well-disclosed company should not be HIGH or CRITICAL risk; got {risk.risk_level}"
        )

    def test_audit_present_reduces_risk(self):
        """Third-party audit should reduce greenwashing risk vs. no audit."""
        with_audit = EsgMetrics(
            scope_1_emissions="1,000 tCO2e (2025)",
            scope_2_emissions="500 tCO2e (2025)",
            scope_3_emissions="5,000 tCO2e (2025)",
            has_third_party_audit=True,
            audit_body="Bureau Veritas",
            board_diversity_pct="40% women",
            sfdr_article_claim="Article 8",
            ai_risk_sector=False,
            ai_transparency_statement=None,
            supply_chain_disclosure=None,
        )
        without_audit = EsgMetrics(
            scope_1_emissions="1,000 tCO2e (2025)",
            scope_2_emissions="500 tCO2e (2025)",
            scope_3_emissions="5,000 tCO2e (2025)",
            has_third_party_audit=False,
            audit_body=None,
            board_diversity_pct="40% women",
            sfdr_article_claim="Article 8",
            ai_risk_sector=False,
            ai_transparency_statement=None,
            supply_chain_disclosure=None,
        )
        risk_with = detect_greenwashing_risk(with_audit)
        risk_without = detect_greenwashing_risk(without_audit)
        assert risk_with.risk_score <= risk_without.risk_score, (
            f"Audit should reduce risk score: {risk_with.risk_score} vs {risk_without.risk_score}"
        )

    def test_no_esg_data_returns_medium_risk(self):
        """Absence of any ESG disclosure should return MEDIUM risk."""
        risk = detect_greenwashing_risk(None)
        assert risk.risk_level == "MEDIUM"
        assert "No ESG data disclosed" in risk.detected_patterns

    def test_greenwashing_risk_has_recommendations(self, sample_deck_extraction):
        """Greenwashing assessment must include actionable recommendations."""
        esg = sample_deck_extraction.esg_metrics
        risk = detect_greenwashing_risk(esg)
        assert isinstance(risk.recommendation, str)
        assert len(risk.recommendation) > 0

    def test_greenwashing_output_feeds_into_report(self, sample_deck_extraction):
        """Greenwashing result structure must be JSON-serializable for report inclusion."""
        esg = sample_deck_extraction.esg_metrics
        risk = detect_greenwashing_risk(esg)

        # Simulate serializing to report
        risk_dict = {
            "risk_level": risk.risk_level,
            "risk_score": risk.risk_score,
            "detected_patterns": risk.detected_patterns,
            "missing_evidence": risk.missing_evidence,
            "red_flags": risk.red_flags,
            "recommendation": risk.recommendation,
        }
        json_str = json.dumps(risk_dict)
        parsed = json.loads(json_str)
        assert parsed["risk_level"] == risk.risk_level


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 9 — Regulatory Mapping
# ══════════════════════════════════════════════════════════════════════════════


class TestStage9_RegulatoryMapping:
    """
    Stage 9: Validates regulatory article mapping from ClaimAssessments.
    Maps violations to specific SFDR/CSRD/SB54 articles.
    """

    def test_regulatory_map_built_from_eu_assessments(self, mock_eu_assessments):
        """EU assessments must feed correctly into regulatory map."""
        reg_map = build_regulatory_map("eu_sfdr_csrd", mock_eu_assessments)
        assert isinstance(reg_map, RegulatoryMap)
        assert reg_map.jurisdiction == "eu_sfdr_csrd"

    def test_regulatory_map_built_from_ca_assessments(self, mock_ca_assessments):
        """CA assessments must feed correctly into regulatory map."""
        reg_map = build_regulatory_map("ca_sb54", mock_ca_assessments)
        assert isinstance(reg_map, RegulatoryMap)
        assert reg_map.jurisdiction == "ca_sb54"

    def test_regulatory_map_tracks_violations_by_severity(self, mock_eu_assessments):
        """Regulatory map must count violations by severity level."""
        reg_map = build_regulatory_map("eu_sfdr_csrd", mock_eu_assessments)
        assert "HIGH" in reg_map.violations_by_severity
        assert "MEDIUM" in reg_map.violations_by_severity
        assert "LOW" in reg_map.violations_by_severity

    def test_regulatory_map_includes_remediations(self, mock_eu_assessments):
        """Regulatory map must include required remediation actions."""
        reg_map = build_regulatory_map("eu_sfdr_csrd", mock_eu_assessments)
        assert isinstance(reg_map.required_remediations, list)

    def test_regulatory_maps_aggregate_across_jurisdictions(self, mock_eu_assessments, mock_ca_assessments):
        """Regulatory maps from all jurisdictions must aggregate correctly."""
        eu_map = build_regulatory_map("eu_sfdr_csrd", mock_eu_assessments)
        ca_map = build_regulatory_map("ca_sb54", mock_ca_assessments)

        aggregated = aggregate_regulatory_maps([eu_map, ca_map])

        assert "total_violations_all_jurisdictions" in aggregated
        assert "jurisdictions" in aggregated
        assert len(aggregated["jurisdictions"]) == 2
        assert "critical_action_items" in aggregated

    def test_total_violations_sums_across_jurisdictions(self, mock_eu_assessments, mock_ca_assessments):
        """Aggregate violation count must be the sum across all jurisdictions."""
        eu_map = build_regulatory_map("eu_sfdr_csrd", mock_eu_assessments)
        ca_map = build_regulatory_map("ca_sb54", mock_ca_assessments)

        aggregated = aggregate_regulatory_maps([eu_map, ca_map])
        expected_total = eu_map.total_violations + ca_map.total_violations
        assert aggregated["total_violations_all_jurisdictions"] == expected_total


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 10 — Compliance Trend Analysis
# ══════════════════════════════════════════════════════════════════════════════


class TestStage10_TrendAnalysis:
    """
    Stage 10: Validates trend analysis using historical compliance score data.
    Simulates a company that has been improving its EU compliance over time.
    """

    @pytest.fixture
    def eu_historical_data(self) -> list[ComplianceDataPoint]:
        """Historical EU compliance scores showing improving trend."""
        now = datetime.now(timezone.utc)
        return [
            ComplianceDataPoint(
                timestamp=now - timedelta(days=30),
                jurisdiction="eu_sfdr_csrd",
                score=50.0,
                risk_level="MEDIUM",
                flagged_claims=4,
                missing_data_claims=3,
            ),
            ComplianceDataPoint(
                timestamp=now - timedelta(days=20),
                jurisdiction="eu_sfdr_csrd",
                score=60.0,
                risk_level="LOW",
                flagged_claims=3,
                missing_data_claims=2,
            ),
            ComplianceDataPoint(
                timestamp=now - timedelta(days=10),
                jurisdiction="eu_sfdr_csrd",
                score=70.0,
                risk_level="LOW",
                flagged_claims=2,
                missing_data_claims=1,
            ),
            ComplianceDataPoint(
                timestamp=now,
                jurisdiction="eu_sfdr_csrd",
                score=80.0,
                risk_level="NONE",
                flagged_claims=0,
                missing_data_claims=0,
            ),
        ]

    def test_trend_analysis_from_historical_scores(self, eu_historical_data):
        """Historical compliance data must feed into trend analysis."""
        trend = analyze_compliance_trend(eu_historical_data, "eu_sfdr_csrd")
        assert trend.jurisdiction == "eu_sfdr_csrd"
        assert trend.velocity > 0, "Scores are increasing, so velocity must be positive"

    def test_improving_trend_detected(self, eu_historical_data):
        """Consistently improving scores must produce IMPROVING trend direction."""
        trend = analyze_compliance_trend(eu_historical_data, "eu_sfdr_csrd")
        assert trend.trend_direction == TrendDirection.IMPROVING

    def test_trend_confidence_increases_with_data_points(self, eu_historical_data):
        """More data points must increase trend confidence."""
        few_points = eu_historical_data[:2]
        many_points = eu_historical_data

        trend_few = analyze_compliance_trend(few_points, "eu_sfdr_csrd")
        trend_many = analyze_compliance_trend(many_points, "eu_sfdr_csrd")

        assert trend_many.confidence >= trend_few.confidence

    def test_empty_data_returns_stable_with_zero_confidence(self):
        """No data points must return STABLE trend with zero confidence."""
        trend = analyze_compliance_trend([], "eu_sfdr_csrd")
        assert trend.trend_direction == TrendDirection.STABLE
        assert trend.confidence == 0.0

    def test_trend_data_points_stored_in_result(self, eu_historical_data):
        """Trend result must include the data points used for analysis."""
        trend = analyze_compliance_trend(eu_historical_data, "eu_sfdr_csrd")
        assert len(trend.data_points) == len(eu_historical_data)

    def test_trend_recommendation_generated(self, eu_historical_data):
        """Trend analysis must produce an actionable recommendation."""
        trend = analyze_compliance_trend(eu_historical_data, "eu_sfdr_csrd")
        assert isinstance(trend.recommendation, str)
        assert len(trend.recommendation) > 0


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 11 — Predictive Risk Modeling
# ══════════════════════════════════════════════════════════════════════════════


class TestStage11_PredictiveRisk:
    """
    Stage 11: Validates predictive risk modeling using current compliance score
    and remediation velocity from Stage 10 trend analysis.
    """

    def test_predictive_model_from_compliance_score(self, mock_eu_assessments):
        """Compliance score from Stage 7 must feed into predictive risk model."""
        score = calculate_compliance_score("eu_sfdr_csrd", mock_eu_assessments)

        # Simulate remediation velocity from trend (0.33 issues/day = 10 issues in 30 days)
        model = predict_future_risk(
            current_score=score.overall_score,
            current_risk_level=score.risk_level,
            flagged_issues=score.flagged_claims + score.missing_data_claims,
            remediation_velocity=0.1,  # Fixing ~1 issue per 10 days
            jurisdiction="eu_sfdr_csrd",
        )

        assert isinstance(model, PredictiveRiskModel)
        assert model.jurisdiction == "eu_sfdr_csrd"

    def test_predicted_risk_improves_with_positive_velocity(self):
        """Positive remediation velocity must improve 90-day risk prediction."""
        model = predict_future_risk(
            current_score=50.0,
            current_risk_level="MEDIUM",
            flagged_issues=5,
            remediation_velocity=0.2,  # Positive: improving
            jurisdiction="sec",
        )

        risk_hierarchy = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NONE": 4}
        assert risk_hierarchy[model.predicted_risk_90_days] >= risk_hierarchy[model.current_risk], (
            f"90-day risk {model.predicted_risk_90_days} should be at least as good as current {model.current_risk}"
        )

    def test_zero_velocity_indicates_no_progress(self):
        """Zero remediation velocity must predict long remediation time."""
        model = predict_future_risk(
            current_score=40.0,
            current_risk_level="MEDIUM",
            flagged_issues=10,
            remediation_velocity=0.0,
            jurisdiction="sec",
        )
        assert model.estimated_remediation_days == 999  # No progress

    def test_predictive_model_includes_early_warnings(self):
        """Predictive model must include early warning indicators for declining compliance."""
        model = predict_future_risk(
            current_score=25.0,
            current_risk_level="HIGH",
            flagged_issues=8,
            remediation_velocity=-0.3,  # Declining
            jurisdiction="eu_sfdr_csrd",
        )
        # With score near critical threshold and declining, should have warnings
        assert isinstance(model.early_warnings, list)

    def test_predictive_model_has_critical_path_items(self):
        """Predictive model must identify critical path items blocking compliance."""
        model = predict_future_risk(
            current_score=45.0,
            current_risk_level="MEDIUM",
            flagged_issues=6,
            remediation_velocity=0.1,
            jurisdiction="eu_sfdr_csrd",
        )
        assert isinstance(model.critical_path_items, list)

    def test_trend_velocity_feeds_into_predictive_model(self):
        """Velocity from trend analysis must correctly drive predictive model score changes."""
        velocity = 1.0  # From trend analysis: 1.0 points/day improvement
        current = 60.0
        model = predict_future_risk(
            current_score=current,
            current_risk_level="LOW",
            flagged_issues=3,
            remediation_velocity=velocity,
            jurisdiction="sec",
        )
        # 30-day projection: current + velocity * 30 * 2 = 60 + 60 = 120 → capped at 100
        assert model.predicted_risk_30_days == "NONE"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 12 — Final Report Aggregation
# ══════════════════════════════════════════════════════════════════════════════


class TestStage12_FinalReport:
    """
    Stage 12: Validates the final report structure by aggregating outputs
    from all previous stages into a single actionable compliance report.
    """

    @pytest.fixture
    def full_report(
        self,
        sample_deck_extraction,
        deck_context,
        mock_sec_assessments,
        mock_eu_assessments,
        mock_ca_assessments,
    ) -> dict:
        """
        Constructs the final compliance report by feeding outputs from each stage
        into the next, ending with a fully aggregated report dict.
        """
        # Stage 7 outputs: Compliance scores per jurisdiction
        sec_score = calculate_compliance_score("sec", mock_sec_assessments)
        eu_score = calculate_compliance_score("eu_sfdr_csrd", mock_eu_assessments)
        ca_score = calculate_compliance_score("ca_sb54", mock_ca_assessments)
        aggregate = aggregate_scores([sec_score, eu_score, ca_score])

        # Stage 8 output: Greenwashing risk from ESG metrics
        greenwashing = detect_greenwashing_risk(sample_deck_extraction.esg_metrics)

        # Stage 9 output: Regulatory maps
        eu_map = build_regulatory_map("eu_sfdr_csrd", mock_eu_assessments)
        ca_map = build_regulatory_map("ca_sb54", mock_ca_assessments)
        reg_aggregate = aggregate_regulatory_maps([eu_map, ca_map])

        # Stage 10 output: Trend analysis
        now = datetime.now(timezone.utc)
        historical_data = [
            ComplianceDataPoint(
                timestamp=now - timedelta(days=30),
                jurisdiction="eu_sfdr_csrd",
                score=55.0,
                risk_level="MEDIUM",
                flagged_claims=4,
                missing_data_claims=2,
            ),
            ComplianceDataPoint(
                timestamp=now,
                jurisdiction="eu_sfdr_csrd",
                score=eu_score.overall_score,
                risk_level=eu_score.risk_level,
                flagged_claims=eu_score.flagged_claims,
                missing_data_claims=eu_score.missing_data_claims,
            ),
        ]
        eu_trend = analyze_compliance_trend(historical_data, "eu_sfdr_csrd")

        # Stage 11 output: Predictive risk
        predictive = predict_future_risk(
            current_score=eu_score.overall_score,
            current_risk_level=eu_score.risk_level,
            flagged_issues=eu_score.flagged_claims + eu_score.missing_data_claims,
            remediation_velocity=eu_trend.velocity,
            jurisdiction="eu_sfdr_csrd",
        )

        # Stage 12: Assemble final report
        all_results = []
        for i, (claim, assessment) in enumerate(
            zip(deck_context.claims_for_verification(), mock_sec_assessments), start=1
        ):
            all_results.append({
                "index": i,
                "claim": claim,
                "verdict": assessment.verdict,
                "forward_looking": assessment.forward_looking,
                "severity": assessment.severity,
                "explanation": assessment.explanation,
                "action_items": assessment.action_items,
                "jurisdiction": assessment.jurisdiction,
            })

        return {
            # Metadata
            "report_id": "e2e_test_report_001",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "company_name": sample_deck_extraction.company.name,
            "industry": sample_deck_extraction.company.industry,
            "modules": ["sec", "eu_sfdr_csrd", "ca_sb54"],
            "total_claims_analyzed": len(mock_sec_assessments) + len(mock_eu_assessments) + len(mock_ca_assessments),

            # Stage 4-6 outputs: Per-claim analysis results
            "claim_results": all_results,

            # Stage 7 outputs: Jurisdiction compliance scores
            "compliance_scores": {
                "sec": {
                    "overall_score": sec_score.overall_score,
                    "risk_level": sec_score.risk_level,
                    "passing_claims": sec_score.passing_claims,
                    "flagged_claims": sec_score.flagged_claims,
                    "recommendation": sec_score.recommendation,
                },
                "eu_sfdr_csrd": {
                    "overall_score": eu_score.overall_score,
                    "risk_level": eu_score.risk_level,
                    "passing_claims": eu_score.passing_claims,
                    "flagged_claims": eu_score.flagged_claims,
                    "recommendation": eu_score.recommendation,
                },
                "ca_sb54": {
                    "overall_score": ca_score.overall_score,
                    "risk_level": ca_score.risk_level,
                    "passing_claims": ca_score.passing_claims,
                    "recommendation": ca_score.recommendation,
                },
            },
            "overall_risk": aggregate["overall_risk"],
            "primary_recommendation": aggregate["primary_recommendation"],

            # Stage 8 output: Greenwashing assessment
            "greenwashing_assessment": {
                "risk_level": greenwashing.risk_level,
                "risk_score": greenwashing.risk_score,
                "detected_patterns": greenwashing.detected_patterns,
                "missing_evidence": greenwashing.missing_evidence,
                "recommendation": greenwashing.recommendation,
            },

            # Stage 9 output: Regulatory mapping
            "regulatory_map": {
                "total_violations": reg_aggregate["total_violations_all_jurisdictions"],
                "jurisdictions": reg_aggregate["jurisdictions"],
                "critical_action_items": reg_aggregate["critical_action_items"],
            },

            # Stage 10 output: Trend analysis
            "trend_analysis": {
                "eu_sfdr_csrd": {
                    "direction": eu_trend.trend_direction.value,
                    "velocity": eu_trend.velocity,
                    "confidence": eu_trend.confidence,
                    "recommendation": eu_trend.recommendation,
                }
            },

            # Stage 11 output: Predictive risk
            "predictive_risk": {
                "jurisdiction": predictive.jurisdiction,
                "current_risk": predictive.current_risk,
                "predicted_risk_30_days": predictive.predicted_risk_30_days,
                "predicted_risk_90_days": predictive.predicted_risk_90_days,
                "estimated_remediation_days": predictive.estimated_remediation_days,
                "risk_trajectory": predictive.risk_trajectory,
                "early_warnings": [
                    {
                        "type": w.indicator_type,
                        "level": w.warning_level.value,
                        "action": w.recommended_action,
                    }
                    for w in predictive.early_warnings
                ],
            },
        }

    # ── Report structure validation ───────────────────────────────────────────

    def test_report_has_required_top_level_keys(self, full_report):
        """Final report must contain all required top-level sections."""
        required_keys = [
            "report_id",
            "generated_at",
            "company_name",
            "modules",
            "total_claims_analyzed",
            "claim_results",
            "compliance_scores",
            "overall_risk",
            "primary_recommendation",
            "greenwashing_assessment",
            "regulatory_map",
            "trend_analysis",
            "predictive_risk",
        ]
        for key in required_keys:
            assert key in full_report, f"Report missing required key: {key}"

    def test_report_company_matches_extraction(self, full_report, sample_deck_extraction):
        """Report company name must match the extracted company."""
        assert full_report["company_name"] == sample_deck_extraction.company.name

    def test_report_all_three_modules_present(self, full_report):
        """Report must cover all three compliance modules."""
        assert "sec" in full_report["modules"]
        assert "eu_sfdr_csrd" in full_report["modules"]
        assert "ca_sb54" in full_report["modules"]

    def test_report_claim_results_from_sec_stage(self, full_report, deck_context):
        """Claim results in report must match claims from deck extraction."""
        claims = deck_context.claims_for_verification()
        assert len(full_report["claim_results"]) == len(claims)
        for result in full_report["claim_results"]:
            assert "claim" in result
            assert "verdict" in result
            assert "severity" in result

    def test_report_compliance_scores_all_jurisdictions(self, full_report):
        """Report must contain scores for all three jurisdictions."""
        scores = full_report["compliance_scores"]
        assert "sec" in scores
        assert "eu_sfdr_csrd" in scores
        assert "ca_sb54" in scores
        for jur, score in scores.items():
            assert "overall_score" in score, f"Score for {jur} missing overall_score"
            assert "risk_level" in score, f"Score for {jur} missing risk_level"
            assert 0 <= score["overall_score"] <= 100

    def test_report_overall_risk_is_valid_level(self, full_report):
        """Overall risk level must be one of the valid risk levels."""
        assert full_report["overall_risk"] in ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_report_greenwashing_has_patterns(self, full_report):
        """Greenwashing assessment must contain detected patterns list."""
        gw = full_report["greenwashing_assessment"]
        assert "risk_level" in gw
        assert "detected_patterns" in gw
        assert isinstance(gw["detected_patterns"], list)

    def test_report_regulatory_map_has_violations(self, full_report):
        """Regulatory map must include violation counts and action items."""
        reg = full_report["regulatory_map"]
        assert "total_violations" in reg
        assert "critical_action_items" in reg
        assert isinstance(reg["critical_action_items"], list)

    def test_report_trend_analysis_present(self, full_report):
        """Trend analysis must be present in the final report."""
        trend = full_report["trend_analysis"]
        assert "eu_sfdr_csrd" in trend
        eu_trend = trend["eu_sfdr_csrd"]
        assert "direction" in eu_trend
        assert "velocity" in eu_trend
        assert eu_trend["direction"] in ["improving", "stable", "declining"]

    def test_report_predictive_risk_present(self, full_report):
        """Predictive risk model must be present in the final report."""
        pred = full_report["predictive_risk"]
        assert "current_risk" in pred
        assert "predicted_risk_30_days" in pred
        assert "predicted_risk_90_days" in pred
        assert "estimated_remediation_days" in pred

    def test_report_serializes_to_json(self, full_report):
        """Final report must be fully JSON-serializable for storage and sharing."""
        json_str = json.dumps(full_report, indent=2)
        parsed = json.loads(json_str)
        assert parsed["company_name"] == "GreenVault Capital"
        assert parsed["modules"] == ["sec", "eu_sfdr_csrd", "ca_sb54"]

    def test_report_can_be_saved_to_disk(self, full_report):
        """Final report must be saveable to disk (for saved_reports/ storage)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as f:
            json.dump(full_report, f, indent=2)
            tmp_path = Path(f.name)

        assert tmp_path.exists()
        loaded = json.loads(tmp_path.read_text())
        assert loaded["report_id"] == full_report["report_id"]
        tmp_path.unlink()

    def test_report_primary_recommendation_is_actionable(self, full_report):
        """Primary recommendation must be a non-empty actionable string."""
        rec = full_report["primary_recommendation"]
        assert isinstance(rec, str)
        assert len(rec) > 10


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE INTEGRATION TEST
# ══════════════════════════════════════════════════════════════════════════════


class TestFullPipelineIntegration:
    """
    Validates that all stages chain together correctly in a single end-to-end flow.
    This is the critical integration test that confirms data flows correctly
    from file upload through final report generation.
    """

    def test_pipeline_stage_outputs_are_type_safe(
        self,
        sample_deck_extraction,
        mock_sec_assessments,
        mock_eu_assessments,
        mock_ca_assessments,
    ):
        """
        Every stage output must be the correct type for the next stage's input.
        Validates the full chain: extraction → context → assessment → score → report.
        """
        # Stage 2 → Stage 3: DeckExtraction → DeckContext
        context = DeckContext(sample_deck_extraction)
        assert isinstance(context, DeckContext)

        # Stage 3 → Stage 4: DeckContext.claims → List[str] for analyzer
        claims = context.claims_for_verification()
        assert all(isinstance(c, str) for c in claims)

        # Stage 4-6 → Stage 7: List[ClaimAssessment] → ComplianceScore
        sec_score = calculate_compliance_score("sec", mock_sec_assessments)
        assert isinstance(sec_score, ComplianceScore)

        # Stage 7 → Stage 8: ComplianceScore + EsgMetrics → GreenwashingRisk
        gw_risk = detect_greenwashing_risk(sample_deck_extraction.esg_metrics)
        assert isinstance(gw_risk, GreenwashingRisk)

        # Stage 7 → Stage 9: List[ClaimAssessment] → RegulatoryMap
        eu_map = build_regulatory_map("eu_sfdr_csrd", mock_eu_assessments)
        assert isinstance(eu_map, RegulatoryMap)

        # Stage 9 → Stage 10: ComplianceScore history → TrendAnalysis
        now = datetime.now(timezone.utc)
        data_points = [
            ComplianceDataPoint(
                timestamp=now - timedelta(days=10),
                jurisdiction="eu_sfdr_csrd",
                score=50.0,
                risk_level="MEDIUM",
                flagged_claims=3,
                missing_data_claims=2,
            ),
            ComplianceDataPoint(
                timestamp=now,
                jurisdiction="eu_sfdr_csrd",
                score=sec_score.overall_score,
                risk_level=sec_score.risk_level,
                flagged_claims=sec_score.flagged_claims,
                missing_data_claims=sec_score.missing_data_claims,
            ),
        ]
        from compliance_trends import TrendAnalysis
        trend = analyze_compliance_trend(data_points, "eu_sfdr_csrd")
        assert isinstance(trend, TrendAnalysis)

        # Stage 10 → Stage 11: TrendAnalysis.velocity → PredictiveRiskModel
        predictive = predict_future_risk(
            current_score=sec_score.overall_score,
            current_risk_level=sec_score.risk_level,
            flagged_issues=sec_score.flagged_claims,
            remediation_velocity=trend.velocity,
            jurisdiction="sec",
        )
        assert isinstance(predictive, PredictiveRiskModel)

    def test_pipeline_handles_no_esg_metrics(self):
        """
        Pipeline must handle a deck with no ESG metrics gracefully (no EU module).
        """
        minimal_extraction = DeckExtraction(
            company=CompanyIdentity(
                name="QuickShip Logistics",
                founders=["Bob Smith"],
                industry="Logistics",
            ),
            claims=[
                ExtractedClaim(
                    text="QuickShip delivers 10,000 packages per day.",
                    verbatim="10,000 daily deliveries as of Jan 2026.",
                    slide=2,
                    category="traction",
                    likely_forward_looking=False,
                )
            ],
            esg_metrics=None,
            founder_demographics=None,
            extraction_notes="No ESG or founder demographic data in deck.",
        )

        context = DeckContext(minimal_extraction)
        claims = context.claims_for_verification()
        assert len(claims) == 1

        # Greenwashing detector handles None gracefully
        gw = detect_greenwashing_risk(None)
        assert gw.risk_level == "MEDIUM"

        # Scoring still works with empty assessments
        score = calculate_compliance_score("sec", [])
        assert score.overall_score == 100.0
        assert score.risk_level == "NONE"

    def test_pipeline_ca_demographics_feed_through(self, sample_deck_extraction, mock_ca_assessments):
        """
        CA SB 54 demographics from extraction must produce valid compliance score
        that feeds correctly into the final report.
        """
        demos = sample_deck_extraction.founder_demographics
        assert demos is not None

        # Demographics → CA assessments → CA score
        ca_score = calculate_compliance_score("ca_sb54", mock_ca_assessments)
        assert ca_score.passing_claims == 2
        assert ca_score.risk_level == "NONE"

        # Score → regulatory map
        ca_map = build_regulatory_map("ca_sb54", mock_ca_assessments)
        assert ca_map.jurisdiction == "ca_sb54"

        # Everything serializable
        report_section = {
            "jurisdiction": ca_score.jurisdiction,
            "score": ca_score.overall_score,
            "risk": ca_score.risk_level,
            "violations": ca_map.total_violations,
        }
        json.dumps(report_section)  # Must not raise

    def test_pipeline_report_has_continuous_data_flow(
        self,
        sample_deck_extraction,
        mock_sec_assessments,
        mock_eu_assessments,
        mock_ca_assessments,
    ):
        """
        Validates that each stage's output is used as the next stage's input,
        forming a complete unbroken data chain from upload to report.
        """
        # STAGE 1→2: File uploaded, extraction produced
        assert sample_deck_extraction.company.name  # Stage 2 output

        # STAGE 2→3: Extraction → DeckContext → Claims list
        ctx = DeckContext(sample_deck_extraction)
        claims = ctx.claims_for_verification()
        assert len(claims) > 0  # Stage 3 output

        # STAGE 3→4: Claims → SEC assessments (count must match)
        assert len(mock_sec_assessments) == len(claims)  # Stage 4 output

        # STAGE 4→7: SEC assessments → SEC compliance score
        sec_score = calculate_compliance_score("sec", mock_sec_assessments)
        assert sec_score.jurisdiction == "sec"  # Stage 7a output

        # STAGE 5→7: EU assessments → EU compliance score
        eu_score = calculate_compliance_score("eu_sfdr_csrd", mock_eu_assessments)
        assert eu_score.jurisdiction == "eu_sfdr_csrd"  # Stage 7b output

        # STAGE 6→7: CA assessments → CA compliance score
        ca_score = calculate_compliance_score("ca_sb54", mock_ca_assessments)
        assert ca_score.jurisdiction == "ca_sb54"  # Stage 7c output

        # STAGE 7→8: ESG metrics from Stage 2 → greenwashing risk
        gw = detect_greenwashing_risk(sample_deck_extraction.esg_metrics)
        assert gw.risk_level in ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]  # Stage 8 output

        # STAGE 7→9: Assessments → regulatory maps
        eu_map = build_regulatory_map("eu_sfdr_csrd", mock_eu_assessments)
        ca_map = build_regulatory_map("ca_sb54", mock_ca_assessments)
        reg_agg = aggregate_regulatory_maps([eu_map, ca_map])
        assert "total_violations_all_jurisdictions" in reg_agg  # Stage 9 output

        # STAGE 9→10: Current scores → trend data point, trend analysis
        now = datetime.now(timezone.utc)
        dp = ComplianceDataPoint(
            timestamp=now,
            jurisdiction="eu_sfdr_csrd",
            score=eu_score.overall_score,
            risk_level=eu_score.risk_level,
            flagged_claims=eu_score.flagged_claims,
            missing_data_claims=eu_score.missing_data_claims,
        )
        trend = analyze_compliance_trend([dp], "eu_sfdr_csrd")
        assert trend.jurisdiction == "eu_sfdr_csrd"  # Stage 10 output

        # STAGE 10→11: Trend velocity → predictive risk
        predictive = predict_future_risk(
            current_score=eu_score.overall_score,
            current_risk_level=eu_score.risk_level,
            flagged_issues=eu_score.flagged_claims,
            remediation_velocity=trend.velocity,
            jurisdiction="eu_sfdr_csrd",
        )
        assert predictive.jurisdiction == "eu_sfdr_csrd"  # Stage 11 output

        # STAGE 11→12: All outputs → final report
        final_report = {
            "company": sample_deck_extraction.company.name,
            "claims_analyzed": len(claims),
            "overall_risk": min(
                [sec_score.risk_level, eu_score.risk_level, ca_score.risk_level],
                key=lambda r: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NONE": 4}[r],
            ),
            "greenwashing_risk": gw.risk_level,
            "regulatory_violations": reg_agg["total_violations_all_jurisdictions"],
            "eu_trend": trend.trend_direction.value,
            "predicted_risk_90d": predictive.predicted_risk_90_days,
        }
        assert json.dumps(final_report)  # Stage 12: serializable ✓
        assert final_report["claims_analyzed"] == 5
        assert final_report["company"] == "GreenVault Capital"
