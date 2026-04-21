import pytest
from unittest.mock import patch, Mock
from analyzer import _resolve_model, analyze_claim, ClaimAssessment
from retriever import Hit


class TestAnalyzer:
    @patch.dict('os.environ', {'ANALYZER_MODEL': 'claude-opus-4'})
    def test_resolve_model_with_env_var(self):
        """Test that _resolve_model uses the ANALYZER_MODEL env var."""
        assert _resolve_model() == 'claude-opus-4'

    @patch.dict('os.environ', {}, clear=True)
    def test_resolve_model_default(self):
        """Test that _resolve_model uses default when no env var."""
        assert _resolve_model() == 'claude-sonnet-4-6'

    @patch('anthropic.Anthropic')
    def test_analyze_claim_success(self, mock_anthropic_class):
        """Test successful claim analysis."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.parsed_output = ClaimAssessment(
            verdict="CONSISTENT",
            explanation="The claim is supported by the filings.",
            forward_looking=True,
            severity="LOW",
            cited_passages=[1]
        )
        mock_client.messages.parse.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        # Create a simple Hit mock
        mock_hit = Mock()
        mock_hit.passage.text = "Revenue was $5M"
        mock_hit.score = 0.9

        claim = "We have $5M ARR"
        hits = [mock_hit]

        result = analyze_claim(mock_client, claim, hits)

        assert result.verdict == "CONSISTENT"
        assert result.forward_looking is True
        mock_client.messages.parse.assert_called_once()

    @patch('llm_local.call_structured')
    @patch('llm_local.is_local_model', return_value=True)
    def test_analyze_claim_local_model(self, mock_is_local, mock_call_structured):
        """Test claim analysis with local model."""
        mock_call_structured.return_value = ClaimAssessment(
            verdict="UNSUPPORTED",
            explanation="No evidence in filings.",
            forward_looking=False,
            severity="MEDIUM",
            cited_passages=[]
        )

        claim = "We will reach $10M ARR next year"
        hits = []

        result = analyze_claim(None, claim, hits, model="llama3.1")

        assert result.verdict == "UNSUPPORTED"
        mock_call_structured.assert_called_once()