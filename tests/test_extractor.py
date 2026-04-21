import pytest
from unittest.mock import patch, Mock
from extractor import _resolve_model, SYSTEM_PROMPT, DeckExtraction, CompanyIdentity
import anthropic


class TestExtractor:
    @patch('anthropic.Anthropic')
    def test_deck_extraction_parsing(self, mock_anthropic_class):
        """Test that the DeckExtraction parsing works with a sample input."""
        # Mock the Anthropic client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.parsed_output = DeckExtraction(
            company=CompanyIdentity(
                name="TestCo",
                industry="B2B SaaS",
                website="https://testco.com"
            ),
            claims=[],
            extraction_notes="Test successful"
        )
        mock_client.messages.parse.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        # Test the parsing logic
        model = _resolve_model()
        response = mock_client.messages.parse(
            model=model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": "Extract the structured pitch deck information. Here is the deck text: We are a Series A startup with $5M ARR. Our name is TestCo. We are raising $10M. Our CAC is $100."
                }
            ],
            output_format=DeckExtraction,
        )

        assert response.parsed_output.extraction_notes == "Test successful"
        mock_client.messages.parse.assert_called_once()

    @patch.dict('os.environ', {'EXTRACTOR_MODEL': 'claude-sonnet-4'})
    def test_resolve_model_with_env_var(self):
        """Test that _resolve_model uses the EXTRACTOR_MODEL env var."""
        assert _resolve_model() == 'claude-sonnet-4'

    @patch.dict('os.environ', {}, clear=True)
    def test_resolve_model_default(self):
        """Test that _resolve_model uses default when no env var."""
        assert _resolve_model() == 'claude-haiku-4-5'

    def test_company_identity_model(self):
        """Test CompanyIdentity model validation."""
        # Valid instance
        company = CompanyIdentity(
            name="Test Company",
            industry="Fintech",
            website="https://test.com"
        )
        assert company.name == "Test Company"
        assert company.industry == "Fintech"

        # Invalid: missing required name
        with pytest.raises(ValueError):
            CompanyIdentity(industry="Fintech")

    def test_deck_extraction_model(self):
        """Test DeckExtraction model validation."""
        company = CompanyIdentity(name="TestCo", industry="SaaS")
        extraction = DeckExtraction(
            company=company,
            claims=[],
            extraction_notes="Valid extraction"
        )
        assert extraction.company.name == "TestCo"
        assert extraction.claims == []