import pytest
from unittest.mock import patch, Mock
from sec import _cik10, lookup_cik, list_filings, fetch_text, chunk_text, Filing


class TestSec:
    def test_cik10_string(self):
        """Test _cik10 with string input."""
        assert _cik10("12345") == "0000012345"
        assert _cik10("1234567890") == "1234567890"

    def test_cik10_int(self):
        """Test _cik10 with int input."""
        assert _cik10(12345) == "0000012345"
        assert _cik10(1234567890) == "1234567890"

    @patch('httpx.get')
    def test_lookup_cik_exact_match(self, mock_get):
        """Test lookup_cik with exact ticker match."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "1": {"cik_str": 12345, "ticker": "AAPL", "title": "Apple Inc."}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = lookup_cik("AAPL")
        assert result == "0000012345"
        mock_get.assert_called_once()

    @patch('httpx.get')
    def test_lookup_cik_no_match(self, mock_get):
        """Test lookup_cik with no match."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "1": {"cik_str": 12345, "ticker": "AAPL", "title": "Apple Inc."}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = lookup_cik("NONEXISTENT")
        assert result is None

    @patch('httpx.get')
    def test_lookup_cik_fuzzy_match(self, mock_get):
        """Test lookup_cik with fuzzy title match."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "1": {"cik_str": 12345, "ticker": "AAPL", "title": "Apple Inc."}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = lookup_cik("apple")
        assert result == "0000012345"

    @patch('httpx.get')
    def test_list_filings_success(self, mock_get):
        """Test list_filings with successful response."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "filings": {
                "recent": {
                    "accessionNumber": ["0001234567-23-000001", "0001234567-23-000002"],
                    "form": ["10-K", "10-Q"],
                    "filingDate": ["2023-12-31", "2023-09-30"],
                    "primaryDocument": ["d1234567.htm", "d2345678.htm"]
                }
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        filings = list_filings("12345", forms=["10-K"], limit=1)

        assert len(filings) == 1
        assert filings[0].cik == "0000012345"
        assert filings[0].form == "10-K"
        assert filings[0].accession == "0001234567-23-000001"

    @patch('httpx.get')
    def test_list_filings_filtered_forms(self, mock_get):
        """Test list_filings filters by form type."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "filings": {
                "recent": {
                    "accessionNumber": ["0001234567-23-000001", "0001234567-23-000002"],
                    "form": ["10-K", "8-K"],
                    "filingDate": ["2023-12-31", "2023-09-30"],
                    "primaryDocument": ["d1234567.htm", "d2345678.htm"]
                }
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        filings = list_filings("12345", forms=["10-K"], limit=5)

        assert len(filings) == 1
        assert filings[0].form == "10-K"

    @patch('httpx.get')
    def test_fetch_text_html(self, mock_get):
        """Test fetch_text with HTML content."""
        mock_response = Mock()
        mock_response.text = "<html><body><script>ignore</script><p>Test content</p></body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        filing = Filing(
            cik="00000012345",
            accession="0001234567-23-000001",
            form="10-K",
            filing_date="2023-12-31",
            primary_doc="d1234567.htm",
            url="https://www.sec.gov/test"
        )

        text = fetch_text(filing)
        assert "Test content" in text
        assert "ignore" not in text

    @patch('httpx.get')
    def test_fetch_text_plain(self, mock_get):
        """Test fetch_text with plain text content."""
        mock_response = Mock()
        mock_response.text = "Plain text content"
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        filing = Filing(
            cik="00000012345",
            accession="0001234567-23-000001",
            form="10-K",
            filing_date="2023-12-31",
            primary_doc="d1234567.txt",
            url="https://www.sec.gov/test"
        )

        text = fetch_text(filing)
        assert text == "Plain text content"

    def test_chunk_text_basic(self):
        """Test chunk_text with basic text."""
        text = "This is a test string that should be chunked into smaller pieces."
        chunks = chunk_text(text, size=20, overlap=5)

        assert len(chunks) > 1
        assert all(len(chunk) <= 20 for chunk in chunks)
        # Check overlap - second chunk should start with some text from first
        if len(chunks) > 1:
            assert chunks[1].startswith(chunks[0][-5:])

    def test_chunk_text_no_overlap(self):
        """Test chunk_text with no overlap."""
        text = "Short text"
        chunks = chunk_text(text, size=10, overlap=0)

        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_chunk_text_empty(self):
        """Test chunk_text with empty/whitespace text."""
        chunks = chunk_text("   \n\n   ", size=10, overlap=2)
        assert chunks == []