import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from retriever import DenseRetriever, Passage, Hit, DEFAULT_MODEL
from sec import Filing


class TestRetriever:
    def test_passage_dataclass(self):
        """Test Passage dataclass creation."""
        filing = Filing(
            cik="00000012345",
            accession="0001234567-23-000001",
            form="10-K",
            filing_date="2023-12-31",
            primary_doc="d1234567.htm",
            url="https://www.sec.gov/test"
        )
        passage = Passage(text="Test passage", filing=filing, chunk_index=0)

        assert passage.text == "Test passage"
        assert passage.filing == filing
        assert passage.chunk_index == 0

    def test_hit_dataclass(self):
        """Test Hit dataclass creation."""
        filing = Filing(
            cik="00000012345",
            accession="0001234567-23-000001",
            form="10-K",
            filing_date="2023-12-31",
            primary_doc="d1234567.htm",
            url="https://www.sec.gov/test"
        )
        passage = Passage(text="Test passage", filing=filing, chunk_index=0)
        hit = Hit(passage=passage, score=0.95)

        assert hit.passage == passage
        assert hit.score == 0.95

    @patch('retriever.SentenceTransformer')
    def test_dense_retriever_init(self, mock_model):
        """Test DenseRetriever initialization."""
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance

        retriever = DenseRetriever("test-model")

        mock_model.assert_called_once_with("test-model")
        assert retriever.model == mock_model_instance
        assert retriever._passages == []
        assert retriever._matrix is None

    @patch('retriever.SentenceTransformer')
    def test_dense_retriever_init_default_model(self, mock_model):
        """Test DenseRetriever initialization with default model."""
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance

        retriever = DenseRetriever()

        mock_model.assert_called_once_with(DEFAULT_MODEL)

    @patch('retriever.SentenceTransformer')
    def test_add_passages(self, mock_model):
        """Test adding passages to the retriever."""
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance

        retriever = DenseRetriever()
        filing = Filing(
            cik="00000012345",
            accession="0001234567-23-000001",
            form="10-K",
            filing_date="2023-12-31",
            primary_doc="d1234567.htm",
            url="https://www.sec.gov/test"
        )
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]

        retriever.add(filing, chunks)

        assert len(retriever._passages) == 3
        assert retriever._passages[0].text == "Chunk 1"
        assert retriever._passages[0].filing == filing
        assert retriever._passages[0].chunk_index == 0
        assert retriever._passages[1].chunk_index == 1
        assert retriever._passages[2].chunk_index == 2

    @patch('retriever.SentenceTransformer')
    def test_build_index(self, mock_model):
        """Test building the index."""
        mock_model_instance = Mock()
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_model_instance.encode.return_value = mock_embeddings
        mock_model.return_value = mock_model_instance

        retriever = DenseRetriever()
        filing = Filing(
            cik="00000012345",
            accession="0001234567-23-000001",
            form="10-K",
            filing_date="2023-12-31",
            primary_doc="d1234567.htm",
            url="https://www.sec.gov/test"
        )
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        retriever.add(filing, chunks)

        retriever.build()

        mock_model_instance.encode.assert_called_once_with(
            ["Chunk 1", "Chunk 2", "Chunk 3"],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        assert retriever._matrix is not None
        np.testing.assert_array_equal(retriever._matrix, mock_embeddings.astype(np.float32))

    @patch('retriever.SentenceTransformer')
    def test_build_empty_index_raises_error(self, mock_model):
        """Test that building an empty index raises ValueError."""
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance

        retriever = DenseRetriever()

        with pytest.raises(ValueError, match="No passages to index"):
            retriever.build()

    @patch('retriever.SentenceTransformer')
    def test_search_without_build_raises_error(self, mock_model):
        """Test that searching without building index raises RuntimeError."""
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance

        retriever = DenseRetriever()

        with pytest.raises(RuntimeError, match="Index not built"):
            retriever.search("test query")

    @patch('retriever.SentenceTransformer')
    def test_search_functionality(self, mock_model):
        """Test search functionality with mocked embeddings."""
        # Mock the model
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance

        # Mock embeddings for passages
        passage_embeddings = np.array([
            [1.0, 0.0],  # Passage 1
            [0.0, 1.0],  # Passage 2
            [0.5, 0.5],  # Passage 3
        ], dtype=np.float32)

        # Mock query embedding
        query_embedding = np.array([[0.8, 0.6]], dtype=np.float32)

        mock_model_instance.encode.side_effect = [passage_embeddings, query_embedding]

        # Set up retriever with passages
        retriever = DenseRetriever()
        filing = Filing(
            cik="00000012345",
            accession="0001234567-23-000001",
            form="10-K",
            filing_date="2023-12-31",
            primary_doc="d1234567.htm",
            url="https://www.sec.gov/test"
        )
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        retriever.add(filing, chunks)
        retriever.build()

        # Perform search
        results = retriever.search("test query", top_k=2)

        assert len(results) == 2
        assert isinstance(results[0], Hit)
        assert isinstance(results[1], Hit)
        assert results[0].passage.text in ["Chunk 1", "Chunk 2", "Chunk 3"]
        assert results[1].passage.text in ["Chunk 1", "Chunk 2", "Chunk 3"]
        assert results[0].score >= results[1].score  # Should be sorted by score

    @patch('retriever.SentenceTransformer')
    def test_search_single_result(self, mock_model):
        """Test search with top_k=1."""
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance

        passage_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        query_embedding = np.array([[0.9, 0.1]], dtype=np.float32)

        mock_model_instance.encode.side_effect = [passage_embeddings, query_embedding]

        retriever = DenseRetriever()
        filing = Filing(
            cik="00000012345",
            accession="0001234567-23-000001",
            form="10-K",
            filing_date="2023-12-31",
            primary_doc="d1234567.htm",
            url="https://www.sec.gov/test"
        )
        chunks = ["Chunk 1", "Chunk 2"]
        retriever.add(filing, chunks)
        retriever.build()

        results = retriever.search("test query", top_k=1)

        assert len(results) == 1
        assert isinstance(results[0], Hit)

    @patch('retriever.SentenceTransformer')
    def test_size_property(self, mock_model):
        """Test the size property."""
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance

        retriever = DenseRetriever()

        assert retriever.size == 0

        filing = Filing(
            cik="00000012345",
            accession="0001234567-23-000001",
            form="10-K",
            filing_date="2023-12-31",
            primary_doc="d1234567.htm",
            url="https://www.sec.gov/test"
        )
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4"]
        retriever.add(filing, chunks)

        assert retriever.size == 4

    @patch('retriever.SentenceTransformer')
    def test_multiple_filings(self, mock_model):
        """Test adding passages from multiple filings."""
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance

        retriever = DenseRetriever()

        filing1 = Filing(
            cik="00000012345",
            accession="0001234567-23-000001",
            form="10-K",
            filing_date="2023-12-31",
            primary_doc="d1234567.htm",
            url="https://www.sec.gov/test1"
        )
        filing2 = Filing(
            cik="00000067890",
            accession="0001234567-23-000002",
            form="10-Q",
            filing_date="2023-09-30",
            primary_doc="d2345678.htm",
            url="https://www.sec.gov/test2"
        )

        retriever.add(filing1, ["Filing1 Chunk1", "Filing1 Chunk2"])
        retriever.add(filing2, ["Filing2 Chunk1"])

        assert retriever.size == 3
        assert retriever._passages[0].filing == filing1
        assert retriever._passages[2].filing == filing2