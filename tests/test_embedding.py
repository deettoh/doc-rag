"""Tests for the embedding service."""

from unittest.mock import MagicMock, patch

import numpy as np

from app.config import settings
from app.services.chunking import ChunkData
from app.services.embedding import EmbeddingService


class TestEmbeddingService:
    """Tests for EmbeddingService."""

    @patch("app.services.embedding.SentenceTransformer")
    def test_generate_embeddings_returns_correct_shape(
        self, mock_sentence_transformer_cls: MagicMock
    ) -> None:
        """Embeddings should have the correct dimension."""
        mock_model = MagicMock()
        # Mock encode to return a numpy array of shape (2, 768)
        mock_model.encode.return_value = np.random.rand(2, settings.embedding_dimension)
        mock_sentence_transformer_cls.return_value = mock_model

        service = EmbeddingService()
        results = service.generate_embeddings(["Hello world", "Test text"])

        assert len(results) == 2
        assert all(len(emb) == settings.embedding_dimension for emb in results)
        assert isinstance(results, list)
        assert isinstance(results[0], list)

    @patch("app.services.embedding.SentenceTransformer")
    def test_generate_embeddings_returns_lists(
        self, mock_sentence_transformer_cls: MagicMock
    ) -> None:
        """Embeddings should be plain lists, not numpy arrays."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_sentence_transformer_cls.return_value = mock_model

        service = EmbeddingService()
        results = service.generate_embeddings(["text"])

        assert isinstance(results[0], list)
        assert isinstance(results[0][0], float)

    @patch("app.services.embedding.SentenceTransformer")
    def test_generate_embeddings_empty_input(
        self, mock_sentence_transformer_cls: MagicMock
    ) -> None:
        """Empty input should return empty list without calling the model."""
        mock_model = MagicMock()
        mock_sentence_transformer_cls.return_value = mock_model

        service = EmbeddingService()
        results = service.generate_embeddings([])

        assert results == []
        mock_model.encode.assert_not_called()

    @patch("app.services.embedding.SentenceTransformer")
    def test_generate_embeddings_calls_model_with_texts(
        self, mock_sentence_transformer_cls: MagicMock
    ) -> None:
        """Model should be called with the exact input texts."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((2, settings.embedding_dimension))
        mock_sentence_transformer_cls.return_value = mock_model

        service = EmbeddingService()
        texts = ["First chunk", "Second chunk"]
        service.generate_embeddings(texts)

        # check that convert_to_numpy=True is passed
        mock_model.encode.assert_called_once_with(texts, convert_to_numpy=True)

    @patch("app.services.embedding.SentenceTransformer")
    def test_embed_chunks_extracts_content(
        self, mock_sentence_transformer_cls: MagicMock
    ) -> None:
        """embed_chunks should extract .content and pass to generate_embeddings."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((2, settings.embedding_dimension))
        mock_sentence_transformer_cls.return_value = mock_model

        service = EmbeddingService()
        chunks = [
            ChunkData(
                document_id=1,
                chunk_index=0,
                content="Hello world",
                page_start=1,
                page_end=1,
                char_start=0,
                char_end=11,
            ),
            ChunkData(
                document_id=1,
                chunk_index=1,
                content="Test text",
                page_start=1,
                page_end=1,
                char_start=12,
                char_end=21,
            ),
        ]
        results = service.embed_chunks(chunks)

        assert len(results) == 2
        mock_model.encode.assert_called_once_with(
            ["Hello world", "Test text"], convert_to_numpy=True
        )

    @patch("app.services.embedding.SentenceTransformer")
    def test_model_loaded_with_correct_name(
        self, mock_sentence_transformer_cls: MagicMock
    ) -> None:
        """Model should be initialized with the configured model name."""
        EmbeddingService()
        mock_sentence_transformer_cls.assert_called_once_with(settings.embedding_model)

    @patch("app.services.embedding.SentenceTransformer")
    def test_single_text_embedding(
        self, mock_sentence_transformer_cls: MagicMock
    ) -> None:
        """Single text should return a list with one embedding."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, settings.embedding_dimension)
        mock_sentence_transformer_cls.return_value = mock_model

        service = EmbeddingService()
        results = service.generate_embeddings(["single text"])

        assert len(results) == 1
        assert len(results[0]) == settings.embedding_dimension
