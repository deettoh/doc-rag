"""Embedding service using Sentence Transformers with BAAI/bge-base-en-v1.5."""

from sentence_transformers import SentenceTransformer

from app.config import settings
from app.services.chunking import ChunkData


class EmbeddingService:
    """Generate text embeddings using a local Sentence Transformer model."""

    def __init__(self, model_name: str = settings.embedding_model) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = settings.embedding_dimension

    def generate_embeddings(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of texts in batches.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once internally

        Returns:
            List of embedding vectors (each 768-dim for bge-base-en-v1.5)
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_chunks(self, chunks: list[ChunkData]) -> list[list[float]]:
        """
        Generate embeddings for a list of ChunkData objects.

        Args:
            chunks: List of ChunkData from ChunkingService

        Returns:
            List of embedding vectors, one per chunk
        """
        texts = [chunk.content for chunk in chunks]
        return self.generate_embeddings(texts)
