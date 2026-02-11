"""Embedding service using Sentence Transformers with BAAI/bge-base-en-v1.5."""

from sentence_transformers import SentenceTransformer

from app.services.chunking import ChunkData

# Model configuration
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = 768


class EmbeddingService:
    """Generate text embeddings using a local Sentence Transformer model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = EMBEDDING_DIMENSION

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each 768-dim for bge-base-en-v1.5)
        """
        if not texts:
            return []

        embeddings = self.model.encode(texts, convert_to_numpy=True)
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
