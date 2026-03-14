"""
RAG-Aya :: Embedder

Supports two backends:
  - Cohere API (embed-multilingual-v3.0)
  - Local sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
"""

import numpy as np
from typing import List

from chunker import Chunk
from logger import init_logger

logger = init_logger(__name__)

BATCH_SIZE = 96


class CohereEmbedder:
    def __init__(self, api_key: str, model: str = "embed-multilingual-v3.0"):
        import cohere
        self.client = cohere.Client(api_key)
        self.model = model
        self.dimension = None

    def embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        texts = [c.text for c in chunks]
        return self._embed_texts(texts, input_type="search_document")

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed_texts([query], input_type="search_query")

    def _embed_texts(self, texts: List[str], input_type: str) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            response = self.client.embed(
                texts=batch,
                model=self.model,
                input_type=input_type,
            )
            all_embeddings.extend(response.embeddings)

        result = np.array(all_embeddings, dtype=np.float32)
        self.dimension = result.shape[1]
        return result


class LocalEmbedder:
    """Local embedder via sentence-transformers. No API key needed."""

    def __init__(self, model: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local embedder: %s", model)
        self.st_model = SentenceTransformer(model)
        self.dimension = self.st_model.get_sentence_embedding_dimension()
        logger.info("Local embedder loaded (dim=%d)", self.dimension)

    def embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        texts = [c.text for c in chunks]
        return self.st_model.encode(texts, show_progress_bar=True).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.st_model.encode([query]).astype(np.float32)
