"""
RAG-Aya :: Embedder

Cohere embed-multilingual-v3.0 for multilingual embeddings.
"""

import numpy as np
from typing import List

import cohere

from chunker import Chunk


BATCH_SIZE = 96  # Cohere API limit per call


class CohereEmbedder:
    def __init__(self, api_key: str, model: str = "embed-multilingual-v3.0"):
        self.client = cohere.Client(api_key)
        self.model = model
        self.dimension = None

    def embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """Embed a list of chunks. Returns (N, dim) float32 array."""
        texts = [c.text for c in chunks]
        return self._embed_texts(texts, input_type="search_document")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns (1, dim) float32 array."""
        return self._embed_texts([query], input_type="search_query")

    def _embed_texts(self, texts: List[str], input_type: str) -> np.ndarray:
        """Batch embed texts via Cohere API."""
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
