"""
RAG-Aya :: Retriever

Vector index with cosine similarity search.
Stores chunks + embeddings, persists to disk.
"""

import json
import os
import numpy as np
from typing import List, Tuple, Optional

from chunker import Chunk
from embedder import CohereEmbedder
from logger import init_logger

logger = init_logger(__name__)


class Retriever:
    def __init__(self, embedder: CohereEmbedder):
        self.embedder = embedder
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def index(self, chunks: List[Chunk]):
        """Embed and index a list of chunks."""
        if not chunks:
            return
        new_embeddings = self.embedder.embed_chunks(chunks)
        if self.embeddings is None:
            self.chunks = chunks
            self.embeddings = new_embeddings
        else:
            self.chunks.extend(chunks)
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        logger.info("Indexed %d chunks (total: %d)", len(chunks), len(self.chunks))

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        """Search for top-k most similar chunks."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_emb = self.embedder.embed_query(query)  # (1, dim)

        # Cosine similarity
        norms_chunks = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms_query = np.linalg.norm(query_emb, axis=1, keepdims=True)
        similarities = (self.embeddings @ query_emb.T) / (norms_chunks * norms_query.T + 1e-8)
        similarities = similarities.squeeze()

        top_k_idx = np.argsort(similarities)[::-1][:k]
        return [(self.chunks[i], float(similarities[i])) for i in top_k_idx]

    def get_context(self, query: str, k: int = 5) -> str:
        """Search and format results as context string."""
        results = self.search(query, k=k)
        if not results:
            return ""
        parts = []
        for chunk, score in results:
            parts.append(f"[{chunk.language}|{chunk.doc_id}] {chunk.text}")
        return "\n\n".join(parts)

    def save(self, path: str):
        """Save index to disk."""
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        chunks_data = [
            {"text": c.text, "doc_id": c.doc_id, "chunk_idx": c.chunk_idx, "language": c.language}
            for c in self.chunks
        ]
        with open(os.path.join(path, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False)
        logger.info("Saved index: %d chunks -> %s", len(self.chunks), path)

    def load(self, path: str):
        """Load index from disk."""
        self.embeddings = np.load(os.path.join(path, "embeddings.npy"))
        with open(os.path.join(path, "chunks.json"), "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        self.chunks = [
            Chunk(text=c["text"], doc_id=c["doc_id"], chunk_idx=c["chunk_idx"], language=c.get("language", "en"))
            for c in chunks_data
        ]
        logger.info("Loaded index: %d chunks from %s", len(self.chunks), path)

    @property
    def stats(self) -> dict:
        return {
            "num_chunks": len(self.chunks),
            "dimension": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "languages": list(set(c.language for c in self.chunks)),
        }
