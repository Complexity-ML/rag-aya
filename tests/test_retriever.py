"""
Tests for retriever.py — Retriever class.

Uses a FakeEmbedder to avoid any Cohere API calls.
The cohere package is mocked at import-time so the test suite works
even when cohere is not installed.
"""

import sys
import types
from unittest.mock import MagicMock

# ── Mock the cohere package before importing retriever/embedder ──
_cohere_mock = MagicMock()
sys.modules.setdefault("cohere", _cohere_mock)

import numpy as np
import pytest

from chunker import Chunk
from retriever import Retriever


# ---------------------------------------------------------------------------
# Fake embedder (no network)
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """Deterministic embedder that returns pre-set vectors."""

    def __init__(self, chunk_vectors: np.ndarray, query_vector: np.ndarray):
        """
        Parameters
        ----------
        chunk_vectors : (N, dim)  embeddings returned by embed_chunks
        query_vector  : (1, dim)  embedding returned by embed_query
        """
        self._chunk_vectors = chunk_vectors
        self._query_vector = query_vector

    def embed_chunks(self, chunks):
        return self._chunk_vectors[: len(chunks)]

    def embed_query(self, query: str):
        return self._query_vector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunks(n: int) -> list:
    return [
        Chunk(text=f"chunk_{i}", doc_id="d1", chunk_idx=i, language="en")
        for i in range(n)
    ]


def _unit(vec):
    """Normalise a vector to unit length."""
    v = np.array(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRetriever:

    def test_index_search_roundtrip(self):
        """Index 3 chunks → search returns the closest match first."""
        chunk_vecs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        query_vec = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        embedder = FakeEmbedder(chunk_vecs, query_vec)
        retriever = Retriever(embedder)
        chunks = _make_chunks(3)
        retriever.index(chunks)

        results = retriever.search("anything", k=3)
        assert len(results) == 3
        # First result should be chunk_0 (vector aligned with query)
        assert results[0][0].text == "chunk_0"

    def test_save_load_persistence(self, tmp_path):
        """save() then load() restores identical state."""
        chunk_vecs = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ], dtype=np.float32)
        query_vec = np.array([[1.0, 0.0]], dtype=np.float32)

        embedder = FakeEmbedder(chunk_vecs, query_vec)
        r1 = Retriever(embedder)
        chunks = _make_chunks(2)
        r1.index(chunks)

        save_dir = str(tmp_path / "idx")
        r1.save(save_dir)

        r2 = Retriever(embedder)
        r2.load(save_dir)

        assert len(r2.chunks) == len(r1.chunks)
        np.testing.assert_array_equal(r2.embeddings, r1.embeddings)
        for a, b in zip(r1.chunks, r2.chunks):
            assert a.text == b.text
            assert a.doc_id == b.doc_id
            assert a.language == b.language

    def test_cosine_similarity_correctness(self):
        """Hand-crafted vectors → verify similarity scores."""
        # Two chunks: one aligned with query, one orthogonal
        chunk_vecs = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        query_vec = np.array([[1.0, 0.0]], dtype=np.float32)

        embedder = FakeEmbedder(chunk_vecs, query_vec)
        retriever = Retriever(embedder)
        retriever.index(_make_chunks(2))

        results = retriever.search("q", k=2)
        # Chunk 0 aligned → similarity ≈ 1.0
        assert results[0][0].text == "chunk_0"
        assert pytest.approx(results[0][1], abs=1e-5) == 1.0
        # Chunk 1 orthogonal → similarity ≈ 0.0
        assert results[1][0].text == "chunk_1"
        assert pytest.approx(results[1][1], abs=1e-5) == 0.0

    def test_top_k_parameter(self):
        """k controls number of results returned."""
        n = 5
        chunk_vecs = np.eye(n, dtype=np.float32)
        query_vec = np.array([[1.0] + [0.0] * (n - 1)], dtype=np.float32)

        embedder = FakeEmbedder(chunk_vecs, query_vec)
        retriever = Retriever(embedder)
        retriever.index(_make_chunks(n))

        assert len(retriever.search("q", k=1)) == 1
        assert len(retriever.search("q", k=3)) == 3

    def test_empty_index(self):
        """Search on an empty retriever returns []."""
        embedder = FakeEmbedder(np.empty((0, 3)), np.zeros((1, 3)))
        retriever = Retriever(embedder)
        assert retriever.search("hello") == []
