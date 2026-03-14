"""
Tests for chunker.py — chunk_text() and chunk_documents().
"""

import pytest
from chunker import Chunk, chunk_text, chunk_documents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_text(length: int, char: str = "a") -> str:
    """Return a string of exactly `length` characters."""
    return char * length


# ---------------------------------------------------------------------------
# chunk_text tests
# ---------------------------------------------------------------------------

class TestChunkText:

    def test_basic_chunking(self):
        """Known 1024-char string → expected number of chunks."""
        text = _make_text(1024)
        chunks = chunk_text(text, doc_id="d1", chunk_size=512, overlap=64)

        # stride = 512 - 64 = 448
        # positions: 0, 448, 896  → 3 chunks
        assert len(chunks) == 3
        assert all(isinstance(c, Chunk) for c in chunks)
        assert chunks[0].doc_id == "d1"
        assert chunks[0].chunk_idx == 0
        assert chunks[1].chunk_idx == 1

    def test_overlap_behavior(self):
        """Last `overlap` chars of chunk N appear at the start of chunk N+1."""
        text = "".join(str(i % 10) for i in range(200))
        overlap = 20
        chunks = chunk_text(text, doc_id="d1", chunk_size=100, overlap=overlap)

        for i in range(len(chunks) - 1):
            tail = chunks[i].text[-overlap:]
            head = chunks[i + 1].text[:overlap]
            assert tail == head, (
                f"Chunk {i} tail != chunk {i+1} head: {tail!r} vs {head!r}"
            )

    def test_empty_string(self):
        """Empty input → no chunks."""
        assert chunk_text("", doc_id="d1") == []

    def test_shorter_than_chunk_size(self):
        """String shorter than chunk_size → single chunk."""
        text = "hello"
        chunks = chunk_text(text, doc_id="d1", chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0].text == "hello"

    def test_single_character(self):
        """Single character → single chunk."""
        chunks = chunk_text("x", doc_id="d1", chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0].text == "x"

    def test_language_propagation(self):
        """language parameter flows into every Chunk.language."""
        chunks = chunk_text("bonjour le monde", doc_id="d1", language="fr")
        assert all(c.language == "fr" for c in chunks)


# ---------------------------------------------------------------------------
# chunk_documents tests
# ---------------------------------------------------------------------------

class TestChunkDocuments:

    def test_language_propagation_via_documents(self):
        """Language from each doc dict propagates to all its chunks."""
        docs = [
            {"id": "en_doc", "text": _make_text(100), "language": "en"},
            {"id": "fr_doc", "text": _make_text(100), "language": "fr"},
        ]
        chunks = chunk_documents(docs, chunk_size=512, overlap=0)
        en_chunks = [c for c in chunks if c.doc_id == "en_doc"]
        fr_chunks = [c for c in chunks if c.doc_id == "fr_doc"]
        assert all(c.language == "en" for c in en_chunks)
        assert all(c.language == "fr" for c in fr_chunks)

    def test_default_language(self):
        """Missing language key defaults to 'en'."""
        docs = [{"id": "d1", "text": "some text"}]
        chunks = chunk_documents(docs)
        assert chunks[0].language == "en"
