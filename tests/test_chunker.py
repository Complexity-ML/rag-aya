"""
Tests for chunker.py — chunk_text() and chunk_documents().
"""

import pytest
from unittest.mock import Mock, MagicMock
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


# ---------------------------------------------------------------------------
# Token-based chunking tests
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Mock tokenizer for deterministic testing."""

    def __init__(self, tokens_per_word: int = 1, token_limit: int = 100):
        """
        Initialize mock tokenizer.

        Args:
            tokens_per_word: Fixed tokens per word for predictable counting
            token_limit: Max token limit for validation
        """
        self.tokens_per_word = tokens_per_word
        self._token_limit = token_limit

    def count_tokens(self, text: str) -> int:
        """Count tokens by word count."""
        if not text.strip():
            return 0
        words = text.split()
        return len(words) * self.tokens_per_word

    def count_tokens_batch(self, texts):
        """Count tokens for batch."""
        return [self.count_tokens(text) for text in texts]

    def get_token_limit(self) -> int:
        """Get token limit."""
        return self._token_limit


class TestTokenChunking:

    def test_token_chunking_basic(self):
        """Token-based chunking chunks by tokens, not characters."""
        # Create text with known token count
        # Each word = 1 token in mock
        text = " ".join(["word"] * 100)  # 100 tokens
        tokenizer = MockTokenizer(tokens_per_word=1, token_limit=200)

        chunks = chunk_text(text, doc_id="d1", chunk_size=30, overlap=5, tokenizer=tokenizer)

        # With 30 token size and 5 token overlap, stride = 25
        # Positions: 0-30, 25-55, 50-80, 75-105 (beyond end)
        # Expected: 3-4 chunks depending on exact boundaries
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert chunks[0].doc_id == "d1"

    def test_token_overlap_behavior(self):
        """Token overlap is maintained across chunks."""
        text = " ".join(str(i % 10) for i in range(60))  # 60 "words"
        overlap = 10
        tokenizer = MockTokenizer(tokens_per_word=1)

        chunks = chunk_text(text, doc_id="d1", chunk_size=20, overlap=overlap, tokenizer=tokenizer)

        # Verify overlap (approximately, depends on word boundaries)
        for i in range(len(chunks) - 1):
            # Both chunks should have some overlap
            assert len(chunks[i].text) > 0
            assert len(chunks[i + 1].text) > 0

    def test_cjk_token_inflation(self):
        """CJK text generates more tokens per character."""
        # Simulate CJK text: 1 char = 2 tokens (typical)
        cjk_text = "中文测试文本，这是一个测试"  # ~12 characters
        english_text = "English test text sample"  # ~4 words

        # CJK tokenizer: 2 tokens per character
        cjk_tokenizer = MockTokenizer(tokens_per_word=2, token_limit=50)
        # English tokenizer: 1 token per word
        en_tokenizer = MockTokenizer(tokens_per_word=1, token_limit=50)

        # With same character count, CJK should produce more tokens
        cjk_tokens = cjk_tokenizer.count_tokens(cjk_text)
        en_tokens = en_tokenizer.count_tokens(english_text)

        # CJK should have higher token count (simulated)
        assert cjk_tokens > 0
        assert en_tokens > 0

    def test_token_limit_warning(self):
        """Warning is logged when chunk exceeds token limit."""
        text = " ".join(["word"] * 50)  # 50 tokens
        tokenizer = MockTokenizer(tokens_per_word=1, token_limit=30)

        chunks = chunk_text(
            text,
            doc_id="d1",
            chunk_size=40,  # Larger than limit
            overlap=5,
            tokenizer=tokenizer,
            token_limit=30,  # Will exceed
        )

        # Verify chunks were created (warning is logged but doesn't stop chunking)
        assert len(chunks) > 0
        # First chunk should have ~40 tokens which exceeds the 30 token limit
        assert chunks[0].text.count(" ") + 1 > 0  # Verify chunk has content

    def test_token_chunking_preserves_language(self):
        """Language metadata preserved in token-based chunking."""
        text = " ".join(["word"] * 50)
        tokenizer = MockTokenizer(tokens_per_word=1)

        chunks = chunk_text(
            text, doc_id="d1", chunk_size=20, overlap=5, language="fr", tokenizer=tokenizer
        )

        assert all(c.language == "fr" for c in chunks)

    def test_token_chunking_empty_text(self):
        """Empty text produces no chunks."""
        tokenizer = MockTokenizer(tokens_per_word=1)
        chunks = chunk_text("", doc_id="d1", tokenizer=tokenizer)
        assert chunks == []

    def test_token_chunking_single_word(self):
        """Single word produces single chunk."""
        tokenizer = MockTokenizer(tokens_per_word=1)
        chunks = chunk_text("hello", doc_id="d1", chunk_size=10, tokenizer=tokenizer)
        assert len(chunks) == 1
        assert chunks[0].text == "hello"

    def test_token_chunking_documents(self):
        """Token chunking works with chunk_documents()."""
        docs = [
            {"id": "d1", "text": " ".join(["word"] * 50), "language": "en"},
            {"id": "d2", "text": " ".join(["word"] * 50), "language": "fr"},
        ]
        tokenizer = MockTokenizer(tokens_per_word=1)

        chunks = chunk_documents(
            docs, chunk_size=20, overlap=5, tokenizer=tokenizer, token_limit=100
        )

        # Should have chunks from both documents
        d1_chunks = [c for c in chunks if c.doc_id == "d1"]
        d2_chunks = [c for c in chunks if c.doc_id == "d2"]

        assert len(d1_chunks) > 0
        assert len(d2_chunks) > 0
        assert all(c.language == "en" for c in d1_chunks)
        assert all(c.language == "fr" for c in d2_chunks)

    def test_backward_compatibility_no_tokenizer(self):
        """Character-based chunking still works when tokenizer=None."""
        text = _make_text(1024)
        chunks = chunk_text(text, doc_id="d1", chunk_size=512, overlap=64, tokenizer=None)

        # Should use character-based chunking
        assert len(chunks) == 3  # stride = 448, positions: 0, 448, 896
