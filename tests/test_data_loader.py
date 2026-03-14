"""
Tests for data_loader.py — load_from_files() and load_wikipedia().

All external dependencies (datasets library) are mocked.
"""

from unittest.mock import patch, MagicMock
import pytest

from data_loader import load_from_files, load_wikipedia


# ---------------------------------------------------------------------------
# load_from_files tests
# ---------------------------------------------------------------------------

class TestLoadFromFiles:

    def test_load_from_files(self, tmp_path):
        """Reads temp .txt files and returns correct documents."""
        f1 = tmp_path / "doc1.txt"
        f2 = tmp_path / "doc2.txt"
        f1.write_text("Hello world", encoding="utf-8")
        f2.write_text("Bonjour le monde", encoding="utf-8")

        docs = load_from_files([str(f1), str(f2)], language="fr")

        assert len(docs) == 2
        assert docs[0]["id"] == str(f1)
        assert docs[0]["text"] == "Hello world"
        assert docs[0]["language"] == "fr"
        assert docs[1]["text"] == "Bonjour le monde"

    def test_load_from_files_empty(self, tmp_path):
        """A file with only whitespace is skipped."""
        f = tmp_path / "blank.txt"
        f.write_text("   \n\n  ", encoding="utf-8")

        docs = load_from_files([str(f)])
        assert len(docs) == 0


# ---------------------------------------------------------------------------
# load_wikipedia tests (mocked)
# ---------------------------------------------------------------------------

class TestLoadWikipedia:

    @patch("data_loader.load_dataset")
    def test_load_wikipedia_mocked(self, mock_load_dataset):
        """Mocked datasets.load_dataset → correct document structure."""
        # Simulate a HuggingFace dataset with 2 rows
        fake_rows = [
            {"text": "A" * 200, "title": "Article_One"},
            {"text": "B" * 200, "title": "Article_Two"},
        ]
        mock_load_dataset.return_value = fake_rows

        docs = load_wikipedia(languages=["en"], max_per_lang=2)

        assert mock_load_dataset.called
        assert len(docs) == 2
        assert docs[0]["language"] == "en"
        assert "Article_One" in docs[0]["id"]
        assert docs[0]["text"] == "A" * 200

    @patch("data_loader.load_dataset")
    def test_load_wikipedia_skips_short(self, mock_load_dataset):
        """Documents with text ≤100 chars are excluded."""
        fake_rows = [
            {"text": "short", "title": "Tiny"},
            {"text": "X" * 200, "title": "Long_Enough"},
        ]
        mock_load_dataset.return_value = fake_rows

        docs = load_wikipedia(languages=["en"], max_per_lang=2)
        assert len(docs) == 1
        assert docs[0]["title"] == "Long_Enough"
