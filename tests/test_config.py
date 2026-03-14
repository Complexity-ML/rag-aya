"""
Tests for config.py — Config dataclass.
"""

import os
import pytest
from config import Config


class TestConfig:

    def test_default_values(self):
        """Config() fields match expected defaults."""
        # Ensure env var doesn't leak into the test
        env_backup = os.environ.pop("COHERE_API_KEY", None)
        try:
            cfg = Config(cohere_api_key="")
            assert cfg.embed_model == "embed-multilingual-v3.0"
            assert cfg.gen_model == "c4ai-aya-23-8b"
            assert cfg.chunk_size == 512
            assert cfg.chunk_overlap == 64
            assert cfg.top_k == 5
            assert cfg.index_path == "index/"
            assert cfg.max_tokens == 512
            assert cfg.temperature == 0.3
            assert cfg.languages == ["en", "fr"]
            assert cfg.max_documents == 100
        finally:
            if env_backup is not None:
                os.environ["COHERE_API_KEY"] = env_backup

    def test_validate_raises_without_api_key(self):
        """validate() raises ValueError when cohere_api_key is empty."""
        cfg = Config(cohere_api_key="")
        with pytest.raises(ValueError, match="COHERE_API_KEY not set"):
            cfg.validate()

    def test_validate_passes_with_api_key(self):
        """validate() does not raise when a key is provided."""
        cfg = Config(cohere_api_key="test-key-12345")
        cfg.validate()  # should not raise
