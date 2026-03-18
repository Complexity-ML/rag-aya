"""
RAG-Aya :: Tokenizer Factory

Provides tokenizer implementations for different embedders:
- CohereTokenizer: Uses Cohere API tokenizer
- LocalTokenizer: Uses sentence-transformers tokenizer
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from logger import init_logger

logger = init_logger(__name__)


class Tokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in a single text string."""
        pass

    @abstractmethod
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens in a batch of text strings."""
        pass

    @abstractmethod
    def get_token_limit(self) -> int:
        """Get maximum safe token limit for this tokenizer's embedder."""
        pass


class CohereTokenizer(Tokenizer):
    """Tokenizer for Cohere API embeddings (embed-multilingual-v3.0)."""

    def __init__(self, cohere_client):
        """
        Initialize Cohere tokenizer.

        Args:
            cohere_client: Initialized cohere.Client instance
        """
        self.client = cohere_client
        self._token_limit = 8000

    def count_tokens(self, text: str) -> int:
        """Count tokens using Cohere's tokenizer."""
        try:
            response = self.client.tokenize(text)
            return len(response.tokens)
        except Exception as e:
            logger.warning("Failed to tokenize with Cohere: %s. Falling back to character estimate.", e)
            # Fallback: rough estimate (0.25 tokens per character for English)
            return max(1, len(text) // 4)

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for multiple texts."""
        return [self.count_tokens(text) for text in texts]

    def get_token_limit(self) -> int:
        """Max safe tokens for embed-multilingual-v3.0 (actual: 8192, safe: 8000)."""
        return self._token_limit


class LocalTokenizer(Tokenizer):
    """Tokenizer for local sentence-transformers embeddings."""

    def __init__(self, st_tokenizer):
        """
        Initialize local tokenizer.

        Args:
            st_tokenizer: The tokenizer from a SentenceTransformer model
        """
        self.tokenizer = st_tokenizer
        self._token_limit = 480

    def count_tokens(self, text: str) -> int:
        """Count tokens using sentence-transformers tokenizer."""
        try:
            # Tokenize and get token count
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            if isinstance(tokens, dict):
                # If dict (some tokenizers return attention_mask, etc.)
                return len(tokens.get("input_ids", []))
            return len(tokens)
        except Exception as e:
            logger.warning("Failed to tokenize locally: %s. Falling back to character estimate.", e)
            # Fallback: rough estimate (0.25 tokens per character for English)
            return max(1, len(text) // 4)

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for multiple texts."""
        return [self.count_tokens(text) for text in texts]

    def get_token_limit(self) -> int:
        """Max safe tokens for MiniLM-L12-v2 (typical limit: 512, safe: 480)."""
        return self._token_limit
