"""
RAG-Aya :: Configuration

Cohere API + pipeline settings.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Cohere API
    cohere_api_key: str = field(default_factory=lambda: os.environ.get("COHERE_API_KEY", ""))
    embed_model: str = "embed-multilingual-v3.0"
    gen_model: str = "c4ai-aya-23-8b"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k: int = 5
    index_path: str = "index/"

    # Generation
    max_tokens: int = 512
    temperature: float = 0.3

    # Data
    languages: List[str] = field(default_factory=lambda: ["en", "fr"])
    max_documents: int = 100

    def validate(self):
        if not self.cohere_api_key:
            raise ValueError("COHERE_API_KEY not set. Export it or pass via .env")
