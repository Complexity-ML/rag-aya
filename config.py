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

    # Local model (aya-offline engine)
    gguf_path: str = ""
    engine_port: int = 8089

    # Local embedder
    local_embedder: bool = False
    local_embed_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # Data
    languages: List[str] = field(default_factory=lambda: ["en", "fr"])
    max_documents: int = 100

    # WMT
    wmt_cache_dir: str = "wmt_data"
    wmt_max_lines: int = 500

    def validate(self, require_cohere: bool = True):
        if require_cohere and not self.cohere_api_key:
            raise ValueError(
                "COHERE_API_KEY not set.\n\n"
                "  1. Get your key at: https://dashboard.cohere.com/api-keys\n"
                "  2. Create a .env file:  cp .env.example .env\n"
                "  3. Add your key:        COHERE_API_KEY=your_key_here\n\n"
                "  Or export directly:      export COHERE_API_KEY=your_key_here"
            )
