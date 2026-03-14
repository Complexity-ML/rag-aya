"""
RAG-Aya :: Pipeline Factory

Shared backend selection logic for main.py and server.py.
"""

from config import Config
from retriever import Retriever
from logger import init_logger

logger = init_logger(__name__)


def build_embedder(config: Config):
    """Build embedder based on config (Cohere API or local)."""
    if config.local_embedder:
        from embedder import LocalEmbedder
        return LocalEmbedder(config.local_embed_model)
    else:
        from embedder import CohereEmbedder
        config.validate(require_cohere=True)
        return CohereEmbedder(config.cohere_api_key, config.embed_model)


def build_generator(config: Config):
    """Build generator based on config (Cohere API or local GGUF engine)."""
    if config.gguf_path:
        from generator import AyaEngineGenerator
        return AyaEngineGenerator(config.gguf_path, port=config.engine_port)
    else:
        from generator import AyaGenerator
        config.validate(require_cohere=True)
        return AyaGenerator(config.cohere_api_key, config.gen_model)


def build_retriever(config: Config, embedder=None):
    """Build retriever with embedder."""
    if embedder is None:
        embedder = build_embedder(config)
    return Retriever(embedder, cache_size=config.cache_size)


def build_pipeline(config: Config):
    """Build full pipeline: embedder + retriever + generator."""
    embedder = build_embedder(config)
    retriever = build_retriever(config, embedder)
    generator = build_generator(config)
    return embedder, retriever, generator
