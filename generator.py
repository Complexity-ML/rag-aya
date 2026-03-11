"""
RAG-Aya :: Generator

Supports two backends:
  - Cohere API (Aya 23)
  - Local GGUF via llama-cpp-python (Tiny Aya)
"""

from typing import List, Optional
from dataclasses import dataclass

from logger import init_logger

logger = init_logger(__name__)


@dataclass
class GenerationResult:
    answer: str
    context: str
    query: str
    model: str


class AyaGenerator:
    """Cohere API backend."""

    def __init__(self, api_key: str, model: str = "c4ai-aya-23-8b"):
        import cohere
        self.client = cohere.Client(api_key)
        self.model = model

    def generate(
        self,
        query: str,
        context: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        language: Optional[str] = None,
    ) -> GenerationResult:
        """Generate an answer using Aya with RAG context."""

        preamble = "You are a helpful multilingual research assistant."
        if language:
            preamble += f" Respond in {language}."

        documents = []
        for i, chunk in enumerate(context.split("\n\n")):
            if chunk.strip():
                documents.append({"title": f"Source {i+1}", "text": chunk.strip()})

        response = self.client.chat(
            model=self.model,
            message=query,
            documents=documents if documents else None,
            preamble=preamble,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return GenerationResult(
            answer=response.text,
            context=context,
            query=query,
            model=self.model,
        )

    def generate_batch(
        self,
        queries: List[str],
        contexts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> List[GenerationResult]:
        """Generate answers for multiple queries."""
        results = []
        for query, context in zip(queries, contexts):
            result = self.generate(query, context, max_tokens=max_tokens, temperature=temperature)
            results.append(result)
        return results


class GGUFGenerator:
    """Local GGUF backend via llama-cpp-python."""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1):
        from llama_cpp import Llama
        logger.info("Loading GGUF model: %s", model_path)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.model_path = model_path
        self.model = model_path.split("/")[-1].split("\\")[-1]
        logger.info("GGUF model loaded: %s", self.model)

    def generate(
        self,
        query: str,
        context: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        language: Optional[str] = None,
    ) -> GenerationResult:
        """Generate an answer using local GGUF model with RAG context."""

        lang_hint = f" Respond in {language}." if language else ""
        prompt = (
            f"You are a helpful multilingual research assistant.{lang_hint}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["Question:", "\n\n\n"],
        )

        answer = output["choices"][0]["text"].strip()

        return GenerationResult(
            answer=answer,
            context=context,
            query=query,
            model=self.model,
        )

    def generate_batch(
        self,
        queries: List[str],
        contexts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> List[GenerationResult]:
        """Generate answers for multiple queries."""
        results = []
        for query, context in zip(queries, contexts):
            result = self.generate(query, context, max_tokens=max_tokens, temperature=temperature)
            results.append(result)
        return results
