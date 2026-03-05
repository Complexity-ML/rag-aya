"""
RAG-Aya :: Generator

Aya 23 via Cohere API for multilingual generation.
"""

from typing import List, Optional
from dataclasses import dataclass

import cohere


@dataclass
class GenerationResult:
    answer: str
    context: str
    query: str
    model: str


class AyaGenerator:
    def __init__(self, api_key: str, model: str = "c4ai-aya-23-8b"):
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

        # Build the prompt with context
        preamble = "You are a helpful multilingual research assistant."
        if language:
            preamble += f" Respond in {language}."

        # Format context as documents for Cohere's RAG
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
