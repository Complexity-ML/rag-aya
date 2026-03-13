"""
RAG-Aya :: Generator

Supports two backends:
  - Cohere API (Aya 23)
  - Aya Engine (custom C inference server from aya-offline)
"""

import json
import subprocess
import time
import urllib.request
import urllib.error
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


class AyaEngineGenerator:
    """Local backend via aya-offline C inference server."""

    def __init__(self, gguf_path: str, engine_dir: str = "", port: int = 8089):
        import os
        self.gguf_path = os.path.abspath(gguf_path)
        self.port = port
        self.url = f"http://localhost:{port}"
        self.model = os.path.basename(gguf_path)
        self._process = None

        # Find the engine executable
        if not engine_dir:
            engine_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine")
        self._exe = None
        for name in ["aya-server.exe", "aya_server.exe", "aya-server"]:
            path = os.path.join(engine_dir, name)
            if os.path.exists(path):
                self._exe = path
                break
        if not self._exe:
            raise FileNotFoundError(f"aya-server not found in {engine_dir}")

        self._start_server()

    def _start_server(self):
        """Start the aya-server process if not already running."""
        if self._is_running():
            logger.info("Aya engine already running on port %d", self.port)
            return

        logger.info("Starting aya engine: %s (port %d)", self._exe, self.port)
        self._process = subprocess.Popen(
            [self._exe, self.gguf_path, str(self.port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Wait for server to be ready
        for _ in range(120):  # up to 60s for model loading
            if self._is_running():
                logger.info("Aya engine ready on port %d", self.port)
                return
            time.sleep(0.5)

        raise RuntimeError("Aya engine failed to start within 60s")

    def _is_running(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.url}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except (urllib.error.URLError, ConnectionError, OSError):
            return False

    def generate(
        self,
        query: str,
        context: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        language: Optional[str] = None,
    ) -> GenerationResult:
        """Generate an answer via the aya engine server."""

        if not language:
            language = "the same language as the question"
        lang_hint = f" Respond in {language}."
        prompt = (
            f"You are a helpful multilingual research assistant.{lang_hint}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        payload = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": 40,
            "top_p": 0.9,
            "min_p": 0.05,
            "min_tokens": 8,
            "repetition_penalty": 1.15, "quality_alpha": 1.0,
        }, ensure_ascii=False).encode("utf-8")

        req = urllib.request.Request(
            f"{self.url}/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode())

        answer = result.get("text", "").strip()

        return GenerationResult(
            answer=answer,
            context=context,
            query=query,
            model=self.model,
        )

    def _build_prompt(self, query: str, context: str, language: Optional[str] = None) -> str:
        lang_hint = f" Respond in {language}." if language else ""
        return (
            f"You are a helpful multilingual research assistant.{lang_hint}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

    def generate_stream(
        self,
        query: str,
        context: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        language: Optional[str] = None,
    ):
        """Stream tokens via SSE. Yields token strings."""
        prompt = self._build_prompt(query, context, language)

        payload = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": 40,
            "top_p": 0.9,
            "min_p": 0.05,
            "min_tokens": 8,
            "repetition_penalty": 1.15, "quality_alpha": 1.0,
            "stream": 1,
        }, ensure_ascii=False).encode("utf-8")

        req = urllib.request.Request(
            f"{self.url}/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=300)
        try:
            while True:
                try:
                    raw = resp.readline()
                except (ConnectionError, OSError):
                    break
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if line == "data: [DONE]":
                    break
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        token = data.get("token", "")
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        continue
        finally:
            try:
                resp.close()
            except (ConnectionError, OSError):
                pass

    def generate_batch(
        self,
        queries: List[str],
        contexts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> List[GenerationResult]:
        results = []
        for query, context in zip(queries, contexts):
            result = self.generate(query, context, max_tokens=max_tokens, temperature=temperature)
            results.append(result)
        return results

    def __del__(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
