"""
RAG-Aya :: aiohttp REST Server

Endpoints:
    POST /index          — Index documents (Wikipedia or text)
    POST /search         — Search chunks by query
    POST /generate       — RAG: search + generate answer
    GET  /stats          — Index statistics
    POST /eval           — Run evaluation
    GET  /health         — Health check

Usage:
    python server.py --port 8080
"""

import argparse
import json
import os
import asyncio
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from aiohttp import web
from dotenv import load_dotenv
load_dotenv()

from config import Config
from chunker import chunk_documents
from embedder import CohereEmbedder
from retriever import Retriever
from generator import AyaGenerator
from evaluate import EvalSample, evaluate_simple
from logger import init_logger

logger = init_logger(__name__)


class RagAyaServer:
    def __init__(self, config: Config):
        config.validate()
        self.config = config
        self.embedder = CohereEmbedder(config.cohere_api_key, config.embed_model)
        self.retriever = Retriever(self.embedder)
        self.generator = AyaGenerator(config.cohere_api_key, config.gen_model)
        self._pool = ThreadPoolExecutor(max_workers=4)

        # Load existing index if available
        index_chunks = os.path.join(config.index_path, "chunks.json")
        if os.path.exists(index_chunks):
            self.retriever.load(config.index_path)

    async def _run_in_pool(self, fn, *args, **kwargs):
        """Run blocking function in thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, partial(fn, *args, **kwargs))

    # ── Handlers ─────────────────────────────────────────────

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "chunks": len(self.retriever.chunks)})

    async def handle_stats(self, request: web.Request) -> web.Response:
        return web.json_response(self.retriever.stats)

    async def handle_index(self, request: web.Request) -> web.Response:
        """
        POST /index
        Body: {"source": "wikipedia", "languages": ["en","fr"], "max_per_lang": 50}
          or: {"documents": [{"id": "...", "text": "...", "language": "en"}, ...]}
          or: {"text": "some text to index", "language": "en"}
        """
        body = await request.json()

        if "text" in body:
            # Index raw text
            documents = [{
                "id": body.get("id", "manual_0"),
                "text": body["text"],
                "language": body.get("language", "en"),
            }]
        elif "documents" in body:
            documents = body["documents"]
        elif body.get("source") == "wikipedia":
            from data_loader import load_wikipedia
            languages = body.get("languages", self.config.languages)
            max_per_lang = body.get("max_per_lang", self.config.max_documents)
            documents = await self._run_in_pool(load_wikipedia, languages, max_per_lang)
        else:
            return web.json_response(
                {"error": "Provide 'text', 'documents', or 'source': 'wikipedia'"},
                status=400,
            )

        chunks = chunk_documents(
            documents,
            self.config.chunk_size,
            self.config.chunk_overlap,
        )

        await self._run_in_pool(self.retriever.index, chunks)
        await self._run_in_pool(self.retriever.save, self.config.index_path)

        return web.json_response({
            "indexed": len(chunks),
            "total_chunks": len(self.retriever.chunks),
        })

    async def handle_search(self, request: web.Request) -> web.Response:
        """
        POST /search
        Body: {"query": "...", "k": 5}
        """
        body = await request.json()
        query = body.get("query", "")
        k = body.get("k", self.config.top_k)

        if not query:
            return web.json_response({"error": "query is required"}, status=400)

        results = await self._run_in_pool(self.retriever.search, query, k)

        return web.json_response({
            "query": query,
            "results": [
                {
                    "text": chunk.text,
                    "doc_id": chunk.doc_id,
                    "language": chunk.language,
                    "score": round(score, 4),
                }
                for chunk, score in results
            ],
        })

    async def handle_generate(self, request: web.Request) -> web.Response:
        """
        POST /generate
        Body: {"query": "...", "k": 5, "language": "en", "max_tokens": 512}
        """
        body = await request.json()
        query = body.get("query", "")
        k = body.get("k", self.config.top_k)
        language = body.get("language")
        max_tokens = body.get("max_tokens", self.config.max_tokens)
        temperature = body.get("temperature", self.config.temperature)

        if not query:
            return web.json_response({"error": "query is required"}, status=400)

        # Retrieve context
        context = await self._run_in_pool(self.retriever.get_context, query, k)

        # Generate with Aya
        result = await self._run_in_pool(
            self.generator.generate,
            query=query,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
            language=language,
        )

        return web.json_response({
            "query": query,
            "answer": result.answer,
            "model": result.model,
            "context_used": bool(context),
        })

    async def handle_eval(self, request: web.Request) -> web.Response:
        """
        POST /eval
        Body: {"queries": [{"question": "...", "language": "en"}, ...]}
          or: {} (uses default test queries)
        """
        body = await request.json() if request.can_read_body else {}

        queries = body.get("queries", [
            {"question": "What is machine learning?", "language": "en"},
            {"question": "Qu'est-ce que l'intelligence artificielle ?", "language": "fr"},
            {"question": "What are neural networks?", "language": "en"},
            {"question": "Comment fonctionne le traitement du langage naturel ?", "language": "fr"},
            {"question": "What is deep learning used for?", "language": "en"},
        ])

        samples = []
        for q in queries:
            question = q["question"]
            lang = q.get("language", "en")
            context = await self._run_in_pool(self.retriever.get_context, question, self.config.top_k)
            contexts_list = [c.strip() for c in context.split("\n\n") if c.strip()]
            result = await self._run_in_pool(
                self.generator.generate, question, context, language=lang,
            )
            samples.append(EvalSample(
                question=question,
                contexts=contexts_list,
                answer=result.answer,
                ground_truth=q.get("ground_truth"),
            ))

        stats = evaluate_simple(samples)

        # Try RAGAS
        ragas_results = None
        try:
            from evaluate import evaluate_ragas
            ragas_results = await self._run_in_pool(evaluate_ragas, samples)
        except Exception as e:
            ragas_results = {"error": str(e)}

        return web.json_response({
            "simple": stats,
            "ragas": ragas_results,
            "samples": [
                {"question": s.question, "answer": s.answer[:200]}
                for s in samples
            ],
        })

    # ── CORS ─────────────────────────────────────────────────

    @staticmethod
    async def handle_options(request: web.Request) -> web.Response:
        return web.Response(status=204, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        })

    @staticmethod
    @web.middleware
    async def cors_middleware(request: web.Request, handler):
        response = await handler(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    # ── App ──────────────────────────────────────────────────

    def create_app(self) -> web.Application:
        app = web.Application(middlewares=[self.cors_middleware])

        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/stats", self.handle_stats)
        app.router.add_post("/index", self.handle_index)
        app.router.add_post("/search", self.handle_search)
        app.router.add_post("/generate", self.handle_generate)
        app.router.add_post("/eval", self.handle_eval)

        # CORS preflight
        for route in ["/index", "/search", "/generate", "/eval"]:
            app.router.add_route("OPTIONS", route, self.handle_options)

        return app


def main():
    parser = argparse.ArgumentParser(description="RAG-Aya Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--index-path", default="index/")
    parser.add_argument("--model", default="c4ai-aya-23-8b")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    config = Config(
        index_path=args.index_path,
        gen_model=args.model,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
    )

    server = RagAyaServer(config)
    app = server.create_app()

    logger.info("RAG-Aya server starting on %s:%d", args.host, args.port)
    logger.info("Model: %s", config.gen_model)
    logger.info("Index: %s (%d chunks)", config.index_path, len(server.retriever.chunks))
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
