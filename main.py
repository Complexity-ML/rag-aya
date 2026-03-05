"""
RAG-Aya :: Main Pipeline

Full RAG pipeline: load data, chunk, embed, index, query, evaluate.

Usage:
    python main.py index                     # Load data + build index
    python main.py query "your question"     # Query the index
    python main.py eval                      # Run RAGAS evaluation
    python main.py demo                      # Interactive demo
"""

import argparse
import os
import sys

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


def build_pipeline(config: Config):
    """Build embedder + retriever + generator."""
    config.validate()
    embedder = CohereEmbedder(config.cohere_api_key, config.embed_model)
    retriever = Retriever(embedder)
    generator = AyaGenerator(config.cohere_api_key, config.gen_model)
    return embedder, retriever, generator


def cmd_index(config: Config):
    """Load data, chunk, embed, save index."""
    from data_loader import load_wikipedia

    embedder, retriever, _ = build_pipeline(config)

    logger.info("Loading documents...")
    documents = load_wikipedia(config.languages, max_per_lang=config.max_documents)

    logger.info("Chunking...")
    chunks = chunk_documents(documents, config.chunk_size, config.chunk_overlap)
    logger.info("%d chunks created", len(chunks))

    logger.info("Embedding + indexing...")
    retriever.index(chunks)

    logger.info("Saving index...")
    retriever.save(config.index_path)
    logger.info("Done!")


def cmd_query(config: Config, query: str):
    """Query the index and generate an answer."""
    embedder, retriever, generator = build_pipeline(config)

    if os.path.exists(os.path.join(config.index_path, "chunks.json")):
        retriever.load(config.index_path)
    else:
        logger.error("No index found. Run 'python main.py index' first.")
        return

    logger.info("Query: %s", query)

    context = retriever.get_context(query, k=config.top_k)
    if not context:
        logger.warning("No relevant documents found.")
        return

    logger.info("Retrieved %d chunks:", config.top_k)
    for chunk, score in retriever.search(query, k=config.top_k):
        logger.info("  [%s] %.3f -- %s...", chunk.language, score, chunk.text[:80])

    logger.info("Generating answer...")
    result = generator.generate(
        query=query,
        context=context,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    logger.info("Answer (%s):", result.model)
    print(result.answer)


def cmd_eval(config: Config):
    """Run evaluation on sample queries."""
    embedder, retriever, generator = build_pipeline(config)

    if os.path.exists(os.path.join(config.index_path, "chunks.json")):
        retriever.load(config.index_path)
    else:
        logger.error("No index found. Run 'python main.py index' first.")
        return

    test_queries = [
        ("What is machine learning?", "en"),
        ("Qu'est-ce que l'intelligence artificielle ?", "fr"),
        ("What are neural networks?", "en"),
        ("Comment fonctionne le traitement du langage naturel ?", "fr"),
        ("What is deep learning used for?", "en"),
    ]

    samples = []
    for query, lang in test_queries:
        logger.info("[%s] %s", lang, query)
        context = retriever.get_context(query, k=config.top_k)
        contexts_list = [c.strip() for c in context.split("\n\n") if c.strip()]
        result = generator.generate(query, context, language=lang)
        logger.info("  -> %s...", result.answer[:100])
        samples.append(EvalSample(
            question=query,
            contexts=contexts_list,
            answer=result.answer,
        ))

    logger.info("Evaluation results (simple):")
    stats = evaluate_simple(samples)
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)

    try:
        from evaluate import evaluate_ragas
        logger.info("RAGAS evaluation:")
        ragas_results = evaluate_ragas(samples)
        for k, v in ragas_results.items():
            logger.info("  %s: %s", k, f"{v:.4f}" if isinstance(v, float) else v)
    except ImportError:
        logger.warning("RAGAS not installed. Run: pip install ragas")
    except Exception as e:
        logger.error("RAGAS error: %s", e)


def cmd_demo(config: Config):
    """Interactive demo loop."""
    embedder, retriever, generator = build_pipeline(config)

    if os.path.exists(os.path.join(config.index_path, "chunks.json")):
        retriever.load(config.index_path)
    else:
        logger.error("No index found. Run 'python main.py index' first.")
        return

    logger.info("RAG-Aya Demo | %d chunks indexed", retriever.stats["num_chunks"])
    print("Type your question (or 'quit' to exit):\n")

    while True:
        query = input("> ").strip()
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        context = retriever.get_context(query, k=config.top_k)
        result = generator.generate(query, context, max_tokens=config.max_tokens)
        print(f"\n{result.answer}\n")


def main():
    parser = argparse.ArgumentParser(description="RAG-Aya Pipeline")
    parser.add_argument("command", choices=["index", "query", "eval", "demo"])
    parser.add_argument("query_text", nargs="?", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-docs", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--index-path", default="index/")
    parser.add_argument("--model", default="c4ai-aya-23-8b")
    args = parser.parse_args()

    config = Config(
        top_k=args.top_k,
        max_documents=args.max_docs,
        chunk_size=args.chunk_size,
        index_path=args.index_path,
        gen_model=args.model,
    )

    try:
        if args.command == "index":
            cmd_index(config)
        elif args.command == "query":
            if not args.query_text:
                logger.error("Usage: python main.py query \"your question\"")
                sys.exit(1)
            cmd_query(config, args.query_text)
        elif args.command == "eval":
            cmd_eval(config)
        elif args.command == "demo":
            cmd_demo(config)
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
