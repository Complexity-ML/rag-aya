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

    print("Loading documents...")
    documents = load_wikipedia(config.languages, max_per_lang=config.max_documents)

    print("Chunking...")
    chunks = chunk_documents(documents, config.chunk_size, config.chunk_overlap)
    print(f"  {len(chunks)} chunks created")

    print("Embedding + indexing...")
    retriever.index(chunks)

    print("Saving index...")
    retriever.save(config.index_path)
    print("Done!")


def cmd_query(config: Config, query: str):
    """Query the index and generate an answer."""
    embedder, retriever, generator = build_pipeline(config)

    if os.path.exists(os.path.join(config.index_path, "chunks.json")):
        retriever.load(config.index_path)
    else:
        print("No index found. Run 'python main.py index' first.")
        return

    print(f"\nQuery: {query}")
    print("-" * 60)

    context = retriever.get_context(query, k=config.top_k)
    if not context:
        print("No relevant documents found.")
        return

    print(f"\nRetrieved {config.top_k} chunks:")
    for chunk, score in retriever.search(query, k=config.top_k):
        print(f"  [{chunk.language}] {score:.3f} — {chunk.text[:80]}...")

    print("\nGenerating answer...")
    result = generator.generate(
        query=query,
        context=context,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    print(f"\nAnswer ({result.model}):")
    print(result.answer)


def cmd_eval(config: Config):
    """Run evaluation on sample queries."""
    embedder, retriever, generator = build_pipeline(config)

    if os.path.exists(os.path.join(config.index_path, "chunks.json")):
        retriever.load(config.index_path)
    else:
        print("No index found. Run 'python main.py index' first.")
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
        print(f"\n  [{lang}] {query}")
        context = retriever.get_context(query, k=config.top_k)
        contexts_list = [c.strip() for c in context.split("\n\n") if c.strip()]
        result = generator.generate(query, context, language=lang)
        print(f"    -> {result.answer[:100]}...")
        samples.append(EvalSample(
            question=query,
            contexts=contexts_list,
            answer=result.answer,
        ))

    print("\n" + "=" * 60)
    print("Evaluation results (simple):")
    stats = evaluate_simple(samples)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    try:
        from evaluate import evaluate_ragas
        print("\nRAGAS evaluation:")
        ragas_results = evaluate_ragas(samples)
        for k, v in ragas_results.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    except ImportError:
        print("\n  RAGAS not installed. Run: pip install ragas")
    except Exception as e:
        print(f"\n  RAGAS error: {e}")


def cmd_demo(config: Config):
    """Interactive demo loop."""
    embedder, retriever, generator = build_pipeline(config)

    if os.path.exists(os.path.join(config.index_path, "chunks.json")):
        retriever.load(config.index_path)
    else:
        print("No index found. Run 'python main.py index' first.")
        return

    print(f"RAG-Aya Demo | {retriever.stats['num_chunks']} chunks indexed")
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

    if args.command == "index":
        cmd_index(config)
    elif args.command == "query":
        if not args.query_text:
            print("Usage: python main.py query \"your question\"")
            sys.exit(1)
        cmd_query(config, args.query_text)
    elif args.command == "eval":
        cmd_eval(config)
    elif args.command == "demo":
        cmd_demo(config)


if __name__ == "__main__":
    main()
