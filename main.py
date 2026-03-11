"""
RAG-Aya :: Main Pipeline

Full RAG pipeline: load WMT data, chunk, embed, index, query, evaluate.

Usage:
    python main.py index --local --lang-pair eng-fra --wmt-dataset news_commentary
    python main.py query "your question" --local --gguf ../tiny-aya-global-q4_k_m.gguf
    python main.py eval  --local --gguf ../tiny-aya-global-q4_k_m.gguf
    python main.py demo  --local --gguf ../tiny-aya-global-q4_k_m.gguf
"""

import argparse
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from config import Config
from chunker import chunk_documents
from retriever import Retriever
from generator import AyaGenerator
from evaluate import EvalSample, evaluate_simple
from logger import init_logger

logger = init_logger(__name__)


def build_pipeline(config: Config):
    """Build embedder + retriever + generator."""
    if config.local_embedder:
        from embedder import LocalEmbedder
        embedder = LocalEmbedder(config.local_embed_model)
    else:
        from embedder import CohereEmbedder
        embedder = CohereEmbedder(config.cohere_api_key, config.embed_model)

    retriever = Retriever(embedder)

    if config.gguf_path:
        from generator import AyaEngineGenerator
        config.validate(require_cohere=not config.local_embedder)
        generator = AyaEngineGenerator(config.gguf_path, port=config.engine_port)
    else:
        config.validate(require_cohere=True)
        generator = AyaGenerator(config.cohere_api_key, config.gen_model)

    return embedder, retriever, generator


def cmd_index(config: Config, lang_pair: str, dataset: str = None):
    """Load WMT25 data, chunk, embed, save index."""
    from data_loader import load_wmt_dataset, load_wmt_recipe

    # Only need embedder + retriever for indexing (no generator)
    if config.local_embedder:
        from embedder import LocalEmbedder
        embedder = LocalEmbedder(config.local_embed_model)
    else:
        from embedder import CohereEmbedder
        config.validate(require_cohere=True)
        embedder = CohereEmbedder(config.cohere_api_key, config.embed_model)

    retriever = Retriever(embedder)

    if dataset:
        logger.info("Loading WMT dataset: %s (%s)...", dataset, lang_pair)
        documents = load_wmt_dataset(
            dataset, lang_pair,
            cache_dir=config.wmt_cache_dir,
            max_lines=config.wmt_max_lines,
        )
    else:
        logger.info("Loading WMT recipe: %s...", lang_pair)
        documents = load_wmt_recipe(
            lang_pair,
            cache_dir=config.wmt_cache_dir,
            max_lines=config.wmt_max_lines,
        )

    if not documents:
        logger.error("No documents loaded.")
        return

    logger.info("Chunking %d documents...", len(documents))
    chunks = chunk_documents(documents, config.chunk_size, config.chunk_overlap)
    logger.info("%d chunks created", len(chunks))

    logger.info("Embedding + indexing...")
    retriever.index(chunks)

    logger.info("Saving index...")
    retriever.save(config.index_path)
    logger.info("Done! Indexed %d chunks from WMT25 %s", len(chunks), lang_pair)


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
    if hasattr(generator, 'generate_stream'):
        print()
        full_answer = ""
        for token in generator.generate_stream(
            query=query, context=context,
            max_tokens=config.max_tokens, temperature=config.temperature,
        ):
            print(token, end="", flush=True)
            full_answer += token
        print("\n")
    else:
        result = generator.generate(
            query=query, context=context,
            max_tokens=config.max_tokens, temperature=config.temperature,
        )
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
        if hasattr(generator, 'generate_stream'):
            print()
            for token in generator.generate_stream(query, context, max_tokens=config.max_tokens):
                print(token, end="", flush=True)
            print("\n")
        else:
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
    # Local mode
    parser.add_argument("--local", action="store_true", help="Use local embedder + GGUF (no API needed)")
    parser.add_argument("--local-embed-model", default="paraphrase-multilingual-MiniLM-L12-v2")
    # GGUF options (uses aya-offline engine)
    parser.add_argument("--gguf", default="", help="Path to GGUF model file")
    parser.add_argument("--engine-port", type=int, default=8089, help="Aya engine server port")
    # WMT options
    parser.add_argument("--lang-pair", default="eng-ara", help="WMT lang pair (e.g. eng-ara, eng-bho)")
    parser.add_argument("--wmt-dataset", default=None, help="Small dataset: news_commentary, ted_talks, wikimatrix")
    parser.add_argument("--wmt-max-lines", type=int, default=500)
    args = parser.parse_args()

    config = Config(
        top_k=args.top_k,
        max_documents=args.max_docs,
        chunk_size=args.chunk_size,
        index_path=args.index_path,
        gen_model=args.model,
        gguf_path=args.gguf,
        engine_port=args.engine_port,
        wmt_max_lines=args.wmt_max_lines,
        local_embedder=args.local,
        local_embed_model=args.local_embed_model,
    )

    try:
        if args.command == "index":
            cmd_index(config, args.lang_pair, args.wmt_dataset)
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
