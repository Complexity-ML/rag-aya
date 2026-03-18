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
from evaluate import EvalSample, evaluate_simple
from pipeline import build_pipeline, build_embedder, build_retriever, build_tokenizer
from logger import init_logger

logger = init_logger(__name__)


def cmd_index(config: Config, lang_pair: str, dataset: str = None):
    """Load WMT25 data, chunk, embed, save index."""
    from data_loader import load_wmt_dataset, load_wmt_recipe

    embedder = build_embedder(config)
    retriever = build_retriever(config, embedder)

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

    # Use token-based chunking if enabled
    tokenizer = build_tokenizer(config, embedder) if config.use_token_chunking else None
    if tokenizer:
        logger.info("Using token-based chunking (size=%d, overlap=%d)", config.token_chunk_size, config.token_overlap)
        chunks = chunk_documents(
            documents,
            chunk_size=config.token_chunk_size,
            overlap=config.token_overlap,
            tokenizer=tokenizer,
            token_limit=embedder.get_token_limit() if config.warn_on_truncation else None,
        )
    else:
        logger.info("Using character-based chunking (size=%d, overlap=%d)", config.chunk_size, config.chunk_overlap)
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

    # Detect query language for chunk filtering
    has_accent = any(c in query for c in "àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ")
    query_lang = "fra" if has_accent else "eng"

    context = retriever.get_context(query, k=config.top_k, prefer_lang=query_lang)
    if not context:
        logger.warning("No relevant documents found.")
        return

    # Log the actual chunks that will be sent (parse from context)
    context_parts = [p.strip() for p in context.split("\n\n") if p.strip()]
    logger.info("Context: %d chunks (lang=%s):", len(context_parts), query_lang)
    for part in context_parts:
        logger.info("  %s...", part[:90])

    logger.info("Generating answer...")
    if hasattr(generator, 'generate_stream'):
        print()
        try:
            for token in generator.generate_stream(
                query=query, context=context,
                max_tokens=config.max_tokens, temperature=config.temperature,
            ):
                print(token, end="", flush=True)
        except KeyboardInterrupt:
            pass
        print("\n")
    else:
        result = generator.generate(
            query=query, context=context,
            max_tokens=config.max_tokens, temperature=config.temperature,
        )
        print(result.answer)


def cmd_eval(config: Config, eval_output: str = ""):
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

    final_results = {"simple": stats}

    try:
        from evaluate import evaluate_ragas
        logger.info("RAGAS evaluation:")
        ragas_results = evaluate_ragas(samples)
        for k, v in ragas_results.items():
            logger.info("  %s: %s", k, f"{v:.4f}" if isinstance(v, float) else v)
        final_results["ragas"] = ragas_results
    except ImportError:
        logger.warning("RAGAS not installed. Run: pip install ragas")
    except Exception as e:
        logger.error("RAGAS error: %s", e)

    if eval_output:
        import json
        try:
            with open(eval_output, "w") as f:
                json.dump(final_results, f, indent=2)
            logger.info("Saved evaluation results to %s", eval_output)
        except Exception as e:
            logger.error("Failed to save evaluation results: %s", e)


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
            try:
                for token in generator.generate_stream(query, context, max_tokens=config.max_tokens):
                    print(token, end="", flush=True)
            except KeyboardInterrupt:
                pass
            print("\n")
        else:
            result = generator.generate(query, context, max_tokens=config.max_tokens)
            print(f"\n{result.answer}\n")


def main():
    parser = argparse.ArgumentParser(description="RAG-Aya Pipeline")
    parser.add_argument("command", choices=["index", "query", "eval", "demo"])
    parser.add_argument("query_text", nargs="?", default="")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-docs", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--token-chunking", action="store_true", help="Use token-based chunking instead of character-based")
    parser.add_argument("--token-chunk-size", type=int, default=300, help="Token chunk size (default: 300)")
    parser.add_argument("--index-path", default="index/")
    parser.add_argument("--model", default="c4ai-aya-23-8b")
    parser.add_argument("--eval-output", default="eval_results.json", help="Path to save evaluation results JSON")
    # Local mode
    parser.add_argument("--local", action="store_true", help="Use local embedder + GGUF (no API needed)")
    parser.add_argument("--local-embed-model", default="paraphrase-multilingual-MiniLM-L12-v2")
    # GGUF options (uses aya-offline engine)
    parser.add_argument("--gguf", default="", help="Path to GGUF model file")
    parser.add_argument("--engine-port", type=int, default=8089, help="Aya engine server port")
    # Generation options
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (default: 0.3)")
    # WMT options
    parser.add_argument("--lang-pair", default="eng-ara", help="WMT lang pair (e.g. eng-ara, eng-bho)")
    parser.add_argument("--wmt-dataset", default=None, help="Small dataset: news_commentary, ted_talks, wikimatrix")
    parser.add_argument("--wmt-max-lines", type=int, default=500)
    args = parser.parse_args()

    # Auto-detect GGUF: if not specified, look for one in current directory
    if not args.gguf:
        import glob as _glob
        gguf_files = _glob.glob("*.gguf")
        if gguf_files:
            args.gguf = gguf_files[0]
            args.local = True
            logger.info("Auto-detected GGUF: %s (local mode)", args.gguf)

    config = Config(
        top_k=args.top_k,
        max_documents=args.max_docs,
        chunk_size=args.chunk_size,
        index_path=args.index_path,
        gen_model=args.model,
        gguf_path=args.gguf,
        engine_port=args.engine_port,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        wmt_max_lines=args.wmt_max_lines,
        local_embedder=args.local,
        local_embed_model=args.local_embed_model,
        use_token_chunking=args.token_chunking,
        token_chunk_size=args.token_chunk_size,
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
            cmd_eval(config, getattr(args, "eval_output", "eval_results.json"))
        elif args.command == "demo":
            cmd_demo(config)
    except KeyboardInterrupt:
        print("\n")
        sys.exit(0)
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
