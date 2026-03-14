"""
RAG-Aya :: Generation Quality Benchmark

Evaluates the full RAG pipeline (retrieval + generation) on a domain-specific
test set drawn from the WMT news commentary gold/finance corpus.

Metrics:
  retrieval_score  — avg cosine similarity of retrieved chunks
  lang_correct     — response language matches query language (0/1)
  clean_response   — no garbage/degeneration patterns detected (0/1)
  context_keywords — fraction of context keywords found in response
  tokens_estimated — estimated token count of response
  score            — composite score [0..1]

Usage:
    python benchmark.py --local --engine-port 8089
    python benchmark.py --local --engine-port 8089 --output results.json
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from config import Config
from pipeline import build_pipeline, build_embedder, build_retriever
from logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Test set — domain specific to WMT gold/finance corpus
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    query: str
    lang: str                        # "eng" or "fra"
    keywords: List[str]              # expected keywords in a good answer
    in_domain: bool = True           # False = out-of-domain (fallback test)
    description: str = ""


TEST_CASES: List[TestCase] = [
    # --- English, in-domain ---
    TestCase(
        query="What is the role of gold as a store of value?",
        lang="eng",
        keywords=["store", "value", "inflation", "wealth", "price"],
        description="Core gold concept",
    ),
    TestCase(
        query="How do interest rates affect gold prices?",
        lang="eng",
        keywords=["interest", "rate", "price", "gold", "monetary"],
        description="Monetary policy + gold",
    ),
    TestCase(
        query="What causes inflation?",
        lang="eng",
        keywords=["inflation", "demand", "supply", "price", "money"],
        description="Inflation causes",
    ),
    TestCase(
        query="Are inflation-indexed bonds a good alternative to gold?",
        lang="eng",
        keywords=["bond", "inflation", "gold", "investment"],
        description="Bonds vs gold comparison",
    ),
    TestCase(
        query="Why do investors buy gold during geopolitical crises?",
        lang="eng",
        keywords=["gold", "crisis", "safe", "investor", "risk"],
        description="Gold as safe haven",
    ),

    # --- French, in-domain ---
    TestCase(
        query="Quel est le rôle de l'or comme réserve de valeur ?",
        lang="fra",
        keywords=["or", "valeur", "inflation", "réserve", "prix"],
        description="Concept fondamental de l'or (FR)",
    ),
    TestCase(
        query="Comment les taux d'intérêt influencent-ils le prix de l'or ?",
        lang="fra",
        keywords=["taux", "intérêt", "prix", "or", "monétaire"],
        description="Politique monétaire + or (FR)",
    ),
    TestCase(
        query="Quelles sont les causes de l'inflation ?",
        lang="fra",
        keywords=["inflation", "demande", "offre", "prix", "monnaie"],
        description="Causes de l'inflation (FR)",
    ),
    TestCase(
        query="L'or est-il un bon investissement en période de crise ?",
        lang="fra",
        keywords=["or", "crise", "investissement", "valeur"],
        description="Or comme investissement (FR)",
    ),

    # --- Out-of-domain ---
    TestCase(
        query="What is the capital of France?",
        lang="eng",
        keywords=["France", "Paris"],
        in_domain=False,
        description="Out-of-domain geography",
    ),
    TestCase(
        query="How do neural networks work?",
        lang="eng",
        keywords=["network", "neural", "learning"],
        in_domain=False,
        description="Out-of-domain ML",
    ),
]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

# Patterns that indicate garbage / degeneration
_GARBAGE_PATTERNS = [
    r"[a-z]{1}\s[a-z]{1}\s[a-z]{1}\s[a-z]{1}",   # spaced single chars: "t o d o"
    r"(.)\1{4,}",                                    # 5+ repeated chars: "mmmmm"
    r"\[WMTN\]|\[MFLT\]|\[QKZ",                     # uppercase tag artifacts
    r"\[\[",                                          # double bracket
    r"(mb){3,}|(vv){3,}|(ww){3,}",                  # repeated 2-char fragments
    r"This fused generation",                         # meta-commentary
    r"\*\*\[Note",                                    # note disclaimers
]
_GARBAGE_RE = [re.compile(p, re.IGNORECASE) for p in _GARBAGE_PATTERNS]

# French character markers (to detect response language)
_FRENCH_MARKERS = set("àâéèêëîïôùûüçœæÀÂÉÈÊËÎÏÔÙÛÜÇ")

# Common English stop words that should appear in any coherent English response
_EN_FUNCTION_WORDS = {"the", "is", "are", "of", "in", "to", "and", "a", "an", "that", "it"}
_FR_FUNCTION_WORDS = {"le", "la", "les", "de", "du", "des", "est", "sont", "et", "un", "une", "que"}


def detect_language(text: str) -> str:
    """Heuristic language detection: returns 'fra' or 'eng'."""
    if not text.strip():
        return "unk"
    # Check for French accents
    accent_count = sum(1 for c in text if c in _FRENCH_MARKERS)
    if accent_count >= 3:
        return "fra"
    words = set(text.lower().split())
    fr_hits = len(words & _FR_FUNCTION_WORDS)
    en_hits = len(words & _EN_FUNCTION_WORDS)
    if fr_hits > en_hits:
        return "fra"
    return "eng"


def is_garbage(text: str) -> Tuple[bool, str]:
    """Returns (is_garbage, reason) for the response."""
    if not text.strip():
        return True, "empty"
    # Very short responses
    if len(text.strip()) < 20:
        return True, "too_short"
    # Check garbage patterns
    for pat in _GARBAGE_RE:
        m = pat.search(text)
        if m:
            return True, f"pattern:{pat.pattern[:30]}"
    # Check for non-Latin script (excluding expected accented French/English)
    non_latin = sum(1 for c in text if ord(c) > 0x2000 and c not in "''""…—–")
    if non_latin > 10:
        return True, f"non_latin_script:{non_latin}_chars"
    return False, ""


def keyword_score(response: str, keywords: List[str]) -> float:
    """Fraction of expected keywords found in response (case-insensitive)."""
    if not keywords:
        return 1.0
    resp_lower = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in resp_lower)
    return hits / len(keywords)


def composite_score(
    retrieval_score: float,
    lang_correct: bool,
    is_clean: bool,
    kw_score: float,
) -> float:
    """Weighted composite score [0..1]."""
    return (
        0.20 * min(retrieval_score, 1.0) +
        0.30 * float(lang_correct) +
        0.30 * float(is_clean) +
        0.20 * kw_score
    )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    query: str
    lang: str
    description: str
    in_domain: bool
    response: str
    retrieval_score: float
    lang_correct: bool
    is_clean: bool
    garbage_reason: str
    kw_score: float
    score: float
    latency_s: float
    n_chunks: int
    chunk_langs: List[str]


def run_benchmark(config: Config, test_cases: List[TestCase]) -> List[BenchResult]:
    embedder, retriever, generator = build_pipeline(config)

    index_file = os.path.join(config.index_path, "chunks.json")
    if not os.path.exists(index_file):
        logger.error("No index found. Run: python main.py index --local")
        sys.exit(1)
    retriever.load(config.index_path)
    logger.info("Index loaded: %d chunks", retriever.stats["num_chunks"])

    results = []
    for i, tc in enumerate(test_cases):
        logger.info(
            "[%d/%d] [%s] %s — %s",
            i + 1, len(test_cases), tc.lang, tc.description, tc.query[:60],
        )

        # Retrieval
        raw_results = retriever.search(tc.query, k=config.top_k * 2)
        lang_map = {"en": "eng", "fr": "fra", "eng": "eng", "fra": "fra"}
        prefer = lang_map.get(tc.lang, tc.lang)
        same_lang = [(c, s) for c, s in raw_results if c.language == prefer]
        selected = same_lang[:config.top_k] if same_lang else raw_results[:config.top_k]
        avg_sim = sum(s for _, s in selected) / len(selected) if selected else 0.0
        chunk_langs = [c.language for c, _ in selected]

        context_parts = [f"[{c.language}|{c.doc_id}] {c.text}" for c, _ in selected]
        context = "\n\n".join(context_parts)

        # Generation
        t0 = time.time()
        try:
            result = generator.generate(
                query=tc.query,
                context=context,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
            response = result.answer.strip()
        except Exception as e:
            logger.error("Generation failed: %s", e)
            response = ""
        latency = time.time() - t0

        # Scoring
        resp_lang = detect_language(response)
        lang_correct = (resp_lang == prefer)
        garbage, garbage_reason = is_garbage(response)
        kw = keyword_score(response, tc.keywords)
        score = composite_score(avg_sim, lang_correct, not garbage, kw)

        br = BenchResult(
            query=tc.query,
            lang=tc.lang,
            description=tc.description,
            in_domain=tc.in_domain,
            response=response,
            retrieval_score=avg_sim,
            lang_correct=lang_correct,
            is_clean=not garbage,
            garbage_reason=garbage_reason,
            kw_score=kw,
            score=score,
            latency_s=latency,
            n_chunks=len(selected),
            chunk_langs=chunk_langs,
        )
        results.append(br)

        # Live preview
        status = "OK" if not garbage and lang_correct else "FAIL"
        logger.info(
            "  [%s] score=%.2f | retr=%.3f | lang=%s(%s) | kw=%.2f | %.1fs",
            status, score, avg_sim, resp_lang, "OK" if lang_correct else "WRONG",
            kw, latency,
        )
        logger.info("  Response: %s...", response[:120].replace("\n", " "))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: List[BenchResult]):
    in_domain  = [r for r in results if r.in_domain]
    out_domain = [r for r in results if not r.in_domain]

    def avg(lst): return sum(lst) / len(lst) if lst else 0.0

    logger.info("=" * 72)
    logger.info(" RAG-Aya :: Benchmark Results")
    logger.info("=" * 72)

    # Per-query table
    header = f"{'#':>2}  {'Lang':>4}  {'Dom':>3}  {'Score':>5}  {'Retr':>5}  {'Lang':>4}  {'KW':>4}  {'Clean':>5}  {'Lat':>5}  Description"
    logger.info(header)
    logger.info("-" * 72)
    for i, r in enumerate(results):
        dom = "IN" if r.in_domain else "OUT"
        lc  = "OK" if r.lang_correct else "FAIL"
        cl  = "OK" if r.is_clean else "FAIL"
        logger.info(
            "%2d  %4s  %3s  %5.2f  %5.3f  %4s  %4.2f  %5s  %4.1fs  %s",
            i + 1, r.lang, dom, r.score, r.retrieval_score,
            lc, r.kw_score, cl, r.latency_s, r.description,
        )

    logger.info("-" * 72)

    # Summary by category
    def summary_line(label, subset):
        if not subset:
            return
        scores  = [r.score for r in subset]
        retr    = [r.retrieval_score for r in subset]
        lang_ok = sum(r.lang_correct for r in subset)
        clean   = sum(r.is_clean for r in subset)
        kw      = [r.kw_score for r in subset]
        lat     = [r.latency_s for r in subset]
        logger.info(
            "%-20s  n=%d  score=%.3f  retr=%.3f  lang=%d/%d  clean=%d/%d  kw=%.3f  lat=%.1fs",
            label, len(subset), avg(scores), avg(retr),
            lang_ok, len(subset), clean, len(subset), avg(kw), avg(lat),
        )

    logger.info("Summary:")
    summary_line("All queries", results)
    summary_line("In-domain", in_domain)
    summary_line("Out-of-domain", out_domain)
    eng = [r for r in results if r.lang == "eng"]
    fra = [r for r in results if r.lang == "fra"]
    summary_line("English", eng)
    summary_line("French", fra)

    # Failures
    failures = [r for r in results if not r.is_clean or not r.lang_correct]
    if failures:
        logger.info("Failures (%d):", len(failures))
        for r in failures:
            reasons = []
            if not r.lang_correct:
                reasons.append(f"wrong_lang ({detect_language(r.response)})")
            if not r.is_clean:
                reasons.append(f"garbage ({r.garbage_reason})")
            logger.info("  [%s] %s: %s", r.lang, r.description, ", ".join(reasons))
    else:
        logger.info("All responses passed quality checks!")

    logger.info("=" * 72)


def save_results(results: List[BenchResult], path: str):
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_queries": len(results),
        "summary": {
            "avg_score": sum(r.score for r in results) / len(results),
            "lang_correct": sum(r.lang_correct for r in results),
            "clean": sum(r.is_clean for r in results),
            "avg_retrieval": sum(r.retrieval_score for r in results) / len(results),
            "avg_kw_score": sum(r.kw_score for r in results) / len(results),
            "avg_latency_s": sum(r.latency_s for r in results) / len(results),
        },
        "results": [
            {
                "query": r.query,
                "lang": r.lang,
                "description": r.description,
                "in_domain": r.in_domain,
                "score": round(r.score, 4),
                "retrieval_score": round(r.retrieval_score, 4),
                "lang_correct": r.lang_correct,
                "is_clean": r.is_clean,
                "garbage_reason": r.garbage_reason,
                "kw_score": round(r.kw_score, 4),
                "latency_s": round(r.latency_s, 2),
                "n_chunks": r.n_chunks,
                "chunk_langs": r.chunk_langs,
                "response": r.response,
            }
            for r in results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Results saved: %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG-Aya Benchmark")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--gguf", default="")
    parser.add_argument("--engine-port", type=int, default=8089)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--index-path", default="index/")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--local-embed-model", default="paraphrase-multilingual-MiniLM-L12-v2")
    args = parser.parse_args()

    if not args.gguf:
        import glob as _glob
        gguf_files = _glob.glob("*.gguf")
        if gguf_files:
            args.gguf = gguf_files[0]
            args.local = True

    config = Config(
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        index_path=args.index_path,
        gguf_path=args.gguf,
        engine_port=args.engine_port,
        local_embedder=args.local,
        local_embed_model=args.local_embed_model,
    )

    logger.info("RAG-Aya Benchmark | %d test cases", len(TEST_CASES))
    logger.info("Model: %s | top_k=%d | max_tokens=%d | temp=%.2f",
                os.path.basename(args.gguf) if args.gguf else "API",
                args.top_k, args.max_tokens, args.temperature)

    results = run_benchmark(config, TEST_CASES)
    print_report(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
