"""
RAG-Aya :: Data Loader

Load documents from Wikipedia, WMT25 parallel data, and local files.
"""

import os
import subprocess
import glob as globlib
from typing import List, Optional

from logger import init_logger

logger = init_logger(__name__)


def load_wikipedia(languages: List[str] = None, max_per_lang: int = 50) -> List[dict]:
    """Load Wikipedia articles via HuggingFace datasets."""
    from datasets import load_dataset

    if languages is None:
        languages = ["en", "fr"]

    documents = []
    for lang in languages:
        logger.info("Loading Wikipedia (%s)...", lang)
        config = f"20220301.{lang}"
        ds = load_dataset("wikipedia", config, split=f"train[:{max_per_lang}]", trust_remote_code=True)
        for i, row in enumerate(ds):
            text = row.get("text", "")
            title = row.get("title", f"wiki_{lang}_{i}")
            if len(text) > 100:
                documents.append({
                    "id": f"wiki_{lang}_{i}_{title[:30]}",
                    "text": text,
                    "title": title,
                    "language": lang,
                })
        count = len([d for d in documents if d["language"] == lang])
        logger.info("Loaded %d articles (%s)", count, lang)

    logger.info("Total documents: %d", len(documents))
    return documents


# ── WMT25 datasets via mtdata ──────────────────────────────────────

# Lightweight datasets good for RAG testing
WMT_RECIPES = {
    "eng-ara": "wmt25-eng-ara",
    "eng-zho": "wmt25-eng-zho",
    "eng-ces": "wmt25-eng-ces",
    "eng-rus": "wmt25-eng-rus",
    "eng-bho": "wmt25-eng-bho",
    "eng-ukr": "wmt25-eng-ukr",
    "eng-kor": "wmt25-eng-kor",
    "eng-jpn": "wmt25-eng-jpn",
}

# Small, clean datasets ideal for testing
WMT_SMALL_DATASETS = {
    "news_commentary": "Statmt-news_commentary-18.1",
    "ted_talks": "Neulab-tedtalks_train-1",
    "wikimatrix": "Facebook-wikimatrix-1",
}


def _ensure_mtdata():
    """Check mtdata is installed."""
    try:
        subprocess.run(["mtdata", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.error("mtdata not installed. Run: pip install mtdata==0.4.3")
        return False


def _ensure_wmt_recipes(cache_dir: str):
    """Download WMT25 recipes config if not present."""
    recipe_path = os.path.join(cache_dir, "mtdata.recipes.wmt25-constrained.yml")
    if not os.path.exists(recipe_path):
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("Downloading WMT25 recipes config...")
        subprocess.run(
            ["wget", "-q", "-O", recipe_path,
             "https://www.statmt.org/wmt25/mtdata/mtdata.recipes.wmt25-constrained.yml"],
            check=True,
        )
    return recipe_path


def load_wmt_recipe(
    lang_pair: str,
    cache_dir: str = "wmt_data",
    max_lines: int = 500,
) -> List[dict]:
    """Load a WMT25 recipe via mtdata get-recipe.

    Args:
        lang_pair: e.g. "eng-ara", "eng-zho", "eng-bho"
        cache_dir: where to download/cache data
        max_lines: max parallel lines to load (keeps it light for RAG testing)
    """
    if not _ensure_mtdata():
        return []

    recipe_id = WMT_RECIPES.get(lang_pair)
    if not recipe_id:
        logger.error("Unknown lang pair: %s. Available: %s", lang_pair, list(WMT_RECIPES.keys()))
        return []

    _ensure_wmt_recipes(cache_dir)

    out_dir = os.path.join(cache_dir, recipe_id)
    if not os.path.exists(out_dir):
        logger.info("Downloading WMT25 recipe: %s ...", recipe_id)
        subprocess.run(
            ["mtdata", "get-recipe", "-ri", recipe_id, "-o", out_dir, "--compress", "--no-merge"],
            cwd=cache_dir,
            check=True,
        )

    return _parse_wmt_dir(out_dir, lang_pair, max_lines)


def load_wmt_dataset(
    dataset_key: str,
    lang_pair: str,
    cache_dir: str = "wmt_data",
    max_lines: int = 500,
) -> List[dict]:
    """Load a single small WMT dataset via mtdata echo.

    Args:
        dataset_key: "news_commentary", "ted_talks", or "wikimatrix"
        lang_pair: e.g. "eng-ara", "eng-ces"
        max_lines: max lines to load
    """
    if not _ensure_mtdata():
        return []

    base_id = WMT_SMALL_DATASETS.get(dataset_key)
    if not base_id:
        logger.error("Unknown dataset: %s. Available: %s", dataset_key, list(WMT_SMALL_DATASETS.keys()))
        return []

    dataset_id = f"{base_id}-{lang_pair}"
    logger.info("Loading WMT dataset: %s (max %d lines)...", dataset_id, max_lines)

    try:
        result = subprocess.run(
            ["mtdata", "echo", dataset_id],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.error("mtdata echo failed: %s", result.stderr[:200])
            return []
    except subprocess.TimeoutExpired:
        logger.error("mtdata echo timed out for %s", dataset_id)
        return []

    src_lang, tgt_lang = lang_pair.split("-")
    documents = []
    for i, line in enumerate(result.stdout.strip().split("\n")):
        if i >= max_lines:
            break
        parts = line.split("\t")
        if len(parts) >= 2:
            src_text, tgt_text = parts[0].strip(), parts[1].strip()
            # Index both sides as separate documents for cross-lingual retrieval
            if len(src_text) > 20:
                documents.append({
                    "id": f"wmt_{dataset_key}_{lang_pair}_{i}_src",
                    "text": src_text,
                    "title": f"{dataset_key} [{src_lang}] #{i}",
                    "language": src_lang,
                })
            if len(tgt_text) > 20:
                documents.append({
                    "id": f"wmt_{dataset_key}_{lang_pair}_{i}_tgt",
                    "text": tgt_text,
                    "title": f"{dataset_key} [{tgt_lang}] #{i}",
                    "language": tgt_lang,
                })

    logger.info("Loaded %d documents from %s", len(documents), dataset_id)
    return documents


def _parse_wmt_dir(path: str, lang_pair: str, max_lines: int) -> List[dict]:
    """Parse downloaded mtdata recipe directory."""
    src_lang, tgt_lang = lang_pair.split("-")
    documents = []

    # mtdata outputs .tsv or separate src/tgt files
    tsv_files = globlib.glob(os.path.join(path, "**", "*.tsv"), recursive=True)
    src_files = globlib.glob(os.path.join(path, f"**/*.{src_lang}"), recursive=True)

    if tsv_files:
        for tsv_path in tsv_files[:3]:  # limit to first 3 dataset files
            dataset_name = os.path.basename(tsv_path).replace(".tsv", "")
            with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        src_text, tgt_text = parts[0], parts[1]
                        if len(src_text) > 20:
                            documents.append({
                                "id": f"wmt_{dataset_name}_{i}_src",
                                "text": src_text,
                                "title": f"{dataset_name} [{src_lang}] #{i}",
                                "language": src_lang,
                            })
                        if len(tgt_text) > 20:
                            documents.append({
                                "id": f"wmt_{dataset_name}_{i}_tgt",
                                "text": tgt_text,
                                "title": f"{dataset_name} [{tgt_lang}] #{i}",
                                "language": tgt_lang,
                            })
    elif src_files:
        for src_path in src_files[:3]:
            tgt_path = src_path.replace(f".{src_lang}", f".{tgt_lang}")
            if not os.path.exists(tgt_path):
                continue
            dataset_name = os.path.basename(src_path).replace(f".{src_lang}", "")
            with open(src_path, "r", encoding="utf-8", errors="ignore") as fs, \
                 open(tgt_path, "r", encoding="utf-8", errors="ignore") as ft:
                for i, (src_line, tgt_line) in enumerate(zip(fs, ft)):
                    if i >= max_lines:
                        break
                    src_text, tgt_text = src_line.strip(), tgt_line.strip()
                    if len(src_text) > 20:
                        documents.append({
                            "id": f"wmt_{dataset_name}_{i}_src",
                            "text": src_text,
                            "title": f"{dataset_name} [{src_lang}] #{i}",
                            "language": src_lang,
                        })
                    if len(tgt_text) > 20:
                        documents.append({
                            "id": f"wmt_{dataset_name}_{i}_tgt",
                            "text": tgt_text,
                            "title": f"{dataset_name} [{tgt_lang}] #{i}",
                            "language": tgt_lang,
                        })

    logger.info("Parsed %d documents from %s", len(documents), path)
    return documents


def load_from_files(paths: List[str], language: str = "en") -> List[dict]:
    """Load documents from local text files."""
    documents = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if text.strip():
            documents.append({
                "id": path,
                "text": text,
                "language": language,
            })
    return documents
