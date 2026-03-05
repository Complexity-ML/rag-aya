"""
RAG-Aya :: Data Loader

Load scientific/Wikipedia documents in FR and EN.
"""

from typing import List


def load_wikipedia(languages: List[str] = None, max_per_lang: int = 50) -> List[dict]:
    """Load Wikipedia articles via HuggingFace datasets."""
    from datasets import load_dataset

    if languages is None:
        languages = ["en", "fr"]

    documents = []
    for lang in languages:
        print(f"  Loading Wikipedia ({lang})...")
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
        print(f"    Loaded {len([d for d in documents if d['language'] == lang])} articles ({lang})")

    print(f"  Total documents: {len(documents)}")
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
