"""
RAG-Aya :: Document Chunker

Split documents into overlapping text chunks for embedding.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_idx: int
    language: str = "en"


def chunk_text(text: str, doc_id: str, chunk_size: int = 512, overlap: int = 64, language: str = "en") -> List[Chunk]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text_str = text[start:end]
        if chunk_text_str.strip():
            chunks.append(Chunk(
                text=chunk_text_str.strip(),
                doc_id=doc_id,
                chunk_idx=idx,
                language=language,
            ))
            idx += 1
        start += chunk_size - overlap
    return chunks


def chunk_documents(documents: List[dict], chunk_size: int = 512, overlap: int = 64) -> List[Chunk]:
    """
    Chunk a list of documents.

    Each document: {"id": str, "text": str, "language": str}
    """
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_text(
            text=doc["text"],
            doc_id=doc["id"],
            chunk_size=chunk_size,
            overlap=overlap,
            language=doc.get("language", "en"),
        )
        all_chunks.extend(doc_chunks)
    return all_chunks
