"""
RAG-Aya :: Document Chunker

Split documents into overlapping text chunks for embedding.
Supports both character-based (default) and token-based chunking.
"""

from dataclasses import dataclass
from typing import List, Optional
from logger import init_logger

logger = init_logger(__name__)


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_idx: int
    language: str = "en"


def _chunk_text_by_characters(text: str, doc_id: str, chunk_size: int = 512, overlap: int = 64, language: str = "en") -> List[Chunk]:
    """Split text into overlapping chunks by character count (legacy mode)."""
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


def _chunk_text_by_tokens(text: str, doc_id: str, chunk_size: int = 300, overlap: int = 50, language: str = "en", tokenizer=None, token_limit: Optional[int] = None) -> List[Chunk]:
    """
    Split text into overlapping chunks by token count.

    Args:
        text: Text to chunk
        doc_id: Document ID
        chunk_size: Target tokens per chunk
        overlap: Overlap tokens between chunks
        language: Language metadata
        tokenizer: Tokenizer instance (required for token counting)
        token_limit: Max tokens allowed (for warnings)

    Returns:
        List of chunks
    """
    if tokenizer is None:
        logger.warning("Token-based chunking requested but no tokenizer provided. Falling back to character-based chunking.")
        return _chunk_text_by_characters(text, doc_id, chunk_size, overlap, language)

    chunks = []
    idx = 0

    # Simple approach: split by finding token boundaries
    # We'll try to approximate token boundaries by splitting on whitespace
    words = text.split()
    if not words:
        return chunks

    current_chunk_words = []
    current_token_count = 0

    for word in words:
        word_tokens = tokenizer.count_tokens(word)
        if current_token_count + word_tokens <= chunk_size:
            current_chunk_words.append(word)
            current_token_count += word_tokens
        else:
            # Finalize current chunk
            if current_chunk_words:
                chunk_text_str = " ".join(current_chunk_words).strip()
                if chunk_text_str:
                    token_count = tokenizer.count_tokens(chunk_text_str)
                    chunks.append(Chunk(
                        text=chunk_text_str,
                        doc_id=doc_id,
                        chunk_idx=idx,
                        language=language,
                    ))

                    # Warn if chunk exceeds token limit
                    if token_limit and token_count > token_limit:
                        logger.warning(
                            "Chunk %d (doc_id=%s) exceeds token limit: %d tokens > %d limit (language=%s)",
                            idx, doc_id, token_count, token_limit, language
                        )

                    idx += 1

            # Start new chunk with overlap
            overlap_tokens = 0
            overlap_words = []
            for ow in reversed(current_chunk_words):
                ow_tokens = tokenizer.count_tokens(ow)
                if overlap_tokens + ow_tokens <= overlap:
                    overlap_words.insert(0, ow)
                    overlap_tokens += ow_tokens
                else:
                    break

            current_chunk_words = overlap_words + [word]
            current_token_count = overlap_tokens + word_tokens

    # Finalize last chunk
    if current_chunk_words:
        chunk_text_str = " ".join(current_chunk_words).strip()
        if chunk_text_str:
            token_count = tokenizer.count_tokens(chunk_text_str)
            chunks.append(Chunk(
                text=chunk_text_str,
                doc_id=doc_id,
                chunk_idx=idx,
                language=language,
            ))

            if token_limit and token_count > token_limit:
                logger.warning(
                    "Chunk %d (doc_id=%s) exceeds token limit: %d tokens > %d limit (language=%s)",
                    idx, doc_id, token_count, token_limit, language
                )

    return chunks


def chunk_text(text: str, doc_id: str, chunk_size: int = 512, overlap: int = 64, language: str = "en", tokenizer=None, token_limit: Optional[int] = None) -> List[Chunk]:
    """
    Split text into overlapping chunks.

    Dispatches to character-based or token-based chunking based on tokenizer presence.

    Args:
        text: Text to chunk
        doc_id: Document ID
        chunk_size: Size per chunk (characters or tokens)
        overlap: Overlap between chunks (characters or tokens)
        language: Language metadata
        tokenizer: Optional tokenizer (if provided, uses token-based chunking)
        token_limit: Max tokens allowed for warning (only used with tokenizer)

    Returns:
        List of chunks
    """
    if tokenizer is not None:
        return _chunk_text_by_tokens(text, doc_id, chunk_size, overlap, language, tokenizer, token_limit)
    else:
        return _chunk_text_by_characters(text, doc_id, chunk_size, overlap, language)


def chunk_documents(documents: List[dict], chunk_size: int = 512, overlap: int = 64, tokenizer=None, token_limit: Optional[int] = None) -> List[Chunk]:
    """
    Chunk a list of documents.

    Each document: {"id": str, "text": str, "language": str}

    Args:
        documents: List of documents to chunk
        chunk_size: Size per chunk (characters or tokens)
        overlap: Overlap between chunks (characters or tokens)
        tokenizer: Optional tokenizer (if provided, uses token-based chunking)
        token_limit: Max tokens allowed for warning (only used with tokenizer)

    Returns:
        List of all chunks from all documents
    """
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_text(
            text=doc["text"],
            doc_id=doc["id"],
            chunk_size=chunk_size,
            overlap=overlap,
            language=doc.get("language", "en"),
            tokenizer=tokenizer,
            token_limit=token_limit,
        )
        all_chunks.extend(doc_chunks)
    return all_chunks
