"""
Chunking strategies for the RAG evaluation harness.

simple_chunk():   fixed-size sliding window over words, with overlap.
semantic_chunk(): splits at sentence-level topic-shift boundaries, detected
                  via embedding similarity between consecutive sentences.
"""

import re
import numpy as np


def simple_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Fixed-size sliding window over whitespace-split words.
    """
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks


_SENTENCE_SPLIT_PATTERN = re.compile(
    r'(?<!\b[A-Z])(?<!\betc)(?<=[.!?])\s+(?=[A-Z`\(\[])'
)


def split_into_sentences(text: str) -> list[str]:
    """
    Lightweight sentence splitter tuned for technical docs: protects inline code spans
    from being split on internal periods.
    """
    text = re.sub(r'`[^`]*`', lambda m: m.group(0).replace('.', '\uE000'), text)
    raw_sentences = _SENTENCE_SPLIT_PATTERN.split(text)
    sentences = [s.replace('\uE000', '.').strip() for s in raw_sentences if s.strip()]
    return sentences


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def semantic_chunk(
    text: str,
    embed_fn,
    similarity_threshold: float = 0.5,
    min_chunk_sentences: int = 2,
    max_chunk_sentences: int = 15,
) -> list[str]:
    """
    Split text into sentences, embed each, and start a new chunk whenever
    similarity between consecutive sentences drops below `similarity_threshold`.
    """
    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        return sentences

    embeddings = embed_fn(sentences)
    chunks, current = [], [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i - 1], embeddings[i])
        should_break = (
            sim < similarity_threshold
            and len(current) >= min_chunk_sentences
        ) or len(current) >= max_chunk_sentences

        if should_break:
            chunks.append(" ".join(current))
            current = []
        current.append(sentences[i])

    if current:
        chunks.append(" ".join(current))
    return chunks
