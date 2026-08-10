"""Ingestion package: Loading, MDX parsing, chunking, and embedding."""
from .loader import DocFile, load_legacy_docs, load_current_docs
from .chunkers import simple_chunk, semantic_chunk, split_into_sentences
from .embedder import embed_texts, get_embedder

__all__ = [
    "DocFile",
    "load_legacy_docs",
    "load_current_docs",
    "load_synthetic_react_docs",
    "simple_chunk",
    "semantic_chunk",
    "split_into_sentences",
    "embed_texts",
    "get_embedder",
]
