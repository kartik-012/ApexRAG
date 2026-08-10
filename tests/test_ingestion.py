"""Unit tests for loader, chunkers, and embedder modules."""

from src.ingestion.loader import load_legacy_docs, load_current_docs
from src.ingestion.chunkers import simple_chunk, semantic_chunk, split_into_sentences
from src.ingestion.embedder import embed_texts

LEGACY_REPO = "data/raw_docs/react-legacy"
CURRENT_REPO = "data/raw_docs/react-dev-current"


def test_real_loader():
    legacy = load_legacy_docs(LEGACY_REPO)
    current = load_current_docs(CURRENT_REPO)
    docs = legacy + current
    assert len(docs) > 0
    assert any(d.source == "legacy" for d in docs)
    assert any(d.source == "current" for d in docs)
    # Verify zero doc_id collisions
    ids = [d.doc_id for d in docs]
    assert len(ids) == len(set(ids)), "doc_id collisions detected!"


def test_simple_chunk():
    text = "Word " * 100
    chunks = simple_chunk(text, chunk_size=20, overlap=5)
    assert len(chunks) > 1


def test_sentence_splitter():
    text = "React is great. `useState` manages state. Does it work? Yes."
    sentences = split_into_sentences(text)
    assert len(sentences) == 4
    assert "`useState` manages state." in sentences


def test_embed_texts():
    texts = ["React UI library", "State management"]
    vecs = embed_texts(texts)
    assert vecs.shape[0] == 2
    assert vecs.shape[1] == 384
