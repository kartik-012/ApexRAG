"""
Embedder wrapper for local embedding generation.

Primary: sentence-transformers with BAAI/bge-small-en-v1.5 (free, CPU-friendly).
Fallback: TF-IDF vectorizer fallback when offline or running lightweight test sweeps.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import EMBEDDING_MODEL_NAME

_model = None
_fallback_tfidf = None


def get_embedder():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception as e:
            print(f"[Warning] SentenceTransformer failed to load ({e}). Using TF-IDF fallback embedder.")
            _model = "FALLBACK"
    return _model


def _tfidf_embed_texts(texts: list[str]) -> np.ndarray:
    global _fallback_tfidf
    if not texts:
        return np.empty((0, 384))
    if _fallback_tfidf is None:
        _fallback_tfidf = TfidfVectorizer(max_features=384)
        vecs = _fallback_tfidf.fit_transform(texts).toarray()
    else:
        try:
            vecs = _fallback_tfidf.transform(texts).toarray()
        except Exception:
            _fallback_tfidf = TfidfVectorizer(max_features=384)
            vecs = _fallback_tfidf.fit_transform(texts).toarray()
    
    if vecs.shape[1] < 384:
        padded = np.zeros((vecs.shape[0], 384))
        padded[:, :vecs.shape[1]] = vecs
        vecs = padded

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Matches embed_fn(sentences) -> np.ndarray interface.
    Normalized embeddings ensure cosine distance == 1 - dot product.
    """
    if not texts:
        return np.empty((0, 384))

    model = get_embedder()
    if model == "FALLBACK" or model is None:
        return _tfidf_embed_texts(texts)

    try:
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs)
    except Exception as e:
        print(f"[Warning] SentenceTransformer embed failed ({e}). Using TF-IDF fallback.")
        return _tfidf_embed_texts(texts)
