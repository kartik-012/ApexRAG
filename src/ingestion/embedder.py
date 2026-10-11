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


_fallback_svd = None


def _tfidf_embed_texts(texts: list[str]) -> np.ndarray:
    global _fallback_tfidf, _fallback_svd
    if not texts:
        return np.empty((0, 384))

    # Fit on corpus-scale batch (e.g. >100 chunks) using bigram TF-IDF + 384-d TruncatedSVD (LSA)
    if _fallback_tfidf is None or len(texts) > 100:
        if len(texts) > 400:
            from sklearn.decomposition import TruncatedSVD
            _fallback_tfidf = TfidfVectorizer(
                max_features=5000,
                stop_words="english",
                sublinear_tf=True,
                ngram_range=(1, 2),
            )
            tfidf_mat = _fallback_tfidf.fit_transform(texts)
            n_comp = min(384, tfidf_mat.shape[1] - 1, tfidf_mat.shape[0] - 1)
            _fallback_svd = TruncatedSVD(n_components=n_comp, random_state=42)
            vecs = _fallback_svd.fit_transform(tfidf_mat)
        else:
            _fallback_tfidf = TfidfVectorizer(max_features=384, sublinear_tf=True)
            _fallback_svd = None
            vecs = _fallback_tfidf.fit_transform(texts).toarray()
    else:
        try:
            tfidf_mat = _fallback_tfidf.transform(texts)
            if _fallback_svd is not None:
                vecs = _fallback_svd.transform(tfidf_mat)
            else:
                vecs = tfidf_mat.toarray()
        except Exception:
            _fallback_tfidf = TfidfVectorizer(max_features=384, sublinear_tf=True)
            _fallback_svd = None
            vecs = _fallback_tfidf.fit_transform(texts).toarray()

    if vecs.shape[1] < 384:
        padded = np.zeros((vecs.shape[0], 384))
        padded[:, :vecs.shape[1]] = vecs
        vecs = padded

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms



def ensure_embedder_fitted(corpus_texts: list[str]) -> None:
    """Ensures the fallback LSA/TF-IDF model is fitted on the corpus when loading an existing ChromaDB index."""
    global _fallback_tfidf
    model = get_embedder()
    if (model == "FALLBACK" or model is None) and _fallback_tfidf is None and corpus_texts:
        _tfidf_embed_texts(corpus_texts)


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

