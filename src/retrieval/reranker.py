"""
Re-ranker — Cross-encoder re-ranking step over top retrieval candidates.

Provides both:
  - Reranker class (used by the pipeline / API) with CrossEncoder + Jaccard fallback.
  - Functional rerank() utility with pluggable score_fn for testing / advanced usage.
  - RerankedResult dataclass for structured output from functional rerank().
  - get_reranker() lazy-loader and real_cross_encoder_score_fn() for production use.
"""

from dataclasses import dataclass
from typing import List, Tuple, Callable

import numpy as np
from src.config import RERANKER_MODEL_NAME

_model = None


def get_reranker(model_name: str = RERANKER_MODEL_NAME):
    """
    Lazy-loads the real cross-encoder. Call this in your actual environment
    (not sandbox) — first call downloads ~80MB of weights, then caches.
    """
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(model_name)
    return _model


@dataclass
class RerankedResult:
    """Structured result from the functional rerank() utility."""
    chunk_id: str
    text: str
    rerank_score: float


def rerank(query: str, candidates: list[dict], score_fn: Callable, top_k: int = 4) -> list[RerankedResult]:
    """
    Functional re-ranking utility with pluggable scoring.

    candidates: list of {"chunk_id": ..., "text": ...} — the candidate pool
                to re-score (e.g. top 20 from hybrid fusion, NOT the whole corpus).
    score_fn: Callable[[list[tuple[str,str]]], list[float]] — takes a list of
              (query, doc_text) pairs, returns one relevance score per pair.
              In production this is CrossEncoder.predict; here it's swappable
              for testing.
    """
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    scores = score_fn(pairs)
    scored = [
        RerankedResult(chunk_id=c["chunk_id"], text=c["text"], rerank_score=float(s))
        for c, s in zip(candidates, scores)
    ]
    scored.sort(key=lambda r: r.rerank_score, reverse=True)
    return scored[:top_k]


def real_cross_encoder_score_fn(pairs: list[tuple[str, str]]) -> list[float]:
    """Production score_fn — wraps the real model. Requires HF access to run."""
    model = get_reranker()
    return model.predict(pairs).tolist()


class Reranker:
    """Pipeline-integrated re-ranker with CrossEncoder + Jaccard word-overlap fallback."""

    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except Exception as e:
            print(f"[Warning] CrossEncoder model failed to load ({e}). Using cosine re-ranker fallback.")
            self.model = None

    def rerank(self, query: str, candidate_chunks: List[Tuple[str, str]], top_k: int = 4) -> List[Tuple[str, str, float]]:
        """
        candidate_chunks: [(chunk_id, chunk_text), ...]
        Returns: [(chunk_id, chunk_text, score), ...] sorted descending by score.
        """
        if not candidate_chunks:
            return []

        if self.model is not None:
            try:
                pairs = [[query, text] for _, text in candidate_chunks]
                scores = self.model.predict(pairs)
                ranked = [
                    (candidate_chunks[i][0], candidate_chunks[i][1], float(scores[i]))
                    for i in range(len(candidate_chunks))
                ]
                ranked.sort(key=lambda x: x[2], reverse=True)
                return ranked[:top_k]
            except Exception as e:
                print(f"[Warning] CrossEncoder prediction error ({e}). Falling back to word overlap scoring.")

        # Fallback scoring using query word overlap / Jaccard similarity
        q_words = set(query.lower().split())
        ranked = []
        for cid, text in candidate_chunks:
            t_words = set(text.lower().split())
            overlap = len(q_words & t_words) / float(len(q_words | t_words) + 1e-9)
            ranked.append((cid, text, float(overlap)))
        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked[:top_k]
