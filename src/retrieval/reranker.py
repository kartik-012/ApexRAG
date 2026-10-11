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
    """Pipeline-integrated re-ranker with CrossEncoder singleton caching + code-aware fallback."""

    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        global _model
        if _model is not None and _model != "FAILED":
            self.model = _model
        elif _model != "FAILED":
            try:
                from sentence_transformers import CrossEncoder
                _model = CrossEncoder(model_name)
                self.model = _model
            except Exception as e:
                print(f"[Warning] CrossEncoder model failed to load ({e}). Using code-aware lexical re-ranker fallback.")
                _model = "FAILED"
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

        # Enhanced fallback scoring: stopword-filtered Jaccard + exact code symbol + bigram boost
        import re
        stopwords = {"what", "how", "does", "do", "is", "the", "a", "an", "in", "to", "of", "for", "with", "when", "why", "can", "you"}
        q_tokens = [w for w in re.findall(r"[a-zA-Z0-9_]+", query.lower()) if w not in stopwords]
        q_words = set(q_tokens) or set(query.lower().split())
        q_bigrams = {f"{q_tokens[i]} {q_tokens[i+1]}" for i in range(len(q_tokens) - 1)}
        code_symbols = set(re.findall(r"`([^`]+)`|\b(use[A-Z]\w+|component[A-Z]\w+|setState|PureComponent|StrictMode|forwardRef)\b", query))
        flat_symbols = {s.lower() for tup in code_symbols for s in tup if s}

        ranked = []
        for rank_idx, (cid, text) in enumerate(candidate_chunks):
            text_low = text.lower()
            t_tokens = [w for w in re.findall(r"[a-zA-Z0-9_]+", text_low) if w not in stopwords]
            t_words = set(t_tokens) or set(text_low.split())
            overlap = len(q_words & t_words) / float(len(q_words | t_words) + 1e-9)
            # Boost exact symbol matches in chunk text or chunk_id
            symbol_boost = sum(0.25 for sym in flat_symbols if sym in text_low or sym in cid.lower())
            # Boost bigram matches
            bigram_boost = sum(0.10 for bg in q_bigrams if bg in text_low)
            # Slight prior for initial hybrid rank position
            prior = 0.05 / (rank_idx + 1)
            score = float(overlap + symbol_boost + bigram_boost + prior)
            ranked.append((cid, text, score))
        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked[:top_k]

