"""
Re-ranker — cross-encoder re-scoring of a candidate pool.

Why this exists on top of hybrid search: vector search and BM25 both score
query and document INDEPENDENTLY, then compare (bi-encoder pattern). A
cross-encoder instead takes (query, document) as a single joint input and
outputs one relevance score directly — it can pick up interaction signals
a bi-encoder throws away entirely (negation, exact entity match, which
sense of an ambiguous term is meant). The cost is it's O(candidates)
forward passes at query time, not one embedding lookup — so we only ever
run it on a pre-filtered candidate pool (e.g. top 20 from hybrid fusion),
never the whole corpus.

This is a convenience wrapper — the canonical implementation lives in
src/retrieval/reranker.py. Import from there for package usage.
"""

# Re-export canonical implementation
from src.retrieval.reranker import (
    Reranker,
    RerankedResult,
    rerank,
    get_reranker,
    real_cross_encoder_score_fn,
)

__all__ = ["Reranker", "RerankedResult", "rerank", "get_reranker", "real_cross_encoder_score_fn"]


if __name__ == "__main__":
    print("=== Testing rerank() logic with stand-in scorer ===")

    candidates = [
        {"chunk_id": "c1", "text": "useState hook manages component state memory in React"},
        {"chunk_id": "c2", "text": "useEffect performs side effects after render"},
        {"chunk_id": "c3", "text": "avoid memory leaks by cleaning up in useEffect return"},
        {"chunk_id": "c4", "text": "class components use componentDidMount lifecycle method"},
    ]

    query = "how do I avoid memory leaks when a component unmounts"

    def stand_in_score_fn(pairs: list[tuple[str, str]]) -> list[float]:
        """
        Deterministic stand-in for CrossEncoder.predict(): scores each pair
        by counting shared meaningful words between query and doc text.
        """
        query_words = set(pairs[0][0].lower().split()) - {"how", "do", "i", "a", "the", "when"}
        scores = []
        for q, doc_text in pairs:
            doc_words = set(doc_text.lower().split())
            overlap = len(query_words & doc_words)
            scores.append(float(overlap))
        return scores

    top_reranked = rerank(query, candidates, stand_in_score_fn, top_k=3)
    print(f"\nRe-ranked top-3 for: '{query}'")
    for r in top_reranked:
        print(f"  score={r.rerank_score:.1f} | {r.chunk_id} | {r.text[:90]}")

    print("\n=== Testing Reranker class fallback ===")
    reranker = Reranker()
    class_candidates = [("c1", "useState hook state memory"), ("c2", "class component setState")]
    ranked = reranker.rerank("useRef hook", class_candidates, top_k=1)
    print(f"Top result: {ranked[0][0]} (score={ranked[0][2]:.3f})")
