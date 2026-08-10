"""
Hybrid search — combines BM25 and Vector rankings via Reciprocal Rank Fusion (RRF).
Formula: RRF_score(doc) = sum_over_rankers( 1 / (k + rank(doc)) )
"""

from dataclasses import dataclass
from src.config import RRF_K_CONSTANT, DEFAULT_TOP_K


@dataclass
class FusedResult:
    chunk_id: str
    fused_score: float
    bm25_rank: int | None      # None if doc absent in BM25 top candidates
    vector_rank: int | None    # None if doc absent in vector top candidates


def reciprocal_rank_fusion(
    bm25_ranked_ids: list[str],
    vector_ranked_ids: list[str],
    k: int = RRF_K_CONSTANT,
    top_k: int = DEFAULT_TOP_K,
) -> list[FusedResult]:
    """
    Combines two rank-ordered lists using Reciprocal Rank Fusion.
    """
    scores: dict[str, float] = {}
    bm25_rank_of: dict[str, int] = {}
    vector_rank_of: dict[str, int] = {}

    for rank, chunk_id in enumerate(bm25_ranked_ids):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        bm25_rank_of[chunk_id] = rank + 1

    for rank, chunk_id in enumerate(vector_ranked_ids):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        vector_rank_of[chunk_id] = rank + 1

    fused = [
        FusedResult(
            chunk_id=cid,
            fused_score=score,
            bm25_rank=bm25_rank_of.get(cid),
            vector_rank=vector_rank_of.get(cid),
        )
        for cid, score in scores.items()
    ]
    fused.sort(key=lambda r: r.fused_score, reverse=True)
    return fused[:top_k]
