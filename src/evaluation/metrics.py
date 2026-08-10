"""
Evaluation metrics calculation engine: Recall@K, MRR, nDCG@K.
"""

import numpy as np


def compute_recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int = 4) -> float:
    """
    Recall@K = |Retrieved@K ∩ Gold| / |Gold|
    """
    if not gold_ids:
        return 1.0  # Void target / non-retrieval ground truth
    retrieved_k = retrieved_ids[:k]
    matched = set(retrieved_k) & gold_ids
    return len(matched) / float(len(gold_ids))


def compute_mrr(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    """
    Mean Reciprocal Rank (MRR) = 1 / rank_of_first_gold_item
    """
    if not gold_ids:
        return 1.0
    for rank, cid in enumerate(retrieved_ids):
        if cid in gold_ids:
            return 1.0 / (rank + 1)
    return 0.0


def compute_ndcg_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int = 4) -> float:
    """
    Normalized Discounted Cumulative Gain (nDCG@K).
    """
    if not gold_ids:
        return 1.0
    dcg = 0.0
    for rank, cid in enumerate(retrieved_ids[:k]):
        if cid in gold_ids:
            dcg += 1.0 / np.log2(rank + 2)

    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold_ids), k)))
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)
