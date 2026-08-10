"""
Retrieval Uncertainty Quantification — Feature 1 of ApexRAG Blueprint.

Quantifies retrieval confidence using:
1. Embedding Spread Score: Pairwise cosine distance among top-K retrieved chunk embeddings.
2. Strategy Agreement Score: Majority consensus among different retrieval strategies.
"""

import numpy as np
from src.config import SPREAD_CONFIDENCE_THRESHOLD, AGREEMENT_CONFIDENCE_THRESHOLD


def embedding_spread_score(top_k_embeddings: list[np.ndarray]) -> float:
    """
    Low spread (tight cluster) = confident, focused retrieval.
    High spread = ambiguous / query poorly matched to corpus.
    Returns average pairwise cosine distance among top-k results.
    """
    n = len(top_k_embeddings)
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        vec_i = np.array(top_k_embeddings[i]).squeeze()
        norm_i = np.linalg.norm(vec_i)
        for j in range(i + 1, n):
            vec_j = np.array(top_k_embeddings[j]).squeeze()
            norm_j = np.linalg.norm(vec_j)
            denom = norm_i * norm_j
            if denom == 0:
                cos_sim = 0.0
            else:
                cos_sim = float(np.dot(vec_i, vec_j) / denom)
            dists.append(1.0 - cos_sim)
    return float(np.mean(dists)) if dists else 0.0


def strategy_agreement_score(results_by_strategy: dict[str, list[str]]) -> float:
    """
    Fraction of strategies whose top-1 result matches the majority top-1.
    1.0 = all strategies agree. 0.25 = total disagreement.
    """
    top1s = [results[0] for results in results_by_strategy.values() if results]
    if not top1s:
        return 0.0
    most_common_count = max(top1s.count(x) for x in set(top1s))
    return float(most_common_count / len(top1s))


def confidence_label(
    spread: float,
    agreement: float,
    spread_threshold: float = SPREAD_CONFIDENCE_THRESHOLD,
    agreement_threshold: float = AGREEMENT_CONFIDENCE_THRESHOLD,
) -> str:
    """
    Categorizes overall retrieval confidence into HIGH, MEDIUM, or LOW.
    """
    if agreement >= agreement_threshold and spread <= spread_threshold:
        return "HIGH"
    if agreement >= 0.25:
        return "MEDIUM"
    return "LOW"
