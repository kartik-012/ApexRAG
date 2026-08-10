"""Unit tests for evaluation metrics, uncertainty quantification, and failure attribution."""

import numpy as np
from src.evaluation.metrics import compute_recall_at_k, compute_mrr, compute_ndcg_at_k
from src.evaluation.uncertainty import embedding_spread_score, strategy_agreement_score, confidence_label
from src.evaluation.failure_attribution import attribute_failure, FailureType


def test_metrics():
    retrieved = ["c1", "c2", "c3", "c4"]
    gold = {"c2", "c5"}

    recall = compute_recall_at_k(retrieved, gold, k=4)
    assert recall == 0.5

    mrr = compute_mrr(retrieved, gold)
    assert mrr == 0.5  # First match c2 is at rank 2 -> 1/2 = 0.5


def test_uncertainty_quantification():
    # Tight cluster -> low spread
    vecs = [np.array([1.0, 0.0]), np.array([0.99, 0.01])]
    spread = embedding_spread_score(vecs)
    assert spread < 0.1

    # Full agreement
    strat_results = {"s1": ["c1"], "s2": ["c1"], "s3": ["c1"]}
    agree = strategy_agreement_score(strat_results)
    assert agree == 1.0

    conf = confidence_label(spread, agree)
    assert conf == "HIGH"


def test_failure_attribution():
    fail_miss = attribute_failure("q1", ["c1", "c2"], {"c9"}, ["c10", "c11"], 0.9, False)
    assert fail_miss == FailureType.RETRIEVAL_MISS

    fail_rank = attribute_failure("q1", ["c1", "c2"], {"c9"}, ["c1", "c2", "c9"], 0.9, False)
    assert fail_rank == FailureType.RETRIEVAL_RANK
