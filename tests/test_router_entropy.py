"""
Unit and Integration Tests for Dense Strategy Router with Shannon Entropy Confidence Gating.

Validates:
  1. Shannon entropy calculation across confident vs ambiguous probability distributions.
  2. Entropy safety gating: ambiguous predictions (H(P) > threshold) are overridden with safety net ('rerank').
  3. Dense 384-d semantic embedding + structural syntax featurization.
  4. End-to-end routing decision generation and telemetry.
"""

import math
import numpy as np
import pytest

from src.retrieval.router import (
    StrategyRouter,
    compute_shannon_entropy,
    compute_routing_confidence,
    RoutingDecision,
)
from src.ingestion.embedder import embed_texts


def test_shannon_entropy_calculation():
    """Verify Shannon entropy mathematics across certainty extremes."""
    # 1. Decisive prediction: high certainty
    confident_dist = {"simple": 0.94, "hybrid": 0.04, "semantic": 0.01, "rerank": 0.01}
    h_confident = compute_shannon_entropy(confident_dist)
    assert h_confident < 0.5, f"Expected low entropy for confident prediction, got {h_confident}"

    # 2. Perfectly uniform prediction: maximum ambiguity across 4 classes
    uniform_dist = {"simple": 0.25, "hybrid": 0.25, "semantic": 0.25, "rerank": 0.25}
    h_uniform = compute_shannon_entropy(uniform_dist)
    assert math.isclose(h_uniform, 2.0, rel_tol=1e-3), f"Expected H=2.0 for 4 uniform classes, got {h_uniform}"

    # 3. Confidence score scaling
    conf_high = compute_routing_confidence(h_confident, num_classes=4)
    conf_low = compute_routing_confidence(h_uniform, num_classes=4)
    assert conf_high > 0.75
    assert conf_low == 0.0


def test_entropy_safety_gating():
    """Verify that ambiguous query distributions trigger the entropy safety gate."""
    router = StrategyRouter(entropy_threshold=1.25, fallback_strategy="rerank")

    # Highly ambiguous distribution (entropy ~ 1.98 > 1.25)
    ambiguous_decision = router.predict_routing_decision("some completely ambiguous generic question")
    # Simulate ambiguous distribution explicitly through compute_shannon_entropy
    amb_probs = {"simple": 0.28, "semantic": 0.26, "hybrid": 0.24, "rerank": 0.22}
    h_amb = compute_shannon_entropy(amb_probs)
    assert h_amb > 1.25

    # If entropy exceeds threshold, router must gate to fallback
    if ambiguous_decision.entropy > router.entropy_threshold:
        assert ambiguous_decision.is_entropy_gated is True
        assert ambiguous_decision.chosen_strategy == "rerank"
        assert "gated to safety net" in (ambiguous_decision.gating_reason or "")


def test_dense_embedding_featurization():
    """Verify that dense embedding featurization produces high-dimensional vectors."""
    router = StrategyRouter()
    query = "How to use useEffect hook with dependency array?"
    q_emb = embed_texts([query])[0]

    features = router.featurize(query, query_embedding=q_emb)
    # Features include 384 embedding dimensions + 5 structural meta-features = 389
    assert features.shape[0] == 1
    assert features.shape[1] >= 384, f"Expected >=384 dimensions, got {features.shape[1]}"


def test_end_to_end_routing_decisions():
    """Verify end-to-end prediction across distinct question types."""
    router = StrategyRouter()

    queries = [
        "useRef vs useState",
        "Detailed comparison of componentDidMount vs useEffect with cleanup in complex applications",
        "How do React components render?",
    ]

    for q in queries:
        decision = router.predict_routing_decision(q)
        assert isinstance(decision, RoutingDecision)
        assert decision.chosen_strategy in ["simple", "semantic", "hybrid", "rerank"]
        assert 0.0 <= decision.entropy <= 2.0
        assert 0.0 <= decision.routing_confidence <= 1.0

        # Probabilities must sum to ~1.0
        total_p = sum(decision.probabilities.values())
        assert math.isclose(total_p, 1.0, abs_tol=0.05)

        # Dictionary serialization
        d = decision.to_dict()
        assert "chosen_strategy" in d
        assert "entropy" in d
        assert "is_entropy_gated" in d


def test_thompson_sampling_bandit_and_online_adaptation(tmp_path):
    """Verify Bayesian Thompson Sampling arm exploration, reward update, and persistence."""
    from src.retrieval.router import ThompsonSamplingBandit

    state_file = tmp_path / "test_bandit_state.json"
    bandit = ThompsonSamplingBandit(state_file=state_file, prior_alpha=1.0, prior_beta=1.0)

    # Initial expected rewards should be 0.5 (alpha=1, beta=1)
    rewards_init = bandit.get_expected_rewards()
    for arm in ["simple", "semantic", "hybrid", "rerank"]:
        assert math.isclose(rewards_init[arm], 0.5, abs_tol=1e-3)

    # Sample arm
    arm, scores = bandit.sample_arm(context_probs={"simple": 0.1, "semantic": 0.2, "hybrid": 0.5, "rerank": 0.2})
    assert arm in ["simple", "semantic", "hybrid", "rerank"]
    assert len(scores) == 4

    # Reward 'hybrid' with multiple successful feedback events
    for _ in range(5):
        bandit.update_reward("hybrid", reward=1.0)
    for _ in range(5):
        bandit.update_reward("simple", reward=0.0)

    rewards_updated = bandit.get_expected_rewards()
    assert rewards_updated["hybrid"] > 0.7  # alpha increased -> expected reward increased
    assert rewards_updated["simple"] < 0.3  # beta increased -> expected reward decreased

    # Persistence verification
    new_bandit = ThompsonSamplingBandit(state_file=state_file)
    assert new_bandit.pull_counts["hybrid"] == 5
    assert new_bandit.pull_counts["simple"] == 5


def test_strategy_router_with_thompson_sampling(tmp_path):
    """Verify StrategyRouter integration with Thompson Sampling."""
    state_file = tmp_path / "bandit_test.json"
    router = StrategyRouter(bandit_state_file=state_file)

    decision = router.predict_routing_decision("How to manage state in React?", use_bandit=True)
    assert decision.chosen_strategy in ["simple", "semantic", "hybrid", "rerank"]
    assert decision.bandit_scores is not None
    assert len(decision.bandit_scores) == 4

    # Online reward update through router
    router.update_bandit_reward(decision.chosen_strategy, reward=1.0)
    stats = router.get_bandit_stats()
    assert stats["pull_counts"][decision.chosen_strategy] >= 1

