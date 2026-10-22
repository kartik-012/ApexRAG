"""
Dynamic Strategy Router with Dense Semantic Embeddings & Shannon Entropy Gating.

Features:
  - Classifies queries into optimal retrieval strategies: 'simple', 'semantic', 'hybrid', 'rerank'.
  - Supports 384-dimensional dense semantic embeddings + structural syntax signals.
  - Quantifies decision ambiguity via Shannon Entropy H(P) over strategy probabilities.
  - Adaptive Entropy Safety Gating: If H(P) > threshold (classifier is uncertain/ambiguous),
    gates the query to the safety net ('rerank' or 'hybrid') to prevent retrieval failures.
"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import scipy.sparse as sp

from src.config import (
    ROUTER_ENTROPY_THRESHOLD,
    ROUTER_FALLBACK_STRATEGY,
    ROUTER_MODEL_PATH,
)


@dataclass
class RoutingDecision:
    """Structured routing outcome with uncertainty metrics and safety gate telemetry."""
    chosen_strategy: str
    raw_strategy: str
    probabilities: Dict[str, float]
    entropy: float
    is_entropy_gated: bool
    gating_reason: Optional[str]
    routing_confidence: float
    bandit_scores: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chosen_strategy": self.chosen_strategy,
            "raw_strategy": self.raw_strategy,
            "probabilities": self.probabilities,
            "entropy": round(self.entropy, 4),
            "is_entropy_gated": self.is_entropy_gated,
            "gating_reason": self.gating_reason,
            "routing_confidence": round(self.routing_confidence, 4),
            "bandit_scores": self.bandit_scores,
        }


def compute_shannon_entropy(probabilities: Union[Dict[str, float], List[float], np.ndarray]) -> float:
    """
    Computes Shannon Entropy in bits (log2) for a probability distribution P:
        H(P) = - sum(p * log2(p)) for p > 0

    For 4 classes:
      - Perfectly confident (e.g. [1.0, 0, 0, 0]) -> H(P) = 0.00 bits
      - Maximum uncertainty (e.g. [0.25, 0.25, 0.25, 0.25]) -> H(P) = 2.00 bits
    """
    if isinstance(probabilities, dict):
        vals = list(probabilities.values())
    else:
        vals = list(probabilities)

    total = sum(vals)
    if total <= 0:
        return 2.0  # Max entropy for 4 classes

    entropy = 0.0
    for v in vals:
        p = v / total
        if p > 1e-9:
            entropy -= p * math.log2(p)
    return float(round(entropy, 4))


def compute_routing_confidence(entropy: float, num_classes: int = 4) -> float:
    """
    Computes normalized confidence score in [0.0, 1.0] derived from Shannon entropy:
        confidence = 1.0 - (entropy / log2(num_classes))

    Linearly maps minimum uncertainty (0.0 bits) to 1.0 confidence, and maximum
    uncertainty (uniform distribution over num_classes) to 0.0 confidence.

    Args:
        entropy: Calculated Shannon entropy in bits (log2).
        num_classes: Total number of discrete candidate routing strategies (default: 4).

    Returns:
        Calibrated routing confidence bounded within [0.0, 1.0].
    """
    max_entropy = math.log2(max(2, num_classes))
    conf = 1.0 - (entropy / max_entropy)
    return float(round(max(0.0, min(1.0, conf)), 4))


class ThompsonSamplingBandit:
    """
    Bayesian Multi-Armed / Contextual Thompson Sampling Bandit for Online Strategy Adaptation.
    
    Maintains Beta(alpha, beta) conjugate priors for each retrieval strategy:
      - 'simple', 'semantic', 'hybrid', 'rerank'
    
    Provides principled exploration vs. exploitation:
      - Explores under-utilized strategies with high posterior variance
      - Exploits top-performing strategies based on observed feedback and evaluation rewards
      - Adapts online continuously without requiring offline retraining.
    """
    def __init__(
        self,
        arms: Optional[List[str]] = None,
        state_file: Union[str, Path] = "models/bandit_state.json",
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        self.arms = arms or ["simple", "semantic", "hybrid", "rerank"]
        self.state_file = Path(state_file)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.alpha: Dict[str, float] = {arm: self.prior_alpha for arm in self.arms}
        self.beta: Dict[str, float] = {arm: self.prior_beta for arm in self.arms}
        self.pull_counts: Dict[str, int] = {arm: 0 for arm in self.arms}
        self.load_state()

    def sample_arm(
        self,
        context_probs: Optional[Dict[str, float]] = None,
        exploration_weight: float = 0.35,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Samples an arm using Thompson Sampling.
        Blends Bayesian Beta posterior samples with supervised context probabilities:
            combined_score(a) = (1 - exploration_weight) * P_context(a) + exploration_weight * theta_sample(a)
        """
        sampled_theta: Dict[str, float] = {}
        for arm in self.arms:
            a = max(0.01, self.alpha.get(arm, self.prior_alpha))
            b = max(0.01, self.beta.get(arm, self.prior_beta))
            sampled_theta[arm] = float(np.random.beta(a, b))

        if context_probs:
            combined_scores: Dict[str, float] = {}
            for arm in self.arms:
                c_p = context_probs.get(arm, 0.25)
                b_s = sampled_theta.get(arm, 0.5)
                combined_scores[arm] = round((1.0 - exploration_weight) * c_p + exploration_weight * b_s, 4)
            best_arm = max(self.arms, key=lambda a: combined_scores[a])
            return best_arm, combined_scores

        best_arm = max(self.arms, key=lambda a: sampled_theta[a])
        return best_arm, {k: round(v, 4) for k, v in sampled_theta.items()}

    def update_reward(self, arm: str, reward: float):
        """Updates Beta posterior for the selected arm based on observed reward in [0.0, 1.0]."""
        if arm not in self.arms:
            return
        reward = max(0.0, min(1.0, float(reward)))
        self.alpha[arm] = self.alpha.get(arm, self.prior_alpha) + reward
        self.beta[arm] = self.beta.get(arm, self.prior_beta) + (1.0 - reward)
        self.pull_counts[arm] = self.pull_counts.get(arm, 0) + 1
        self.save_state()

    def get_expected_rewards(self) -> Dict[str, float]:
        """Returns posterior mean E[theta] = alpha / (alpha + beta) for each arm."""
        return {
            arm: round(self.alpha[arm] / (self.alpha[arm] + self.beta[arm]), 4)
            for arm in self.arms
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "arms": self.arms,
            "expected_rewards": self.get_expected_rewards(),
            "pull_counts": self.pull_counts,
            "alpha": {k: round(v, 3) for k, v in self.alpha.items()},
            "beta": {k: round(v, 3) for k, v in self.beta.items()},
        }

    def save_state(self):
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(self.get_stats(), indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[Warning] Failed to persist bandit state: {e}")

    def load_state(self) -> bool:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                for arm in self.arms:
                    if "alpha" in data and arm in data["alpha"]:
                        self.alpha[arm] = float(data["alpha"][arm])
                    if "beta" in data and arm in data["beta"]:
                        self.beta[arm] = float(data["beta"][arm])
                    if "pull_counts" in data and arm in data["pull_counts"]:
                        self.pull_counts[arm] = int(data["pull_counts"][arm])
                return True
            except Exception as e:
                print(f"[Warning] Failed to load bandit state: {e}")
        return False


class StrategyRouter:
    """
    Dense Strategy Router with Shannon Entropy Confidence Gating and Thompson Sampling Bandits.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = ROUTER_MODEL_PATH,
        entropy_threshold: float = ROUTER_ENTROPY_THRESHOLD,
        fallback_strategy: str = ROUTER_FALLBACK_STRATEGY,
        bandit_state_file: Union[str, Path] = "models/bandit_state.json",
    ):
        self.model_path = Path(model_path)
        self.entropy_threshold = entropy_threshold
        self.fallback_strategy = fallback_strategy
        self.bandit = ThompsonSamplingBandit(state_file=bandit_state_file)

        self.vectorizer = None
        self.classifier = None
        self.feature_type = "tfidf"  # "dense" or "tfidf"
        self.dimension = 0
        self.is_loaded = False

        self.load_model()

    def load_model(self) -> bool:
        """Loads trained router model from disk."""
        if self.model_path.exists():
            try:
                data = joblib.load(self.model_path)
                self.vectorizer = data.get("vectorizer")
                self.classifier = data.get("classifier")
                self.feature_type = data.get("feature_type", "tfidf" if self.vectorizer is not None else "dense")
                self.dimension = data.get("dimension", 0)
                self.is_loaded = self.classifier is not None
                return self.is_loaded
            except Exception as e:
                print(f"[Warning] Failed to load strategy router model: {e}")
        self.is_loaded = False
        return False

    def featurize(self, question: str, query_embedding: Optional[np.ndarray] = None) -> Any:
        """
        Featurizes query into dense or sparse representation concatenated
        with lexical syntax meta-features.
        """
        query_length = len(question.split())
        has_named_entity = int(bool(re.search(r"\b[A-Z][a-zA-Z0-9_]*\b", question)))
        has_code_symbol = int(bool(re.search(
            r"`[^`]+`|\b(use[A-Z]\w+|component[A-Z]\w+|React\.\w+|setState|PureComponent|StrictMode|forwardRef)\b",
            question,
        )))
        has_comparison = int(bool(re.search(r"\b(vs|versus|difference|compare|prefer|instead of)\b", question.lower())))
        has_number = int(bool(re.search(r"\d+", question)))

        meta_features = np.array([[query_length, has_named_entity, has_code_symbol, has_comparison, has_number]], dtype=np.float32)

        if self.feature_type == "dense":
            if query_embedding is None:
                # Lazy-import to prevent circular imports
                from src.ingestion.embedder import embed_texts
                query_embedding = embed_texts([question])[0]

            dense_vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
            return np.hstack([dense_vec, meta_features])

        if self.vectorizer is not None:
            text_vec = self.vectorizer.transform([question])
            return sp.hstack([text_vec, meta_features])

        return meta_features

    def predict_routing_decision(
        self,
        question: str,
        query_embedding: Optional[np.ndarray] = None,
        use_bandit: bool = False,
    ) -> RoutingDecision:
        """
        Predicts retrieval strategy and evaluates Shannon entropy to gate uncertain predictions.
        Optionally uses Bayesian Thompson Sampling for online exploration.
        """
        all_strats = ["simple", "semantic", "hybrid", "rerank"]

        if use_bandit:
            base_decision = self.predict_routing_decision(question, query_embedding=query_embedding, use_bandit=False)
            bandit_arm, bandit_scores = self.bandit.sample_arm(context_probs=base_decision.probabilities)
            if base_decision.is_entropy_gated:
                return RoutingDecision(
                    chosen_strategy=base_decision.chosen_strategy,
                    raw_strategy=base_decision.raw_strategy,
                    probabilities=base_decision.probabilities,
                    entropy=base_decision.entropy,
                    is_entropy_gated=True,
                    gating_reason=base_decision.gating_reason,
                    routing_confidence=base_decision.routing_confidence,
                    bandit_scores=bandit_scores,
                )

            return RoutingDecision(
                chosen_strategy=bandit_arm,
                raw_strategy=base_decision.raw_strategy,
                probabilities=base_decision.probabilities,
                entropy=base_decision.entropy,
                is_entropy_gated=False,
                gating_reason="Selected via Bayesian Thompson Sampling exploration",
                routing_confidence=base_decision.routing_confidence,
                bandit_scores=bandit_scores,
            )

        if self.is_loaded:
            try:
                X = self.featurize(question, query_embedding=query_embedding)
                raw_pred = str(self.classifier.predict(X)[0])
                prob_map = {s: 0.0 for s in all_strats}

                if hasattr(self.classifier, "predict_proba"):
                    probs = self.classifier.predict_proba(X)[0]
                    classes = [str(c) for c in self.classifier.classes_]
                    for c, p in zip(classes, probs):
                        prob_map[c] = round(float(p), 4)
                else:
                    prob_map[raw_pred] = 1.0

                entropy = compute_shannon_entropy(prob_map)
                routing_conf = compute_routing_confidence(entropy, num_classes=len(all_strats))

                # Shannon Entropy Safety Gate
                if entropy > self.entropy_threshold:
                    gated_strategy = self.fallback_strategy
                    reason = (
                        f"High Shannon entropy ({entropy:.3f} > {self.entropy_threshold:.2f}): "
                        f"strategy distribution is ambiguous, gated to safety net '{gated_strategy}'."
                    )
                    return RoutingDecision(
                        chosen_strategy=gated_strategy,
                        raw_strategy=raw_pred,
                        probabilities=prob_map,
                        entropy=entropy,
                        is_entropy_gated=True,
                        gating_reason=reason,
                        routing_confidence=routing_conf,
                    )

                return RoutingDecision(
                    chosen_strategy=raw_pred,
                    raw_strategy=raw_pred,
                    probabilities=prob_map,
                    entropy=entropy,
                    is_entropy_gated=False,
                    gating_reason=None,
                    routing_confidence=routing_conf,
                )
            except Exception as e:
                print(f"[Warning] ML router prediction failed ({e}), using calibrated heuristic.")

        # Calibrated Heuristic Fallback with Entropy Gating
        return self._heuristic_decision(question)

    def _heuristic_decision(self, question: str) -> RoutingDecision:
        """Heuristic rule-based routing with entropy quantification."""
        words = question.split()
        has_hook_or_api = bool(re.search(
            r"`[^`]+`|\b(use[A-Z]\w+|component[A-Z]\w+|React\.\w+|setState|StrictMode|forwardRef)\b",
            question,
        ))
        has_comparison = bool(re.search(r"\b(vs|versus|difference|compare|prefer|instead of)\b", question.lower()))

        if len(words) <= 4 and has_hook_or_api and not has_comparison:
            prob_map = {"simple": 0.65, "hybrid": 0.22, "semantic": 0.08, "rerank": 0.05}
            raw_pred = "simple"
        elif has_comparison or len(words) >= 12 or "difference between" in question.lower():
            prob_map = {"rerank": 0.70, "hybrid": 0.18, "semantic": 0.08, "simple": 0.04}
            raw_pred = "rerank"
        elif has_hook_or_api and len(words) <= 10:
            prob_map = {"hybrid": 0.60, "rerank": 0.22, "simple": 0.10, "semantic": 0.08}
            raw_pred = "hybrid"
        else:
            # Ambiguous / generic query
            prob_map = {"semantic": 0.42, "hybrid": 0.32, "rerank": 0.18, "simple": 0.08}
            raw_pred = "semantic"

        entropy = compute_shannon_entropy(prob_map)
        routing_conf = compute_routing_confidence(entropy, num_classes=4)

        if entropy > self.entropy_threshold:
            gated_strategy = self.fallback_strategy
            reason = (
                f"High Shannon entropy ({entropy:.3f} > {self.entropy_threshold:.2f}): "
                f"ambiguous distribution gated to safety net '{gated_strategy}'."
            )
            return RoutingDecision(
                chosen_strategy=gated_strategy,
                raw_strategy=raw_pred,
                probabilities=prob_map,
                entropy=entropy,
                is_entropy_gated=True,
                gating_reason=reason,
                routing_confidence=routing_conf,
            )

        return RoutingDecision(
            chosen_strategy=raw_pred,
            raw_strategy=raw_pred,
            probabilities=prob_map,
            entropy=entropy,
            is_entropy_gated=False,
            gating_reason=None,
            routing_confidence=routing_conf,
        )

    def update_bandit_reward(self, arm: str, reward: float):
        """Updates Thompson Sampling posterior based on generation or user feedback reward."""
        self.bandit.update_reward(arm, reward)

    def get_bandit_stats(self) -> Dict[str, Any]:
        """Returns Thompson Sampling bandit arms, expected rewards, and pull counts."""
        return self.bandit.get_stats()

    def predict_strategy_with_proba(
        self,
        question: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Backward-compatible API returning (chosen_strategy, probabilities).
        """
        decision = self.predict_routing_decision(question, query_embedding=query_embedding)
        return decision.chosen_strategy, decision.probabilities

    def predict_strategy(
        self,
        question: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> str:
        """
        Backward-compatible API returning chosen strategy string.
        """
        return self.predict_routing_decision(question, query_embedding=query_embedding).chosen_strategy

