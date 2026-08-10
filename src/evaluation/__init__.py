"""Evaluation package: Metrics, Retrieval Uncertainty, Failure Attribution, Pareto Cost Tracking, Harness Runner & Reporter."""
from .metrics import compute_recall_at_k, compute_mrr, compute_ndcg_at_k
from .uncertainty import embedding_spread_score, strategy_agreement_score, confidence_label
from .failure_attribution import attribute_failure, FailureType
from .cost_latency_tracker import measure_strategy, select_pareto_optimal
from .run_eval import evaluate_harness
from .report import generate_eval_markdown_report

__all__ = [
    "compute_recall_at_k",
    "compute_mrr",
    "compute_ndcg_at_k",
    "embedding_spread_score",
    "strategy_agreement_score",
    "confidence_label",
    "attribute_failure",
    "FailureType",
    "measure_strategy",
    "select_pareto_optimal",
    "evaluate_harness",
    "generate_eval_markdown_report",
]
