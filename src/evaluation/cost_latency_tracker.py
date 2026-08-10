"""
Cost-Accuracy-Latency Pareto Optimization — Feature 6 of ApexRAG Blueprint.
"""

import time
from typing import Callable, Dict, Any, List


def measure_strategy(strategy_fn: Callable[[str], Any], question: str) -> Dict[str, Any]:
    """
    Measures execution latency and compute time for a given retrieval strategy function.
    """
    start = time.perf_counter()
    result = strategy_fn(question)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return {
        "result": result,
        "latency_ms": latency_ms,
        "compute_cost_sec": latency_ms / 1000.0,  # $0 marginal cost; CPU-seconds measured
    }


def select_pareto_optimal(results: List[Dict[str, Any]], max_latency_ms: float = 300.0) -> Dict[str, Any]:
    """
    Objective: maximize recall_at_k subject to latency_ms <= max_latency_ms.
    Used in production infra to select operational config within SLA limits.
    """
    feasible = [r for r in results if r.get("latency_ms", float("inf")) <= max_latency_ms]
    if not feasible:
        return min(results, key=lambda r: r.get("latency_ms", float("inf")))
    return max(feasible, key=lambda r: r.get("recall_at_k", 0.0))
