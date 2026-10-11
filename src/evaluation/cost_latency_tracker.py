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


# Standard Model Pricing per 1M tokens (USD)
# Source: Industry standard API pricing schedules (Prompt / Completion / Embedding)
MODEL_PRICING_PER_1M: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00, "embedding": 0.02},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "embedding": 0.02},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00, "embedding": 0.02},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00, "embedding": 0.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25, "embedding": 0.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00, "embedding": 0.02},
    "text-embedding-3-large": {"input": 0.13, "output": 0.00, "embedding": 0.13},
    "local": {"input": 0.00, "output": 0.00, "embedding": 0.00},
    "default": {"input": 0.15, "output": 0.60, "embedding": 0.02},
}


def estimate_tokens(text: str) -> int:
    """Heuristic token estimation: ~1 token per 3.8-4 characters or 0.75 words."""
    if not text:
        return 0
    char_estimate = len(text) / 3.8
    word_estimate = len(text.split()) * 1.3
    return max(1, int((char_estimate + word_estimate) / 2.0))


def estimate_query_cost_usd(
    tokens_in: int = 0,
    tokens_out: int = 0,
    embedding_tokens: int = 0,
    model_name: str = "gpt-4o-mini",
) -> Dict[str, float]:
    """
    Computes precise itemized USD cost for RAG generation and embedding operations.
    Returns input_cost_usd, output_cost_usd, embedding_cost_usd, and total_cost_usd.
    """
    key = (model_name or "default").lower()
    pricing = MODEL_PRICING_PER_1M.get(key)
    if not pricing:
        matched = False
        for k, p in MODEL_PRICING_PER_1M.items():
            if k in key:
                pricing = p
                matched = True
                break
        if not matched:
            pricing = MODEL_PRICING_PER_1M["default"]

    input_cost = (tokens_in / 1_000_000.0) * pricing.get("input", 0.0)
    output_cost = (tokens_out / 1_000_000.0) * pricing.get("output", 0.0)
    embedding_cost = (embedding_tokens / 1_000_000.0) * pricing.get("embedding", 0.02)
    total_cost = input_cost + output_cost + embedding_cost

    return {
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "embedding_cost_usd": round(embedding_cost, 6),
        "total_cost_usd": round(total_cost, 6),
    }


def estimate_cost_attribution(
    question: str,
    context_chunks: List[str],
    answer: str,
    model_name: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Comprehensive RAG turn cost attribution calculator:
    Breaks down prompt, context, generation, and embedding tokens and their financial cost.
    """
    prompt_overhead_tokens = 60
    query_tokens = estimate_tokens(question)
    context_text = "\n\n".join(context_chunks) if context_chunks else ""
    context_tokens = estimate_tokens(context_text)
    tokens_in = prompt_overhead_tokens + query_tokens + context_tokens
    tokens_out = estimate_tokens(answer)
    embedding_tokens = query_tokens

    cost_dict = estimate_query_cost_usd(
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        embedding_tokens=embedding_tokens,
        model_name=model_name,
    )

    return {
        "model_name": model_name,
        "token_counts": {
            "query_tokens": query_tokens,
            "context_tokens": context_tokens,
            "prompt_overhead_tokens": prompt_overhead_tokens,
            "total_input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "embedding_tokens": embedding_tokens,
            "total_tokens": tokens_in + tokens_out + embedding_tokens,
        },
        **cost_dict,
    }

