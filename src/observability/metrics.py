"""
Thread-Safe Prometheus-Compatible Metrics Registry for ApexRAG.

Emits standard Prometheus exposition text format (0.0.4 / OpenMetrics specification)
and provides JSON snapshots for real-time Web Studio telemetry without requiring
external dependencies.
"""

import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class Metric:
    def __init__(self, name: str, description: str, metric_type: str, label_names: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.metric_type = metric_type
        self.label_names = tuple(sorted(label_names or []))
        self._lock = threading.Lock()

    def _format_labels(self, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return ""
        items = [f'{k}="{labels[k]}"' for k in sorted(labels.keys())]
        return "{" + ",".join(items) + "}"


class Counter(Metric):
    def __init__(self, name: str, description: str, label_names: Optional[List[str]] = None):
        super().__init__(name, description, "counter", label_names)
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = {}

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """
        Increments the counter metric monotonically.

        Args:
            amount: Non-negative value to increment (default: 1.0).
            labels: Label key-value pairs scoping the observation.

        Raises:
            ValueError: If amount is negative, preserving Prometheus counter semantics.
        """
        if amount < 0:
            raise ValueError("Counters can only increase")
        label_key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[label_key] = self._values.get(label_key, 0.0) + amount

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        label_key = tuple(sorted((labels or {}).items()))
        with self._lock:
            return self._values.get(label_key, 0.0)

    def render(self) -> List[str]:
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            if not self._values:
                lines.append(f"{self.name} 0")
            else:
                for label_key, val in sorted(self._values.items()):
                    lbl_dict = dict(label_key)
                    lbl_str = self._format_labels(lbl_dict)
                    val_str = f"{int(val)}" if val.is_integer() else f"{val:.4f}"
                    lines.append(f"{self.name}{lbl_str} {val_str}")
        return lines


class Gauge(Metric):
    def __init__(self, name: str, description: str, label_names: Optional[List[str]] = None):
        super().__init__(name, description, "gauge", label_names)
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = {}

    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        label_key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[label_key] = float(value)

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        label_key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[label_key] = self._values.get(label_key, 0.0) + amount

    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        label_key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[label_key] = self._values.get(label_key, 0.0) - amount

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        label_key = tuple(sorted((labels or {}).items()))
        with self._lock:
            return self._values.get(label_key, 0.0)

    def render(self) -> List[str]:
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} gauge",
        ]
        with self._lock:
            if not self._values:
                lines.append(f"{self.name} 0")
            else:
                for label_key, val in sorted(self._values.items()):
                    lbl_dict = dict(label_key)
                    lbl_str = self._format_labels(lbl_dict)
                    val_str = f"{int(val)}" if val.is_integer() else f"{val:.4f}"
                    lines.append(f"{self.name}{lbl_str} {val_str}")
        return lines


class Histogram(Metric):
    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__(name, description, "histogram", label_names)
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._counts: Dict[Tuple[Tuple[str, str], ...], Dict[float, int]] = {}
        self._sums: Dict[Tuple[Tuple[str, str], ...], float] = {}
        self._totals: Dict[Tuple[Tuple[str, str], ...], int] = {}

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None):
        label_key = tuple(sorted((labels or {}).items()))
        with self._lock:
            if label_key not in self._counts:
                self._counts[label_key] = {b: 0 for b in self.buckets}
                self._sums[label_key] = 0.0
                self._totals[label_key] = 0

            self._sums[label_key] += value
            self._totals[label_key] += 1
            for b in self.buckets:
                if value <= b:
                    self._counts[label_key][b] += 1

    def get_stats(self, labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        label_key = tuple(sorted((labels or {}).items()))
        with self._lock:
            total = self._totals.get(label_key, 0)
            s = self._sums.get(label_key, 0.0)
            return {
                "count": total,
                "sum": round(s, 4),
                "avg": round(s / total, 4) if total > 0 else 0.0,
            }

    def render(self) -> List[str]:
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            if not self._counts:
                # Render zero default
                for b in self.buckets:
                    lines.append(f'{self.name}_bucket{{le="{b}"}} 0')
                lines.append(f'{self.name}_bucket{{le="+Inf"}} 0')
                lines.append(f"{self.name}_sum 0")
                lines.append(f"{self.name}_count 0")
            else:
                for label_key, b_counts in sorted(self._counts.items()):
                    lbl_dict = dict(label_key)
                    for b in self.buckets:
                        count_val = b_counts[b]
                        b_lbls = {**lbl_dict, "le": str(b)}
                        lines.append(f"{self.name}_bucket{self._format_labels(b_lbls)} {count_val}")
                    inf_lbls = {**lbl_dict, "le": "+Inf"}
                    total = self._totals[label_key]
                    lines.append(f"{self.name}_bucket{self._format_labels(inf_lbls)} {total}")
                    lines.append(f"{self.name}_sum{self._format_labels(lbl_dict)} {self._sums[label_key]:.4f}")
                    lines.append(f"{self.name}_count{self._format_labels(lbl_dict)} {total}")
        return lines


class MetricsRegistry:
    """Central singleton Prometheus registry for ApexRAG."""

    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._init_standard_metrics()

    def _init_standard_metrics(self):
        # 1. API request metrics
        self.requests_total = Counter(
            "apexrag_requests_total",
            "Total count of HTTP requests processed by ApexRAG endpoints.",
            ["endpoint", "method", "status"],
        )
        self.request_duration = Histogram(
            "apexrag_request_duration_seconds",
            "Latency of HTTP requests processed by ApexRAG endpoints in seconds.",
            ["endpoint"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        # 2. Pipeline Stage Duration Histograms
        self.stage_duration = Histogram(
            "apexrag_stage_duration_seconds",
            "Internal execution duration of individual RAG pipeline stages in seconds.",
            ["stage"],
            buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )

        # 3. Strategy Router & Entropy Metrics
        self.router_decisions = Counter(
            "apexrag_router_decisions_total",
            "Count of retrieval strategy selections made by the Dense Strategy Router.",
            ["strategy"],
        )
        self.router_entropy = Histogram(
            "apexrag_router_entropy",
            "Shannon entropy distribution H(P) of router prediction probability vectors.",
            buckets=(0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0),
        )
        self.entropy_fallbacks = Counter(
            "apexrag_entropy_fallbacks_total",
            "Count of high-entropy queries safety-diverted to Cross-Encoder reranking.",
        )

        # 4. Semantic Cache Metrics
        self.cache_events = Counter(
            "apexrag_cache_events_total",
            "Total count of semantic query cache hits and misses.",
            ["event"],
        )

        # 5. Token Generation Metrics
        self.tokens_generated = Counter(
            "apexrag_tokens_generated_total",
            "Total tokens generated by LLM backends.",
            ["backend", "model"],
        )

        # 6. Active Telemetry Gauges
        self.active_requests = Gauge(
            "apexrag_active_requests",
            "Number of HTTP requests currently being handled.",
        )
        self.active_semaphore_permits = Gauge(
            "apexrag_active_semaphore_permits",
            "Available permits in the bounded concurrency semaphore.",
        )

        # 7. Ragas-Style Generation Evaluation Metrics
        self.eval_faithfulness = Gauge(
            "apexrag_eval_faithfulness",
            "Latest Ragas-style faithfulness factual consistency score (0.0 - 1.0).",
        )
        self.eval_answer_relevance = Gauge(
            "apexrag_eval_answer_relevance",
            "Latest Ragas-style answer relevance query alignment score (0.0 - 1.0).",
        )
        self.eval_context_recall = Gauge(
            "apexrag_eval_context_recall",
            "Latest Ragas-style context recall reference ground truth score (0.0 - 1.0).",
        )

        # Register all standard metrics
        for attr, obj in list(self.__dict__.items()):
            if isinstance(obj, Metric):
                self._metrics[obj.name] = obj

    def register(self, metric: Metric):
        self._metrics[metric.name] = metric

    def render_prometheus(self) -> str:
        """Renders complete Prometheus exposition text."""
        blocks = []
        for name in sorted(self._metrics.keys()):
            metric = self._metrics[name]
            lines = metric.render()
            if lines:
                blocks.append("\n".join(lines))
        return "\n\n".join(blocks) + "\n"

    def snapshot(self) -> Dict[str, Any]:
        """Returns structured JSON summary for dashboard / status telemetry."""
        # Calculate cache hit ratio
        cache_hits = self.cache_events.get({"event": "hit"})
        cache_misses = self.cache_events.get({"event": "miss"})
        total_cache = cache_hits + cache_misses
        hit_ratio = round(cache_hits / total_cache, 4) if total_cache > 0 else 0.0

        # Stages
        stages = {}
        for stg in ["embed", "dense_search", "bm25_search", "hybrid_rrf", "rerank", "router", "llm_generate"]:
            stat = self.stage_duration.get_stats({"stage": stg})
            if stat["count"] > 0:
                stages[stg] = {
                    "count": stat["count"],
                    "avg_ms": round(stat["avg"] * 1000.0, 2),
                    "total_sec": stat["sum"],
                }

        # Strategies
        strats = {
            s: int(self.router_decisions.get({"strategy": s}))
            for s in ["simple", "semantic", "hybrid", "rerank"]
        }

        return {
            "timestamp": time.time(),
            "cache": {
                "hits": int(cache_hits),
                "misses": int(cache_misses),
                "hit_ratio": hit_ratio,
            },
            "stages": stages,
            "router_decisions": strats,
            "entropy_fallbacks": int(self.entropy_fallbacks.get()),
            "eval_scores": {
                "faithfulness": round(self.eval_faithfulness.get(), 4),
                "answer_relevance": round(self.eval_answer_relevance.get(), 4),
                "context_recall": round(self.eval_context_recall.get(), 4),
            },
        }


# Global Metrics Registry Singleton
metrics_registry = MetricsRegistry()


def render_prometheus_metrics() -> str:
    return metrics_registry.render_prometheus()


def record_request_metric(endpoint: str, method: str, status_code: int, duration_sec: float):
    metrics_registry.requests_total.inc(labels={"endpoint": endpoint, "method": method, "status": str(status_code)})
    metrics_registry.request_duration.observe(duration_sec, labels={"endpoint": endpoint})


def record_stage_duration(stage: str, duration_sec: float):
    metrics_registry.stage_duration.observe(duration_sec, labels={"stage": stage})


def record_router_decision(strategy: str, entropy: Optional[float] = None, is_fallback: bool = False):
    metrics_registry.router_decisions.inc(labels={"strategy": strategy})
    if entropy is not None:
        metrics_registry.router_entropy.observe(entropy)
    if is_fallback:
        metrics_registry.entropy_fallbacks.inc()


def record_cache_event(hit: bool):
    metrics_registry.cache_events.inc(labels={"event": "hit" if hit else "miss"})


def record_tokens_generated(backend: str, model: str, token_count: int):
    metrics_registry.tokens_generated.inc(amount=float(token_count), labels={"backend": backend, "model": model})


def record_evaluation_scores(faithfulness: float, relevance: float, recall: float):
    metrics_registry.eval_faithfulness.set(faithfulness)
    metrics_registry.eval_answer_relevance.set(relevance)
    metrics_registry.eval_context_recall.set(recall)
