"""
Production Observability, Distributed Tracing (OpenTelemetry) and Prometheus Metrics for ApexRAG.
"""

from src.observability.tracer import (
    ApexTracer,
    get_tracer,
    trace_span,
    get_current_trace_id,
    get_recent_traces,
    get_trace_by_id,
)
from src.observability.metrics import (
    metrics_registry,
    render_prometheus_metrics,
    record_request_metric,
    record_stage_duration,
    record_router_decision,
    record_cache_event,
    record_tokens_generated,
    record_evaluation_scores,
)

__all__ = [
    "ApexTracer",
    "get_tracer",
    "trace_span",
    "get_current_trace_id",
    "get_recent_traces",
    "get_trace_by_id",
    "metrics_registry",
    "render_prometheus_metrics",
    "record_request_metric",
    "record_stage_duration",
    "record_router_decision",
    "record_cache_event",
    "record_tokens_generated",
    "record_evaluation_scores",
]
