"""
Unit and Integration Tests for Solution 7: Production Observability, OpenTelemetry Tracing & Prometheus Metrics.
"""

import asyncio
import time
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.observability.tracer import (
    ApexTracer,
    TraceBuffer,
    get_tracer,
    trace_span,
    get_current_trace_id,
    get_recent_traces,
    get_trace_by_id,
    clear_trace_buffer,
)
from src.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    metrics_registry,
    render_prometheus_metrics,
    record_request_metric,
    record_stage_duration,
    record_router_decision,
    record_cache_event,
)


@pytest.fixture(autouse=True)
def reset_observability_state():
    """Reset traces buffer before each test."""
    clear_trace_buffer()
    yield


def test_opentelemetry_tracer_initialization_and_attributes():
    """Verifies that ApexTracer initializes OpenTelemetry and records spans with attributes."""
    tracer = get_tracer()
    assert tracer is not None
    assert tracer.tracer is not None

    with trace_span("unit_test.dense_search", {"strategy": "semantic", "top_k": 5}) as span:
        current_tid = get_current_trace_id()
        assert len(current_tid) == 32
        assert all(c in "0123456789abcdef" for c in current_tid)

    traces = get_recent_traces(limit=5)
    assert len(traces) >= 1
    t = traces[0]
    assert t["trace_id"] == current_tid
    assert t["root_name"] == "unit_test.dense_search"
    assert t["status"] == "OK"
    assert t["span_count"] >= 1


def test_async_trace_span_context_manager():
    """Verifies that trace_span works asynchronously and captures nested parent-child hierarchies."""
    parent_tid = None

    async def _run():
        nonlocal parent_tid
        async with trace_span("parent_query_flow", {"env": "test"}):
            parent_tid = get_current_trace_id()
            await asyncio.sleep(0.01)

            async with trace_span("child_retrieval_step", {"step": 1}):
                await asyncio.sleep(0.01)

    asyncio.run(_run())

    traces = get_recent_traces(limit=5)
    assert len(traces) >= 1
    found = [t for t in traces if t["trace_id"] == parent_tid]
    assert len(found) == 1
    t = found[0]
    assert t["root_name"] == "parent_query_flow"
    assert t["span_count"] == 2

    # Verify parent-child linkage in spans
    spans_by_name = {s["name"]: s for s in t["spans"]}
    assert "parent_query_flow" in spans_by_name
    assert "child_retrieval_step" in spans_by_name

    parent_span = spans_by_name["parent_query_flow"]
    child_span = spans_by_name["child_retrieval_step"]
    assert child_span["parent_span_id"] == parent_span["span_id"]


def test_waterfall_timeline_computation():
    """Verifies waterfall timeline calculations with relative offsets, durations, and percentages."""
    with trace_span("waterfall_root"):
        time.sleep(0.01)
        with trace_span("step_1"):
            time.sleep(0.01)
        with trace_span("step_2"):
            time.sleep(0.01)

    traces = get_recent_traces(limit=1)
    assert len(traces) == 1
    waterfall = traces[0]["waterfall"]
    assert len(waterfall) == 3

    for item in waterfall:
        assert "name" in item
        assert "offset_ms" in item
        assert item["offset_ms"] >= 0.0
        assert "duration_ms" in item
        assert "pct_of_total" in item
        assert 0.0 <= item["pct_of_total"] <= 100.0


def test_trace_error_status_and_exception_handling():
    """Verifies that exceptions inside trace_span mark the span and trace as ERROR."""
    tid = None
    try:
        with trace_span("faulty_stage") as s:
            tid = get_current_trace_id()
            raise ValueError("Simulated retrieval timeout")
    except ValueError:
        pass

    assert tid is not None
    trace_obj = get_trace_by_id(tid)
    assert trace_obj is not None
    assert trace_obj["status"] == "ERROR"
    faulty_span = trace_obj["spans"][0]
    assert faulty_span["status"] == "ERROR"


def test_prometheus_metrics_registry_and_exposition():
    """Tests Counter, Gauge, and Histogram metrics and verifies standard Prometheus text formatting."""
    reg = MetricsRegistry()

    # Counter test
    reg.requests_total.inc(labels={"endpoint": "/query", "method": "POST", "status": "200"})
    assert reg.requests_total.get({"endpoint": "/query", "method": "POST", "status": "200"}) == 1.0

    # Gauge test
    reg.active_requests.set(4)
    assert reg.active_requests.get() == 4.0
    reg.active_requests.dec()
    assert reg.active_requests.get() == 3.0

    # Histogram test
    reg.stage_duration.observe(0.025, labels={"stage": "dense_search"})
    reg.stage_duration.observe(0.045, labels={"stage": "dense_search"})
    stats = reg.stage_duration.get_stats({"stage": "dense_search"})
    assert stats["count"] == 2
    assert abs(stats["sum"] - 0.070) < 1e-4

    # Prometheus exposition text formatting
    prom_text = reg.render_prometheus()
    assert "# HELP apexrag_requests_total" in prom_text
    assert "# TYPE apexrag_requests_total counter" in prom_text
    assert 'apexrag_requests_total{endpoint="/query",method="POST",status="200"} 1' in prom_text
    assert "# HELP apexrag_stage_duration_seconds" in prom_text
    assert "# TYPE apexrag_stage_duration_seconds histogram" in prom_text
    assert 'apexrag_stage_duration_seconds_bucket{le="0.05",stage="dense_search"} 2' in prom_text
    assert 'apexrag_stage_duration_seconds_count{stage="dense_search"} 2' in prom_text


def test_api_metrics_endpoint():
    """Tests GET /metrics endpoint returns valid Prometheus 0.0.4 exposition text."""
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert "# HELP apexrag_requests_total" in body
    assert "# TYPE apexrag_requests_total counter" in body
    assert "apexrag_stage_duration_seconds" in body
    assert "apexrag_active_requests" in body


def test_api_traces_endpoints():
    """Tests GET /traces and GET /traces/{trace_id} endpoints."""
    client = TestClient(app)

    # Perform a health request to generate a trace
    res = client.get("/health")
    assert res.status_code == 200
    trace_id_header = res.headers.get("X-Trace-Id")
    assert trace_id_header is not None
    assert len(trace_id_header) == 32

    # Query /traces
    list_res = client.get("/traces?limit=10")
    assert list_res.status_code == 200
    data = list_res.json()
    assert "traces" in data
    assert "count" in data
    assert data["count"] >= 1

    # Query /traces/{trace_id}
    detail_res = client.get(f"/traces/{trace_id_header}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["trace_id"] == trace_id_header
    assert "waterfall" in detail
    assert "spans" in detail


def test_middleware_trace_headers_and_payload_injection():
    """Verifies that API requests receive X-Trace-Id headers and /query response body includes trace_id."""
    client = TestClient(app)

    payload = {
        "question": "What is the useEffect dependency array in React?",
        "strategy": "auto",
        "top_k": 2,
    }
    res = client.post("/query", json=payload)
    assert res.status_code == 200

    # Headers
    assert "X-Trace-Id" in res.headers
    trace_id = res.headers["X-Trace-Id"]
    assert len(trace_id) == 32
    assert "X-Response-Time-Ms" in res.headers

    # Payload
    body = res.json()
    assert "trace_id" in body
    assert body["trace_id"] == trace_id
    assert "stage_latencies_ms" in body
