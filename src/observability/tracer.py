"""
OpenTelemetry-Compliant Distributed Tracing Engine & Waterfall Collector for ApexRAG.

Provides:
  - Standard W3C Trace Context propagation (trace_id, span_id, parent_span_id)
  - Synchronous & Asynchronous `trace_span` context managers
  - Thread-safe in-memory ring buffer of completed traces for API / UI inspection
  - Automatic waterfall timeline computation for flamegraph visualization
"""

import asyncio
from collections import deque
from datetime import datetime, timezone
import os
import threading
import time
from typing import Any, Dict, List, Optional
import uuid

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode, Status
    from opentelemetry.sdk.trace import TracerProvider, SpanProcessor, ReadableSpan
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.resources import Resource
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False


class TraceRecord:
    """Represents a finalized trace with its child spans and waterfall timeline."""

    def __init__(self, trace_id: str, root_name: str):
        self.trace_id = trace_id
        self.root_name = root_name
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration_ms: float = 0.0
        self.status: str = "OK"
        self.spans: List[Dict[str, Any]] = []
        self.waterfall: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "root_name": self.root_name,
            "start_time_iso": datetime.fromtimestamp(self.start_time, tz=timezone.utc).isoformat() if self.start_time > 0 else "",
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "span_count": len(self.spans),
            "spans": self.spans,
            "waterfall": self.waterfall,
        }


class TraceBuffer:
    """
    Thread-safe FIFO ring buffer storing completed distributed traces.

    Maintains a fixed-size collections.deque bounded by max_history to prevent
    unbounded memory growth under sustained production request volumes.
    Guarded by an internal threading.Lock for safe concurrent access across worker threads.
    """

    def __init__(self, max_history: int = 100):
        self._max_history = max_history
        self._lock = threading.Lock()
        self._traces: deque[TraceRecord] = deque(maxlen=max_history)
        self._trace_map: Dict[str, TraceRecord] = {}
        self._pending_spans: Dict[str, List[Dict[str, Any]]] = {}

    def record_span(self, span_dict: Dict[str, Any]):
        trace_id = span_dict["trace_id"]
        with self._lock:
            if trace_id not in self._pending_spans:
                self._pending_spans[trace_id] = []
            self._pending_spans[trace_id].append(span_dict)

            # If this is a root span (no parent) or a completed trace
            if span_dict.get("parent_span_id") is None:
                self._finalize_trace(trace_id, root_span=span_dict)

    def _finalize_trace(self, trace_id: str, root_span: Optional[Dict[str, Any]] = None):
        spans = self._pending_spans.pop(trace_id, [])
        if not spans:
            return

        # Sort spans by start timestamp
        spans.sort(key=lambda s: s["start_ns"])
        root = root_span or spans[0]
        root_start_ns = root["start_ns"]
        root_end_ns = max(s["end_ns"] for s in spans)
        duration_ms = (root_end_ns - root_start_ns) / 1e6
        if duration_ms < 0:
            duration_ms = 0.0

        trace_rec = TraceRecord(trace_id=trace_id, root_name=root.get("name", "pipeline"))
        trace_rec.start_time = root_start_ns / 1e9
        trace_rec.end_time = root_end_ns / 1e9
        trace_rec.duration_ms = duration_ms
        trace_rec.status = "ERROR" if any(s.get("status") == "ERROR" for s in spans) else "OK"

        # Build waterfall with relative offsets
        waterfall = []
        for s in spans:
            offset_ms = max(0.0, (s["start_ns"] - root_start_ns) / 1e6)
            s_dur = max(0.0, (s["end_ns"] - s["start_ns"]) / 1e6)
            pct = round((s_dur / duration_ms * 100.0), 1) if duration_ms > 0 else 100.0
            waterfall.append({
                "span_id": s["span_id"],
                "parent_span_id": s.get("parent_span_id"),
                "name": s["name"],
                "offset_ms": round(offset_ms, 2),
                "duration_ms": round(s_dur, 2),
                "pct_of_total": pct,
                "status": s.get("status", "OK"),
                "attributes": s.get("attributes", {}),
            })

        trace_rec.spans = spans
        trace_rec.waterfall = waterfall

        # Append to ring buffer
        self._traces.append(trace_rec)
        self._trace_map[trace_id] = trace_rec
        if len(self._trace_map) > self._max_history * 2:
            # Evict stale map keys
            valid_ids = {t.trace_id for t in self._traces}
            self._trace_map = {k: v for k, v in self._trace_map.items() if k in valid_ids}

    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            recs = list(self._traces)[-limit:]
            recs.reverse()
            return [r.to_dict() for r in recs]

    def get_by_id(self, trace_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            rec = self._trace_map.get(trace_id)
            return rec.to_dict() if rec else None

    def clear(self):
        with self._lock:
            self._traces.clear()
            self._trace_map.clear()
            self._pending_spans.clear()


class BufferSpanProcessor(SpanProcessor):
    """Custom OpenTelemetry SpanProcessor sending completed spans to TraceBuffer."""

    def __init__(self, buffer: TraceBuffer):
        self.buffer = buffer

    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span: ReadableSpan):
        try:
            trace_id_str = f"{span.context.trace_id:032x}"
            span_id_str = f"{span.context.span_id:016x}"
            parent_id_str = f"{span.parent.span_id:016x}" if span.parent else None

            status_str = "OK"
            if span.status.status_code == StatusCode.ERROR:
                status_str = "ERROR"

            events_list = []
            for ev in getattr(span, "events", []):
                events_list.append({
                    "name": ev.name,
                    "timestamp": ev.timestamp / 1e9,
                    "attributes": dict(ev.attributes or {}),
                })

            span_dict = {
                "trace_id": trace_id_str,
                "span_id": span_id_str,
                "parent_span_id": parent_id_str,
                "name": span.name,
                "start_ns": span.start_time,
                "end_ns": span.end_time,
                "status": status_str,
                "attributes": dict(span.attributes or {}),
                "events": events_list,
            }
            self.buffer.record_span(span_dict)
        except Exception:
            pass

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000):
        return True


class ApexTracer:
    """Singleton OpenTelemetry Tracer with ring-buffer storage and trace-context helpers."""

    def __init__(self):
        self.buffer = TraceBuffer(max_history=100)
        self._provider = TracerProvider(
            resource=Resource.create({"service.name": "apexrag", "service.version": "2.0.0"})
        )
        self._processor = BufferSpanProcessor(self.buffer)
        self._provider.add_span_processor(self._processor)
        self.tracer = self._provider.get_tracer("apexrag")
        trace.set_tracer_provider(self._provider)

    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        span = self.tracer.start_as_current_span(name)
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        return span


# Global Singleton
_apex_tracer = ApexTracer()


def get_tracer() -> ApexTracer:
    return _apex_tracer


def get_current_trace_id() -> str:
    """Returns the current active OpenTelemetry 32-char hex trace_id, or generates a fresh one."""
    current_span = trace.get_current_span()
    ctx = current_span.get_span_context() if current_span else None
    if ctx and ctx.trace_id:
        return f"{ctx.trace_id:032x}"
    return uuid.uuid4().hex


def get_current_span_id() -> str:
    """Returns the current active OpenTelemetry 16-char hex span_id, or empty string."""
    current_span = trace.get_current_span()
    ctx = current_span.get_span_context() if current_span else None
    if ctx and ctx.span_id:
        return f"{ctx.span_id:016x}"
    return ""


def get_recent_traces(limit: int = 20) -> List[Dict[str, Any]]:
    return _apex_tracer.buffer.get_recent(limit=limit)


def get_trace_by_id(trace_id: str) -> Optional[Dict[str, Any]]:
    return _apex_tracer.buffer.get_by_id(trace_id)


def clear_trace_buffer():
    _apex_tracer.buffer.clear()


class trace_span:
    """
    Unified Sync & Async Context Manager for OpenTelemetry distributed tracing.

    Usage:
        # Synchronous:
        with trace_span("retrieval.dense_search", {"top_k": 5}) as span:
            results = dense_search(query)

        # Asynchronous:
        async with trace_span("generation.llm_call", {"model": "llama3"}) as span:
            response = await generate_async(...)
    """

    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.name = name
        self.attributes = attributes or {}
        self._cm = None
        self._span = None

    def __enter__(self):
        tracer = _apex_tracer.tracer
        self._cm = tracer.start_as_current_span(self.name)
        self._span = self._cm.__enter__()
        for k, v in self.attributes.items():
            if v is not None:
                self._span.set_attribute(str(k), v if isinstance(v, (bool, str, int, float)) else str(v))
        return self._span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None and self._span is not None:
            self._span.record_exception(exc_val)
            self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))
        return self._cm.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)
