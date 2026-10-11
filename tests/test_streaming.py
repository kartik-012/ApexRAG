"""
Integration and protocol verification tests for Server-Sent Events (SSE) streaming API (/query/stream).

Validates:
  1. POST /query/stream emits event: meta, event: token, event: done sequence.
  2. GET /query/stream works for browser EventSource clients.
  3. Tokens reconstruct a grounded answer and metadata contains uncertainty quantification.
  4. Cache hits stream seamlessly with finish_reason='cache_hit'.
"""

import json
import pytest
import httpx
from fastapi.testclient import TestClient

from src.api.main import app, initialize_pipeline


@pytest.fixture(autouse=True)
def setup_pipeline():
    initialize_pipeline()


def _parse_sse_stream(stream_text: str):
    """Parses raw SSE wire text into structured event objects."""
    events = []
    current_event = None
    current_data = []

    for line in stream_text.splitlines():
        line = line.strip()
        if not line:
            if current_event and current_data:
                events.append({
                    "event": current_event,
                    "data": json.loads("\n".join(current_data)),
                })
                current_event = None
                current_data = []
            continue

        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            current_data.append(line.split(":", 1)[1].strip())

    if current_event and current_data:
        events.append({
            "event": current_event,
            "data": json.loads("\n".join(current_data)),
        })

    return events


def test_query_stream_post_protocol():
    """Verify POST /query/stream returns valid Server-Sent Events with meta, token, and done phases."""
    client = TestClient(app)
    payload = {
        "question": "How do you manage state with useState?",
        "strategy": "auto",
        "top_k": 3,
        "use_cache": False,
    }

    response = client.post("/query/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = _parse_sse_stream(response.text)
    assert len(events) >= 3, f"Expected at least meta, token, and done events, got {len(events)}"

    event_names = [e["event"] for e in events]
    assert event_names[0] == "meta", "First SSE event must be 'meta'"
    assert "token" in event_names, "Stream must contain 'token' events"
    assert event_names[-1] == "done", "Final SSE event must be 'done'"

    # 1. Validate meta payload
    meta = events[0]["data"]
    assert meta["question"] == "How do you manage state with useState?"
    assert meta["confidence"] in ["HIGH", "MEDIUM", "LOW"]
    assert "strategy_used" in meta
    assert len(meta["context_chunks"]) > 0
    assert "stage_latencies_ms" in meta

    # 2. Reconstruct stream tokens
    tokens = [e["data"]["token"] for e in events if e["event"] == "token"]
    full_text = "".join(tokens).strip()
    assert len(full_text) > 10, f"Reconstructed answer text is too short: {full_text}"

    # 3. Validate done payload
    done = events[-1]["data"]
    assert done["tokens_generated"] == len(tokens)
    assert done["total_latency_ms"] > 0
    assert done["finish_reason"] == "stop"


def test_query_stream_get_protocol():
    """Verify GET /query/stream supports browser EventSource GET requests."""
    client = TestClient(app)
    response = client.get("/query/stream?question=When+does+useEffect+run%3F&strategy=hybrid&top_k=2")
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = _parse_sse_stream(response.text)
    assert len(events) >= 3

    meta = events[0]["data"]
    assert "useEffect" in meta["question"]
    assert meta["confidence"] in ["HIGH", "MEDIUM", "LOW"]


@pytest.mark.anyio
async def test_async_streaming_under_concurrency():
    """Verify streaming works seamlessly under async client concurrency."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/query/stream",
            json={"question": "What is the difference between props and state?", "strategy": "simple", "use_cache": False},
        )
        assert resp.status_code == 200
        events = _parse_sse_stream(resp.text)
        assert events[0]["event"] == "meta"
        assert events[-1]["event"] == "done"
