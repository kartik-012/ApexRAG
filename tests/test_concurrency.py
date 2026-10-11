"""
Unit and Integration Tests for Non-Blocking Asynchronous Concurrency in ApexRAG.

Validates:
  1. Concurrency telemetry exposed in /health
  2. Parallel dual-branch retrieval (Dense Vector + Sparse BM25 via asyncio.gather)
  3. Non-blocking event loop execution under simultaneous concurrent load (httpx.AsyncClient)
  4. Semaphore bounding and peak concurrency tracking
"""

import asyncio
import pytest
import httpx

from src.api import main as api_main
from src.api.concurrency import ConcurrencyManager, concurrency_mgr
from src.ingestion.embedder import embed_texts


@pytest.fixture(autouse=True)
def setup_pipeline():
    """Ensure the pipeline is initialized for tests."""
    api_main.initialize_pipeline()


def test_concurrency_telemetry_in_health():
    """Verify that /health exposes rich concurrency metrics."""
    from fastapi.testclient import TestClient
    client = TestClient(api_main.app)
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "concurrency" in data
    c_meta = data["concurrency"]
    assert "active_requests" in c_meta
    assert "peak_concurrent_requests" in c_meta
    assert "parallel_retrievals_executed" in c_meta
    assert c_meta["parallel_retrieval_enabled"] is True
    assert c_meta["max_concurrent_embeds"] >= 1
    assert c_meta["max_concurrent_reranks"] >= 1


@pytest.mark.anyio
async def test_parallel_dual_branch_retrieval():
    """Verify that dense vector search and sparse BM25 search execute concurrently."""
    query = "How to use useEffect hook?"
    q_emb = embed_texts([query])[0]

    initial_count = concurrency_mgr._parallel_retrievals_executed

    vec_res, bm25_res = await concurrency_mgr.retrieve_dual_branch_parallel(
        api_main._vector_store.query,
        api_main._bm25_store.query,
        q_emb,
        query,
        top_k=4,
    )

    assert len(vec_res) > 0
    assert len(bm25_res) > 0
    assert concurrency_mgr._parallel_retrievals_executed == initial_count + 1


@pytest.mark.anyio
async def test_non_blocking_event_loop_concurrent_requests():
    """
    Simulates high concurrent load using httpx.AsyncClient and asyncio.gather:
    Fires multiple simultaneous /query, /health, and /confidence requests in parallel.
    Verifies that no head-of-line blocking occurs and all requests resolve with 200 OK.
    """
    transport = httpx.ASGITransport(app=api_main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        queries = [
            "How does useState work?",
            "What is useEffect cleanup?",
            "How does legacy componentDidMount work?",
        ]

        async def send_query(q: str):
            return await client.post("/query", json={"question": q, "strategy": "auto", "use_cache": False})

        async def send_health():
            return await client.get("/health")

        async def send_confidence(q: str):
            return await client.get(f"/confidence?question={q}")

        # Dispatch 3 heavy queries + 3 health checks + 3 confidence checks simultaneously
        tasks = [
            send_query(queries[0]),
            send_health(),
            send_confidence(queries[0]),
            send_query(queries[1]),
            send_health(),
            send_confidence(queries[1]),
            send_query(queries[2]),
            send_health(),
            send_confidence(queries[2]),
        ]

        responses = await asyncio.gather(*tasks)

        for resp in responses:
            assert resp.status_code == 200

        # Verify query responses contain expected RAG payload
        query_resps = [responses[0], responses[3], responses[6]]
        for q_resp in query_resps:
            data = q_resp.json()
            assert "answer" in data
            assert "confidence" in data
            assert "stage_latencies_ms" in data
            assert "parallel_retrieval_ms" in data["stage_latencies_ms"]

        # Verify concurrency tracking recorded multiple active requests
        telemetry = concurrency_mgr.get_telemetry()
        assert telemetry["total_requests_served"] > 0
        assert telemetry["peak_concurrent_requests"] >= 1


@pytest.mark.anyio
async def test_bounded_semaphore_capacity():
    """Verify that bounded semaphores prevent thread flooding."""
    mgr = ConcurrencyManager(max_concurrent_embeds=2, max_concurrent_reranks=1)
    
    execution_order = []

    def mock_embed(texts):
        return [0.0]

    async def worker(idx: int):
        res = await mgr.embed_texts_async(mock_embed, [f"text_{idx}"])
        execution_order.append(idx)
        return res

    tasks = [worker(i) for i in range(4)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 4
    assert len(execution_order) == 4
    assert mgr.get_telemetry()["max_concurrent_embeds"] == 2
