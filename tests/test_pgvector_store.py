"""
Unit tests for Solution 2: PostgreSQL + pgvector Production Vector Store & Dual Backend.
"""

import numpy as np
from src.retrieval.pgvector_store import PgVectorStore
from src.retrieval.vector_store import VectorStore
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_pgvector_store_initialization_graceful_offline():
    """PgVectorStore must handle offline/unreachable database gracefully without crashing."""
    pg_offline = PgVectorStore(dsn="postgresql://invalid:invalid@localhost:5433/nonexistent")
    assert pg_offline.is_healthy() is False
    assert pg_offline.count() == 0
    assert pg_offline.get_all_embeddings_for_ids(["any_id"]) == []


def test_vector_store_dual_backend_fallback():
    """VectorStore must automatically select pgvector when healthy or fall back to Chroma/in-memory."""
    vs = VectorStore(backend_preference="auto")
    assert vs.backend in ("pgvector", "chroma", "in_memory")
    assert isinstance(vs.is_pgvector, bool)

    # Test adding and querying chunks through the unified abstraction
    chunk_ids = ["test_chunk_01", "test_chunk_02"]
    texts = [
        "useEffect synchronizes a component with external network systems.",
        "useRef provides a mutable reference without re-rendering.",
    ]
    embeddings = np.array([
        [1.0] + [0.0] * 383,
        [0.0, 1.0] + [0.0] * 382,
    ])
    metadatas = [
        {"source": "current", "title": "useEffect"},
        {"source": "current", "title": "useRef"},
    ]

    vs.add_chunks(chunk_ids, texts, embeddings, metadatas)
    assert vs.count() >= 2

    # Query with cosine similarity
    query_vec = np.array([1.0] + [0.0] * 383)
    results = vs.query(query_vec, top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "test_chunk_01"
    assert "useEffect" in results[0].text

    # Test embedding lookup for uncertainty spread calculation
    top_embeddings = vs.get_all_embeddings_for_ids(["test_chunk_01"])
    assert len(top_embeddings) == 1
    assert len(top_embeddings[0]) == 384


def test_health_endpoint_reports_vector_backend():
    """FastAPI /health endpoint must expose the active vector store backend and pgvector status."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "vector_backend" in data
    assert "is_pgvector" in data
    assert data["vector_backend"] in ("pgvector", "chroma", "in_memory")
