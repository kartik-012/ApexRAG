"""Integration tests for FastAPI endpoints using TestClient."""

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root_dashboard_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    assert "docs_url" in res.json()


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_query_endpoint():
    payload = {"question": "How to handle state in class components?", "strategy": "auto"}
    res = client.post("/query", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert "answer" in body
    assert "confidence" in body
    assert "spread_score" in body


def test_confidence_endpoint():
    res = client.get("/confidence?question=useRef vs useState")
    assert res.status_code == 200
    body = res.json()
    assert body["question"] == "useRef vs useState"
    assert body["confidence"] in ["HIGH", "MEDIUM", "LOW"]
