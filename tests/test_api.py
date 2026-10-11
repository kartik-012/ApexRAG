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
    assert "cost_attribution" in body
    assert body["cost_attribution"]["total_cost_usd"] >= 0
    assert "token_counts" in body["cost_attribution"]



def test_confidence_endpoint():
    res = client.get("/confidence?question=useRef vs useState")
    assert res.status_code == 200
    body = res.json()
    assert body["question"] == "useRef vs useState"
    assert body["confidence"] in ["HIGH", "MEDIUM", "LOW"]

    post_res = client.post("/confidence", json={"question": "useRef vs useState"})
    assert post_res.status_code == 200
    assert post_res.json()["confidence"] in ["HIGH", "MEDIUM", "LOW"]


def test_dashboard_ui_endpoint():
    res = client.get("/dashboard")
    assert res.status_code == 200
    assert "ApexRAG Studio" in res.text


def test_compare_strategies_endpoint():
    res = client.post("/compare", json={"question": "When does useEffect cleanup run?", "top_k": 3})
    assert res.status_code == 200
    body = res.json()
    assert "strategies" in body
    for strat in ["simple", "semantic", "hybrid", "rerank", "router"]:
        assert strat in body["strategies"]


def test_debate_and_adversarial_endpoints():
    deb = client.post(
        "/debate",
        json={
            "question": "What is useState?",
            "context": "useState is a React Hook that lets you add a state variable to your component.",
            "answer": "useState adds a state variable to a component.",
        },
    )
    assert deb.status_code == 200
    assert "faithful" in deb.json()

    adv = client.post("/adversarial", json={"question": "What is useState?"})
    assert adv.status_code == 200
    assert len(adv.json()["variants"]) == 3


def test_user_feedback_endpoints():
    feedback_payload = {
        "trace_id": "test-trace-123",
        "rating": 5,
        "category": "perfect",
        "comment": "Accurate and well-grounded response.",
        "question": "What is useEffect?",
        "answer": "useEffect manages side effects.",
    }
    post_res = client.post("/feedback", json=feedback_payload)
    assert post_res.status_code == 200
    body = post_res.json()
    assert body["status"] == "success"
    assert body["entry"]["rating"] == 5

    get_res = client.get("/feedback?limit=10")
    assert get_res.status_code == 200
    get_body = get_res.json()
    assert get_body["total"] >= 1
    assert any(e["trace_id"] == "test-trace-123" for e in get_body["entries"])


