"""
Unit and Integration Tests for Corrective RAG (CRAG) Self-Reflective Evaluation.
"""

from src.retrieval.crag import (
    RetrievalGrade,
    CRAGResult,
    compute_chunk_relevance,
    refine_knowledge_units,
    reformulate_query,
    execute_crag_pipeline,
)
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_compute_chunk_relevance():
    query = "How does useEffect hook manage dependencies?"
    relevant_chunk = "useEffect is a React hook that takes a function and a dependency array to run side effects."
    irrelevant_chunk = "The quick brown fox jumps over the lazy dog in the forest."

    rel_score = compute_chunk_relevance(query, relevant_chunk)
    irrel_score = compute_chunk_relevance(query, irrelevant_chunk)

    assert rel_score > 0.60
    assert irrel_score < 0.15


def test_crag_correct_grade():
    query = "What is useState hook?"
    chunks = [
        "useState is a React Hook that lets you add a state variable to your functional component.",
        "To declare a state variable, call useState at the top level of your component.",
    ]
    result = execute_crag_pipeline(query, chunks)
    assert result.grade == RetrievalGrade.CORRECT
    assert result.action_taken == "direct_generate"
    assert result.confidence_score >= 0.55
    assert len(result.refined_chunks) == 2


def test_crag_ambiguous_grade_and_knowledge_refinement():
    query = "React hooks rules"
    chunks = [
        "React hooks have two main rules: only call hooks at the top level, and only call hooks from React functions. We also went to lunch yesterday and discussed server configurations.",
    ]
    result = execute_crag_pipeline(query, chunks, threshold_correct=0.85, threshold_ambiguous=0.25)
    assert result.grade in [RetrievalGrade.AMBIGUOUS, RetrievalGrade.CORRECT]
    assert len(result.refined_chunks) == 1


def test_crag_incorrect_grade_and_query_reformulation():
    query = "can you please tell me how does python pandas dataframe work"
    chunks = [
        "React Virtual DOM is an in-memory representation of real DOM elements.",
    ]
    result = execute_crag_pipeline(query, chunks, threshold_correct=0.80, threshold_ambiguous=0.40)
    assert result.grade == RetrievalGrade.INCORRECT
    assert result.action_taken == "query_reformulation"
    assert result.reformulated_query is not None
    # Conversational filler stripped
    assert "can you please tell me" not in result.reformulated_query.lower()


def test_api_query_with_crag_telemetry():
    res = client.post(
        "/query",
        json={
            "question": "Explain useEffect cleanup function",
            "enable_crag": True,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "crag_telemetry" in data
    assert data["crag_telemetry"]["grade"] in ["CORRECT", "AMBIGUOUS", "INCORRECT"]
    assert "action_taken" in data["crag_telemetry"]
    assert "confidence_score" in data["crag_telemetry"]
