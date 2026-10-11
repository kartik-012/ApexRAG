"""
Unit and Integration Tests for Ragas-Style End-to-End Generation Evaluation.

Validates:
  1. Faithfulness scoring (factual consistency of generated answer with retrieved context).
  2. Answer relevance scoring (semantic alignment between question and generated response).
  3. Context recall scoring against reference ground-truth answer.
  4. Root-cause failure diagnosis (differentiating Retrieval Miss from LLM Hallucination).
  5. POST /eval/generation API endpoint integration.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.evaluation.generation_eval import (
    compute_faithfulness,
    compute_answer_relevance,
    compute_context_recall,
    diagnose_pipeline_failure,
    evaluate_generation_sample,
    evaluate_generation_batch,
)

SAMPLE_CONTEXT = [
    "In React function components, the `useState` Hook lets you declare a state variable. "
    "Calling `useState` returns a pair: the current state value and an updater function. "
    "If you update state to the identical value using Object.is comparison, React bails out of re-rendering.",
    "The `useEffect` Hook lets you perform side effects in functional components. "
    "The cleanup function executes before the component unmounts and before re-running on subsequent renders."
]


def test_faithfulness_faithful_response():
    """Verify that an answer directly grounded in retrieved context scores high faithfulness."""
    faithful_answer = (
        "The useState hook lets you declare a state variable in functional components and returns the current state and an updater. "
        "React will bail out of re-rendering if updated to the identical value based on Object.is."
    )
    score, verified = compute_faithfulness(faithful_answer, SAMPLE_CONTEXT)
    assert score >= 0.80, f"Expected high faithfulness for grounded answer, got {score}"
    assert len(verified) >= 1
    assert any(c.is_supported for c in verified)


def test_faithfulness_hallucinated_response():
    """Verify that ungrounded or contradictory claims score low faithfulness."""
    hallucinated_answer = (
        "React uses the experimental useMachine hook to compile WebAssembly modules into GPU shaders. "
        "Component state is permanently serialized into a SQLite database on the client file system."
    )
    score, verified = compute_faithfulness(hallucinated_answer, SAMPLE_CONTEXT)
    assert score < 0.50, f"Expected low faithfulness for hallucinated answer, got {score}"
    assert any(not c.is_supported for c in verified)


def test_answer_relevance():
    """Verify semantic relevance scoring on topic vs off topic responses."""
    question = "How does useState work in React?"
    relevant_answer = "useState is a React Hook that declares a state variable and returns an updater function to trigger re-renders."
    off_topic_answer = "The boiling temperature of water at sea level is exactly 100 degrees Celsius under standard atmospheric pressure."

    rel_score = compute_answer_relevance(question, relevant_answer)
    off_score = compute_answer_relevance(question, off_topic_answer)

    assert rel_score > off_score, f"Relevant ({rel_score}) should exceed off-topic ({off_score})"
    assert rel_score >= 0.50
    assert off_score < 0.35


def test_context_recall():
    """Verify that context recall measures coverage of ground truth reference assertions."""
    gold_answer = "Calling useState returns the state variable and an updater function. Updating to an identical value bails out of rendering."
    recall_score, verified = compute_context_recall(gold_answer, SAMPLE_CONTEXT)
    assert recall_score >= 0.80, f"Expected high context recall, got {recall_score}"


def test_root_cause_diagnosis():
    """Verify pipeline failure attribution distinguishes retrieval miss from LLM hallucination."""
    # Case 1: High context recall (retriever did its job), but low faithfulness -> LLM Hallucinated
    diag_hallucination = diagnose_pipeline_failure(
        context_recall=0.85,
        faithfulness=0.30,
        answer_relevance=0.75,
    )
    assert diag_hallucination == "GENERATION_HALLUCINATION"

    # Case 2: Low context recall (retriever failed to find the facts) -> Retrieval Miss
    diag_retrieval_miss = diagnose_pipeline_failure(
        context_recall=0.20,
        faithfulness=0.85,
        answer_relevance=0.75,
    )
    assert diag_retrieval_miss == "RETRIEVAL_FAILURE"

    # Case 3: Both recall and faithfulness are strong -> SUCCESS
    diag_success = diagnose_pipeline_failure(
        context_recall=0.90,
        faithfulness=0.95,
        answer_relevance=0.85,
    )
    assert diag_success == "SUCCESS"


def test_batch_generation_evaluation():
    """Verify macro-level batch evaluation across multiple samples."""
    samples = [
        {
            "question": "What does useState do?",
            "answer": "useState declares state in function components and returns an updater function.",
            "context": SAMPLE_CONTEXT,
            "ground_truth": "useState adds a state variable to a function component.",
        },
        {
            "question": "When does useEffect cleanup execute?",
            "answer": "The cleanup function runs before the component unmounts.",
            "context": SAMPLE_CONTEXT,
            "ground_truth": "The cleanup function runs before unmount and prior to subsequent effect executions.",
        },
    ]
    report = evaluate_generation_batch(samples)
    assert report["samples_count"] == 2
    assert report["mean_faithfulness"] >= 0.70
    assert report["mean_answer_relevance"] >= 0.50
    assert "diagnoses_breakdown" in report


def test_api_eval_generation_endpoint():
    """Verify POST /eval/generation endpoint returns valid Ragas-style metrics."""
    client = TestClient(app)
    payload = {
        "question": "How to declare state in functional React components?",
        "answer": "Use the useState Hook to declare state variables in function components.",
        "context": SAMPLE_CONTEXT,
        "ground_truth": "useState lets you add state variables to functional components.",
    }
    res = client.post("/eval/generation", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert "faithfulness" in body
    assert "answer_relevance" in body
    assert "context_recall" in body
    assert "diagnosis" in body
    assert body["diagnosis"] in ["SUCCESS", "RETRIEVAL_FAILURE", "GENERATION_HALLUCINATION", "COMPOUND_FAILURE"]
