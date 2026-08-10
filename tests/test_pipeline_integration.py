"""
Integration test for the full evaluation harness end-to-end pipeline.
Verifies that evaluate_harness() runs without errors and produces valid metrics.
"""

from src.evaluation.run_eval import evaluate_harness
from src.evaluation.report import generate_eval_markdown_report


def test_evaluate_harness_runs(sample_qa_pairs):
    """Full harness must execute and return structured results for all strategies."""
    result = evaluate_harness(qa_pairs=sample_qa_pairs)

    assert "timestamp" in result
    assert "num_questions" in result
    assert result["num_questions"] == len(sample_qa_pairs)

    summary = result["summary"]
    for strat in ["simple", "semantic", "hybrid", "rerank", "router"]:
        assert strat in summary
        assert "avg_recall_at_4" in summary[strat]
        assert "avg_mrr" in summary[strat]
        assert "accuracy" in summary[strat]

    # Metrics must be in [0, 1]
    for strat, metrics in summary.items():
        assert 0.0 <= metrics["avg_recall_at_4"] <= 1.0
        assert 0.0 <= metrics["avg_mrr"] <= 1.0
        assert 0.0 <= metrics["accuracy"] <= 1.0


def test_report_generation(sample_qa_pairs):
    """Report generator must return a non-empty markdown string."""
    result = evaluate_harness(qa_pairs=sample_qa_pairs)
    report = generate_eval_markdown_report(result)

    assert isinstance(report, str)
    assert "ApexRAG" in report
    assert "| Strategy" in report
    assert "Failure Attribution" in report
