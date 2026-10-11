"""
Tests for Automated CI/CD Evaluation Quality Gate.
"""

from src.evaluation.ci_gate import run_ci_eval_gate


def test_ci_eval_gate_execution():
    # Run gate with realistic baseline thresholds
    passed = run_ci_eval_gate(
        sample_size=3,
        min_faithfulness=0.20,
        min_recall=0.20,
        min_relevance=0.20,
        output_report_path="reports/test_ci_gate_report.md",
    )
    assert passed is True


def test_ci_eval_gate_fails_on_high_threshold():
    # A threshold of 0.99 must fail and block the merge
    passed = run_ci_eval_gate(
        sample_size=2,
        min_faithfulness=0.99,
        min_recall=0.99,
        min_relevance=0.99,
        output_report_path="reports/test_ci_gate_fail_report.md",
    )
    assert passed is False
