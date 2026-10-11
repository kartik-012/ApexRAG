"""
Evaluation Report Generator — formats evaluation harness JSON outputs into Markdown reports.
"""

from typing import Dict, Any


def generate_eval_markdown_report(eval_data: Dict[str, Any]) -> str:
    """
    Generates GitHub Flavored Markdown evaluation report table.
    """
    timestamp = eval_data.get("timestamp", "N/A")
    num_qs = eval_data.get("num_questions", 0)
    summary = eval_data.get("summary", {})

    report_lines = [
        f"# ApexRAG Evaluation Harness Benchmark Report",
        f"**Run Timestamp**: `{timestamp}`  ",
        f"**Benchmark Questions Evaluated**: `{num_qs}`  ",
        f"**Infra Cost**: `$0.00 / INR 0.00` (100% Local)",
        "",
        "## Retrieval Strategy Performance Comparison",
        "",
        "| Strategy | Recall@4 | MRR | nDCG@4 | Accuracy | Avg Latency (ms) | Primary Failure Mode |",
        "|---|---|---|---|---|---|---|",
    ]

    for strat, metrics in summary.items():
        recall = metrics.get("avg_recall_at_4", 0.0)
        mrr = metrics.get("avg_mrr", 0.0)
        ndcg = metrics.get("avg_ndcg_at_4", 0.0)
        acc = metrics.get("accuracy", 0.0)
        lat = metrics.get("avg_latency_ms", 0.0)
        failures = metrics.get("failure_attributions", {})
        nonzero_failures = {k: v for k, v in failures.items() if v > 0}
        top_failure = max(nonzero_failures.items(), key=lambda x: x[1])[0] if nonzero_failures else "NONE"

        report_lines.append(
            f"| **{strat.upper()}** | {recall:.4f} | {mrr:.4f} | {ndcg:.4f} | {acc * 100:.1f}% | {lat:.1f} ms | `{top_failure}` |"
        )

    pareto = eval_data.get("pareto_optimal")
    if pareto:
        report_lines.extend([
            "",
            f"**Pareto-Optimal SLA Config (≤300ms)**: `{pareto.get('strategy', 'hybrid').upper()}` "
            f"(Recall@4: `{pareto.get('recall_at_k', 0.0):.4f}`, Latency: `{pareto.get('latency_ms', 0.0):.1f} ms`)",
        ])

    report_lines.extend([
        "",
        "## Failure Attribution Breakdown",
        "",
        "| Strategy | Retrieval Miss | Retrieval Rank | Chunk Boundary | Hallucination |",
        "|---|---|---|---|---|",
    ])

    for strat, metrics in summary.items():
        failures = metrics.get("failure_attributions", {})
        miss = failures.get("RETRIEVAL_MISS", 0)
        rank = failures.get("RETRIEVAL_RANK", 0)
        bound = failures.get("CHUNK_BOUNDARY", 0)
        halluc = failures.get("GENERATION_HALLUCINATION", 0)
        report_lines.append(f"| **{strat.upper()}** | {miss} | {rank} | {bound} | {halluc} |")

    report_lines.extend([
        "",
        "> [!NOTE]",
        "> Evaluation completed automatically via ApexRAG Evaluation Engine. All runs are saved to `results/eval_runs/`.",
    ])

    return "\n".join(report_lines)

