"""
Automated CI/CD Evaluation Quality Gate (Senior/Staff AI Engineering Standard).

Executes Ragas-style automated evaluation across the verified ground-truth dataset.
Enforces quality thresholds on:
  - Faithfulness (Factual consistency, hallucination defense) >= 0.75
  - Context Recall (Retrieval coverage of ground truth) >= 0.70
  - Answer Relevance (Query alignment) >= 0.70

Fails CI pipeline with exit code 1 if any metric breaches SLA thresholds,
blocking regression pull requests before production deployment.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

from src.config import GROUND_TRUTH_DIR, EVAL_RUNS_DIR
from src.evaluation.generation_eval import (
    evaluate_generation_sample,
    evaluate_generation_batch,
    GenerationEvalResult,
)
from src.retrieval.bm25_store import BM25Store
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.generation.llm_client import generate_answer

# CI SLA Quality Gate Thresholds
DEFAULT_MIN_FAITHFULNESS = 0.75
DEFAULT_MIN_CONTEXT_RECALL = 0.70
DEFAULT_MIN_ANSWER_RELEVANCE = 0.70


def run_ci_eval_gate(
    sample_size: int = 25,
    min_faithfulness: float = DEFAULT_MIN_FAITHFULNESS,
    min_recall: float = DEFAULT_MIN_CONTEXT_RECALL,
    min_relevance: float = DEFAULT_MIN_ANSWER_RELEVANCE,
    output_report_path: Optional[str] = None,
) -> bool:
    """
    Runs the automated evaluation quality gate and returns True if passed, False if failed.
    """
    qa_path = GROUND_TRUTH_DIR / "qa_pairs.json"
    if not qa_path.exists():
        print(f"[CI Gate Error] Ground truth QA dataset not found at {qa_path}")
        return False

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    test_samples = qa_pairs[:sample_size]
    print(f"================================================================================")
    print(f"  ApexRAG CI/CD Automated Evaluation Quality Gate")
    print(f"  Testing {len(test_samples)} verified ground truth pairs")
    print(f"  Thresholds: Faithfulness >= {min_faithfulness}, Recall >= {min_recall}, Relevance >= {min_relevance}")
    print(f"================================================================================")

    import src.api.main as api_main
    from src.ingestion.embedder import embed_texts

    api_main.initialize_pipeline()
    records = []
    t0 = time.time()

    for idx, sample in enumerate(test_samples, 1):
        q = sample["question"]
        gold_ans = sample.get("answer", "")

        # Dual-branch retrieval
        bm_res = api_main._bm25_store.query(q, top_k=5)
        q_emb = embed_texts([q])[0]
        vec_res = api_main._vector_store.query(q_emb, top_k=5)

        bm_ids = [cid for cid, _ in bm_res]
        vec_ids = [r.chunk_id for r in vec_res]
        fused = reciprocal_rank_fusion(bm_ids, vec_ids, k=60, top_k=5)

        # Parent Document Context Expansion
        context_texts = []
        seen_p = set()
        for f in fused:
            cid = f.chunk_id
            pid = api_main._id_to_meta.get(cid, {}).get("parent_id", cid)
            if pid in api_main._parent_id_to_text and pid not in seen_p:
                seen_p.add(pid)
                context_texts.append(api_main._parent_id_to_text[pid])
            elif cid in api_main._id_to_text and cid not in seen_p:
                seen_p.add(cid)
                context_texts.append(api_main._id_to_text[cid])

        context_str = "\n\n".join(context_texts) if context_texts else "React useState and useEffect manage state and effects."
        pred_ans = generate_answer(q, [context_str])

        records.append({
            "question": q,
            "answer": pred_ans,
            "context": context_str,
            "ground_truth": gold_ans,
        })
        print(f"  [{idx}/{len(test_samples)}] Evaluated query: '{q[:45]}...'")

    eval_summary = evaluate_generation_batch(records)
    total_time = round(time.time() - t0, 2)

    faith = eval_summary.get("mean_faithfulness", 0.0) or 0.0
    recall = eval_summary.get("mean_context_recall", 0.0) or 0.0
    relevance = eval_summary.get("mean_answer_relevance", 0.0) or 0.0

    faith_pass = faith >= min_faithfulness
    recall_pass = recall >= min_recall
    relevance_pass = relevance >= min_relevance
    all_passed = faith_pass and recall_pass and relevance_pass

    sample_cnt = eval_summary.get("samples_count", len(records))
    diag_breakdown = eval_summary.get("diagnoses_breakdown", {})

    print("\n--------------------------------------------------------------------------------")
    print(f"  CI/CD Quality Gate Results (Duration: {total_time}s)")
    print(f"  • Faithfulness:      {faith:.4f}  (Req: >= {min_faithfulness})  -> {'[PASS]' if faith_pass else '[FAIL]'}")
    print(f"  • Context Recall:    {recall:.4f}  (Req: >= {min_recall})  -> {'[PASS]' if recall_pass else '[FAIL]'}")
    print(f"  • Answer Relevance:  {relevance:.4f}  (Req: >= {min_relevance})  -> {'[PASS]' if relevance_pass else '[FAIL]'}")
    print(f"  • Total Processed:   {sample_cnt}")
    print(f"  • Failures by Type:  {diag_breakdown}")
    print("--------------------------------------------------------------------------------")

    # Generate Markdown PR summary artifact
    md_report = f"""# ApexRAG CI/CD Quality Gate Report

| Metric | Target SLA | Measured Score | Status |
|---|---|---|---|
| **Faithfulness (Anti-Hallucination)** | &ge; {min_faithfulness:.2f} | **{faith:.4f}** | {'`PASS` :white_check_mark:' if faith_pass else '`FAIL` :x:'} |
| **Context Recall (Retrieval)** | &ge; {min_recall:.2f} | **{recall:.4f}** | {'`PASS` :white_check_mark:' if recall_pass else '`FAIL` :x:'} |
| **Answer Relevance (Intent)** | &ge; {min_relevance:.2f} | **{relevance:.4f}** | {'`PASS` :white_check_mark:' if relevance_pass else '`FAIL` :x:'} |

- **Total Test Pairs Evaluated**: {sample_cnt}
- **Execution Latency**: {total_time}s
- **Gate Verdict**: **{'ALL CHECKS PASSED - MERGE PERMITTED' if all_passed else 'QUALITY GATE BREACHED - MERGE BLOCKED'}**
- **Diagnosis Breakdown**:
  - `SUCCESS`: {diag_breakdown.get('SUCCESS', 0)}
  - `RETRIEVAL_FAILURE`: {diag_breakdown.get('RETRIEVAL_FAILURE', 0)}
  - `GENERATION_HALLUCINATION`: {diag_breakdown.get('GENERATION_HALLUCINATION', 0)}
  - `COMPOUND_FAILURE`: {diag_breakdown.get('COMPOUND_FAILURE', 0)}
"""

    if output_report_path:
        out_p = Path(output_report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(md_report, encoding="utf-8")
        print(f"  Report written to: {out_p}")

    report_file = EVAL_RUNS_DIR / f"ci_gate_{int(time.time())}.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps({
        "all_passed": all_passed,
        "metrics": {
            "faithfulness": faith,
            "context_recall": recall,
            "answer_relevance": relevance,
        },
        "thresholds": {
            "min_faithfulness": min_faithfulness,
            "min_context_recall": min_recall,
            "min_answer_relevance": min_relevance,
        },
        "eval_summary": eval_summary,
    }, indent=2), encoding="utf-8")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ApexRAG CI/CD Automated Evaluation Quality Gate")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of ground-truth QA pairs to evaluate (default: 20)")
    parser.add_argument("--min-faithfulness", type=float, default=DEFAULT_MIN_FAITHFULNESS)
    parser.add_argument("--min-recall", type=float, default=DEFAULT_MIN_CONTEXT_RECALL)
    parser.add_argument("--min-relevance", type=float, default=DEFAULT_MIN_ANSWER_RELEVANCE)
    parser.add_argument("--output-report", type=str, default="reports/ci_eval_gate_summary.md")

    args = parser.parse_args()
    success = run_ci_eval_gate(
        sample_size=args.sample_size,
        min_faithfulness=args.min_faithfulness,
        min_recall=args.min_recall,
        min_relevance=args.min_relevance,
        output_report_path=args.output_report,
    )

    if not success:
        print("\n[CI Gate Error] One or more quality metrics failed SLA thresholds. Exiting with code 1.")
        sys.exit(1)
    else:
        print("\n[CI Gate Success] All quality gates passed! Pull request approved.")
        sys.exit(0)
