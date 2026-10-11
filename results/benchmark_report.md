# ApexRAG Evaluation Harness Benchmark Report
**Run Timestamp**: `2026-10-11T16:26:45.681746`  
**Benchmark Questions Evaluated**: `100`  
**Infra Cost**: `$0.00 / INR 0.00` (100% Local)

## Retrieval Strategy Performance Comparison

| Strategy | Recall@4 | MRR | nDCG@4 | Accuracy | Avg Latency (ms) | Primary Failure Mode |
|---|---|---|---|---|---|---|
| **SIMPLE** | 0.6500 | 0.4501 | 0.4850 | 65.0% | 13.1 ms | `RETRIEVAL_MISS` |
| **SEMANTIC** | 0.6100 | 0.4347 | 0.4661 | 61.0% | 9.3 ms | `RETRIEVAL_MISS` |
| **HYBRID** | 0.6100 | 0.4586 | 0.4804 | 61.0% | 22.4 ms | `RETRIEVAL_MISS` |
| **RERANK** | 0.6700 | 0.4540 | 0.4996 | 67.0% | 23.4 ms | `RETRIEVAL_MISS` |
| **ROUTER** | 0.7100 | 0.5040 | 0.5487 | 71.0% | 23.5 ms | `RETRIEVAL_MISS` |

**Pareto-Optimal SLA Config (≤300ms)**: `ROUTER` (Recall@4: `0.7100`, Latency: `23.5 ms`)

## Failure Attribution Breakdown

| Strategy | Retrieval Miss | Retrieval Rank | Chunk Boundary | Hallucination |
|---|---|---|---|---|
| **SIMPLE** | 22 | 13 | 0 | 0 |
| **SEMANTIC** | 28 | 11 | 0 | 0 |
| **HYBRID** | 24 | 15 | 0 | 0 |
| **RERANK** | 25 | 8 | 0 | 0 |
| **ROUTER** | 23 | 6 | 0 | 0 |

> [!NOTE]
> Evaluation completed automatically via ApexRAG Evaluation Engine. All runs are saved to `results/eval_runs/`.