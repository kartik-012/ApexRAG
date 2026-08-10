# ApexRAG Evaluation Harness Benchmark Report
**Run Timestamp**: `2026-08-10T15:44:10.533740`  
**Benchmark Questions Evaluated**: `100`  
**Infra Cost**: `$0.00 / INR 0.00` (100% Local)

## Retrieval Strategy Performance Comparison

| Strategy | Recall@4 | MRR | Accuracy | Primary Failure Mode |
|---|---|---|---|---|
| **SIMPLE** | 0.2000 | 0.1421 | 20.0% | `RETRIEVAL_MISS` |
| **SEMANTIC** | 0.1100 | 0.0830 | 11.0% | `RETRIEVAL_MISS` |
| **HYBRID** | 0.1900 | 0.1297 | 19.0% | `RETRIEVAL_MISS` |
| **RERANK** | 0.1200 | 0.0985 | 12.0% | `RETRIEVAL_MISS` |
| **ROUTER** | 0.1400 | 0.1071 | 14.0% | `RETRIEVAL_MISS` |

## Failure Attribution Breakdown

| Strategy | Retrieval Miss | Retrieval Rank | Chunk Boundary | Hallucination |
|---|---|---|---|---|
| **SIMPLE** | 74 | 6 | 0 | 0 |
| **SEMANTIC** | 82 | 7 | 0 | 0 |
| **HYBRID** | 73 | 8 | 0 | 0 |
| **RERANK** | 73 | 15 | 0 | 0 |
| **ROUTER** | 78 | 8 | 0 | 0 |

> [!NOTE]
> Evaluation completed automatically via ApexRAG Evaluation Engine. All runs are saved to `results/eval_runs/`.