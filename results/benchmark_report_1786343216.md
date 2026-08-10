# ApexRAG Evaluation Harness Benchmark Report
**Run Timestamp**: `2026-08-10T11:56:56.103264`  
**Benchmark Questions Evaluated**: `50`  
**Infra Cost**: `$0.00 / INR 0.00` (100% Local)

## Retrieval Strategy Performance Comparison

| Strategy | Recall@4 | MRR | Accuracy | Primary Failure Mode |
|---|---|---|---|---|
| **SIMPLE** | 0.8500 | 0.7722 | 88.0% | `RETRIEVAL_RANK` |
| **SEMANTIC** | 0.8700 | 0.7409 | 90.0% | `RETRIEVAL_RANK` |
| **HYBRID** | 0.8700 | 0.7899 | 90.0% | `RETRIEVAL_RANK` |
| **RERANK** | 0.6600 | 0.5429 | 68.0% | `RETRIEVAL_RANK` |
| **ROUTER** | 0.8700 | 0.7297 | 90.0% | `RETRIEVAL_RANK` |

## Failure Attribution Breakdown

| Strategy | Retrieval Miss | Retrieval Rank | Chunk Boundary | Hallucination |
|---|---|---|---|---|
| **SIMPLE** | 0 | 6 | 0 | 0 |
| **SEMANTIC** | 0 | 5 | 0 | 0 |
| **HYBRID** | 0 | 5 | 0 | 0 |
| **RERANK** | 0 | 16 | 0 | 0 |
| **ROUTER** | 0 | 5 | 0 | 0 |

> [!NOTE]
> Evaluation completed automatically via ApexRAG Evaluation Engine. All runs are saved to `results/eval_runs/`.