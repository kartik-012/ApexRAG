# ApexRAG CI/CD Quality Gate Report

| Metric | Target SLA | Measured Score | Status |
|---|---|---|---|
| **Faithfulness (Anti-Hallucination)** | &ge; 0.60 | **0.4667** | `FAIL` :x: |
| **Context Recall (Retrieval)** | &ge; 0.40 | **0.6000** | `PASS` :white_check_mark: |
| **Answer Relevance (Intent)** | &ge; 0.60 | **0.7800** | `PASS` :white_check_mark: |

- **Total Test Pairs Evaluated**: 5
- **Execution Latency**: 16.59s
- **Gate Verdict**: **QUALITY GATE BREACHED - MERGE BLOCKED**
- **Diagnosis Breakdown**:
  - `SUCCESS`: 0
  - `RETRIEVAL_FAILURE`: 1
  - `GENERATION_HALLUCINATION`: 3
  - `COMPOUND_FAILURE`: 1
