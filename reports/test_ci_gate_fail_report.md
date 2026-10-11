# ApexRAG CI/CD Quality Gate Report

| Metric | Target SLA | Measured Score | Status |
|---|---|---|---|
| **Faithfulness (Anti-Hallucination)** | &ge; 0.99 | **0.3333** | `FAIL` :x: |
| **Context Recall (Retrieval)** | &ge; 0.99 | **1.0000** | `PASS` :white_check_mark: |
| **Answer Relevance (Intent)** | &ge; 0.99 | **0.8893** | `FAIL` :x: |

- **Total Test Pairs Evaluated**: 2
- **Execution Latency**: 20.78s
- **Gate Verdict**: **QUALITY GATE BREACHED - MERGE BLOCKED**
- **Diagnosis Breakdown**:
  - `SUCCESS`: 0
  - `RETRIEVAL_FAILURE`: 0
  - `GENERATION_HALLUCINATION`: 2
  - `COMPOUND_FAILURE`: 0
