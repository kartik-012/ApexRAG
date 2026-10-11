# ApexRAG CI/CD Quality Gate Report

| Metric | Target SLA | Measured Score | Status |
|---|---|---|---|
| **Faithfulness (Anti-Hallucination)** | &ge; 0.20 | **0.3333** | `PASS` :white_check_mark: |
| **Context Recall (Retrieval)** | &ge; 0.20 | **0.6667** | `PASS` :white_check_mark: |
| **Answer Relevance (Intent)** | &ge; 0.20 | **0.9262** | `PASS` :white_check_mark: |

- **Total Test Pairs Evaluated**: 3
- **Execution Latency**: 43.96s
- **Gate Verdict**: **ALL CHECKS PASSED - MERGE PERMITTED**
- **Diagnosis Breakdown**:
  - `SUCCESS`: 0
  - `RETRIEVAL_FAILURE`: 0
  - `GENERATION_HALLUCINATION`: 2
  - `COMPOUND_FAILURE`: 1
