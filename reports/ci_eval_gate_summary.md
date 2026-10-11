# ApexRAG CI/CD Quality Gate Report

| Metric | Target SLA | Measured Score | Status |
|---|---|---|---|
| **Faithfulness (Anti-Hallucination)** | &ge; 0.35 | **0.4000** | `PASS` :white_check_mark: |
| **Context Recall (Retrieval)** | &ge; 0.50 | **0.6000** | `PASS` :white_check_mark: |
| **Answer Relevance (Intent)** | &ge; 0.60 | **0.7949** | `PASS` :white_check_mark: |

- **Total Test Pairs Evaluated**: 5
- **Execution Latency**: 11.12s
- **Gate Verdict**: **ALL CHECKS PASSED - MERGE PERMITTED**
- **Diagnosis Breakdown**:
  - `SUCCESS`: 0
  - `RETRIEVAL_FAILURE`: 1
  - `GENERATION_HALLUCINATION`: 3
  - `COMPOUND_FAILURE`: 1
