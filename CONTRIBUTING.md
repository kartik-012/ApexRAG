<div align="center">

# 🤝 Contributing to ApexRAG

**We welcome contributions that maintain the research rigor and $0 infrastructure constraint of this project.**

</div>

---

## 📋 Table of Contents

1. [Before You Start](#before-you-start)
2. [Development Setup](#development-setup)
3. [Project Philosophy](#project-philosophy)
4. [Code Standards](#code-standards)
5. [Testing Requirements](#testing-requirements)
6. [Branching & PR Conventions](#branching--pr-conventions)
7. [Architecture Decision Records](#architecture-decision-records)
8. [Module Map](#module-map)
9. [CI/CD Gates](#cicd-gates)
10. [Good First Issues](#good-first-issues)

---

## Before You Start

Please read these documents before opening a PR:

| Document | Purpose |
|---|---|
| [`DECISIONS.md`](DECISIONS.md) | Why key architectural choices were made |
| [`SOURCES.md`](SOURCES.md) | How the corpus was constructed and verified |
| [`README.md`](README.md) | Full system architecture, benchmarks, and design rationale |

> **Hard constraint:** The `$0 / ₹0` infrastructure cost is a non-negotiable project invariant. Any contribution that introduces paid API calls (OpenAI, Cohere, Pinecone, etc.) will not be merged.

---

## Development Setup

```bash
# 1. Fork & clone
git clone https://github.com/kartik-012/ApexRAG.git
cd ApexRAG

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Clone the React documentation corpora
git clone --depth 1 --branch v17.0.2 \
  https://github.com/reactjs/reactjs.org.git \
  data/raw_docs/react-legacy

git clone --depth 1 \
  https://github.com/reactjs/react.dev.git \
  data/raw_docs/react-dev-current

# 5. Build the BM25 + ChromaDB indexes
python scripts/build_index.py \
  --legacy-dir  data/raw_docs/react-legacy \
  --current-dir data/raw_docs/react-dev-current \
  --chunk-size  200 \
  --overlap     30

# 6. Run the full test suite to verify setup
python -m pytest tests/ -v
```

> 💡 **Tip:** You do **not** need Ollama running for tests. The LLM layer has a deterministic mock fallback that activates automatically in CI mode.

---

## Project Philosophy

ApexRAG was built around three engineering principles. All contributions must respect them:

### 1. 🔬 Research-Grade Observability
Every retrieval decision must be fully attributable. Avoid abstractions that hide what's happening inside the pipeline. If Recall@4 drops, a contributor must be able to identify the *exact* line of code responsible.

### 2. 💸 $0 Infrastructure Cost
All models, databases, and inference engines run locally on CPU. This enables:
- Full reproducibility in isolated CI (no API keys needed)
- Zero recurring costs for iterative development
- Complete data sovereignty

### 3. ✅ Deterministic Evaluation
The ground truth (`data/ground_truth/qa_pairs.json`) is **immutable** — never generated or modified by an LLM. Recall@4 and MRR are computed from exact chunk ID matching. No probabilistic LLM judge is in the retrieval evaluation loop.

---

## Code Standards

### Language & Types

- **Python 3.10+** required — use modern union types (`X | Y`), built-in generics (`list[str]`, `dict[str, Any]`)
- All public function signatures must be **fully type-annotated**
- Every public function must have a **docstring** (purpose, params, return value, raises)

### Style

- Formatter: `black` (line length 100)
- Linter: `ruff`
- Run before committing:

```bash
black src/ tests/ scripts/
ruff check src/ tests/ scripts/
```

### Constraints

| Rule | Reason |
|---|---|
| No paid API calls | $0 infrastructure invariant |
| No `load_synthetic_react_docs()` or equivalent | Evaluation validity — only real corpus |
| Every external dependency must have a local fallback | CI must run without GPU or network |
| `assert len(ids) == len(set(ids))` must stay in `loader.py` | Prevents silent `doc_id` collisions |
| `data/ground_truth/qa_pairs.json` is read-only | Immutable ground truth |

### Commit Messages

Use the imperative mood, describing *what the commit does*, not what you did:

```
✅  Add uncertainty quantification endpoint to FastAPI
✅  Fix ANSI padding misalignment in generate_qa_key.py
✅  Refactor BM25 index to support incremental updates
❌  Fixed a bug
❌  Changes
```

---

## Testing Requirements

### Rules

- Every new feature or bug fix **must** include at least one test in `tests/`
- Tests must pass with **Ollama offline** — use the mock LLM (`LLM_MOCK=true` is auto-set in pytest via `conftest.py`)
- No test may modify `data/ground_truth/qa_pairs.json`
- ChromaDB tests must use a **temporary isolated path** — never the production persistent store

### Running Tests

```bash
# Full suite (25 tests, ~87% coverage)
python -m pytest tests/ -v

# Specific module
python -m pytest tests/test_retrieval.py -v

# With coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing

# Fast mode (skip slow integration tests)
python -m pytest tests/ -v -m "not slow"
```

### Test Categories

| File | Tests |
|---|---|
| `test_ingestion.py` | MDX parsing, chunking, embedding, doc_id uniqueness |
| `test_retrieval.py` | BM25, vector, hybrid RRF, cross-encoder, router |
| `test_evaluation.py` | Recall@K, MRR, failure attribution, uncertainty |
| `test_api.py` | FastAPI endpoints, request/response schema |
| `test_llm_generation.py` | Ollama mock, prompt grounding |
| `test_advanced_features.py` | Adversarial, counterfactual, drift, debate |
| `test_pipeline_integration.py` | End-to-end pipeline smoke test |

---

## Branching & PR Conventions

### Branch Naming

```
feature/add-nDCG-metric
fix/bm25-tokenization-unicode
refactor/decouple-router-from-eval
docs/update-architecture-diagram
```

### PR Checklist

Before opening a PR, confirm:

- [ ] `python -m pytest tests/ -v` passes with zero failures
- [ ] `black` and `ruff` produce no output
- [ ] New code is fully type-annotated and docstringed
- [ ] No paid API calls introduced
- [ ] `data/ground_truth/qa_pairs.json` unchanged
- [ ] CI eval-gate (Recall@4 ≥ 0.75 on HYBRID) passes
- [ ] PR description explains *why*, not just *what*

### Review Process

1. Open a PR against `main`
2. CI runs: `pytest` + eval gate (HYBRID Recall@4 ≥ 0.75)
3. One reviewer approval required
4. Squash-merge preferred for clean history

---

## Architecture Decision Records

Any significant design trade-off **must** be documented in [`DECISIONS.md`](DECISIONS.md) before merging. Use this format:

```markdown
## ADR 00X: Short Descriptive Title

- **Context**: What problem or situation prompted this decision?
- **Decision**: What was chosen?
- **Rationale**: Why this over the rejected alternatives?
- **Consequences**: What are the trade-offs, costs, or limitations?
```

**Examples of changes that require an ADR:**
- Switching embedding models
- Changing chunking strategy or parameters
- Adding a new retrieval strategy
- Changing the evaluation metric definition
- Replacing ChromaDB with another vector store

---

## Module Map

| Module | Purpose | Key Entry Points |
|---|---|---|
| `src/config.py` | All path constants and environment config | `CHROMA_PATH`, `BM25_PATH`, `LLM_MODEL` |
| `src/ingestion/loader.py` | MDX parser → `DocFile` objects | `load_legacy_docs()`, `load_current_docs()` |
| `src/ingestion/chunkers.py` | Word-window and semantic chunking | `simple_chunk()`, `semantic_chunk()` |
| `src/ingestion/embedder.py` | sentence-transformers + TF-IDF fallback | `embed()` |
| `src/retrieval/bm25_store.py` | BM25Okapi keyword index | `BM25Store.retrieve()` |
| `src/retrieval/vector_store.py` | ChromaDB persistent client | `VectorStore.retrieve()` |
| `src/retrieval/hybrid.py` | Reciprocal Rank Fusion (k=60) | `hybrid_retrieve()` |
| `src/retrieval/reranker.py` | Cross-encoder + Jaccard CI fallback | `rerank()` |
| `src/retrieval/router.py` | LogisticRegression strategy dispatcher | `Router.predict()`, `Router.train()` |
| `src/generation/llm_client.py` | Ollama wrapper + deterministic mock | `generate()` |
| `src/evaluation/run_eval.py` | 5-strategy evaluation harness | `run_evaluation()` |
| `src/evaluation/metrics.py` | Recall@K, MRR, nDCG@K | `recall_at_k()`, `mrr()` |
| `src/evaluation/uncertainty.py` | Embedding spread + strategy agreement | `compute_uncertainty()` |
| `src/evaluation/failure_attribution.py` | Causal failure root-cause classifier | `classify_failure()` |
| `src/evaluation/cost_latency_tracker.py` | Pareto-optimal config selector | `track()`, `pareto_select()` |
| `src/evaluation/report.py` | Markdown benchmark report generator | `generate_report()` |
| `src/adversarial/generate_variants.py` | Paraphrase / negation / multi-hop variants | `generate_variants()` |
| `src/counterfactual/generate_decoys.py` | Near-duplicate distractor injector | `inject_decoys()` |
| `src/drift/check_drift.py` | Embedding centroid drift detector | `check_drift()` |
| `src/debate/multi_judge.py` | Dual-LLM faithfulness debate | `judge()` |
| `src/active_learning/flag_for_review.py` | Ground truth quality flagging | `flag_for_review()` |
| `src/api/main.py` | FastAPI: `/query` `/confidence` `/eval` `/health` | `app` |
| `scripts/build_index.py` | CLI: build BM25 + ChromaDB indexes | `main()` |
| `scripts/run_full_pipeline.py` | 4-phase pipeline orchestrator | `main()` |
| `scripts/train_router.py` | Train LogisticRegression strategy router | `main()` |
| `scripts/generate_qa_key.py` | Interactive ground truth authoring tool | `main()` |

---

## CI/CD Gates

### `eval-gate.yml` — Runs on every push and PR

```
pytest --tb=short   →  All 25 tests must pass
run_full_pipeline   →  HYBRID Recall@4 ≥ 0.75  (merge gate)
```

### `drift-check.yml` — Runs weekly (Sunday midnight UTC)

```
check_drift.py  →  If centroid drift > 0.15, auto-opens a GitHub Issue
```

Any PR that causes the eval gate to fail will be blocked from merging regardless of review status.

---

## Good First Issues

Looking for a way to contribute? These are well-scoped and don't require deep familiarity with the whole codebase:

| Issue | Module | Difficulty |
|---|---|---|
| Add `nDCG@K` metric alongside Recall@K and MRR | `src/evaluation/metrics.py` | 🟢 Easy |
| Add `--strategy` flag to `run_full_pipeline.py` CLI | `scripts/run_full_pipeline.py` | 🟢 Easy |
| Write docstrings for all functions in `src/drift/` | `src/drift/` | 🟢 Easy |
| Add semantic chunking ablation to benchmark report | `src/ingestion/chunkers.py` | 🟡 Medium |
| Improve router feature set with bigram TF-IDF | `src/retrieval/router.py` | 🟡 Medium |
| Add RAGAS integration as a comparison module | `src/evaluation/` | 🔴 Advanced |
| Support incremental index updates without full rebuild | `src/retrieval/vector_store.py` | 🔴 Advanced |

---

<div align="center">

**Thank you for helping make ApexRAG better.**

*Every contribution that improves retrieval accuracy — even by 1% — represents real engineering progress.*

</div>
