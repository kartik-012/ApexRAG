# Contributing to ApexRAG

Thank you for your interest in contributing! Please follow this guide to maintain code quality and project standards.

---

## 🚀 Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/apexrag.git
cd apexrag

# 2. Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Run tests to verify your setup
python -m pytest tests/ -v
```

---

## 📐 Code Standards

- **Python 3.11+** — use union types (`X | Y`), `list[str]`, `dict[str, Any]` (not `List`, `Dict`).
- **Docstrings** — every public function must have a docstring explaining purpose, params, and return value.
- **Type Hints** — all function signatures must be fully type-annotated.
- **No paid APIs** — the $0 constraint is a hard requirement. No OpenAI/Cohere/Pinecone API calls anywhere.
- **Fallback-first** — every external dependency (Ollama, ChromaDB, sentence-transformers) must have a graceful local fallback so CI runs without a GPU or running services.

---

## 🧪 Testing Requirements

- All new features must include at least one unit test in `tests/`.
- Tests must pass without Ollama running — use the mock LLM fallback.
- Run the full suite before submitting a PR:

```bash
python -m pytest tests/ -v
```

---

## 🌿 Branching & PR Conventions

- Branch naming: `feature/short-description`, `fix/bug-description`, `refactor/scope`
- Commit messages: imperative mood — `Add uncertainty quantification to /query endpoint`
- All PRs require the CI eval-gate to pass before merging.

---

## 📝 Architecture Decision Records

For any significant design trade-off, add an entry to [`DECISIONS.md`](DECISIONS.md) following the existing ADR format:

```markdown
## ADR XXX: Title

- **Context**: Why was this decision needed?
- **Decision**: What was chosen?
- **Rationale**: Why this over alternatives?
- **Consequences**: What are the trade-offs?
```

---

## 🗂️ Module Map

| Module | Purpose |
|---|---|
| `src/config.py` | All environment config, path constants |
| `src/ingestion/` | Doc loading, chunking, embedding |
| `src/retrieval/` | BM25, vector, hybrid, re-rank, router |
| `src/generation/` | Ollama LLM wrapper + mock fallback |
| `src/evaluation/` | Metrics, uncertainty, failure attribution, report |
| `src/adversarial/` | Auto-generates adversarial test question variants |
| `src/counterfactual/` | Decoy distractor generation and evaluation |
| `src/drift/` | Embedding-space corpus drift detection |
| `src/active_learning/` | Ground truth quality flagging |
| `src/debate/` | Multi-judge faithfulness arbitration |
| `src/api/` | FastAPI production server |
| `scripts/` | Build, train, and pipeline runner scripts |
| `tests/` | Full pytest unit + integration test suite |

---

Thank you for helping make ApexRAG better!
