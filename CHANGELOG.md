# Changelog

All notable changes to ApexRAG are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-08-10

### Added

#### Core Pipeline
- `src/ingestion/loader.py` — MDX parser, YAML frontmatter extraction, synthetic React corpus (8 verified docs)
- `src/ingestion/chunkers.py` — `simple_chunk()` (sliding word window) + `semantic_chunk()` (embedding similarity split)
- `src/ingestion/embedder.py` — `BAAI/bge-small-en-v1.5` via sentence-transformers; TF-IDF fallback for offline CI
- `src/retrieval/vector_store.py` — ChromaDB persistent client; in-memory fallback when ChromaDB unavailable
- `src/retrieval/bm25_store.py` — Native BM25Okapi implementation + `rank_bm25` library binding
- `src/retrieval/hybrid.py` — Reciprocal Rank Fusion (RRF, k=60)
- `src/retrieval/reranker.py` — `cross-encoder/ms-marco-MiniLM-L-6-v2`; Jaccard word-overlap fallback
- `src/generation/llm_client.py` — Ollama wrapper with deterministic mock for offline/CI execution

#### 8 Advanced Features
- **Feature 1**: `src/evaluation/uncertainty.py` — Embedding Spread Score + Strategy Agreement Score + confidence labelling
- **Feature 2**: `src/adversarial/generate_variants.py` — Paraphrase / negated / multi-hop variant generator
- **Feature 3**: `src/evaluation/failure_attribution.py` — Causal root-cause diagnosis per failing question
- **Feature 4**: `src/retrieval/router.py` — Learned LogisticRegression strategy router (TF-IDF + metadata)
- **Feature 5**: `src/drift/check_drift.py` — Embedding centroid shift drift detector
- **Feature 6**: `src/evaluation/cost_latency_tracker.py` — Pareto-optimal config selector under latency SLA
- **Feature 7**: `src/counterfactual/generate_decoys.py` — Near-duplicate decoy distractor injector
- **Feature 8a**: `src/active_learning/flag_for_review.py` — Ground truth quality disagreement flagging
- **Feature 8b**: `src/debate/multi_judge.py` — Dual-LLM faithfulness debate with arbitration

#### Evaluation & Reporting
- `src/evaluation/metrics.py` — Recall@K, MRR, nDCG@K
- `src/evaluation/run_eval.py` — Full 5-strategy harness orchestrator, saves timestamped JSON
- `src/evaluation/report.py` — GitHub Flavored Markdown benchmark report generator

#### API
- `src/api/main.py` — FastAPI production server: `/query`, `/confidence`, `/eval`, `/health`
- Uncertainty confidence flag on `/query` with `LOW` warning message

#### Scripts
- `scripts/build_index.py` — CLI index builder with `--legacy-dir`, `--current-dir`, `--chunk-size`, `--overlap`
- `scripts/run_full_pipeline.py` — 4-phase pipeline orchestrator with `--save-report`, `--adversarial` flags
- `scripts/generate_qa_key.py` — Router training CSV generator from eval logs or synthetic bootstrap
- `scripts/train_router.py` — Router LogisticRegression trainer with synthetic seed data fallback

#### Ground Truth Data
- `data/ground_truth/qa_pairs.json` — 50 hand-curated React documentation QA benchmark pairs
- `data/ground_truth/qa_pairs_adversarial.json` — Pre-generated adversarial variants
- `data/ground_truth/qa_pairs_counterfactual.json` — Pre-generated decoy test set

#### CI/CD
- `.github/workflows/eval-gate.yml` — Runs pytest + full pipeline on every push/PR
- `.github/workflows/drift-check.yml` — Weekly corpus drift detection with auto-GitHub-issue on failure

#### Tests (8 modules, full coverage)
- `tests/conftest.py` — Shared fixtures: `sample_qa_pairs`, `synthetic_docs`
- `tests/test_ingestion.py` — Loader, chunker, embedder unit tests
- `tests/test_retrieval.py` — BM25, RRF, re-ranker, router unit tests
- `tests/test_evaluation.py` — Metrics, uncertainty, failure attribution unit tests
- `tests/test_api.py` — FastAPI integration tests via TestClient
- `tests/test_llm_generation.py` — LLM client + mock fallback unit tests
- `tests/test_advanced_features.py` — Adversarial, counterfactual, debate, active learning tests
- `tests/test_pipeline_integration.py` — Full harness end-to-end integration test

#### Documentation
- `README.md` — Full architecture, benchmarks, quickstart, repo tree, config, resume bullets
- `DECISIONS.md` — ADR 001–004 architectural trade-off rationale
- `CONTRIBUTING.md` — Setup guide, code standards, testing requirements, module map
- `SOURCES.md` — Verified corpus source repositories and verified counterfactual pairs
- `CHANGELOG.md` — This file

---

## Roadmap

### [1.1.0] — Planned
- [ ] RAGAS integration for faithfulness metric comparison
- [ ] Per-question detailed result logging for router training data export
- [ ] Streamlit dashboard for live benchmark visualization
- [ ] Support for custom corpora (PDF, HTML ingestion)
