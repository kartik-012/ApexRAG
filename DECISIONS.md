# Architecture Decision Records (ADR) & System Rationale

## ADR 001: Enforcing strict $0 / ₹0 Marginal Infrastructure Spend

- **Context**: Standard enterprise RAG evaluation systems rely on OpenAI (`text-embedding-3-small`, `gpt-4o`) and hosted vector stores (Pinecone, Weaviate), costing hundreds of dollars per benchmark run.
- **Decision**: Replace all paid cloud dependencies with CPU-friendly local open-source models:
  - Embeddings: `BAAI/bge-small-en-v1.5` via `sentence-transformers` (runs on CPU, 384 dim).
  - Vector DB: `ChromaDB` (local embedded file-based sqlite vector store with cosine metric).
  - Keyword Search: `rank_bm25` (in-memory BM25Okapi).
  - Re-ranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`.
  - LLM & Judges: Local `Ollama` running `llama3.1:8b` and `phi3:mini`.
- **Consequences**: Zero recurring API bills, 100% data privacy/sovereignty, fully reproducible in isolated CI environments.

---

## ADR 002: Reciprocal Rank Fusion (RRF) over Weighted Score Averaging

- **Context**: Combining keyword search (BM25) and dense vector search (embeddings) requires merging two distinct result rankings.
- **Decision**: Use Reciprocal Rank Fusion (RRF) with constant $k=60$ ($RRF\_score = \sum \frac{1}{k + rank}$).
- **Rationale**: BM25 scores are unbounded $[0, \infty)$, whereas cosine similarities are bounded $[-1, 1]$. Score averaging requires fragile hyperparameter tuning per corpus update. RRF operates purely on ordinal rank positions, making it immune to score scaling mismatch.

---

## ADR 003: Multi-Judge Dual-Architecture Faithfulness Debate

- **Context**: Using a single LLM to evaluate its own faithfulness introduces self-preference bias.
- **Decision**: Pair two architecturally distinct local models (`llama3.1:8b` vs `phi3:mini`). When judge verdicts disagree, a third arbitration pass is triggered with full transcripts.
- **Rationale**: Cross-model verification significantly reduces false positive faithfulness approvals compared to single-model checking.

---

## ADR 004: Learned Dynamic Strategy Router vs Static Retrieval Strategy

- **Context**: No single retrieval strategy (BM25, vector, hybrid, or re-rank) dominates across all query types.
- **Decision**: Train a scikit-learn `LogisticRegression` classifier on past evaluation failure attribution logs to dynamically route queries at runtime based on query length, named entity presence, and TF-IDF features.
- **Trade-off**: Adds a lightweight (~2ms) inference overhead before retrieval, but optimizes Recall@K across heterogeneous technical questions.
