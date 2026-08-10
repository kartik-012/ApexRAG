# RAG Evaluation Harness — Advanced Edition (₹0 / $0 Cost Blueprint)

Every tool below is free and runs locally. No paid API calls anywhere in this stack.

---

## 0. The ₹0 Cost Rule — How It's Enforced

| Component | Paid default | ₹0 replacement |
|---|---|---|
| LLM (generation + judge calls) | OpenAI API | **Ollama**, running `llama3.1:8b` or `phi3:mini` locally |
| Embeddings | OpenAI `text-embedding-3-small` | **`BAAI/bge-small-en-v1.5`** via `sentence-transformers`, runs on CPU |
| Vector DB | Pinecone/Weaviate (hosted, paid tiers) | **Chroma** (local file-based) or **PGVector** (local Docker Postgres) |
| Re-ranker | Cohere Rerank API | **`cross-encoder/ms-marco-MiniLM-L-6-v2`**, local via `sentence-transformers` |
| Keyword search | Elastic Cloud | **`rank_bm25`**, pure Python, in-memory |
| CI/CD | — | **GitHub Actions** — free unlimited minutes on public repos |
| Scheduled jobs (drift detection) | Cron server | **GitHub Actions `schedule` trigger** — free |
| Dynamic router model | — | **scikit-learn** `LogisticRegression`, trained on your own eval logs, free |

**Requirement**: Ollama must be installed and running (`ollama serve`) — free, one-time local install, no account/API key needed. Everything else is `pip install`.

---

## 1. Full Repository Structure

```
rag-eval-harness/
├── README.md
├── DECISIONS.md
├── requirements.txt
├── docker-compose.yml              # postgres+pgvector, local only
├── .env.example
├── .github/workflows/
│   ├── eval-gate.yml               # runs on every push
│   └── drift-check.yml             # scheduled, weekly
├── data/
│   ├── raw_docs/
│   └── ground_truth/
│       ├── qa_pairs.json           # your 50 hand-written pairs
│       ├── qa_pairs_adversarial.json   # auto-generated paraphrases (Feature 2)
│       └── qa_pairs_counterfactual.json # decoy-doc test set (Feature 7)
├── src/
│   ├── config.py
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── chunkers.py             # simple_chunk(), semantic_chunk()
│   │   └── embedder.py             # wraps bge-small
│   ├── retrieval/
│   │   ├── vector_store.py
│   │   ├── bm25_store.py
│   │   ├── hybrid.py               # reciprocal rank fusion
│   │   ├── reranker.py             # cross-encoder
│   │   └── router.py               # Feature 4: dynamic strategy router
│   ├── generation/
│   │   └── llm_client.py           # wraps local Ollama calls
│   ├── evaluation/
│   │   ├── metrics.py              # recall@k, MRR
│   │   ├── uncertainty.py          # Feature 1
│   │   ├── failure_attribution.py  # Feature 3
│   │   ├── cost_latency_tracker.py # Feature 6
│   │   ├── run_eval.py             # ties it all together
│   │   └── report.py
│   ├── adversarial/
│   │   └── generate_variants.py    # Feature 2: paraphrase/negation/multi-hop generator
│   ├── counterfactual/
│   │   └── generate_decoys.py      # Feature 7: near-duplicate distractor injector
│   ├── drift/
│   │   └── check_drift.py          # Feature 5: embedding-space drift detector
│   ├── active_learning/
│   │   └── flag_for_review.py      # Feature 8: self-improving ground truth
│   ├── debate/
│   │   └── multi_judge.py          # Feature 8 extension: two-judge faithfulness debate
│   └── api/
│       └── main.py                 # FastAPI, exposes /query + /confidence
├── results/
│   ├── eval_runs/                  # timestamped JSON per run
│   ├── drift_logs/
│   └── router_training_data.csv
├── models/
│   └── strategy_router.pkl         # trained sklearn classifier
├── tests/
└── scripts/
    ├── build_index.py
    ├── generate_qa_key.py
    └── train_router.py
```

---

## 2. requirements.txt (all free, no paid keys required)

```
fastapi
uvicorn
psycopg2-binary
pgvector
chromadb
sentence-transformers
rank_bm25
ollama
scikit-learn
pandas
numpy
python-dotenv
pytest
apscheduler
```

---

## 3. Base Pipeline (unchanged core — see prior blueprint for full code)

Ingestion → chunk (simple/semantic) → embed (bge-small) → store (Chroma/PGVector) → retrieve via 4 strategies (simple / semantic / hybrid / rerank) → generate answer via local Ollama model. This part stays as previously specified — the 8 features below sit on top of it.

---

## FEATURE 1 — Retrieval Uncertainty Quantification

**Goal**: know when the system doesn't know, instead of answering confidently on bad retrieval.

**`src/evaluation/uncertainty.py`**
```python
import numpy as np

def embedding_spread_score(top_k_embeddings: list[np.ndarray]) -> float:
    """
    Low spread (tight cluster) = confident retrieval.
    High spread = ambiguous / query poorly matched to corpus.
    Returns average pairwise cosine distance among top-k results.
    """
    n = len(top_k_embeddings)
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            cos_sim = np.dot(top_k_embeddings[i], top_k_embeddings[j]) / (
                np.linalg.norm(top_k_embeddings[i]) * np.linalg.norm(top_k_embeddings[j])
            )
            dists.append(1 - cos_sim)
    return float(np.mean(dists))


def strategy_agreement_score(results_by_strategy: dict[str, list[str]]) -> float:
    """
    Fraction of strategies whose top-1 result matches the majority top-1.
    1.0 = all 4 strategies agree. 0.25 = total disagreement.
    """
    top1s = [results[0] for results in results_by_strategy.values() if results]
    if not top1s:
        return 0.0
    most_common_count = max(top1s.count(x) for x in set(top1s))
    return most_common_count / len(top1s)


def confidence_label(spread: float, agreement: float,
                      spread_threshold: float = 0.35,
                      agreement_threshold: float = 0.5) -> str:
    if agreement >= agreement_threshold and spread <= spread_threshold:
        return "HIGH"
    if agreement >= 0.25:
        return "MEDIUM"
    return "LOW"
```

**Wired into `/query` endpoint**: when confidence is `LOW`, the API response includes `"warning": "Low retrieval confidence — answer may be unreliable, consider rephrasing"` instead of a plain answer. This is the single most demoable feature — show it live in your README as a GIF: a clean question gets a confident answer, a vague/out-of-scope question gets flagged.

---

## FEATURE 2 — Adversarial Self-Testing Loop

**Goal**: don't just test the 50 questions you wrote — test paraphrases, negations, and multi-hop rewrites your local LLM generates automatically.

**`src/adversarial/generate_variants.py`**
```python
import ollama, json

VARIANT_PROMPT = """Given this question and answer, generate 3 variants:
1. A paraphrase (same meaning, different wording)
2. A negated/trick version (changes meaning subtly, should NOT retrieve the same answer)
3. A multi-hop version (requires combining this fact with a related concept)

Question: {question}
Answer: {answer}

Return strict JSON: {{"paraphrase": "...", "negated": "...", "multi_hop": "..."}}"""

def generate_adversarial_set(qa_pairs: list[dict]) -> list[dict]:
    variants = []
    for pair in qa_pairs:
        prompt = VARIANT_PROMPT.format(question=pair["question"], answer=pair["answer"])
        response = ollama.chat(model="llama3.1:8b", messages=[{"role": "user", "content": prompt}])
        parsed = json.loads(response["message"]["content"])
        for variant_type, question_text in parsed.items():
            variants.append({
                "id": f"{pair['id']}_{variant_type}",
                "question": question_text,
                "gold_chunk_ids": pair["gold_chunk_ids"] if variant_type != "negated" else [],
                "variant_type": variant_type,
                "source_id": pair["id"],
            })
    return variants
```

**Why the negated variant matters**: if your system retrieves the *same* chunk for a negated question that should have a different (or no) answer, that's a real, measurable robustness failure — and almost no portfolio project tests for it. Run once, cache to `qa_pairs_adversarial.json`, re-run in CI like the base set.

---

## FEATURE 3 — Causal Failure Attribution

**Goal**: for every wrong answer, output *which stage* failed, not just a pass/fail score.

**`src/evaluation/failure_attribution.py`**
```python
from enum import Enum

class FailureType(Enum):
    RETRIEVAL_MISS = "gold chunk never retrieved at all"
    RETRIEVAL_RANK = "gold chunk retrieved but ranked below top-k"
    CHUNK_BOUNDARY = "answer split across two adjacent chunks, neither complete"
    GENERATION_HALLUCINATION = "correct context retrieved, LLM answer not grounded in it"
    GROUND_TRUTH_AMBIGUOUS = "question has multiple valid answers not captured in gold set"

def attribute_failure(question_id: str, retrieved_ids: list[str], gold_ids: set[str],
                       full_ranked_ids: list[str], faithfulness_score: float,
                       answer_correct: bool) -> FailureType | None:
    if answer_correct:
        return None
    if not (set(retrieved_ids) & gold_ids):
        if set(full_ranked_ids[:20]) & gold_ids:
            return FailureType.RETRIEVAL_RANK
        return FailureType.RETRIEVAL_MISS
    if faithfulness_score < 0.5:
        return FailureType.GENERATION_HALLUCINATION
    return FailureType.CHUNK_BOUNDARY  # heuristic fallback — flag for manual review
```

**Output**: your eval report doesn't just say "72% accuracy" — it says "72% accuracy: 14% retrieval-miss, 8% ranking issue, 4% hallucination, 2% ambiguous ground truth." That breakdown, plotted as a stacked bar per strategy, is the single most senior-looking chart you can put in a README.

---

## FEATURE 4 — Dynamic Strategy Router (learned, not static)

**Goal**: pick the best retrieval strategy per incoming query instead of shipping one fixed winner.

**`scripts/train_router.py`**
```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# router_training_data.csv built from your eval logs across all strategies:
# columns: question, query_length, has_named_entity, has_number, best_strategy
df = pd.read_csv("results/router_training_data.csv")

vectorizer = TfidfVectorizer(max_features=200)
X_text = vectorizer.fit_transform(df["question"])
X_meta = df[["query_length", "has_named_entity", "has_number"]].values

import scipy.sparse as sp
X = sp.hstack([X_text, X_meta])
y = df["best_strategy"]

clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

joblib.dump({"vectorizer": vectorizer, "classifier": clf}, "models/strategy_router.pkl")
```

**How training data is generated**: this is not guesswork — it comes directly from your Feature 3 eval logs. For every question in your ground truth + adversarial sets, you already know which of the 4 strategies got the highest recall/MRR for that specific question. That becomes your label. No extra annotation needed — the router trains on data your eval harness already produced.

**At inference**: `router.py` loads the pickle, featurizes the incoming query, and predicts which strategy to run — instead of always running all 4 or always using one static winner.

---

## FEATURE 5 — Continuous Production Drift Detection

**Goal**: catch the failure mode CI/CD can't see — the corpus and ground truth going stale over time, with no code change at all.

**`src/drift/check_drift.py`**
```python
import numpy as np, json
from src.ingestion.embedder import embed_texts

def compute_drift(current_chunks: list[str], baseline_embeddings: np.ndarray) -> dict:
    current_embeddings = embed_texts(current_chunks)
    baseline_centroid = baseline_embeddings.mean(axis=0)
    current_centroid = current_embeddings.mean(axis=0)
    centroid_shift = float(np.linalg.norm(baseline_centroid - current_centroid))

    # Per-ground-truth-question check: has the top-retrieved chunk for
    # any question changed since baseline, without a code change?
    return {
        "centroid_shift": centroid_shift,
        "flagged": centroid_shift > 0.15,  # threshold tuned from your own baseline run
    }
```

**`.github/workflows/drift-check.yml`**
```yaml
name: Weekly Drift Check
on:
  schedule:
    - cron: '0 3 * * 1'   # every Monday, free on GitHub Actions
  workflow_dispatch: {}

jobs:
  drift:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: python -m src.drift.check_drift
      - name: Open issue if drift detected
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner, repo: context.repo.repo,
              title: 'Corpus drift detected — ground truth may be stale',
              body: 'Weekly drift check flagged significant embedding-space shift. Review qa_pairs.json.'
            })
```
Free, needs no server — GitHub Actions' own scheduler runs it and auto-opens a GitHub issue when drift crosses your threshold.

---

## FEATURE 6 — Cost-Accuracy-Latency Pareto Optimization

**Goal**: report the full trade-off surface, and justify your production config with an explicit objective function — not "highest number wins."

**`src/evaluation/cost_latency_tracker.py`**
```python
import time

def measure_strategy(strategy_fn, question: str) -> dict:
    start = time.perf_counter()
    result = strategy_fn(question)
    latency_ms = (time.perf_counter() - start) * 1000
    # local models = $0 marginal cost; track "compute cost" as CPU-seconds instead
    # so the trade-off is still real and measurable even at ₹0 spend
    return {"result": result, "latency_ms": latency_ms}

def select_pareto_optimal(results: list[dict], max_latency_ms: float = 300) -> dict:
    """
    Objective: maximize recall@k subject to latency_ms <= max_latency_ms.
    This is the real decision-making pattern used in production infra —
    not 'pick the highest accuracy number', but 'pick the best accuracy
    within a hard operational constraint'.
    """
    feasible = [r for r in results if r["latency_ms"] <= max_latency_ms]
    if not feasible:
        return min(results, key=lambda r: r["latency_ms"])
    return max(feasible, key=lambda r: r["recall_at_k"])
```

Since every model here is local/free, your "cost" axis becomes **CPU time / compute cost** instead of $/query — same trade-off logic, same chart, zero real spend. Report it as: "at ₹0 infra cost, re-ranking adds 140ms latency for a 4pt recall gain — I chose to accept it because my target use case tolerates sub-second responses."

---

## FEATURE 7 — Counterfactual Retrieval Testing

**Goal**: test whether your system gets confused by near-duplicate/superseded documents — a real, common production failure (old policy vs. new policy both in the corpus).

**`src/counterfactual/generate_decoys.py`**
```python
import ollama, json

DECOY_PROMPT = """Take this factual chunk and create a near-duplicate version
with ONE key fact altered (a number, date, or name changed), keeping
everything else nearly identical in wording and structure.

Original: {chunk_text}

Return only the altered chunk text."""

def generate_decoy(chunk_text: str) -> str:
    response = ollama.chat(model="llama3.1:8b",
                            messages=[{"role": "user", "content": DECOY_PROMPT.format(chunk_text=chunk_text)}])
    return response["message"]["content"].strip()

def build_counterfactual_test_set(qa_pairs: list[dict], chunks_by_id: dict) -> list[dict]:
    test_set = []
    for pair in qa_pairs:
        gold_chunk = chunks_by_id[pair["gold_chunk_ids"][0]]
        decoy_text = generate_decoy(gold_chunk)
        test_set.append({
            "question": pair["question"],
            "real_gold_id": pair["gold_chunk_ids"][0],
            "decoy_chunk_text": decoy_text,
            "decoy_chunk_id": f"decoy_{pair['id']}",
        })
    return test_set
```

**Eval logic**: inject each decoy into the vector store alongside the real corpus, then re-run retrieval. Score = did the system retrieve/cite the *real* chunk, or get fooled by the near-duplicate? Report as a separate metric: "Real-vs-decoy discrimination accuracy: 78%." This is a distinct, rarely-tested failure mode — most people never even think about it.

---

## FEATURE 8 — Self-Improving Ground Truth + Multi-Judge Faithfulness Debate

**8a. Active learning on the ground truth set** (`src/active_learning/flag_for_review.py`)
```python
def flag_disagreements(eval_results: list[dict], confidence_threshold: float = 0.3) -> list[dict]:
    """
    If the system's own uncertainty score strongly disagrees with the
    human-labeled 'correct/incorrect' judgment (e.g., system is highly
    confident but marked wrong, or very uncertain but marked correct),
    flag that question for human re-review. This is how the ground truth
    set itself improves over time instead of being frozen at 50 static Qs.
    """
    flagged = []
    for r in eval_results:
        disagreement = abs(r["system_confidence"] - (1.0 if r["marked_correct"] else 0.0))
        if disagreement > (1 - confidence_threshold):
            flagged.append({**r, "reason": "high confidence/correctness mismatch — review needed"})
    return flagged
```

**8b. Multi-judge debate for faithfulness** (`src/debate/multi_judge.py`)
```python
import ollama

def debate_faithfulness(question: str, context: str, answer: str) -> dict:
    judge_prompt = """Context: {context}
Answer: {answer}
Question: {question}
Is every claim in the answer directly supported by the context? Answer YES or NO and explain in one sentence."""

    judge_a = ollama.chat(model="llama3.1:8b",
        messages=[{"role": "user", "content": judge_prompt.format(context=context, answer=answer, question=question)}])
    judge_b = ollama.chat(model="phi3:mini",   # DIFFERENT model = independent judge, reduces shared bias
        messages=[{"role": "user", "content": judge_prompt.format(context=context, answer=answer, question=question)}])

    verdict_a = "YES" in judge_a["message"]["content"].upper()
    verdict_b = "YES" in judge_b["message"]["content"].upper()

    if verdict_a == verdict_b:
        return {"faithful": verdict_a, "agreement": True}

    # disagreement -> arbitrate with a third pass, both opinions shown
    arbitration_prompt = f"""Two reviewers disagree on whether this answer is grounded.
Reviewer A: {judge_a['message']['content']}
Reviewer B: {judge_b['message']['content']}
Give the final YES/NO verdict."""
    final = ollama.chat(model="llama3.1:8b", messages=[{"role": "user", "content": arbitration_prompt}])
    return {"faithful": "YES" in final["message"]["content"].upper(), "agreement": False}
```

Using two *different* local models (`llama3.1:8b` and `phi3:mini`, both free via Ollama) as independent judges is what makes this a real debate rather than one model checking itself — shared-model self-evaluation has known blind spots, using a different architecture as the second judge measurably reduces that.

---

## 4. Updated README Structure (add these on top of the 4 base sections)

5. **Confidence & Uncertainty** — screenshot of a low-confidence flagged response
6. **Failure Attribution Breakdown** — stacked bar chart of failure types per strategy
7. **Robustness Report** — accuracy on base questions vs. adversarial/paraphrased vs. counterfactual/decoy questions (three separate numbers, not one)
8. **Router Performance** — accuracy of dynamic routing vs. best static strategy
9. **Drift Log** — link to `results/drift_logs/`, show it caught something real if it did

---

## 5. Honest Scope Note

Not every feature needs to ship simultaneously. Realistic build order for maximum interview leverage per hour invested:

1. Base pipeline (must work first)
2. Feature 1 (Uncertainty) — highest visible impact, ~1 day
3. Feature 3 (Failure Attribution) — pairs naturally with #1, ~1 day
4. Feature 6 (Cost-Latency Pareto) — cheap to add once metrics exist, ~half day
5. Feature 2 (Adversarial) — ~1 day
6. Feature 7 (Counterfactual) — ~1 day
7. Feature 4 (Router) — needs data from 2/3/7 first, ~1-2 days
8. Feature 5 (Drift) — set-and-forget once written, ~half day
9. Feature 8 (Active learning + multi-judge) — most complex, do last, ~1-2 days

All 8 are genuinely rare in candidate portfolios. Even shipping 4-5 of them well, with honest numbers and a clear DECISIONS.md, puts you ahead of the vast majority of "I built a RAG chatbot" submissions — and every tool used above costs ₹0.
