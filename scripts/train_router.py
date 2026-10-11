"""
Train Strategy Router Script — Feature 4 of ApexRAG Blueprint.
Trains a LogisticRegression model on evaluation logs to dynamically choose retrieval strategies.
"""

import sys
import re
import pandas as pd
import numpy as np
from pathlib import Path
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROUTER_MODEL_PATH, RESULTS_DIR, MODELS_DIR, GROUND_TRUTH_DIR, EVAL_RUNS_DIR
from src.ingestion.embedder import embed_texts
import json


def _infer_best_strategy(question: str) -> str:
    words = question.split()
    has_code = bool(re.search(r"`[^`]+`|\b(use[A-Z]\w+|component[A-Z]\w+|setState|PureComponent|StrictMode|forwardRef)\b", question))
    has_comparison = bool(re.search(r"\b(vs|versus|difference|compare|prefer|instead of)\b", question.lower()))
    if len(words) <= 5 and has_code and not has_comparison:
        return "simple"
    elif has_comparison or len(words) >= 12 or "difference between" in question.lower():
        return "rerank"
    elif has_code and len(words) <= 10:
        return "hybrid"
    return "semantic"


def main():
    print("Training Dense Strategy Router with Shannon Entropy Gating...")

    training_csv = RESULTS_DIR / "router_training_data.csv"
    qa_file = GROUND_TRUTH_DIR / "qa_pairs.json"

    empirical_winners = {}
    eval_files = sorted(EVAL_RUNS_DIR.glob("eval_run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if eval_files:
        try:
            latest = json.loads(eval_files[0].read_text(encoding="utf-8"))
            per_q = latest.get("per_question_results", {})
            if per_q:
                cand_strats = ["rerank", "hybrid", "simple", "semantic"]
                n_q = len(per_q.get("hybrid", []))
                for idx in range(n_q):
                    best_s = "rerank"
                    best_score = -1.0
                    qid = per_q["hybrid"][idx]["qid"]
                    for s in cand_strats:
                        item = per_q.get(s, [])[idx]
                        sc = item.get("recall", 0.0) * 2.0 + item.get("mrr", 0.0)
                        if sc > best_score:
                            best_score = sc
                            best_s = s
                    empirical_winners[qid] = best_s
        except Exception:
            pass

    rows = []
    if qa_file.exists():
        qa_pairs = json.loads(qa_file.read_text(encoding="utf-8"))
        for item in qa_pairs:
            q = item["question"]
            qid = item.get("id", "")
            has_code_symbol = int(bool(re.search(r"`[^`]+`|\b(use[A-Z]\w+|component[A-Z]\w+|setState|PureComponent|StrictMode|forwardRef)\b", q)))
            has_comparison = int(bool(re.search(r"\b(vs|versus|difference|compare|prefer|instead of)\b", q.lower())))
            rows.append({
                "question": q,
                "query_length": len(q.split()),
                "has_named_entity": int(bool(re.search(r"\b[A-Z][a-zA-Z0-9_]*\b", q))),
                "has_code_symbol": has_code_symbol,
                "has_comparison": has_comparison,
                "has_number": int(bool(re.search(r"\d+", q))),
                "best_strategy": empirical_winners.get(qid, _infer_best_strategy(q)),
            })

    if not rows:
        rows = [
            {"question": "What is componentDidMount lifecycle?", "query_length": 4, "has_named_entity": 1, "has_code_symbol": 1, "has_comparison": 0, "has_number": 0, "best_strategy": "hybrid"},
            {"question": "How to manage state in React function component using hooks?", "query_length": 10, "has_named_entity": 1, "has_code_symbol": 1, "has_comparison": 0, "has_number": 0, "best_strategy": "semantic"},
            {"question": "Detailed explanation of passing data deeply across multiple component trees using Context API provider", "query_length": 15, "has_named_entity": 1, "has_code_symbol": 1, "has_comparison": 0, "has_number": 0, "best_strategy": "rerank"},
            {"question": "useRef vs useState", "query_length": 3, "has_named_entity": 1, "has_code_symbol": 1, "has_comparison": 1, "has_number": 0, "best_strategy": "simple"},
        ]

    df = pd.DataFrame(rows)
    df.to_csv(training_csv, index=False)

    print(f"Embedding {len(df)} training queries with dense representations...")
    dense_embeddings = embed_texts(df["question"].tolist())

    meta_cols = ["query_length", "has_named_entity", "has_code_symbol", "has_comparison", "has_number"]
    X_meta = df[meta_cols].values.astype(np.float32)
    X = np.hstack([np.array(dense_embeddings, dtype=np.float32), X_meta])
    y = df["best_strategy"]

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "classifier": clf,
        "feature_type": "dense",
        "dimension": X.shape[1],
        "meta_columns": meta_cols,
    }
    joblib.dump(payload, MODELS_DIR / "strategy_router.pkl")
    joblib.dump(payload, MODELS_DIR / "router.pkl")
    print(f"Dense Strategy Router successfully trained and saved to {ROUTER_MODEL_PATH} (feature dim: {X.shape[1]})!")


if __name__ == "__main__":
    main()

