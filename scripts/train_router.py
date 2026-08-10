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

from src.config import ROUTER_MODEL_PATH, RESULTS_DIR


def main():
    print("Training Strategy Router ML Model...")

    training_csv = RESULTS_DIR / "router_training_data.csv"
    if not training_csv.exists():
        # Create synthetic training set if evaluation log CSV isn't created yet
        sample_data = [
            {"question": "What is componentDidMount lifecycle?", "query_length": 4, "has_named_entity": 1, "has_number": 0, "best_strategy": "hybrid"},
            {"question": "How to manage state in React function component using hooks?", "query_length": 10, "has_named_entity": 1, "has_number": 0, "best_strategy": "semantic"},
            {"question": "Detailed explanation of passing data deeply across multiple component trees using Context API provider", "query_length": 15, "has_named_entity": 1, "has_number": 0, "best_strategy": "rerank"},
            {"question": "useRef vs useState", "query_length": 3, "has_named_entity": 1, "has_number": 0, "best_strategy": "simple"},
        ]
        df = pd.DataFrame(sample_data)
        df.to_csv(training_csv, index=False)
    else:
        df = pd.read_csv(training_csv)

    vectorizer = TfidfVectorizer(max_features=200)
    X_text = vectorizer.fit_transform(df["question"])
    X_meta = df[["query_length", "has_named_entity", "has_number"]].values
    X = sp.hstack([X_text, X_meta])
    y = df["best_strategy"]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    ROUTER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "classifier": clf}, ROUTER_MODEL_PATH)
    print(f"Strategy Router trained successfully and saved to {ROUTER_MODEL_PATH}!")


if __name__ == "__main__":
    main()
