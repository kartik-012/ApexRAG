"""
Dynamic Strategy Router — selects optimal retrieval strategy per query.
Feature 4 of ApexRAG Evaluation Harness.
"""

import re
import joblib
import scipy.sparse as sp
from pathlib import Path
from src.config import ROUTER_MODEL_PATH


class StrategyRouter:
    def __init__(self, model_path: str | Path = ROUTER_MODEL_PATH):
        self.model_path = Path(model_path)
        self.vectorizer = None
        self.classifier = None
        self.is_loaded = False
        self.load_model()

    def load_model(self) -> bool:
        if self.model_path.exists():
            try:
                data = joblib.load(self.model_path)
                self.vectorizer = data.get("vectorizer")
                self.classifier = data.get("classifier")
                self.is_loaded = (self.vectorizer is not None and self.classifier is not None)
                return self.is_loaded
            except Exception as e:
                print(f"[Warning] Failed to load strategy router model: {e}")
        self.is_loaded = False
        return False

    def featurize(self, question: str):
        query_length = len(question.split())
        has_named_entity = int(bool(re.search(r'\b[A-Z][a-zA-Z0-9_]*\b', question)))
        has_number = int(bool(re.search(r'\d+', question)))
        meta_features = [[query_length, has_named_entity, has_number]]

        if self.vectorizer is not None:
            text_vec = self.vectorizer.transform([question])
            return sp.hstack([text_vec, meta_features])
        return meta_features

    def predict_strategy(self, question: str) -> str:
        """
        Predicts best retrieval strategy: 'simple', 'semantic', 'hybrid', or 'rerank'.
        Uses trained model if available; falls back to heuristic rules otherwise.
        """
        if self.is_loaded:
            try:
                X = self.featurize(question)
                return str(self.classifier.predict(X)[0])
            except Exception as e:
                print(f"[Warning] Router prediction failed ({e}), using heuristic.")

        # Heuristic fallback
        words = question.split()
        if len(words) < 5 or re.search(r'\b(use[A-Z]\w+|componentDid\w+|React\.\w+)\b', question):
            return "hybrid"  # exact keyword + semantic fusion works best for code identifiers
        elif len(words) > 12:
            return "rerank"  # long complex queries benefit from cross-encoder re-ranking
        else:
            return "semantic"
