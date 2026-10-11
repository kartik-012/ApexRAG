"""
BM25 keyword search — exact term match retrieval.
Pure-Python BM25Okapi implementation with rank_bm25 fallback.
"""

import math
import re
from typing import List, Tuple

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


_CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str) -> List[str]:
    """Lowercase word/identifier tokenizer preserving underscores and expanding camelCase symbols."""
    raw_tokens = _TOKEN_PATTERN.findall(text)
    tokens: List[str] = []
    for tok in raw_tokens:
        low = tok.lower()
        tokens.append(low)
        # Expand camelCase (e.g. useEffect -> use, effect; componentDidMount -> component, did, mount)
        parts = _CAMEL_SPLIT.split(tok)
        if len(parts) > 1:
            for p in parts:
                pl = p.lower()
                if len(pl) > 2 and pl != "use":
                    tokens.append(pl)
    return tokens


class NativeBM25Okapi:
    """Pure Python BM25Okapi implementation (zero external package dependencies)."""
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / float(self.corpus_size) if self.corpus_size > 0 else 1.0

        self.doc_freqs: List[dict] = []
        self.nd: dict = {}

        for doc in corpus:
            freq = {}
            for term in doc:
                freq[term] = freq.get(term, 0) + 1
            self.doc_freqs.append(freq)
            for term in freq:
                self.nd[term] = self.nd.get(term, 0) + 1

        self.idf: dict = {}
        for term, n in self.nd.items():
            # Standard BM25 IDF formula
            self.idf[term] = math.log((self.corpus_size - n + 0.5) / (n + 0.5) + 1.0)

    def get_scores(self, query: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for term in query:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, freq_dict in enumerate(self.doc_freqs):
                freq = freq_dict.get(term, 0)
                if freq == 0:
                    continue
                doc_len = self.doc_lengths[i]
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                scores[i] += idf * (numerator / denominator)
        return scores


class BM25Store:
    def __init__(self):
        self.bm25 = None
        self.chunk_ids: List[str] = []
        self.metadatas: List[dict] = []
        self.tokenized_corpus: List[List[str]] = []

    def build(self, chunk_ids: List[str], texts: List[str], metadatas: List[dict] | None = None) -> None:
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must be the same length")
        self.chunk_ids = chunk_ids
        self.metadatas = metadatas if metadatas is not None else [{} for _ in chunk_ids]
        self.tokenized_corpus = [tokenize(t) for t in texts]
        if self.tokenized_corpus:
            try:
                from rank_bm25 import BM25Okapi
                self.bm25 = BM25Okapi(self.tokenized_corpus)
            except Exception:
                self.bm25 = NativeBM25Okapi(self.tokenized_corpus)

    def query(self, query_text: str, top_k: int = 4, where: dict | None = None) -> List[Tuple[str, float]]:
        """Returns [(chunk_id, bm25_score), ...] sorted descending by score."""
        if self.bm25 is None or not self.chunk_ids:
            return []
        tokenized_query = tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)
        pairs = []
        for idx, (cid, sc) in enumerate(zip(self.chunk_ids, scores)):
            if where and idx < len(self.metadatas):
                meta = self.metadatas[idx] or {}
                if not all(meta.get(k) == v for k, v in where.items()):
                    continue
            pairs.append((cid, float(sc)))
        ranked = sorted(pairs, key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
