"""
Corrective RAG (CRAG) — Self-Reflective Retrieval Evaluation & Knowledge Refinement.
Reference: Yan et al., "Corrective Retrieval Augmented Generation" (2024).

Features:
  1. Retrieval Grader: Evaluates document relevance to the query into 3 grades:
     - CORRECT: Documents directly satisfy the query -> Direct generation.
     - AMBIGUOUS: Documents are partially relevant -> Knowledge Striping & Sentence Refinement.
     - INCORRECT: Documents are irrelevant -> Agentic Query Reformulation & safe fallback.
  2. Knowledge Refinement: Strips extraneous sentences from ambiguous chunks.
  3. Query Reformulation: Rewrites low-scoring queries for secondary retrieval or safe refusal.
"""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class RetrievalGrade(str, Enum):
    CORRECT = "CORRECT"
    AMBIGUOUS = "AMBIGUOUS"
    INCORRECT = "INCORRECT"


@dataclass
class CRAGResult:
    """Telemetry and outcome of the Corrective RAG evaluation pipeline."""
    grade: RetrievalGrade
    confidence_score: float
    chunk_scores: List[float]
    action_taken: str
    refined_chunks: List[str]
    reformulated_query: Optional[str] = None
    is_refusal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grade": self.grade.value,
            "confidence_score": round(self.confidence_score, 4),
            "chunk_scores": [round(s, 4) for s in self.chunk_scores],
            "action_taken": self.action_taken,
            "refined_chunk_count": len(self.refined_chunks),
            "reformulated_query": self.reformulated_query,
            "is_refusal": self.is_refusal,
        }


STOPWORDS = {
    "how", "does", "do", "what", "is", "the", "a", "an", "in", "to", "and", "or",
    "of", "for", "with", "that", "this", "can", "you", "i", "we", "at", "by", "from",
    "it", "as", "be", "are", "was", "were", "tell", "me", "please"
}


def _stem(token: str) -> str:
    """Simple rule-based suffix normalization for technical concepts."""
    t = token.lower()
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("ing") and len(t) > 5:
        return t[:-3]
    if t.endswith("ed") and len(t) > 4:
        return t[:-2]
    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        return t[:-1]
    return t


def _tokenize(text: str, remove_stopwords: bool = True) -> set:
    """Lowercases, cleans, and extracts stemmed technical tokens."""
    tokens = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return {_stem(t) for t in tokens}


def compute_chunk_relevance(query: str, chunk: str) -> float:
    """
    Computes a calibrated relevance score in [0.0, 1.0] between query and chunk.
    Balances stemmed keyword overlap, technical symbol detection, and exact phrases.
    """
    if not query or not chunk:
        return 0.0

    q_tokens = _tokenize(query, remove_stopwords=True)
    c_tokens = _tokenize(chunk, remove_stopwords=False)

    if not q_tokens:
        # Fallback to raw tokens if query was only stopwords
        q_tokens = _tokenize(query, remove_stopwords=False)
    if not q_tokens or not c_tokens:
        return 0.0

    # 1. Stemmed Token Overlap (Recall against query keywords)
    matched_tokens = 0
    for qt in q_tokens:
        if qt in c_tokens or any(qt in ct or ct in qt for ct in c_tokens):
            matched_tokens += 1
    overlap_ratio = matched_tokens / len(q_tokens)

    # 2. Technical symbol weighting (e.g. use..., component..., React..., state, props, effect, hook)
    symbols = re.findall(
        r"\b(use[a-z0-9_]+|component[a-z0-9_]+|react\.[a-z0-9_]+|state|props|effect|ref|context|hooks?)\b",
        query.lower(),
    )
    symbol_boost = 0.0
    if symbols:
        matched_symbols = sum(1 for s in symbols if s in chunk.lower() or _stem(s) in c_tokens)
        symbol_boost = (matched_symbols / len(symbols)) * 0.30

    # 3. Exact phrase match boost
    phrase_boost = 0.0
    q_clean = query.strip().lower()
    if len(q_clean.split()) <= 6 and q_clean in chunk.lower():
        phrase_boost = 0.20

    combined = (overlap_ratio * 0.65) + symbol_boost + phrase_boost
    return float(max(0.0, min(1.0, combined)))


def refine_knowledge_units(chunks: List[str], query: str, min_sentence_score: float = 0.15) -> List[str]:
    """
    Knowledge Striping (CRAG Phase 2):
    Splits retrieved chunks into individual sentences/units and removes irrelevant filler.
    Retains only informative sentences that share conceptual overlap with the query.
    """
    refined: List[str] = []
    q_tokens = _tokenize(query, remove_stopwords=True)
    if not q_tokens:
        q_tokens = _tokenize(query, remove_stopwords=False)

    for chunk in chunks:
        # Split on sentence boundaries while preserving code fences
        sentences = re.split(r"(?<=[.!?\n])\s+", chunk)
        kept_sentences = []

        for sent in sentences:
            sent_str = sent.strip()
            if not sent_str:
                continue
            # Keep intact code blocks
            if "```" in sent_str or "{" in sent_str or ";" in sent_str:
                kept_sentences.append(sent_str)
                continue
            s_tokens = _tokenize(sent_str, remove_stopwords=False)
            if s_tokens and any(qt in s_tokens for qt in q_tokens):
                kept_sentences.append(sent_str)

        if kept_sentences:
            refined.append(" ".join(kept_sentences))
        else:
            # Fallback to entire chunk if over-filtered
            refined.append(chunk)

    return refined if refined else chunks


def reformulate_query(query: str) -> str:
    """
    Agentic Query Reformulation (CRAG Phase 3):
    Rewrites an ineffective query by stripping conversational filler and
    expanding canonical technical terminology.
    """
    cleaned = re.sub(
        r"\b(can you please tell me|can you tell me|how do i|please explain|what is the meaning of|show me how to|tell me about|explain)\b",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip()

    # Expand common abbreviations
    cleaned = re.sub(r"\bprops\b", "properties props", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bstate\b", "component state useState", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\beffect\b", "side effects useEffect", cleaned, flags=re.IGNORECASE)

    return cleaned if cleaned else query


def execute_crag_pipeline(
    query: str,
    retrieved_chunks: List[str],
    threshold_correct: float = 0.55,
    threshold_ambiguous: float = 0.30,
) -> CRAGResult:
    """
    Executes the end-to-end Corrective RAG evaluation and action workflow:
      1. Evaluates per-chunk and overall relevance score.
      2. If CORRECT (score >= threshold_correct): direct pass-through.
      3. If AMBIGUOUS (threshold_ambiguous <= score < threshold_correct): knowledge refinement.
      4. If INCORRECT (score < threshold_ambiguous): triggers query reformulation.
    """
    if not retrieved_chunks:
        return CRAGResult(
            grade=RetrievalGrade.INCORRECT,
            confidence_score=0.0,
            chunk_scores=[],
            action_taken="query_reformulation",
            refined_chunks=[],
            reformulated_query=reformulate_query(query),
            is_refusal=True,
        )

    chunk_scores = [compute_chunk_relevance(query, c) for c in retrieved_chunks]
    max_score = max(chunk_scores) if chunk_scores else 0.0
    mean_score = float(np.mean(chunk_scores)) if chunk_scores else 0.0
    # Blend max and mean score (giving more weight to the best matching chunk)
    aggregate_score = (max_score * 0.7) + (mean_score * 0.3)

    if aggregate_score >= threshold_correct:
        return CRAGResult(
            grade=RetrievalGrade.CORRECT,
            confidence_score=aggregate_score,
            chunk_scores=chunk_scores,
            action_taken="direct_generate",
            refined_chunks=retrieved_chunks,
            reformulated_query=None,
            is_refusal=False,
        )

    elif aggregate_score >= threshold_ambiguous:
        refined = refine_knowledge_units(retrieved_chunks, query)
        return CRAGResult(
            grade=RetrievalGrade.AMBIGUOUS,
            confidence_score=aggregate_score,
            chunk_scores=chunk_scores,
            action_taken="knowledge_refinement",
            refined_chunks=refined,
            reformulated_query=None,
            is_refusal=False,
        )

    else:
        new_query = reformulate_query(query)
        return CRAGResult(
            grade=RetrievalGrade.INCORRECT,
            confidence_score=aggregate_score,
            chunk_scores=chunk_scores,
            action_taken="query_reformulation",
            refined_chunks=retrieved_chunks,
            reformulated_query=new_query,
            is_refusal=False,
        )
