"""
End-to-End Generation Evaluation Engine (Ragas-Style).

Provides:
  - Faithfulness: Factual consistency of generated answers with retrieved context (claim extraction & entailment verification).
  - Answer Relevance: Semantic alignment between user question and generated answer.
  - Context Recall: Degree to which retrieved context contains the reference ground-truth facts.
  - Root-Cause Failure Attribution: Distinguishes Retrieval Misses from LLM Hallucinations.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.ingestion.embedder import embed_texts


@dataclass
class ClaimVerification:
    """Individual atomic claim verification detail."""
    claim: str
    is_supported: bool
    similarity_score: float
    supporting_snippet: Optional[str] = None


@dataclass
class GenerationEvalResult:
    """Structured end-to-end evaluation result for a single generation."""
    question: str
    answer: str
    faithfulness: float
    answer_relevance: float
    context_recall: Optional[float]
    claims_verified: List[ClaimVerification]
    diagnosis: str  # "SUCCESS", "RETRIEVAL_FAILURE", "GENERATION_HALLUCINATION", "COMPOUND_FAILURE"
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "context_recall": round(self.context_recall, 4) if self.context_recall is not None else None,
            "claims_count": len(self.claims_verified),
            "claims_supported": sum(1 for c in self.claims_verified if c.is_supported),
            "diagnosis": self.diagnosis,
            "details": self.details,
        }


def _split_into_claims(text: str) -> List[str]:
    """
    Decomposes an answer or reference text into discrete atomic factual statements.
    Filters out conversational fluff and formatting artifacts.
    """
    cleaned = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    claims = []

    fluff_patterns = [
        r"^in summary",
        r"^to conclude",
        r"^based on the (documentation|provided context)",
        r"^overall",
        r"^in general",
    ]

    for s in raw_sentences:
        s_clean = s.strip()
        if len(s_clean) < 15 or len(s_clean.split()) < 3:
            continue
        # Check if sentence is purely conversational boilerplate
        if any(re.match(p, s_clean.lower()) for p in fluff_patterns) and len(s_clean.split()) <= 6:
            continue
        claims.append(s_clean)

    return claims or [text.strip()] if text.strip() else []


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Computes cosine similarity between two 1D vectors."""
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "with", "from", "of", "by", "as",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "shall", "should", "may", "might", "must", "can", "could",
    "it", "its", "they", "their", "them", "this", "that", "these", "those", "which", "who",
    "and", "or", "but", "if", "because", "so", "than", "too", "very", "also", "how", "what",
}


def _stem(w: str) -> str:
    """Lightweight morphological suffix normalizer for fast zero-cost matching."""
    for suffix in ("ing", "tions", "tion", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return w[:-len(suffix)]
    return w


def _extract_content_words(text: str) -> set[str]:
    """Extracts non-trivial content words, stripping punctuation and applying light suffix stemming."""
    words = re.findall(r"[a-zA-Z0-9_.]+", text.lower())
    return {_stem(w) for w in words if len(w) > 1 and w not in STOPWORDS}


def compute_faithfulness(
    answer: str,
    context: Union[List[str], str],
    entailment_threshold: float = 0.45,
) -> Tuple[float, List[ClaimVerification]]:
    """
    Computes Faithfulness (Factual Consistency) score in [0.0, 1.0].
    Decomposes the answer into atomic statements and verifies each against context
    using hybrid lexical claim coverage + semantic embedding similarity.

    Faithfulness = |Supported Claims| / |Total Claims|
    """
    if isinstance(context, str):
        context_chunks = [c.strip() for c in context.split("\n\n") if c.strip()]
    else:
        context_chunks = [c.strip() for c in context if c.strip()]

    if not context_chunks:
        return 0.0, []

    claims = _split_into_claims(answer)
    if not claims:
        return 1.0, []

    # Batch embed claims and context chunks
    all_texts = claims + context_chunks
    embeddings = embed_texts(all_texts)
    claim_embs = embeddings[:len(claims)]
    context_embs = embeddings[len(claims):]

    verified_claims: List[ClaimVerification] = []

    # Extract code symbols from context
    ctx_full = " ".join(context_chunks).lower()
    code_sym_pattern = r"`([^`]+)`|\b(use[A-Z]\w+|component[A-Z]\w+|setState|PureComponent|StrictMode|forwardRef)\b"
    ctx_symbols = {
        s.lower()
        for tup in re.findall(code_sym_pattern, ctx_full)
        for s in tup if s
    }

    ctx_word_sets = [_extract_content_words(c) for c in context_chunks]

    for claim, c_emb in zip(claims, claim_embs):
        claim_low = claim.lower()
        claim_words = _extract_content_words(claim_low)
        best_sim = 0.0
        best_cov = 0.0
        best_snippet = None

        for idx, (ctx_text, ctx_emb, ctx_words) in enumerate(zip(context_chunks, context_embs, ctx_word_sets)):
            sim = _cosine_similarity(c_emb, ctx_emb)
            cov = len(claim_words & ctx_words) / float(len(claim_words) or 1)
            if sim > best_sim:
                best_sim = sim
                best_snippet = ctx_text[:180] + ("..." if len(ctx_text) > 180 else "")
            if cov > best_cov:
                best_cov = cov

        # Check code symbol conflict (claim mentions a specific hook/API not present in context)
        claim_symbols = {
            s.lower()
            for tup in re.findall(code_sym_pattern, claim_low)
            for s in tup if s
        }
        symbol_conflict = bool(claim_symbols and not (claim_symbols & ctx_symbols))

        # Hybrid support score: 65% lexical claim coverage + 35% semantic embedding similarity
        support_score = 0.65 * best_cov + 0.35 * max(0.0, best_sim)
        is_supported = (support_score >= entailment_threshold) and not symbol_conflict

        verified_claims.append(
            ClaimVerification(
                claim=claim,
                is_supported=is_supported,
                similarity_score=round(support_score, 4),
                supporting_snippet=best_snippet if is_supported else None,
            )
        )

    supported_count = sum(1 for c in verified_claims if c.is_supported)
    faithfulness_score = supported_count / float(len(verified_claims))
    return float(round(faithfulness_score, 4)), verified_claims


def compute_answer_relevance(question: str, answer: str) -> float:
    """
    Computes Answer Relevance score in [0.0, 1.0].
    Measures semantic alignment and keyword intent coverage between user question and generated response.
    """
    if not question.strip() or not answer.strip():
        return 0.0

    # Low relevance penalty for non-informative refusal answers
    refusal_phrases = [
        "i do not know",
        "i don't know",
        "not sufficient information",
        "information is unavailable",
        "does not contain sufficient information",
    ]
    if any(p in answer.lower() for p in refusal_phrases) and len(answer.split()) < 25:
        return 0.15

    q_words = _extract_content_words(question)
    a_words = _extract_content_words(answer)
    lex_overlap = len(q_words & a_words) / float(len(q_words) or 1)

    embeddings = embed_texts([question, answer])
    q_emb, a_emb = embeddings[0], embeddings[1]
    sem_sim = max(0.0, _cosine_similarity(q_emb, a_emb))

    # Hybrid answer relevance with calibrated dynamic range
    base_relevance = 0.55 * lex_overlap + 0.45 * sem_sim
    scaled_relevance = min(1.0, base_relevance * 1.35)
    return float(round(scaled_relevance, 4))


def compute_context_recall(
    ground_truth_answer: str,
    context: Union[List[str], str],
    attribution_threshold: float = 0.45,
) -> Tuple[float, List[ClaimVerification]]:
    """
    Computes Context Recall in [0.0, 1.0].
    Measures the fraction of reference ground-truth assertions present in retrieved context.
    """
    if not ground_truth_answer.strip():
        return 1.0, []

    return compute_faithfulness(
        answer=ground_truth_answer,
        context=context,
        entailment_threshold=attribution_threshold,
    )


def diagnose_pipeline_failure(
    context_recall: Optional[float],
    faithfulness: float,
    answer_relevance: float,
    threshold: float = 0.65,
) -> str:
    """
    Pinpoints root cause:
      - SUCCESS: Context was retrieved and answer is faithful.
      - RETRIEVAL_FAILURE: Context lacked facts, causing generation to miss.
      - GENERATION_HALLUCINATION: Context had facts, but LLM hallucinated ungrounded claims.
      - COMPOUND_FAILURE: Both retrieval and generation failed.
    """
    recall_ok = (context_recall is None) or (context_recall >= threshold)
    faith_ok = (faithfulness >= threshold) and (answer_relevance >= 0.40)

    if recall_ok and faith_ok:
        return "SUCCESS"
    elif not recall_ok and faith_ok:
        return "RETRIEVAL_FAILURE"
    elif recall_ok and not faith_ok:
        return "GENERATION_HALLUCINATION"
    else:
        return "COMPOUND_FAILURE"


def evaluate_generation_sample(
    question: str,
    answer: str,
    context: Union[List[str], str],
    ground_truth: Optional[str] = None,
) -> GenerationEvalResult:
    """
    Executes full tri-metric Ragas-style evaluation on a single sample.
    """
    faithfulness, verified_claims = compute_faithfulness(answer, context)
    relevance = compute_answer_relevance(question, answer)

    recall = None
    if ground_truth:
        recall, _ = compute_context_recall(ground_truth, context)

    diagnosis = diagnose_pipeline_failure(
        context_recall=recall,
        faithfulness=faithfulness,
        answer_relevance=relevance,
    )

    details = {
        "claims_total": len(verified_claims),
        "claims_supported": sum(1 for c in verified_claims if c.is_supported),
        "unsupported_claims": [c.claim for c in verified_claims if not c.is_supported],
    }

    return GenerationEvalResult(
        question=question,
        answer=answer,
        faithfulness=faithfulness,
        answer_relevance=relevance,
        context_recall=recall,
        claims_verified=verified_claims,
        diagnosis=diagnosis,
        details=details,
    )


def evaluate_generation_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates a batch of RAG generation samples and computes macro-averaged metrics.
    """
    results: List[GenerationEvalResult] = []
    diagnoses: Dict[str, int] = {
        "SUCCESS": 0,
        "RETRIEVAL_FAILURE": 0,
        "GENERATION_HALLUCINATION": 0,
        "COMPOUND_FAILURE": 0,
    }

    for s in samples:
        res = evaluate_generation_sample(
            question=s.get("question", ""),
            answer=s.get("answer", ""),
            context=s.get("context", s.get("context_chunks", [])),
            ground_truth=s.get("ground_truth", s.get("reference_answer")),
        )
        results.append(res)
        diagnoses[res.diagnosis] = diagnoses.get(res.diagnosis, 0) + 1

    n = len(results)
    if n == 0:
        return {"samples_count": 0}

    avg_faith = sum(r.faithfulness for r in results) / float(n)
    avg_relevance = sum(r.answer_relevance for r in results) / float(n)
    recall_vals = [r.context_recall for r in results if r.context_recall is not None]
    avg_recall = (sum(recall_vals) / float(len(recall_vals))) if recall_vals else None

    return {
        "samples_count": n,
        "mean_faithfulness": round(avg_faith, 4),
        "mean_answer_relevance": round(avg_relevance, 4),
        "mean_context_recall": round(avg_recall, 4) if avg_recall is not None else None,
        "diagnoses_breakdown": diagnoses,
        "success_rate": round(diagnoses.get("SUCCESS", 0) / float(n), 4),
        "samples": [r.to_dict() for r in results],
    }
