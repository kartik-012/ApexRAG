"""
Causal Failure Attribution — Feature 3 of ApexRAG Blueprint.

Attributes wrong answers to specific pipeline stages (retrieval miss, rank issue, chunk boundary, hallucination).
"""

from enum import Enum


class FailureType(Enum):
    RETRIEVAL_MISS = "gold chunk never retrieved at all"
    RETRIEVAL_RANK = "gold chunk retrieved but ranked below top-k"
    CHUNK_BOUNDARY = "answer split across two adjacent chunks, neither complete"
    GENERATION_HALLUCINATION = "correct context retrieved, LLM answer not grounded in it"
    GROUND_TRUTH_AMBIGUOUS = "question has multiple valid answers not captured in gold set"


def attribute_failure(
    question_id: str,
    retrieved_ids: list[str],
    gold_ids: set[str],
    full_ranked_ids: list[str],
    faithfulness_score: float,
    answer_correct: bool,
) -> FailureType | None:
    """
    Diagnoses exact pipeline failure root cause across retrieval and generation boundaries.

    Categorization taxonomy:
      1. RETRIEVAL_MISS: Gold chunk absent across entire candidate retrieval pool.
      2. RETRIEVAL_RANK: Gold chunk retrieved within candidate pool (top-20) but suppressed below cutoff K.
      3. GENERATION_HALLUCINATION: Gold chunk retrieved, but generator produced ungrounded claims (faithfulness < 0.5).
      4. CHUNK_BOUNDARY: Relevant text split across boundary seams, degrading answer completeness.

    Args:
        question_id: Stable identifier for benchmark question.
        retrieved_ids: Top-K retrieved chunk IDs.
        gold_ids: Ground-truth chunk IDs.
        full_ranked_ids: Extended candidate pool (e.g. top-20 before cutoff).
        faithfulness_score: Calibrated factual grounding score in [0.0, 1.0].
        answer_correct: Ground-truth validation correctness boolean.

    Returns:
        FailureType enum diagnosing root cause, or None if pipeline succeeded.
    """
    if answer_correct and (set(retrieved_ids) & gold_ids):
        return None

    if not (set(retrieved_ids) & gold_ids):
        if set(full_ranked_ids[:20]) & gold_ids:
            return FailureType.RETRIEVAL_RANK
        return FailureType.RETRIEVAL_MISS

    if faithfulness_score < 0.5:
        return FailureType.GENERATION_HALLUCINATION

    return FailureType.CHUNK_BOUNDARY
