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
    Diagnoses exact root cause when a question fails accuracy/recall checks.
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
