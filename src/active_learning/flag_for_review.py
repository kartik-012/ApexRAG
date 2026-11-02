"""
Active Learning Ground Truth Flags — Feature 8a of ApexRAG Blueprint.

Flags ground truth questions for human review when system confidence strongly disagrees with marked correctness.
"""

from typing import List, Dict, Any


def flag_disagreements(eval_results: List[Dict[str, Any]], confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
    """
    Flags evaluation questions where system confidence sharply conflicts with ground-truth correctness.

    Triages edge cases for human-in-the-loop review:
      1. High Confidence False Positives: System reports HIGH confidence, but output fails ground-truth checks.
      2. Low Confidence True Positives: System reports LOW confidence, but answer is evaluated as correct.

    Args:
        eval_results: List of evaluation result records with 'confidence' and 'correct' fields.
        confidence_threshold: Maximum allowable disagreement margin before flagging.

    Returns:
        List of flagged question records annotated with triage reasons.
    """
    flagged = []
    for r in eval_results:
        sys_conf = 1.0 if r.get("confidence") == "HIGH" else (0.5 if r.get("confidence") == "MEDIUM" else 0.0)
        is_correct = 1.0 if r.get("correct") else 0.0
        disagreement = abs(sys_conf - is_correct)
        if disagreement > (1.0 - confidence_threshold):
            flagged.append({
                **r,
                "reason": "high confidence/correctness mismatch — review ground truth annotation",
            })
    return flagged
