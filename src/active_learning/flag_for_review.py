"""
Active Learning Ground Truth Flags — Feature 8a of ApexRAG Blueprint.

Flags ground truth questions for human review when system confidence strongly disagrees with marked correctness.
"""

from typing import List, Dict, Any


def flag_disagreements(eval_results: List[Dict[str, Any]], confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
    """
    Flags questions where system confidence conflicts with evaluation judgment.
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
