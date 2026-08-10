"""
Continuous Production Drift Detection — Feature 5 of ApexRAG Blueprint.

Detects embedding-space centroid shifts and corpus staleness without code changes.
"""

import json
import numpy as np
from datetime import datetime
from src.config import DRIFT_THRESHOLD, DRIFT_LOGS_DIR
from src.ingestion.embedder import embed_texts


def compute_drift(current_chunks: list[str], baseline_embeddings: np.ndarray) -> dict:
    """
    Computes centroid shift in embedding space against reference baseline embeddings.
    """
    current_embeddings = embed_texts(current_chunks)
    if len(current_embeddings) == 0 or len(baseline_embeddings) == 0:
        return {"centroid_shift": 0.0, "flagged": False}

    baseline_centroid = baseline_embeddings.mean(axis=0)
    current_centroid = current_embeddings.mean(axis=0)
    centroid_shift = float(np.linalg.norm(baseline_centroid - current_centroid))

    result = {
        "timestamp": datetime.now().isoformat(),
        "centroid_shift": round(centroid_shift, 4),
        "threshold": DRIFT_THRESHOLD,
        "flagged": centroid_shift > DRIFT_THRESHOLD,
    }

    # Log results
    log_file = DRIFT_LOGS_DIR / f"drift_check_{int(datetime.now().timestamp())}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    # Smoke test for CLI / GitHub Action execution
    sample_current = ["React hooks state management update", "useRef ref access DOM"]
    baseline = embed_texts(["Class components state lifecycle", "React Component legacy base class"])
    res = compute_drift(sample_current, baseline)
    print(f"Drift Check Result: Centroid Shift={res['centroid_shift']} | Flagged={res['flagged']}")
