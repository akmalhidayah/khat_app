"""Step 6 — Out-of-Distribution detection via Maximum Softmax Probability."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import numpy as np

from services.khat_recognition_messages import MSG_LOW_DATASET_SIMILARITY, REJECTION_LOW_SIMILARITY
from services.validation_pipeline.utils import env_float, rejected

logger = logging.getLogger(__name__)


def maximum_softmax_probability(probabilities: Sequence[float]) -> float:
    if probabilities is None:
        return 0.0
    arr = np.asarray(list(probabilities), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.max(arr))


def check_msp_ood(
    probabilities: Sequence[float],
    *,
    predicted_class: Optional[str] = None,
) -> Dict:
    """Reject when max softmax probability is below the configured floor."""
    min_conf = env_float("VALIDATION_MSP_MIN", 0.45)
    confidence = maximum_softmax_probability(probabilities)
    if confidence < min_conf:
        payload = rejected(
            MSG_LOW_DATASET_SIMILARITY,
            stage="ood_msp",
            details={
                "confidence": round(confidence, 4),
                "min_confidence": min_conf,
                "predicted_class": predicted_class,
            },
        )
        payload["rejection_code"] = REJECTION_LOW_SIMILARITY
        payload["confidence"] = round(confidence, 4)
        return payload
    return {
        "status": "Accepted",
        "accepted": True,
        "stage": "ood_msp",
        "confidence": round(confidence, 4),
        "min_confidence": min_conf,
        "predicted_class": predicted_class,
    }
