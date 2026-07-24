"""Shared helpers for the pre-CNN input validation pipeline."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def rejected(reason: str, *, stage: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": "Rejected",
        "reason": reason,
        "stage": stage,
        "accepted": False,
    }
    if details:
        payload["details"] = details
    return payload


def accepted(
    *,
    predicted_class: str,
    confidence: float,
    similarity: float,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": "Accepted",
        "class": predicted_class,
        "confidence": float(confidence),
        "similarity": float(similarity),
        "accepted": True,
    }
    if details:
        payload["details"] = details
    return payload


def env_flag(name: str, default: str = "1") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)
