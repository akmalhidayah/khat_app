"""Orchestrates the full pre-/post-CNN input validation pipeline."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

from services.khat_recognition_messages import (
    MSG_LOW_DATASET_SIMILARITY,
    MSG_NOT_ARABIC_IMAGE,
    MSG_NOT_ARABIC_SCRIPT,
    MSG_OUTSIDE_TRAINED_CLASSES,
    REJECTION_LOW_SIMILARITY,
    REJECTION_NOT_ARABIC_IMAGE,
    REJECTION_NOT_ARABIC_SCRIPT,
    REJECTION_OUTSIDE_CLASSES,
)
from services.validation_pipeline.arabic_script import detect_arabic_script
from services.validation_pipeline.object_detector import detect_forbidden_objects
from services.validation_pipeline.ood_detector import check_msp_ood
from services.validation_pipeline.similarity import cosine_similarity_to_dataset
from services.validation_pipeline.utils import accepted, env_flag
from services.validation_pipeline.validation import validate_image_file

logger = logging.getLogger(__name__)


def pipeline_enabled() -> bool:
    return env_flag("INPUT_VALIDATION_PIPELINE", "1")


def validate_before_cnn(image_path: str) -> Dict[str, Any]:
    """Steps 1–3: file validation, Arabic script, forbidden objects."""
    steps = []

    file_result = validate_image_file(image_path)
    steps.append(file_result)
    if not file_result.get("accepted"):
        # Unreadable / invalid file is treated as non-Arabic image content.
        reason = file_result.get("reason") or MSG_NOT_ARABIC_IMAGE
        return {
            **file_result,
            "reason": MSG_NOT_ARABIC_IMAGE if "corrupted" not in str(reason).lower() else reason,
            "rejection_code": REJECTION_NOT_ARABIC_IMAGE,
            "steps": steps,
        }

    arabic_result = detect_arabic_script(image_path)
    steps.append(arabic_result)
    if not arabic_result.get("accepted"):
        return {**arabic_result, "steps": steps}

    object_result = detect_forbidden_objects(image_path)
    steps.append(object_result)
    if not object_result.get("accepted"):
        return {**object_result, "steps": steps}

    return {
        "status": "Accepted",
        "accepted": True,
        "stage": "pre_cnn",
        "steps": steps,
    }


def validate_distribution(
    *,
    embedding,
    probabilities: Sequence[float],
    predicted_class: Optional[str],
    classifier=None,
    config=None,
) -> Dict[str, Any]:
    """Steps 5–6: cosine similarity gallery + MSP OOD."""
    steps = []
    sim_result = cosine_similarity_to_dataset(
        embedding,
        classifier=classifier,
        config=config,
    )
    steps.append(sim_result)
    if not sim_result.get("accepted"):
        return {**sim_result, "steps": steps}

    ood_result = check_msp_ood(probabilities, predicted_class=predicted_class)
    steps.append(ood_result)
    if not ood_result.get("accepted"):
        return {**ood_result, "steps": steps}

    similarity = float(sim_result.get("similarity") or 0.0)
    confidence = float(ood_result.get("confidence") or 0.0)
    return accepted(
        predicted_class=predicted_class or "",
        confidence=confidence,
        similarity=similarity,
        details={"steps": steps},
    )


def _resolve_rejection_code(pipeline_result: Dict[str, Any]) -> str:
    code = pipeline_result.get("rejection_code")
    if code:
        return str(code)
    stage = str(pipeline_result.get("stage") or "")
    reason = str(pipeline_result.get("reason") or "")
    if stage in {"object_detection", "file"} or reason == MSG_NOT_ARABIC_IMAGE:
        return REJECTION_NOT_ARABIC_IMAGE
    if stage == "arabic_script" or reason == MSG_NOT_ARABIC_SCRIPT:
        return REJECTION_NOT_ARABIC_SCRIPT
    if stage in {"similarity", "ood_msp"} or reason == MSG_LOW_DATASET_SIMILARITY:
        return REJECTION_LOW_SIMILARITY
    return REJECTION_OUTSIDE_CLASSES


def to_flask_rejection(pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
    """Map pipeline rejection into existing Flask result keys."""
    code = _resolve_rejection_code(pipeline_result)
    if code == REJECTION_NOT_ARABIC_SCRIPT:
        reason = MSG_NOT_ARABIC_SCRIPT
        input_status = "non_khat"
    elif code == REJECTION_NOT_ARABIC_IMAGE:
        reason = MSG_NOT_ARABIC_IMAGE
        input_status = "non_khat"
    elif code == REJECTION_LOW_SIMILARITY:
        reason = MSG_LOW_DATASET_SIMILARITY
        input_status = "unrecognized"
    else:
        reason = pipeline_result.get("reason") or MSG_OUTSIDE_TRAINED_CLASSES
        input_status = "unrecognized"

    return {
        "input_status": input_status,
        "is_khat": False,
        "predicted_class": None,
        "confidence": None,
        "scores": None,
        "probabilities": None,
        "sorted_probabilities": None,
        "top2_margin": None,
        "top2_margin_pct": None,
        "reliability": "Rejected",
        "rejection_reason": reason,
        "rejection_code": code,
        "message": reason,
        "status_label": "Khat Tidak Dikenali" if input_status == "unrecognized" else None,
        "stage2_ran": False,
        "stage2_message": (
            "Klasifikasi diblokir oleh pipeline validasi input "
            "(bukan tulisan Arab / di luar dataset / kemiripan rendah)."
        ),
        "validation_pipeline": pipeline_result,
        "khat_probability": 0.0,
        "non_khat_probability": 1.0,
        "detection_status": "Rejected Non-Khat" if input_status == "non_khat" else "Unrecognized",
        "detection_decision": "Ditolak",
        "stage2_permission": "Blocked",
    }
