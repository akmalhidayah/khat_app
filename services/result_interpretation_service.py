import json
import os
from typing import Any, Dict, List, Optional, Tuple

from flask import current_app

from services.khat_detector_service import resolve_detection
from services.khat_characteristics_service import build_characteristics_context
from services.dataset_similarity_service import build_similarity_context
from services.metrics_service import ACCURACY_TARGET
from services.model_context_service import get_model_context, get_model_display_label, get_model_short_name, is_teachable_machine_active
from services.teachable_machine_service import get_tm_display_info

CLASS_DISPLAY = {
    "naskhi": "Naskhi",
    "diwani": "Diwani",
    "diwani_jali": "Diwani Jali",
    "tsuluts": "Tsuluts",
}

CLASS_COLORS = {
    "naskhi": {"bar": "naskhi", "hex": "#059669"},
    "diwani": {"bar": "diwani", "hex": "#4f46e5"},
    "diwani_jali": {"bar": "diwani-jali", "hex": "#d97706"},
    "tsuluts": {"bar": "tsuluts", "hex": "#0891b2"},
}

MODE_READINESS = {
    "ultra_fast": {
        "label": "Demo Model",
        "badge": "demo",
        "tone": "warning",
        "message": "This prediction was generated using a Demo model and may not represent final research performance.",
    },
    "fast": {
        "label": "Fast Experimental Model",
        "badge": "fast",
        "tone": "orange",
        "message": "This prediction was generated using a Fast/Demo model and may not represent final research performance.",
    },
    "research": {
        "label": "Research Model",
        "badge": "research",
        "tone": "success",
        "message": "This prediction was generated using the research model.",
    },
    "normal": {
        "label": "Research Model",
        "badge": "research",
        "tone": "success",
        "message": "This prediction was generated using the research model.",
    },
    "teachable_machine": {
        "label": "Teachable Machine Model",
        "badge": "research",
        "tone": "success",
        "message": "This prediction was generated using the integrated Teachable Machine model.",
    },
}


def display_name(class_key: Optional[str]) -> str:
    if not class_key:
        return "Unknown"
    return CLASS_DISPLAY.get(class_key, class_key.replace("_", " ").title())


def parse_filename_class(filename: str) -> Tuple[Optional[str], Optional[str]]:
    if not filename:
        return None, None

    lower = filename.lower().replace("-", "_")
    compact = lower.replace(" ", "_")

    prefix_map = (
        ("diwani_jali_", "diwani_jali"),
        ("naskhi_", "naskhi"),
        ("diwani_", "diwani"),
        ("tsuluts_", "tsuluts"),
    )
    for prefix, class_key in prefix_map:
        if compact.startswith(prefix):
            return class_key, display_name(class_key)

    diwani_jali_tokens = ("diwani_jali", "diwanijali", "diwani jali")
    for token in diwani_jali_tokens:
        normalized = token.replace(" ", "_")
        if normalized in compact or token in filename.lower():
            return "diwani_jali", display_name("diwani_jali")

    if "naskhi" in lower or "naski" in lower:
        return "naskhi", display_name("naskhi")

    if "diwani" in lower and "jali" not in lower:
        return "diwani", display_name("diwani")

    for token in ("tsuluts", "tsuluth", "sulus", "thuluth"):
        if token in lower:
            return "tsuluts", display_name("tsuluts")

    return None, None


def resolve_expected_class(
    filename: str,
    manual_expected_class: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], str]:
    """Return expected class key, display name, and source: manual | filename | none."""
    manual_key = normalize_manual_expected_class(manual_expected_class)
    if manual_key:
        return manual_key, display_name(manual_key), "manual"

    filename_key, filename_display = parse_filename_class(filename or "")
    if filename_key:
        return filename_key, filename_display, "filename"

    return None, None, "external"


def normalize_manual_expected_class(value: Optional[str]) -> Optional[str]:
    key = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not key or key in ("unknown", "none", "-", ""):
        return None
    if key in CLASS_DISPLAY:
        return key
    aliases = {"diwanijali": "diwani_jali", "tsuluth": "tsuluts", "thuluth": "tsuluts"}
    return aliases.get(key)


def compute_known_class_status(
    confidence_pct: float,
    margin_pct: float,
    has_expected_class: bool = False,
    prediction_match: bool = True,
) -> str:
    if has_expected_class and not prediction_match:
        return "Needs Review — Label Mismatch"
    if confidence_pct >= 85 and margin_pct >= 15:
        return "Strong Known Class"
    if confidence_pct >= 70 and margin_pct >= 10:
        return "Likely Known Class"
    if 60 <= confidence_pct < 70:
        return "Uncertain Class"
    return "Possible Unknown / Needs Review"


def build_dataset_bias_warning() -> Optional[str]:
    try:
        from services.dataset_quality_service import load_quality_report

        report = load_quality_report(current_app.config)
        imbalance = float(report.get("class_imbalance_percent") or 0)
        if imbalance <= 25:
            return None
        counts = report.get("per_class_counts") or report.get("class_counts") or {}
        minority = [display_name(k) for k, v in counts.items() if v == min(counts.values())] if counts else []
        minority_text = ", ".join(minority[:2]) if minority else "Naskhi dan Diwani"
        return (
            f"Distribusi dataset belum seimbang (selisih {imbalance:.0f}%). "
            f"Kelas dengan jumlah data lebih besar dapat lebih sering diprediksi oleh model. "
            f"Disarankan menambah data pada kelas minoritas seperti {minority_text}."
        )
    except Exception:
        return None


def build_misclassification_explanation(predicted_display: str, expected_display: str) -> str:
    return (
        f"Model kemungkinan menangkap pola visual yang mirip dengan kelas {predicted_display}, "
        f"seperti komposisi dekoratif atau bentuk tarikan huruf tertentu. "
        f"Namun berdasarkan label pembanding ({expected_display}) atau validasi manual, "
        f"gambar ini perlu ditinjau ulang."
    )


def _model_confidence_label(confidence_pct: float) -> str:
    if confidence_pct >= 85:
        return "Strong Model Confidence"
    if confidence_pct >= 70:
        return "Moderate Model Confidence"
    return "Weak Model Confidence"


def apply_stage1_combined_decision(
    validation_fields: Dict[str, Any],
    detection_status: Optional[str] = None,
    input_status: Optional[str] = None,
    detection_explanation: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge Stage 1 Khat detection outcome with Stage 2 validation fields."""
    updated = dict(validation_fields)
    status = detection_status or input_status or "khat"

    if status == "non_khat" or input_status == "non_khat":
        updated["final_decision"] = "Classification Blocked"
        updated["review_status"] = "Rejected"
        updated["manual_review_required"] = False
        updated["confidence_label"] = "Rejected"
        return updated

    if input_status == "unrecognized" or status == "unrecognized":
        from services.khat_recognition_messages import STATUS_LABEL
        updated["final_decision"] = STATUS_LABEL
        updated["review_status"] = "Rejected"
        updated["manual_review_required"] = False
        return updated

    # Borderline/uncertain Stage-1 outcomes are treated as rejected for labeling.
    if status in ("borderline_khat", "uncertain_khat", "uncertain") or input_status == "uncertain":
        updated["final_decision"] = "Classification Blocked"
        updated["review_status"] = "Rejected"
        updated["manual_review_required"] = False
        updated["confidence_label"] = "Rejected"
        if detection_explanation:
            existing = updated.get("explanation_text") or ""
            if detection_explanation not in existing:
                updated["explanation_text"] = f"{detection_explanation} {existing}".strip()
        return updated

    is_confirmed = status in ("confirmed_khat", "moderate_khat", "khat")
    prediction_match = bool(validation_fields.get("prediction_match"))
    has_expected = bool(validation_fields.get("expected_class"))

    if is_confirmed and prediction_match and has_expected:
        pass

    if detection_explanation:
        existing = updated.get("explanation_text") or ""
        if detection_explanation not in existing:
            updated["explanation_text"] = f"{detection_explanation} {existing}".strip()

    return updated


def compute_final_decision(
    prediction_match: bool,
    confidence_pct: float,
    has_expected_class: bool,
    filename_mismatch: bool,
) -> str:
    if has_expected_class and filename_mismatch:
        return "Not Automatically Valid"
    if confidence_pct < 70:
        return "Low Confidence Review"
    if prediction_match and confidence_pct >= 85:
        return "Accepted"
    if prediction_match and confidence_pct >= 70:
        return "Accepted with Caution"
    if not has_expected_class:
        if confidence_pct >= 85:
            return "Accepted with Caution"
        if confidence_pct >= 70:
            return "Accepted with Caution"
        return "Low Confidence Review"
    return "Low Confidence Review"


def build_explanation_text(
    predicted_display: str,
    expected_display: Optional[str],
    confidence_pct: float,
    prediction_match: bool,
    has_expected_class: bool,
    margin_pct: float,
    is_demo: bool = False,
) -> str:
    if has_expected_class and not prediction_match and expected_display:
        if confidence_pct >= 85:
            conf_text = f"high confidence ({confidence_pct:.2f}%)"
        elif confidence_pct >= 70:
            conf_text = f"moderate confidence ({confidence_pct:.2f}%)"
        else:
            conf_text = f"low confidence ({confidence_pct:.2f}%)"

        explanation = (
            f"The model predicted this image as {predicted_display} with {conf_text}. "
            f"However, the expected class from the filename is {expected_display}. "
            f"Prediction Mismatch. Manual review required."
        )
        if is_demo:
            explanation += " This prediction was generated using a demo/fast model and is not suitable for final research use."
        return explanation

    if prediction_match and confidence_pct >= 70:
        return (
            f"The model predicted this image as {predicted_display} with moderate confidence. "
            f"The predicted class matches the expected class from the filename, so the result is accepted with caution."
        )

    if prediction_match and confidence_pct >= 85:
        return (
            f"The model predicted this image as {predicted_display} with high confidence ({confidence_pct:.2f}%). "
            f"The predicted class matches the expected class from the filename, so the result is accepted."
        )

    if prediction_match:
        return (
            f"The model predicted this image as {predicted_display} with low confidence ({confidence_pct:.2f}%). "
            f"The predicted class matches the expected class from the filename. "
            f"Prediction Match but Low Confidence. Manual review recommended."
        )

    if confidence_pct >= 70:
        return (
            f"The model predicts this image as {predicted_display} with moderate confidence ({confidence_pct:.2f}%). "
            f"No filename-based expected class was detected for validation."
        )

    return (
        f"The model prediction is uncertain. Confidence is low ({confidence_pct:.2f}%) "
        f"and the result should not be treated as final without manual review."
    )


def _confidence_category(confidence_pct: float) -> str:
    if confidence_pct >= 85:
        return "Strong Prediction"
    if confidence_pct >= 70:
        return "Moderate Prediction"
    return "Weak Prediction"


def compute_validation_fields(
    filename: str,
    predicted_class: Optional[str],
    confidence: Optional[float],
    margin_pct: float = 0.0,
    top_2_class: Optional[str] = None,
    top_2_score: Optional[float] = None,
    manual_expected_class: Optional[str] = None,
) -> Dict[str, Any]:
    expected_class, expected_display, expected_source = resolve_expected_class(
        filename or "",
        manual_expected_class=manual_expected_class,
    )
    confidence_pct = (confidence or 0) * 100 if confidence is not None and confidence <= 1 else (confidence or 0)
    prediction_match = bool(expected_class and predicted_class and expected_class == predicted_class)
    has_expected_class = bool(expected_class)
    predicted_display = display_name(predicted_class)
    known_class_status = compute_known_class_status(
        confidence_pct,
        margin_pct,
        has_expected_class=has_expected_class,
        prediction_match=prediction_match,
    )
    is_external = expected_source == "external"
    external_warning = (
        "Gambar ini tidak memiliki label pembanding. Walaupun model memberikan confidence tinggi, "
        "hasil tetap perlu divalidasi manual karena gambar eksternal dapat memiliki gaya visual "
        "yang berbeda dari dataset training."
        if is_external
        else None
    )
    no_comparator_message = (
        "Tidak ada label pembanding. Hasil prediksi perlu divalidasi secara manual."
        if is_external
        else None
    )

    base = {
        "expected_class": expected_class,
        "expected_class_display": expected_display,
        "expected_source": expected_source,
        "manual_expected_class": normalize_manual_expected_class(manual_expected_class),
        "has_expected_class": has_expected_class,
        "is_external_image": is_external,
        "external_warning": external_warning,
        "no_comparator_message": no_comparator_message,
        "known_class_status": known_class_status,
        "top_2_class": top_2_class,
        "top_2_score": top_2_score,
        "top_2_margin": margin_pct,
        "prediction_match": prediction_match,
    }

    if has_expected_class and not prediction_match:
        mismatch_explanation = build_misclassification_explanation(predicted_display, expected_display or "")
        return {
            **base,
            "validation_status": "Prediction Mismatch",
            "review_status": "Manual Review Required",
            "confidence_label": "Mismatch Detected",
            "confidence_interpretation": "Mismatch Detected",
            "confidence_tone": "danger",
            "model_confidence_label": _model_confidence_label(confidence_pct),
            "final_decision": "Manual Review Required",
            "manual_review_required": True,
            "misclassification_explanation": mismatch_explanation,
            "explanation_text": build_explanation_text(
                predicted_display,
                expected_display,
                confidence_pct,
                False,
                True,
                margin_pct,
            ),
        }

    if prediction_match:
        if confidence_pct >= 85:
            review_status = "No Review Needed"
            final_decision = "Accepted"
            confidence_label = "Strong Prediction"
            tone = "high"
            manual_review_required = False
        elif confidence_pct >= 70:
            review_status = "Review Optional"
            final_decision = "Accepted with Caution"
            confidence_label = "Moderate Prediction"
            tone = "moderate"
            manual_review_required = False
        else:
            review_status = "Manual Review Recommended"
            final_decision = "Low Confidence Review"
            confidence_label = "Weak Prediction"
            tone = "low"
            manual_review_required = True
            explanation_override = (
                f"The model predicted this image as {predicted_display} with low confidence ({confidence_pct:.2f}%). "
                f"The predicted class matches the expected class from the filename. "
                f"Prediction Match but Low Confidence. Manual review recommended."
            )

        return {
            **base,
            "validation_status": "Prediction Match",
            "review_status": review_status,
            "confidence_label": confidence_label,
            "confidence_interpretation": confidence_label,
            "confidence_tone": tone,
            "model_confidence_label": _model_confidence_label(confidence_pct),
            "final_decision": final_decision,
            "manual_review_required": manual_review_required,
            "explanation_text": explanation_override if confidence_pct < 70 else build_explanation_text(
                predicted_display,
                expected_display,
                confidence_pct,
                True,
                True,
                margin_pct,
            ),
        }

    # No expected class provided
    confidence_label = _confidence_category(confidence_pct)
    if confidence_pct >= 85:
        review_status = "Review Optional"
        final_decision = "Accepted with Caution"
        manual_review_required = False
    elif confidence_pct >= 70:
        review_status = "Review Optional"
        final_decision = "Accepted with Caution"
        manual_review_required = False
    else:
        review_status = "Manual Review Recommended"
        final_decision = "Low Confidence Review"
        manual_review_required = True

    return {
        **base,
        "validation_status": "No Expected Class Provided",
        "review_status": review_status,
        "confidence_label": confidence_label,
        "confidence_interpretation": confidence_label,
        "confidence_tone": "moderate" if confidence_pct >= 70 else "low",
        "model_confidence_label": _model_confidence_label(confidence_pct),
        "final_decision": final_decision,
        "manual_review_required": manual_review_required,
        "explanation_text": no_comparator_message or build_explanation_text(
            predicted_display,
            expected_display,
            confidence_pct,
            False,
            False,
            margin_pct,
        ),
    }


def _reliability_from_confidence(
    confidence_pct: float,
    margin_pct: float = 0.0,
    prediction_match: bool = True,
    has_expected_class: bool = False,
    predicted_display: str = "",
    expected_display: Optional[str] = None,
) -> Dict[str, str]:
    if has_expected_class and not prediction_match:
        model_conf = _model_confidence_label(confidence_pct)
        return {
            "level": "Manual Review Required",
            "tone": "danger",
            "color": "danger",
            "message": (
                f"Predicted class ({predicted_display}) does not match the expected class "
                f"from the filename ({expected_display}). High model confidence does not imply correctness."
            ),
            "sub_message": f"{model_conf}, but Prediction Mismatch",
        }
    if prediction_match and confidence_pct >= 85 and margin_pct >= 50:
        return {
            "level": "Strong Prediction",
            "tone": "high",
            "color": "success",
            "message": "High confidence and the prediction matches the filename-based expected class.",
            "sub_message": None,
        }
    if prediction_match and confidence_pct >= 85:
        return {
            "level": "Strong Prediction",
            "tone": "high",
            "color": "success",
            "message": "High confidence and the prediction matches the filename-based expected class.",
            "sub_message": None,
        }
    if prediction_match and confidence_pct >= 70:
        return {
            "level": "Moderate Prediction",
            "tone": "moderate",
            "color": "info",
            "message": "The predicted class matches the filename-based expected class with moderate confidence.",
            "sub_message": None,
        }
    return {
        "level": "Weak Prediction",
        "tone": "low",
        "color": "warning",
        "message": "Low confidence. Manual verification is strongly recommended.",
        "sub_message": None,
    }


def _margin_analysis(margin_pct: float, filename_mismatch: bool = False) -> Dict[str, Any]:
    if filename_mismatch:
        return {
            "level": "mismatch",
            "label": "Mismatch — margin not used for validity",
            "is_high_ambiguity": False,
            "is_moderate_ambiguity": False,
            "interpretation": (
                "Top-2 margin reflects model separation only and does not validate correctness "
                "when the predicted class differs from the expected class."
            ),
        }
    if margin_pct >= 50:
        return {
            "level": "stable",
            "label": "Clear separation",
            "is_high_ambiguity": False,
            "is_moderate_ambiguity": False,
            "interpretation": "The top class is clearly ahead of the second strongest class.",
        }
    if margin_pct >= 20:
        return {
            "level": "stable",
            "label": "Stable separation",
            "is_high_ambiguity": False,
            "is_moderate_ambiguity": False,
            "interpretation": "The top class leads the second class with reasonable separation.",
        }
    if margin_pct >= 10:
        return {
            "level": "moderate",
            "label": "Moderate separation",
            "is_high_ambiguity": False,
            "is_moderate_ambiguity": True,
            "interpretation": "The top two classes are reasonably separated, but some visual overlap may remain.",
        }
    return {
        "level": "high",
        "label": "High ambiguity",
        "is_high_ambiguity": True,
        "is_moderate_ambiguity": True,
        "interpretation": "Ambiguous result — manual review recommended.",
    }


def _quality_status(
    confidence_pct: float,
    margin_pct: float,
    val_acc: Optional[float],
    is_demo: bool,
    is_research: bool,
    filename_mismatch: bool = False,
) -> Dict[str, str]:
    if filename_mismatch:
        return {
            "label": "Not Automatically Valid",
            "tone": "danger",
            "message": "Prediction does not match the filename-based expected class and cannot be accepted automatically.",
        }

    if confidence_pct < 60 or (confidence_pct < 70 and margin_pct < 10):
        return {
            "label": "Unreliable Result",
            "tone": "danger",
            "message": "This prediction should not be treated as final without manual verification.",
        }

    if (
        confidence_pct >= 85
        and val_acc is not None
        and val_acc >= ACCURACY_TARGET
        and is_research
        and not is_demo
    ):
        return {
            "label": "Strong Result",
            "tone": "success",
            "message": "Confidence and model performance support treating this as a strong candidate result.",
        }

    if confidence_pct >= 70 and margin_pct >= 15 and not is_demo:
        return {
            "label": "Acceptable Result",
            "tone": "info",
            "message": "The prediction is acceptable, though confirmation is still good practice.",
        }

    return {
        "label": "Needs Manual Review",
        "tone": "warning",
        "message": "Manual review is recommended before using this result in research or reporting.",
    }


def _model_readiness(model_ctx: Dict[str, Any]) -> Dict[str, Any]:
    mode_key = model_ctx.get("training_mode_key", "unknown")
    base = MODE_READINESS.get(mode_key, {
        "label": "Unknown Model",
        "badge": "unknown",
        "tone": "muted",
        "message": "Model training mode could not be determined.",
    })

    val_acc = model_ctx.get("validation_accuracy")
    using_best = model_ctx.get("using_best_model", False)
    is_research = model_ctx.get("is_research_model", False)

    if (
        using_best
        and is_research
        and val_acc is not None
        and float(val_acc) >= ACCURACY_TARGET
    ):
        return {
            "label": "Best Validated Model",
            "badge": "best",
            "tone": "premium",
            "message": "This prediction used the best validated research model checkpoint.",
        }

    return dict(base)


def _build_prediction_summary(
    top_display: str,
    expected_display: Optional[str],
    confidence_pct: float,
    margin_pct: float,
    is_demo: bool,
    prediction_match: bool,
    has_expected_class: bool,
) -> str:
    return build_explanation_text(
        top_display,
        expected_display,
        confidence_pct,
        prediction_match,
        has_expected_class,
        margin_pct,
        is_demo,
    )


def _mismatch_recommended_actions(expected_display: Optional[str], predicted_display: str) -> List[str]:
    expected = expected_display or "the expected class"
    return [
        "Check whether the uploaded filename label is correct.",
        f"Verify whether the image visually belongs to {expected} or {predicted_display}.",
        "Add more training images for visually similar classes.",
        "Balance the dataset between Diwani, Diwani Jali, Naskhi, and Tsuluts.",
        "Retrain the model and evaluate it using a separate test dataset.",
    ]


def _build_warnings(
    confidence_pct: float,
    margin_pct: float,
    margin_info: Dict[str, Any],
    requires_manual_review: bool,
    filename_mismatch: bool,
    filename_expected_display: Optional[str],
    predicted_display: str,
    model_readiness: Dict[str, Any],
    is_demo: bool,
    val_acc: Optional[float],
    test_acc: Optional[float],
) -> List[Dict[str, str]]:
    warnings: List[Dict[str, str]] = []
    val_pct = (val_acc * 100) if val_acc is not None else None
    test_pct = (test_acc * 100) if test_acc is not None else None

    if filename_mismatch:
        pass
    elif requires_manual_review and confidence_pct < 70:
        if margin_info["is_high_ambiguity"]:
            warnings.append({
                "priority": "1",
                "type": "reliability",
                "tone": "warning",
                "title": "Prediction reliability",
                "message": (
                    f"Confidence is {confidence_pct:.2f}% with a top-2 margin of only {margin_pct:.2f}%. "
                    "The model is considering more than one visually similar class."
                ),
            })
        elif confidence_pct < 70:
            warnings.append({
                "priority": "1",
                "type": "reliability",
                "tone": "danger",
                "title": "Low prediction confidence",
                "message": (
                    f"Confidence is {confidence_pct:.2f}%, which is below the recommended threshold. "
                    "Please verify this result manually."
                ),
            })
        elif confidence_pct < 85 and margin_pct < 20:
            warnings.append({
                "priority": "1",
                "type": "reliability",
                "tone": "warning",
                "title": "Moderate prediction confidence",
                "message": (
                    f"Confidence is {confidence_pct:.2f}%. The result is usable for exploration, "
                    "but manual review is recommended."
                ),
            })

    if is_demo:
        warnings.append({
            "priority": "2",
            "type": "model_mode",
            "tone": "danger",
            "title": "Demo model only — not suitable for final classification",
            "message": (
                "This model was trained using Fast/Demo mode. "
                "Please train using Research Accuracy Mode before using the system for final prediction."
            ),
        })

    perf_messages = []
    if val_pct is not None and val_pct < ACCURACY_TARGET * 100:
        perf_messages.append(f"validation accuracy is {val_pct:.2f}%")
    if test_pct is not None and test_pct < ACCURACY_TARGET * 100:
        perf_messages.append(f"test accuracy is {test_pct:.2f}%")

    if perf_messages:
        warnings.append({
            "priority": "3",
            "type": "performance",
            "tone": "warning",
            "title": "Model performance below research target",
            "message": (
                f"Latest {' and '.join(perf_messages)} (target ≥ {ACCURACY_TARGET * 100:.0f}%). "
                "Consider research-mode retraining before relying on predictions."
            ),
        })
    elif confidence_pct >= 85 and val_pct is not None and val_pct < 70:
        warnings.append({
            "priority": "3",
            "type": "performance",
            "tone": "warning",
            "title": "High confidence, low model validation",
            "message": (
                f"Although this prediction confidence is {confidence_pct:.2f}%, the model's validation "
                f"accuracy is only {val_pct:.2f}%. Treat this result with caution."
            ),
        })

    return warnings


def _recommended_actions(
    confidence_pct: float,
    margin_pct: float,
    requires_manual_review: bool,
    is_demo: bool,
    val_acc: Optional[float],
    filename_mismatch: bool = False,
    expected_display: Optional[str] = None,
    predicted_display: str = "",
) -> List[str]:
    if filename_mismatch:
        return _mismatch_recommended_actions(expected_display, predicted_display)

    research_ready = (
        confidence_pct >= 85
        and margin_pct >= 50
        and not is_demo
        and val_acc is not None
        and val_acc >= ACCURACY_TARGET
    )

    if research_ready:
        return [
            "Save and keep this result in classification history.",
            "Continue classification on additional samples for comparison.",
            "Use the prediction as a strong candidate result in your research notes.",
        ]

    actions = [
        "Review the uploaded image manually and compare it with the predicted class.",
        "Run model evaluation to inspect per-class performance and confusion patterns.",
        "Retrain the model using Research Accuracy Mode with sufficient epochs.",
        "Improve dataset balance and review possible mislabeled samples.",
    ]
    if is_demo:
        actions.insert(2, "Switch from Fast/Demo training to Research Accuracy Mode for reliable predictions.")
    if margin_pct < 15:
        actions.insert(1, "Inspect the top-two classes carefully because their scores are close.")
    if confidence_pct < 70:
        actions.insert(0, "Upload a clearer image or verify that preprocessing matches training conditions.")
    return actions[:6]


def _load_evaluation_context() -> Dict[str, Any]:
    path = current_app.config["EVALUATION_RESULT_PATH"]
    if not os.path.isfile(path):
        return {"available": False}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        config = data.get("config", {})
        return {
            "available": True,
            "accuracy": data.get("accuracy"),
            "test_accuracy": data.get("accuracy"),
            "precision": data.get("precision"),
            "recall": data.get("recall"),
            "f1_score": data.get("f1_score"),
            "f1_macro": data.get("f1_macro"),
            "evaluated_at": config.get("evaluated_at") or data.get("evaluated_at"),
        }
    except (json.JSONDecodeError, OSError):
        return {"available": False}


def build_image_metadata(image_abs_path: str) -> Dict[str, Any]:
    meta = {
        "exists": False,
        "width": None,
        "height": None,
        "format": None,
        "size_bytes": None,
        "size_display": None,
    }
    if not image_abs_path or not os.path.isfile(image_abs_path):
        return meta

    meta["exists"] = True
    meta["size_bytes"] = os.path.getsize(image_abs_path)
    size = meta["size_bytes"]
    if size < 1024:
        meta["size_display"] = f"{size} B"
    elif size < 1024 * 1024:
        meta["size_display"] = f"{size / 1024:.1f} KB"
    else:
        meta["size_display"] = f"{size / (1024 * 1024):.2f} MB"

    ext = os.path.splitext(image_abs_path)[1].lower().lstrip(".")
    meta["format"] = ext.upper() if ext else "Unknown"

    try:
        from PIL import Image

        with Image.open(image_abs_path) as img:
            meta["width"], meta["height"] = img.size
            if img.format:
                meta["format"] = img.format
    except Exception:
        pass
    return meta


def build_result_analysis(result_row, saved: bool = True, tta_used: bool = False) -> Dict[str, Any]:
    class_index_map = {"naskhi": 0, "diwani": 1, "diwani_jali": 2, "tsuluts": 3}
    try:
        path = current_app.config["CLASS_INDICES_PATH"]
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            indices = data.get("class_indices", data)
            class_index_map = {k: int(v) for k, v in indices.items()}
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        pass

    scores = {
        "naskhi": float(result_row.naskhi_score or 0),
        "diwani": float(result_row.diwani_score or 0),
        "diwani_jali": float(result_row.diwani_jali_score or 0),
        "tsuluts": float(result_row.tsuluts_score or 0),
    }

    sorted_probs = sorted(
        [
            {
                "key": key,
                "label": display_name(key),
                "index": class_index_map.get(key, -1),
                "probability": value,
                "percent": round(value * 100, 2),
                "color": CLASS_COLORS.get(key, {}).get("bar", ""),
                "hex": CLASS_COLORS.get(key, {}).get("hex", "#64748b"),
            }
            for key, value in scores.items()
        ],
        key=lambda item: item["probability"],
        reverse=True,
    )
    for rank, item in enumerate(sorted_probs, start=1):
        item["rank"] = rank

    filename_key, filename_expected_display = parse_filename_class(result_row.filename or "")
    predicted_key = result_row.predicted_class
    manual_expected = getattr(result_row, "manual_expected_class", None)

    top = sorted_probs[0]
    second = sorted_probs[1] if len(sorted_probs) > 1 else None
    third = sorted_probs[2] if len(sorted_probs) > 2 else None
    fourth = sorted_probs[3] if len(sorted_probs) > 3 else None

    top_confidence_pct = top["percent"]
    second_confidence_pct = second["percent"] if second else None
    third_confidence_pct = third["percent"] if third else None
    fourth_confidence_pct = fourth["percent"] if fourth else None
    margin_difference = round(top_confidence_pct - (second_confidence_pct or 0), 2)

    validation_fields = compute_validation_fields(
        result_row.uploaded_filename or result_row.filename or "",
        predicted_key,
        result_row.confidence_score if result_row.confidence_score is not None else result_row.confidence,
        margin_pct=margin_difference,
        top_2_class=second["key"] if second else None,
        top_2_score=second["probability"] if second else None,
        manual_expected_class=manual_expected,
    )

    stage1_resolved = resolve_detection(float(result_row.khat_probability or 0))
    validation_fields = apply_stage1_combined_decision(
        validation_fields,
        detection_status=result_row.detection_status or result_row.input_status,
        input_status=result_row.input_status,
        detection_explanation=stage1_resolved.get("explanation_text"),
    )
    if saved:
        if result_row.review_status:
            validation_fields["review_status"] = result_row.review_status
        if result_row.final_decision:
            validation_fields["final_decision"] = result_row.final_decision
        if result_row.manual_review_required is not None:
            validation_fields["manual_review_required"] = bool(result_row.manual_review_required)
        if getattr(result_row, "known_class_status", None):
            validation_fields["known_class_status"] = result_row.known_class_status
        if getattr(result_row, "validation_status", None) == "User Corrected Label":
            validation_fields["validation_status"] = result_row.validation_status

    expected_key = validation_fields.get("expected_class")
    expected_display = validation_fields.get("expected_class_display")
    has_expected_class = bool(expected_key)
    filename_mismatch = validation_fields.get("validation_status") == "Prediction Mismatch"
    prediction_mismatch = filename_mismatch

    for item in sorted_probs:
        item["is_predicted"] = item["key"] == predicted_key
        item["is_expected"] = item["key"] == expected_key
        item["is_match"] = item["is_predicted"] and item["is_expected"]
        if prediction_mismatch and item["is_predicted"]:
            item["highlight"] = "mismatch"
        elif item["is_match"]:
            item["highlight"] = "match"
        elif item["is_expected"]:
            item["highlight"] = "expected"
        elif item["is_predicted"]:
            item["highlight"] = "predicted"
        else:
            item["highlight"] = ""

    expected_rank = next((item["rank"] for item in sorted_probs if item["key"] == expected_key), None)
    expected_not_top = bool(expected_key and expected_rank and expected_rank > 1)

    reliability = _reliability_from_confidence(
        top_confidence_pct,
        margin_difference,
        prediction_match=validation_fields["prediction_match"],
        has_expected_class=has_expected_class,
        predicted_display=display_name(predicted_key),
        expected_display=expected_display,
    )
    margin_info = _margin_analysis(margin_difference, filename_mismatch=prediction_mismatch)

    model_ctx = get_model_context()
    model_readiness = _model_readiness(model_ctx)
    eval_ctx = _load_evaluation_context()

    val_acc = model_ctx.get("validation_accuracy")
    test_acc = eval_ctx.get("test_accuracy") or model_ctx.get("test_accuracy")
    is_demo = model_ctx.get("is_demo_model", False)
    is_research = model_ctx.get("is_research_model", False)

    stage1_status = str(
        result_row.detection_status
        or result_row.input_status
        or ""
    ).strip().lower()

    stage1_requires_manual_review = stage1_status in {
        "moderate_khat",
        "uncertain_khat",
        "borderline_khat",
    }

    requires_manual_review = bool(
        validation_fields.get("manual_review_required")
        or prediction_mismatch
        or stage1_requires_manual_review
        or top_confidence_pct < 70
    )

    if requires_manual_review:
        validation_fields["manual_review_required"] = True
        validation_fields["review_status"] = "Manual Review Required"

        if validation_fields.get("final_decision") in {
            None,
            "",
            "Accepted",
            "Accepted with Caution",
        }:
            validation_fields["final_decision"] = (
                "Manual Review Required"
            )

    review_badge_text = validation_fields["review_status"]

    is_final_research_model = is_research and not is_demo
    not_final_model_message = None
    if is_demo:
        not_final_model_message = "This is not a final research model. Demo/Fast training is for testing only."
    elif not is_research:
        not_final_model_message = "This is not a final research model."

    quality_status = _quality_status(
        top_confidence_pct, margin_difference, val_acc, is_demo, is_research, prediction_mismatch
    )

    warnings = _build_warnings(
        top_confidence_pct,
        margin_difference,
        margin_info,
        requires_manual_review,
        prediction_mismatch,
        expected_display,
        display_name(predicted_key),
        model_readiness,
        is_demo,
        val_acc,
        test_acc,
    )

    similarity_ctx = build_similarity_context(result_row)
    if similarity_ctx.get("available") and similarity_ctx.get("message"):
        score = float(similarity_ctx.get("similarity_score") or 0)
        if score >= 85:
            warnings.append(
                {
                    "priority": "2",
                    "type": "similarity",
                    "tone": similarity_ctx.get("tone", "warn"),
                    "title": "Training data similarity",
                    "message": similarity_ctx["message"],
                }
            )

    explanation_text = (
        result_row.explanation_text
        or validation_fields.get("explanation_text")
        or _build_prediction_summary(
            top["label"],
            expected_display,
            top_confidence_pct,
            margin_difference,
            is_demo,
            validation_fields["prediction_match"],
            has_expected_class,
        )
    )

    prediction_summary = explanation_text

    active_model_path = model_ctx.get("active_model_path") or current_app.config["MODEL_PATH"]
    image_abs = os.path.join(current_app.config["BASE_DIR"], result_row.image_path)
    image_meta = build_image_metadata(image_abs)

    close_competitor = margin_difference < 10 and second is not None
    arch_key = model_ctx.get("model_architecture_key", "efficientnetb0")
    arch_label = model_ctx.get("model_architecture", "EfficientNetB0")
    preprocessing_mode = getattr(result_row, "preprocessing_mode", None) or "standard"
    if arch_key == "teachable_machine":
        image_size = model_ctx.get("teachable_machine_image_size", 224)
        preprocessing_steps = [
            "EXIF orientation correction",
            "Convert to RGB",
            f"Resize to {image_size} × {image_size} pixels",
            "Normalize pixel values to [-1, 1] (Teachable Machine standard)",
        ]
    else:
        preprocessing_steps = [
            "EXIF orientation correction",
            "Convert to RGB",
            "Trim excessive white borders",
            "Resize with aspect-ratio preservation",
            "Pad to 224 × 224 square canvas",
            f"{arch_label} preprocess_input normalization",
        ]
    if tta_used:
        preprocessing_steps.append("Test-Time Augmentation averaging (5 variants)")
    if preprocessing_mode == "manuscript":
        preprocessing_steps.insert(1, "Manuscript mode: aggressive border trim and content-area focus")

    model_short = get_model_short_name(model_ctx)
    tm_model_info = get_tm_display_info() if is_teachable_machine_active(model_ctx) else None
    model_display = (
        tm_model_info["model_architecture"]
        if tm_model_info
        else get_model_display_label(model_ctx)
    )
    show_actions_card = prediction_mismatch or top_confidence_pct < 70 or validation_fields.get("is_external_image")
    dataset_bias_warning = build_dataset_bias_warning()
    caution_note = (
        "Result blocked because Stage 1 Khat detection failed (non-calligraphy / out of domain)."
        if stage1_resolved.get("detection_status") in ("borderline_khat", "uncertain_khat", "uncertain", "non_khat")
        or (result_row.input_status or "") in ("non_khat", "uncertain", "unrecognized")
        else (
            "Result accepted with caution because confidence is moderate."
            if validation_fields["prediction_match"] and 70 <= top_confidence_pct < 85
            else None
        )
    )

    return {
        "sorted_probabilities": sorted_probs,
        "predicted_class": predicted_key,
        "predicted_display_name": top["label"],
        "top_class": top["key"],
        "top_class_display": top["label"],
        "top_confidence": top["probability"],
        "top_confidence_pct": top_confidence_pct,
        "second_class": second["key"] if second else None,
        "second_class_display": second["label"] if second else None,
        "second_confidence": second["probability"] if second else None,
        "second_confidence_pct": second_confidence_pct,
        "third_class": third["key"] if third else None,
        "third_class_display": third["label"] if third else None,
        "third_confidence": third["probability"] if third else None,
        "third_confidence_pct": third_confidence_pct,
        "fourth_class": fourth["key"] if fourth else None,
        "fourth_class_display": fourth["label"] if fourth else None,
        "fourth_confidence": fourth["probability"] if fourth else None,
        "fourth_confidence_pct": fourth_confidence_pct,
        "confidence_margin_top2": margin_difference,
        "margin_difference": margin_difference,
        "margin_level": margin_info["level"],
        "margin_interpretation": margin_info["interpretation"],
        "reliability_level": reliability["level"],
        "reliability_tone": reliability["tone"],
        "reliability_color": reliability["color"],
        "reliability_message": reliability["message"],
        "reliability_sub_message": reliability.get("sub_message"),
        "is_low_confidence": top_confidence_pct < 70,
        "is_ambiguous": margin_info["is_high_ambiguity"],
        "is_moderate_ambiguity": margin_info["is_moderate_ambiguity"],
        "close_competitor": close_competitor,
        "requires_manual_review": requires_manual_review,
        "manual_review_label": review_badge_text,
        "review_badge_text": review_badge_text,
        "filename_expected_class": filename_key,
        "filename_expected_display": filename_expected_display,
        "filename_mismatch": prediction_mismatch,
        "prediction_mismatch": prediction_mismatch,
        "has_expected_class": has_expected_class,
        "expected_source": validation_fields.get("expected_source"),
        "manual_expected_class": manual_expected,
        "manual_expected_display": display_name(manual_expected) if manual_expected else None,
        "is_external_image": validation_fields.get("is_external_image"),
        "external_warning": validation_fields.get("external_warning"),
        "no_comparator_message": validation_fields.get("no_comparator_message"),
        "known_class_status": validation_fields.get("known_class_status"),
        "misclassification_explanation": validation_fields.get("misclassification_explanation"),
        "dataset_bias_warning": dataset_bias_warning,
        "preprocessing_mode": preprocessing_mode,
        "source_type": getattr(result_row, "source_type", None) or validation_fields.get("expected_source"),
        "correction_label": getattr(result_row, "correction_label", None),
        "correction_notes": getattr(result_row, "correction_notes", None),
        "show_mismatch_card": prediction_mismatch and has_expected_class,
        "filename_mismatch_message": (
            "Prediksi Tidak Sesuai Label Pembanding — Manual Review Required."
            if prediction_mismatch and expected_display
            else None
        ),
        "validation_status": validation_fields["validation_status"],
        "review_status": validation_fields["review_status"],
        "confidence_label": validation_fields.get("confidence_label"),
        "expected_class": result_row.expected_class or validation_fields["expected_class"],
        "expected_class_display": display_name(result_row.expected_class or validation_fields["expected_class"]),
        "prediction_match": validation_fields["prediction_match"],
        "confidence_interpretation": validation_fields.get("confidence_interpretation"),
        "confidence_interpretation_tone": validation_fields["confidence_tone"],
        "model_confidence_label": validation_fields.get("model_confidence_label"),
        "final_decision": validation_fields.get("final_decision"),
        "explanation_text": explanation_text,
        "expected_rank": expected_rank,
        "expected_not_top": expected_not_top,
        "model_display_name": model_display,
        "model_short_name": model_short,
        "tm_model_info": tm_model_info,
        "active_prediction_engine": (
            "Teachable Machine TensorFlow.js"
            if is_teachable_machine_active(model_ctx)
            else f"Keras ({model_short})"
        ),
        "saved_label": (
            "Saved for Review" if prediction_mismatch and saved
            else f"Saved to History #{result_row.id}" if saved
            else None
        ),
        "show_recommended_actions_card": show_actions_card,
        "caution_note": caution_note,
        "prediction_quality_summary": prediction_summary,
        "interpretation_text": prediction_summary,
        "recommended_actions": _recommended_actions(
            top_confidence_pct,
            margin_difference,
            requires_manual_review,
            is_demo,
            val_acc,
            filename_mismatch=prediction_mismatch,
            expected_display=expected_display,
            predicted_display=display_name(predicted_key),
        ),
        "warnings": warnings,
        "quality_warnings": [w["message"] for w in warnings],
        "quality_status": quality_status,
        "model_readiness": model_readiness,
        "model_mode": model_readiness["label"],
        "model_mode_badge": model_readiness["badge"],
        "model_mode_tone": model_readiness["tone"],
        "latest_validation_accuracy": val_acc,
        "latest_validation_accuracy_pct": round(val_acc * 100, 2) if val_acc is not None else None,
        "latest_test_accuracy": test_acc,
        "latest_test_accuracy_pct": round(test_acc * 100, 2) if test_acc is not None else None,
        "top2_comparison_message": margin_info["interpretation"],
        "ambiguous_message": (
            "The model is considering more than one visually similar class."
            if margin_info["is_high_ambiguity"]
            else None
        ),
        "saved_to_history": saved,
        "result_id": result_row.id,
        "tta_used": tta_used,
        "image_meta": image_meta,
        "model_exists": model_ctx.get("model_exists", False),
        "using_best_model": model_ctx.get("using_best_model", False),
        "model_name": (
            tm_model_info["model_name"]
            if tm_model_info
            else model_short
        ),
        "model_source": (
            result_row.model_source
            or (tm_model_info["model_source"] if tm_model_info else "Keras Transfer Learning")
        ),
        "model_runtime": (
            result_row.model_runtime
            or (tm_model_info["model_runtime"] if tm_model_info else "TensorFlow Keras")
        ),
        "model_input_size": (
            result_row.model_input_size
            or (tm_model_info["model_input_size"] if tm_model_info else "224x224")
        ),
        "model_path": active_model_path,
        "model_architecture": arch_label,
        "model_architecture_key": model_ctx.get("model_architecture_key", "vgg16"),
        "training_mode_key": model_ctx.get("training_mode_key"),
        "training_date": model_ctx.get("training_date"),
        "evaluation": eval_ctx,
        "training": {
            "training_mode": model_ctx.get("training_mode_label"),
            "training_mode_key": model_ctx.get("training_mode_key"),
            "validation_accuracy": val_acc,
            "training_date": model_ctx.get("training_date"),
        },
        "model_context": model_ctx,
        "preprocessing_steps": preprocessing_steps,
        "class_index_map": class_index_map,
        "show_retrain_research": is_demo or (val_acc is not None and val_acc < ACCURACY_TARGET),
        "is_final_research_model": is_final_research_model,
        "not_final_model_message": not_final_model_message,
        "below_target_message": (
            "Model performance is below target. Many predictions may still be wrong."
            if (test_acc is not None and test_acc < ACCURACY_TARGET)
            or (val_acc is not None and val_acc < ACCURACY_TARGET)
            else None
        ),
        "khat_characteristics": build_characteristics_context(
            predicted_key,
            expected_class=result_row.expected_class or validation_fields["expected_class"],
            validation_status=validation_fields["validation_status"],
            filename_mismatch=prediction_mismatch,
        ),
        "similarity": similarity_ctx,
    }
