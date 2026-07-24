"""Detailed algorithm calculation breakdown for classification transparency."""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import current_app

from services.khat_detector_service import get_thresholds, is_detector_available
from services.model_context_service import get_model_context, is_teachable_machine_active
from services.result_interpretation_service import (
    CLASS_DISPLAY,
    build_image_metadata,
    build_result_analysis,
    display_name,
    resolve_expected_class,
)
from services.teachable_machine_service import get_tm_display_info


def _confidence_pct(confidence: Optional[float]) -> float:
    if confidence is None:
        return 0.0
    return confidence * 100 if confidence <= 1 else float(confidence)


def _margin_pct(margin: Optional[float]) -> float:
    if margin is None:
        return 0.0
    return margin * 100 if margin <= 1 else float(margin)


def _parse_softmax_scores_json(raw: Any) -> Dict[str, float]:
    if not raw:
        return {}
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    scores: Dict[str, float] = {}
    for key, value in data.items():
        if value is None:
            continue
        try:
            scores[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return scores


def _format_softmax_display(value: float) -> Dict[str, str]:
    v = max(float(value or 0), 0.0)
    pct = v * 100
    if v >= 0.0001:
        score_display = f"{v:.4f}"
    elif v > 0:
        score_display = f"{v:.2e}"
    else:
        score_display = "0.0000"
    if pct >= 0.01:
        percent_display = f"{pct:.2f}%"
    elif v > 0:
        percent_display = "< 0.01%"
    else:
        percent_display = "0.00%"
    return {"softmax_score_display": score_display, "percent_display": percent_display}


def algorithm_confidence_category(confidence_pct: float) -> Dict[str, str]:
    if confidence_pct >= 85:
        return {"label": "Strong Prediction", "tone": "success"}
    if confidence_pct >= 70:
        return {"label": "Moderate Prediction", "tone": "info"}
    if confidence_pct >= 60:
        return {"label": "Weak Prediction", "tone": "warning"}
    return {"label": "Low Confidence", "tone": "danger"}


def algorithm_known_class_status(
    confidence_pct: float,
    margin_pct: float,
    has_expected_class: bool = False,
    prediction_match: bool = True,
) -> str:
    if has_expected_class and not prediction_match:
        return "Needs Review — Label Mismatch"
    if confidence_pct >= 85 and margin_pct >= 15:
        return "Known Class"
    if confidence_pct >= 70 and margin_pct >= 10:
        return "Likely Known Class"
    if confidence_pct >= 60:
        return "Uncertain Class"
    if confidence_pct < 60 or margin_pct < 10:
        return "Possible Unknown Class"
    return "Uncertain Class"


def algorithm_margin_interpretation(margin_pct: float) -> Dict[str, str]:
    if margin_pct >= 15:
        return {
            "level": "strong",
            "label": "Relatively strong prediction",
            "explanation": "A large margin indicates that the model strongly favors one class.",
        }
    if margin_pct >= 10:
        return {
            "level": "moderate",
            "label": "Moderate prediction",
            "explanation": "The model favors the top class, but separation from the second class is moderate.",
        }
    return {
        "level": "ambiguous",
        "label": "Ambiguous prediction",
        "explanation": "A small margin indicates that the model is uncertain between two classes and manual review is recommended.",
    }


def algorithm_final_decision(
    predicted_class: Optional[str],
    expected_class: Optional[str],
    confidence_pct: float,
    margin_pct: float,
    input_status: str = "khat",
    detection_status: Optional[str] = None,
    validation_status: Optional[str] = None,
) -> Dict[str, str]:
    if input_status == "non_khat" or detection_status == "non_khat":
        return {
            "decision": "Rejected as Non-Khat",
            "reason": "Stage 1 detected the input as non-Arabic Khat. Classification was rejected or blocked.",
        }

    has_expected = bool(expected_class)
    prediction_match = bool(expected_class and predicted_class and expected_class == predicted_class)
    is_uncertain_stage1 = detection_status in ("borderline_khat", "uncertain_khat", "uncertain")

    if has_expected and not prediction_match:
        return {
            "decision": "Manual Review Required",
            "reason": "The predicted class does not match the expected class. Manual review is required even when confidence is high.",
        }

    if is_uncertain_stage1:
        return {
            "decision": "Manual Review Required",
            "reason": "Stage 1 Khat detection is borderline or uncertain. The result should be reviewed manually.",
        }

    if confidence_pct < 60 or margin_pct < 10:
        if not has_expected:
            return {
                "decision": "Manual Review Recommended",
                "reason": "Confidence is weak or the top-two margin is small, and no reference label is available for automatic validation.",
            }
        return {
            "decision": "Manual Review Recommended",
            "reason": "Confidence is weak or the top-two margin is small. Manual verification is recommended.",
        }

    if prediction_match and confidence_pct >= 85 and margin_pct >= 15:
        return {
            "decision": "Accepted",
            "reason": "The predicted class matches the expected class with strong confidence and a clear top-two margin.",
        }

    if not has_expected:
        if confidence_pct >= 70:
            return {
                "decision": "Accepted with Caution",
                "reason": "The model produced a moderate or strong confidence score, but no expected class is available to automatically validate the result.",
            }
        return {
            "decision": "Manual Review Recommended",
            "reason": "No expected class is available and confidence is below the recommended threshold.",
        }

    if prediction_match and confidence_pct >= 70:
        return {
            "decision": "Accepted with Caution",
            "reason": "The predicted class matches the expected class, but confidence or margin suggests cautious interpretation.",
        }

    if 70 <= confidence_pct < 85 and not has_expected:
        return {
            "decision": "Review Optional",
            "reason": "Confidence is moderate, the expected class is unknown, and no serious issue was detected.",
        }

    return {
        "decision": "Manual Review Recommended",
        "reason": "The result should be reviewed before use in research or reporting.",
    }


def build_indonesian_interpretation(
    predicted_display: str,
    confidence_pct: float,
    margin_pct: float,
    second_display: Optional[str],
    confidence_label: str,
    final_decision: str,
    has_expected: bool,
    prediction_match: bool,
    expected_display: Optional[str] = None,
) -> str:
    if has_expected and not prediction_match:
        return (
            f"Walaupun confidence tinggi ({confidence_pct:.2f}%), hasil prediksi ({predicted_display}) "
            f"berbeda dari label pembanding ({expected_display or 'referensi'}). "
            f"Oleh karena itu, hasil tidak boleh dianggap otomatis valid dan keputusan akhir adalah Manual Review Required."
        )

    threshold_note = ""
    if confidence_pct >= 85:
        threshold_note = f"Nilai ini termasuk {confidence_label} karena berada di atas ambang 85%."
    elif confidence_pct >= 70:
        threshold_note = f"Nilai ini termasuk {confidence_label} (70%–84%)."
    elif confidence_pct >= 60:
        threshold_note = f"Nilai ini termasuk {confidence_label} (60%–69%)."
    else:
        threshold_note = f"Nilai ini termasuk {confidence_label} (di bawah 60%)."

    margin_text = (
        f"Top-2 margin sebesar {margin_pct:.2f}% menunjukkan bahwa prediksi {predicted_display} "
        f"jauh lebih dominan dibanding kelas kedua ({second_display})."
        if second_display and margin_pct >= 15
        else (
            f"Top-2 margin sebesar {margin_pct:.2f}% menunjukkan ada tumpang tindih antara "
            f"{predicted_display} dan {second_display or 'kelas kedua'}."
            if second_display
            else f"Top-2 margin adalah {margin_pct:.2f}%."
        )
    )

    caution = ""
    if not has_expected and "Caution" in final_decision:
        caution = (
            " Namun, karena label pembanding tidak tersedia, hasil diterima dengan catatan "
            "dan dapat direview manual."
        )
    elif final_decision == "Manual Review Recommended":
        caution = " Review manual disarankan sebelum menggunakan hasil ini."
    elif final_decision == "Manual Review Required":
        caution = " Keputusan akhir memerlukan review manual."

    return (
        f"Model memprediksi gambar sebagai {predicted_display} dengan confidence {confidence_pct:.2f}%. "
        f"{threshold_note} {margin_text}{caution}"
    ).strip()


def _confidence_threshold_line(confidence_pct: float, label: str) -> str:
    if label == "Strong Prediction":
        return f"{confidence_pct:.2f}% >= 85%"
    if label == "Moderate Prediction":
        return f"70% <= {confidence_pct:.2f}% < 85%"
    if label == "Weak Prediction":
        return f"60% <= {confidence_pct:.2f}% < 70%"
    return f"{confidence_pct:.2f}% < 60%"


def build_academic_interpretation(
    predicted_display: str,
    confidence_pct: float,
    margin_pct: float,
    second_display: Optional[str],
    confidence_label: str,
    final_decision: str,
    has_expected: bool,
    prediction_match: bool,
    expected_display: Optional[str] = None,
) -> str:
    if has_expected and not prediction_match:
        return (
            f"Although the confidence score is {confidence_pct:.2f}%, the predicted class ({predicted_display}) "
            f"does not match the expected class ({expected_display or 'reference label'}). "
            f"Therefore, the final decision is Manual Review Required."
        )

    margin_text = (
        f"The Top-2 margin of {margin_pct:.2f}% indicates that {predicted_display} is much more dominant than "
        f"the second-ranked class, {second_display}."
        if second_display and margin_pct >= 15
        else (
            f"The Top-2 margin of {margin_pct:.2f}% suggests some overlap between {predicted_display} "
            f"and {second_display or 'the second-ranked class'}."
            if second_display
            else f"The Top-2 margin is {margin_pct:.2f}%."
        )
    )

    threshold_note = ""
    if confidence_pct >= 85:
        threshold_note = "This score is categorized as a Strong Prediction because it is above the 85% threshold."
    elif confidence_pct >= 70:
        threshold_note = "This score is categorized as a Moderate Prediction (70%–84%)."
    elif confidence_pct >= 60:
        threshold_note = "This score is categorized as a Weak Prediction (60%–69%)."
    else:
        threshold_note = "This score is categorized as Low Confidence (below 60%)."

    caution = ""
    if not has_expected and "Caution" in final_decision:
        caution = " However, because no expected class is available, the result is accepted with caution and may still be reviewed manually."
    elif final_decision == "Manual Review Recommended":
        caution = " Manual review is recommended before treating this result as final."
    elif "Unknown" in final_decision or margin_pct < 10:
        caution = (
            " The image may be outside the trained classes or may have visual characteristics "
            "that are not strongly represented in the training dataset."
        )

    return (
        f"The model predicts this image as {predicted_display} with a confidence score of {confidence_pct:.2f}%. "
        f"{threshold_note} {margin_text}{caution}"
    ).strip()


def _input_source_label(source: Optional[str]) -> str:
    mapping = {
        "upload": "Upload",
        "dataset": "Dataset",
        "external_url": "External URL",
        "external": "External URL",
    }
    return mapping.get((source or "upload").lower(), "Upload")


def _expected_source_label(source: Optional[str]) -> str:
    mapping = {
        "manual": "Manual Expected Class",
        "filename": "Filename Prefix",
        "dataset": "Dataset Label",
        "external": "Unknown",
        "none": "Unknown",
    }
    return mapping.get((source or "external").lower(), "Unknown")


def _scores_from_row(result_row) -> Dict[str, float]:
    labels = list(current_app.config.get("CLASS_LABELS", []))
    scores = _parse_softmax_scores_json(getattr(result_row, "softmax_scores_json", None))
    if scores:
        return {label: float(scores.get(label, 0) or 0) for label in labels}

    legacy = {
        "naskhi": float(result_row.naskhi_score or result_row.probability_naskhi or 0),
        "diwani": float(result_row.diwani_score or result_row.probability_diwani or 0),
        "diwani_jali": float(result_row.diwani_jali_score or result_row.probability_diwani_jali or 0),
        "tsuluts": float(result_row.tsuluts_score or result_row.probability_tsuluts or 0),
    }
    if any(legacy.values()):
        return legacy

    if result_row.confidence is not None and result_row.predicted_class:
        scores = {label: 0.0 for label in labels}
        scores[result_row.predicted_class] = float(result_row.confidence)
        if result_row.top_2_class and result_row.top_2_score is not None:
            scores[result_row.top_2_class] = float(result_row.top_2_score)
        return scores
    return legacy


def build_algorithm_calculation(
    result_row,
    analysis: Optional[Dict[str, Any]] = None,
    detection: Optional[Dict[str, Any]] = None,
    tta_used: bool = False,
    input_source: Optional[str] = None,
) -> Dict[str, Any]:
    if analysis is None:
        analysis = build_result_analysis(result_row, saved=True, tta_used=tta_used)

    image_abs = os.path.join(current_app.config["BASE_DIR"], result_row.image_path or "")
    image_meta = analysis.get("image_meta") or build_image_metadata(image_abs)

    scores = _scores_from_row(result_row)
    sorted_probs = []
    for key, value in scores.items():
        display = _format_softmax_display(value)
        sorted_probs.append(
            {
                "key": key,
                "label": display_name(key),
                "softmax_score": round(value, 4),
                "percent": round(value * 100, 2),
                **display,
            }
        )
    sorted_probs.sort(key=lambda item: item["softmax_score"], reverse=True)
    for rank, item in enumerate(sorted_probs, start=1):
        item["rank"] = rank

    top = sorted_probs[0]
    softmax_peaked = top["softmax_score"] >= 0.99
    second = sorted_probs[1] if len(sorted_probs) > 1 else None
    confidence_pct = top["percent"]
    margin_pct = round(confidence_pct - (second["percent"] if second else 0), 2)

    expected_key = result_row.expected_class or analysis.get("expected_class")
    expected_display = display_name(expected_key) if expected_key else "Unknown"
    expected_source = result_row.source_type or analysis.get("expected_source") or "external"
    has_expected = bool(expected_key)
    prediction_match = bool(
        expected_key and result_row.predicted_class and expected_key == result_row.predicted_class
    )
    validation_status = result_row.validation_status or analysis.get("validation_status") or (
        "Prediction Match" if prediction_match else (
            "Prediction Mismatch" if has_expected else "No Expected Class Provided"
        )
    )

    conf_cat = algorithm_confidence_category(confidence_pct)
    margin_info = algorithm_margin_interpretation(margin_pct)
    known_status = algorithm_known_class_status(
        confidence_pct, margin_pct, has_expected_class=has_expected, prediction_match=prediction_match
    )

    input_status = result_row.input_status or "khat"
    detection_status = result_row.detection_status or input_status
    final = algorithm_final_decision(
        result_row.predicted_class,
        expected_key,
        confidence_pct,
        margin_pct,
        input_status=input_status,
        detection_status=detection_status,
        validation_status=validation_status,
    )
    if result_row.final_decision:
        final["decision"] = result_row.final_decision

    model_ctx = get_model_context()
    tm_info = get_tm_display_info() if is_teachable_machine_active(model_ctx) else None
    model_input_size = (
        result_row.model_input_size
        or (tm_info.get("model_input_size") if tm_info else "224x224")
        or "224x224"
    )
    model_input_display = model_input_size.replace("x", " × ") + " px"
    processed_size = model_input_display

    preprocessing_mode = getattr(result_row, "preprocessing_mode", None) or analysis.get("preprocessing_mode") or "standard"
    preprocessing_steps = analysis.get("preprocessing_steps") or []
    if getattr(result_row, "preprocessing_steps_json", None):
        try:
            stored_steps = json.loads(result_row.preprocessing_steps_json)
            if isinstance(stored_steps, list) and stored_steps:
                preprocessing_steps = stored_steps
        except (json.JSONDecodeError, TypeError):
            pass

    thresholds = get_thresholds()
    khat_prob = float(result_row.khat_probability or 0)
    khat_pct = round(khat_prob * 100, 2)
    non_khat_pct = round((1 - khat_prob) * 100, 2) if result_row.khat_probability is not None else None
    detector_available = is_detector_available()

    src = input_source or getattr(result_row, "input_source", None) or "upload"
    created = result_row.created_at.strftime("%Y-%m-%d %H:%M:%S") if result_row.created_at else "—"

    steps: List[Dict[str, Any]] = []

    steps.append({
        "step": 1,
        "key": "image_input",
        "name": "Image Input",
        "status": "completed",
        "explanation": "The image is received by the system as classification input and will be preprocessed before being passed into the model.",
        "fields": [
            {"label": "Filename", "value": result_row.uploaded_filename or result_row.filename or "—"},
            {"label": "Input Source", "value": _input_source_label(src)},
            {"label": "Format", "value": image_meta.get("format") or "—"},
            {
                "label": "Original Size",
                "value": (
                    f"{image_meta['width']} × {image_meta['height']} px"
                    if image_meta.get("width") and image_meta.get("height")
                    else "—"
                ),
            },
            {"label": "File Size", "value": image_meta.get("size_display") or "—"},
            {"label": "Classification Time", "value": created},
        ],
        "result": "Image accepted as classification input.",
    })

    norm_note = "pixel range 0–1 (Teachable Machine)" if tm_info else "architecture-specific normalization"
    preprocess_explanation = (
        f"The image is adjusted to the model input size of {model_input_display} so that it matches the input format required by the Teachable Machine model."
        if tm_info
        else f"The image is adjusted to the model input size of {model_input_display} to match the CNN model requirements."
    )
    steps.append({
        "step": 2,
        "key": "preprocessing",
        "name": "Image Preprocessing",
        "status": "completed",
        "explanation": preprocess_explanation,
        "fields": [
            {
                "label": "Original Size",
                "value": (
                    f"{image_meta['width']} × {image_meta['height']} px"
                    if image_meta.get("width") and image_meta.get("height")
                    else "—"
                ),
            },
            {"label": "Model Input Size", "value": model_input_display},
            {"label": "Color Mode", "value": "RGB"},
            {"label": "Normalization", "value": norm_note},
            {
                "label": "Preprocessing Mode",
                "value": "Document / Manuscript Mode" if preprocessing_mode == "manuscript" else "Standard",
            },
            {"label": "Aspect Ratio Handling", "value": "Contain with padding (Teachable Machine)" if tm_info else "Center crop / pad to square"},
        ],
        "steps_list": preprocessing_steps,
        "result": f"Image preprocessed to {model_input_display}.",
    })

    if detector_available and result_row.khat_probability is not None:
        accept_pct = round(thresholds["accept"] * 100, 2)
        reject_pct = round(thresholds["reject"] * 100, 2)
        borderline_pct = round(thresholds["borderline"] * 100, 2)
        if khat_pct >= accept_pct:
            stage1_status = "Confirmed Khat"
            stage1_result = f"{khat_pct:.2f}% >= {accept_pct:.2f}% — input classified as Confirmed Khat and continues to Khat type classification."
        elif khat_pct >= reject_pct:
            stage1_status = "Uncertain Input"
            stage1_result = f"{khat_pct:.2f}% is between reject and accept thresholds — continue with caution / manual review."
        else:
            stage1_status = "Non-Khat"
            stage1_result = f"{khat_pct:.2f}% < {reject_pct:.2f}% — input classified as Non-Khat."

        steps.append({
            "step": 3,
            "key": "khat_validation",
            "name": "Khat / Non-Khat Validation",
            "status": "completed",
            "explanation": "Stage 1 validates whether the input resembles Arabic Khat calligraphy before type classification.",
            "fields": [
                {"label": "Khat Probability", "value": f"{khat_pct:.2f}%"},
                {"label": "Non-Khat Probability", "value": f"{non_khat_pct:.2f}%" if non_khat_pct is not None else "—"},
                {"label": "Accept Threshold", "value": f"{accept_pct:.2f}%"},
                {"label": "Reject Threshold", "value": f"{reject_pct:.2f}%"},
                {"label": "Borderline Range", "value": f"{borderline_pct:.2f}% – {accept_pct - 0.01:.2f}%"},
                {"label": "Detection Decision", "value": result_row.detection_decision or stage1_status},
            ],
            "formula": "Non-Khat Probability = 100% − Khat Probability",
            "calculation": f"Khat Probability = {khat_pct:.2f}%",
            "result": stage1_result,
            "decision_status": stage1_status,
        })
    else:
        steps.append({
            "step": 3,
            "key": "khat_validation",
            "name": "Khat / Non-Khat Validation",
            "status": "skipped",
            "explanation": "Stage 1 Khat / Non-Khat detection is not available or was not recorded for this prediction.",
            "fields": [],
            "result": "Skipped — detector not available.",
        })

    steps.append({
        "step": 4,
        "key": "tensor_preparation",
        "name": "Tensor Preparation",
        "status": "completed",
        "explanation": "The preprocessed image is converted into a tensor so it can be processed by the TensorFlow.js model.",
        "fields": [
            {"label": "Model Input Size", "value": model_input_display},
            {"label": "Tensor Shape", "value": "[1, 224, 224, 3]"},
            {"label": "Channel", "value": "RGB"},
            {"label": "Runtime", "value": result_row.model_runtime or (tm_info.get("model_runtime") if tm_info else "TensorFlow.js")},
        ],
        "result": "Tensor prepared for model inference.",
    })

    labels = ["diwani", "diwani jali", "naskhi", "tsuluts"]
    steps.append({
        "step": 5,
        "key": "model_prediction",
        "name": "Model Prediction",
        "status": "completed",
        "explanation": "The model generates probability scores for each trained class. The class with the highest probability is selected as the main prediction.",
        "fields": [
            {"label": "Model Source", "value": result_row.model_source or (tm_info.get("model_source") if tm_info else "Teachable Machine")},
            {"label": "Model Name", "value": result_row.model_name or (tm_info.get("model_name") if tm_info else "tm-my-image-model")},
            {"label": "Runtime", "value": result_row.model_runtime or (tm_info.get("model_runtime") if tm_info else "TensorFlow.js")},
            {"label": "Model Type", "value": "Teachable Machine Image Model" if tm_info else "CNN Transfer Learning"},
            {"label": "Labels", "value": ", ".join(labels)},
        ],
        "result": f"Predicted class: {display_name(result_row.predicted_class)}.",
    })

    steps.append({
        "step": 6,
        "key": "softmax",
        "name": "Softmax Probability",
        "status": "completed",
        "explanation": "Softmax converts model output scores into probabilities that sum to 1. Each percentage equals the softmax score × 100.",
        "fields": [],
        "formula": "Probability (%) = Softmax Score × 100",
        "softmax_table": sorted_probs,
        "result": f"Highest probability: {top['label']} ({top['percent']:.2f}%).",
    })

    ranking_lines = [f"Rank {item['rank']}: {item['label']} — {item['percent']:.2f}%" for item in sorted_probs]
    steps.append({
        "step": 7,
        "key": "class_ranking",
        "name": "Class Ranking",
        "status": "completed",
        "explanation": "The class with the highest probability is selected as the predicted class.",
        "ranking": ranking_lines,
        "result": f"Top ranked class: {top['label']}.",
    })

    steps.append({
        "step": 8,
        "key": "top2_margin",
        "name": "Top-2 Margin Calculation",
        "status": "completed",
        "explanation": margin_info["explanation"],
        "fields": [
            {"label": "Top-1 Class", "value": top["label"]},
            {"label": "Top-1 Score", "value": f"{top['percent']:.2f}%"},
            {"label": "Top-2 Class", "value": second["label"] if second else "—"},
            {"label": "Top-2 Score", "value": f"{second['percent']:.2f}%" if second else "—"},
        ],
        "formula": "Top-2 Margin = Top-1 Score − Top-2 Score",
        "calculation": (
            f"Top-2 Margin = {top['percent']:.2f}% − {second['percent']:.2f}% = {margin_pct:.2f}%"
            if second
            else f"Top-2 Margin = {margin_pct:.2f}%"
        ),
        "margin_level": margin_info["label"],
        "result": margin_info["label"],
    })

    steps.append({
        "step": 9,
        "key": "confidence_category",
        "name": "Confidence Category",
        "status": "completed",
        "explanation": "The confidence category is determined from the Top-1 probability score using fixed research thresholds.",
        "fields": [
            {"label": "Confidence Score", "value": f"{confidence_pct:.2f}%"},
            {"label": "Confidence Level", "value": conf_cat["label"]},
        ],
        "result": f"{conf_cat['label']} (confidence {confidence_pct:.2f}%).",
        "badge": conf_cat,
    })

    steps.append({
        "step": 10,
        "key": "expected_validation",
        "name": "Expected Class Validation",
        "status": "completed",
        "explanation": (
            "Because no expected class is available, the prediction result cannot be automatically validated against a reference label."
            if not has_expected
            else "The predicted class is compared against the expected class from the highest-priority available source."
        ),
        "fields": [
            {"label": "Predicted Class", "value": display_name(result_row.predicted_class)},
            {"label": "Expected Class", "value": expected_display},
            {"label": "Expected Class Source", "value": _expected_source_label(expected_source)},
            {"label": "Validation Status", "value": validation_status},
        ],
        "formula": "Validation = Predicted Class == Expected Class",
        "result": validation_status,
    })

    steps.append({
        "step": 11,
        "key": "known_class",
        "name": "Known Class Status",
        "status": "completed",
        "explanation": "Known class status estimates whether the image likely belongs to one of the trained classes based on confidence and margin.",
        "fields": [
            {"label": "Confidence", "value": f"{confidence_pct:.2f}%"},
            {"label": "Top-2 Margin", "value": f"{margin_pct:.2f}%"},
            {"label": "Known Class Status", "value": known_status},
        ],
        "result": known_status,
    })

    steps.append({
        "step": 12,
        "key": "final_decision",
        "name": "Final Decision",
        "status": "completed",
        "explanation": "Final decision combines confidence, top-2 margin, expected class validation, and Stage 1 detection outcome.",
        "fields": [
            {"label": "Final Decision", "value": final["decision"]},
        ],
        "result": final["reason"],
    })

    similarity = analysis.get("similarity") or {}
    predicted_display = display_name(result_row.predicted_class)
    interpretation = build_academic_interpretation(
        predicted_display,
        confidence_pct,
        margin_pct,
        second["label"] if second else None,
        conf_cat["label"],
        final["decision"],
        has_expected,
        prediction_match,
        expected_display if has_expected else None,
    )
    interpretation_id = build_indonesian_interpretation(
        predicted_display,
        confidence_pct,
        margin_pct,
        second["label"] if second else None,
        conf_cat["label"],
        final["decision"],
        has_expected,
        prediction_match,
        expected_display if has_expected else None,
    )

    validation_explanation_id = (
        "Karena tidak ada label pembanding, sistem tidak dapat memvalidasi hasil secara otomatis."
        if not has_expected
        else (
            "Predicted Class sama dengan Expected Class, sehingga validasi otomatis berhasil."
            if prediction_match
            else "Predicted Class tidak sama dengan Expected Class, sehingga hasil harus direview manual."
        )
    )

    final_explanation_id = final["reason"]
    if not has_expected and "Caution" in final["decision"]:
        final_explanation_id = (
            "Model memberikan confidence tinggi, tetapi karena label pembanding tidak tersedia, "
            "hasil diterima dengan catatan."
        )
    elif has_expected and not prediction_match:
        final_explanation_id = (
            "Walaupun confidence tinggi, prediksi tidak sesuai label pembanding. "
            "Keputusan akhir adalah Manual Review Required."
        )

    softmax_with_status = []
    for item in sorted_probs:
        row = dict(item)
        row["status"] = "Top Prediction" if item["rank"] == 1 else ""
        softmax_with_status.append(row)

    pipeline_steps = [
        {
            "step": 1,
            "name": "Input Image",
            "explanation": "Gambar diterima oleh sistem sebagai input klasifikasi.",
            "status": "Completed",
        },
        {
            "step": 2,
            "name": "Preprocessing",
            "explanation": (
                f"Gambar dikonversi ke RGB, diresize ke {model_input_display}, "
                "dan dinormalisasi sebelum masuk ke model."
            ),
            "status": "Completed",
        },
        {
            "step": 3,
            "name": "Tensor Preparation",
            "explanation": "Gambar dipersiapkan sebagai tensor input dengan bentuk [1, 224, 224, 3].",
            "status": "Completed",
        },
        {
            "step": 4,
            "name": "Model Prediction",
            "explanation": "Model Teachable Machine TensorFlow.js menghasilkan probabilitas untuk setiap kelas.",
            "status": "Completed",
        },
        {
            "step": 5,
            "name": "Softmax Output",
            "explanation": "Skor keluaran model diubah menjadi probabilitas softmax yang totalnya 1.",
            "status": "Completed",
        },
        {
            "step": 6,
            "name": "Class Ranking",
            "explanation": "Kelas diurutkan dari probabilitas tertinggi ke terendah.",
            "status": "Completed",
        },
        {
            "step": 7,
            "name": "Margin Calculation",
            "explanation": "Selisih antara skor Top-1 dan Top-2 dihitung sebagai Top-2 Margin.",
            "status": "Completed",
        },
        {
            "step": 8,
            "name": "Validation",
            "explanation": "Prediksi dibandingkan dengan Expected Class jika tersedia.",
            "status": "Completed",
        },
        {
            "step": 9,
            "name": "Final Decision",
            "explanation": "Keputusan akhir ditentukan dari confidence, margin, validasi, dan deteksi Khat.",
            "status": "Completed",
        },
    ]

    summary_grid = [
        {"label": "Predicted Class", "value": predicted_display},
        {"label": "Confidence", "value": f"{confidence_pct:.2f}%"},
        {"label": "Top-2 Class", "value": second["label"] if second else "—"},
        {"label": "Top-2 Score", "value": f"{second['percent']:.2f}%" if second else "—"},
        {"label": "Top-2 Margin", "value": f"{margin_pct:.2f}%"},
        {"label": "Expected Class", "value": expected_display},
        {"label": "Validation Status", "value": validation_status},
        {"label": "Final Decision", "value": final["decision"]},
    ]

    summary_table = [
        {"component": "Input Size", "value": model_input_display, "explanation": "Model input size"},
        {"component": "Predicted Class", "value": display_name(result_row.predicted_class), "explanation": "Class with the highest probability"},
        {"component": "Top-1 Score", "value": f"{top['percent']:.2f}%", "explanation": "Highest probability score"},
        {"component": "Top-2 Class", "value": second["label"] if second else "—", "explanation": "Closest second class"},
        {"component": "Top-2 Score", "value": f"{second['percent']:.2f}%" if second else "—", "explanation": "Second highest probability"},
        {"component": "Top-2 Margin", "value": f"{margin_pct:.2f}%", "explanation": "Difference between Top-1 and Top-2"},
        {"component": "Confidence Level", "value": conf_cat["label"], "explanation": "Based on Top-1 score"},
        {"component": "Expected Class", "value": expected_display, "explanation": _expected_source_label(expected_source)},
        {"component": "Validation Status", "value": validation_status, "explanation": "Automatic validation outcome"},
        {"component": "Known Class Status", "value": known_status, "explanation": "Estimate of trained-class membership"},
        {"component": "Final Decision", "value": final["decision"], "explanation": final["reason"]},
    ]

    return {
        "title": "Detail Perhitungan Algoritma",
        "subtitle": (
            "Rincian proses inferensi model, perhitungan probabilitas, margin, "
            "validasi label, dan keputusan akhir sistem."
        ),
        "technical_honesty_note": (
            "Catatan: Bagian ini menampilkan perhitungan inferensi dan logika keputusan berdasarkan "
            "keluaran model. Sistem tidak menampilkan perhitungan internal layer-by-layer dari "
            "neural network Teachable Machine."
        ),
        "steps": steps,
        "pipeline_steps": pipeline_steps,
        "summary_grid": summary_grid,
        "softmax_table": softmax_with_status,
        "softmax_peaked": softmax_peaked,
        "softmax_peaked_note": (
            "Distribusi softmax sangat tajam: model hampir 100% yakin pada kelas teratas. "
            "Kelas lain bukan benar-benar nol — nilainya sangat kecil (orde 10⁻¹² atau lebih) "
            "sehingga tampilan 4 desimal membulatkannya menjadi 0.0000."
            if softmax_peaked
            else None
        ),
        "ranking": ranking_lines,
        "ranking_items": sorted_probs,
        "summary_table": summary_table,
        "predicted_class": predicted_display,
        "confidence": confidence_pct,
        "top1_class": top["label"],
        "top1_score": top["percent"],
        "top2_class": second["label"] if second else None,
        "top2_score": second["percent"] if second else None,
        "top2_margin": margin_pct,
        "expected_class": expected_display,
        "expected_class_source": _expected_source_label(expected_source),
        "validation_status": validation_status,
        "validation_explanation": validation_explanation_id,
        "confidence_level": conf_cat["label"],
        "confidence_calculation": {
            "confidence_pct": confidence_pct,
            "threshold_line": _confidence_threshold_line(confidence_pct, conf_cat["label"]),
            "category": conf_cat["label"],
            "badge_tone": conf_cat["tone"],
        },
        "margin_calculation": {
            "top1_score": top["percent"],
            "top2_score": second["percent"] if second else 0,
            "margin": margin_pct,
            "formula": "Top-2 Margin = Top-1 Score − Top-2 Score",
            "calculation_line": (
                f"Top-2 Margin = {top['percent']:.2f}% − {second['percent']:.2f}% = {margin_pct:.2f}%"
                if second
                else f"Top-2 Margin = {margin_pct:.2f}%"
            ),
            "interpretation": (
                "Margin yang besar menunjukkan bahwa prediksi utama jauh lebih dominan dibanding kelas kedua. "
                "Margin yang kecil menunjukkan prediksi ambigu."
            ),
        },
        "final_decision_block": {
            "confidence_level": conf_cat["label"],
            "top2_margin": f"{margin_pct:.2f}%",
            "validation_status": validation_status,
            "known_class_status": known_status,
            "final_decision": final["decision"],
            "explanation": final_explanation_id,
        },
        "formulas": [
            {"title": "Probability Percentage", "code": "Probability (%) = Softmax Score × 100"},
            {"title": "Top-2 Margin", "code": "Top-2 Margin = Top-1 Score − Top-2 Score"},
            {"title": "Prediction Validation", "code": "Validation = Predicted Class == Expected Class"},
            {
                "title": "Confidence Category",
                "code": (
                    "if confidence >= 85:\n"
                    "    Strong Prediction\n"
                    "elif confidence >= 70:\n"
                    "    Moderate Prediction\n"
                    "elif confidence >= 60:\n"
                    "    Weak Prediction\n"
                    "else:\n"
                    "    Low Confidence"
                ),
            },
        ],
        "formulas_id": [
            {"title": "Probability Percentage", "code": "Probability (%) = Softmax Score × 100"},
            {"title": "Top-2 Margin", "code": "Top-2 Margin = Top-1 Score − Top-2 Score"},
            {"title": "Validation", "code": "Validation = Predicted Class == Expected Class"},
            {
                "title": "Confidence Category",
                "code": (
                    "if confidence >= 85:\n"
                    "    Strong Prediction\n"
                    "elif confidence >= 70:\n"
                    "    Moderate Prediction\n"
                    "elif confidence >= 60:\n"
                    "    Weak Prediction\n"
                    "else:\n"
                    "    Low Confidence"
                ),
            },
        ],
        "interpretation": interpretation,
        "interpretation_id": interpretation_id,
        "final_decision": final,
        "confidence_badge": conf_cat,
        "known_class_status": known_status,
        "similarity": {
            "available": bool(similarity.get("available")),
            "method": "Perceptual Hash (pHash)" if similarity.get("available") else None,
            "similarity_status": similarity.get("similarity_status"),
            "similarity_score_pct": similarity.get("similarity_score_pct"),
            "nearest_dataset_image": similarity.get("nearest_dataset_image"),
            "nearest_dataset_class": similarity.get("nearest_dataset_class"),
            "risk_level": similarity.get("risk_level"),
            "message": similarity.get("message") or (
                None if similarity.get("available") else "Dataset similarity checking is not enabled."
            ),
        },
    }


def safe_build_algorithm_calculation(
    result_row,
    analysis: Optional[Dict[str, Any]] = None,
    detection: Optional[Dict[str, Any]] = None,
    tta_used: bool = False,
    input_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Build algorithm calculation payload; never raises to callers."""
    try:
        return build_algorithm_calculation(
            result_row,
            analysis=analysis,
            detection=detection,
            tta_used=tta_used,
            input_source=input_source,
        )
    except Exception:
        try:
            current_app.logger.exception("build_algorithm_calculation failed for row %s", getattr(result_row, "id", "?"))
        except RuntimeError:
            pass
        if analysis is None:
            analysis = build_result_analysis(result_row, saved=True, tta_used=tta_used)
        return _fallback_algorithm_calculation(result_row, analysis)


def _fallback_algorithm_calculation(result_row, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal calculation payload from analysis when full build fails."""
    sorted_probs = analysis.get("sorted_probabilities") or []
    top = sorted_probs[0] if sorted_probs else {}
    second = sorted_probs[1] if len(sorted_probs) > 1 else None
    confidence_pct = float(top.get("percent") or 0)
    margin_pct = round(confidence_pct - float(second.get("percent") or 0), 2) if second else 0.0
    expected_key = result_row.expected_class or analysis.get("expected_class")
    expected_display = display_name(expected_key) if expected_key else "Unknown"
    validation_status = analysis.get("validation_status") or "No Expected Class Provided"
    conf_cat = algorithm_confidence_category(confidence_pct)
    final_decision = result_row.final_decision or analysis.get("final_decision") or "Accepted with Caution"
    predicted_display = display_name(result_row.predicted_class)

    softmax_rows = []
    for item in sorted_probs:
        pct = float(item.get("percent") or 0)
        softmax_rows.append({
            "key": item.get("key"),
            "label": item.get("label"),
            "softmax_score": round(pct / 100, 4),
            "percent": pct,
            "rank": item.get("rank"),
            "status": "Top Prediction" if item.get("rank") == 1 else "",
        })

    calc = {
        "title": "Detail Perhitungan Algoritma",
        "subtitle": (
            "Rincian proses inferensi model, perhitungan probabilitas, margin, "
            "validasi label, dan keputusan akhir sistem."
        ),
        "technical_honesty_note": (
            "Catatan: Bagian ini menampilkan perhitungan inferensi dan logika keputusan berdasarkan "
            "keluaran model. Sistem tidak menampilkan perhitungan internal layer-by-layer dari "
            "neural network Teachable Machine."
        ),
        "summary_grid": [
            {"label": "Predicted Class", "value": predicted_display},
            {"label": "Confidence", "value": f"{confidence_pct:.2f}%"},
            {"label": "Top-2 Class", "value": second.get("label") if second else "—"},
            {"label": "Top-2 Score", "value": f"{float(second.get('percent') or 0):.2f}%" if second else "—"},
            {"label": "Top-2 Margin", "value": f"{margin_pct:.2f}%"},
            {"label": "Expected Class", "value": expected_display},
            {"label": "Validation Status", "value": validation_status},
            {"label": "Final Decision", "value": final_decision},
        ],
        "pipeline_steps": [],
        "softmax_table": softmax_rows,
        "ranking_items": softmax_rows,
        "confidence_calculation": {
            "confidence_pct": confidence_pct,
            "threshold_line": _confidence_threshold_line(confidence_pct, conf_cat["label"]),
            "category": conf_cat["label"],
            "badge_tone": conf_cat["tone"],
        },
        "margin_calculation": {
            "top1_score": confidence_pct,
            "top2_score": float(second.get("percent") or 0) if second else 0,
            "margin": margin_pct,
            "formula": "Top-2 Margin = Top-1 Score − Top-2 Score",
            "calculation_line": f"Top-2 Margin = {confidence_pct:.2f}% − {float(second.get('percent') or 0):.2f}% = {margin_pct:.2f}%" if second else f"Top-2 Margin = {margin_pct:.2f}%",
            "interpretation": (
                "Margin yang besar menunjukkan bahwa prediksi utama jauh lebih dominan dibanding kelas kedua. "
                "Margin yang kecil menunjukkan prediksi ambigu."
            ),
        },
        "validation_status": validation_status,
        "validation_explanation": (
            "Karena tidak ada label pembanding, sistem tidak dapat memvalidasi hasil secara otomatis."
            if not expected_key
            else "Prediksi dibandingkan dengan label pembanding yang tersedia."
        ),
        "expected_class": expected_display,
        "expected_class_source": analysis.get("expected_source") or "Unknown",
        "final_decision_block": {
            "confidence_level": conf_cat["label"],
            "top2_margin": f"{margin_pct:.2f}%",
            "validation_status": validation_status,
            "known_class_status": analysis.get("known_class_status") or "—",
            "final_decision": final_decision,
            "explanation": "Perhitungan detail direkonstruksi dari data hasil klasifikasi yang tersimpan.",
        },
        "formulas_id": [
            {"title": "Probability Percentage", "code": "Probability (%) = Softmax Score × 100"},
            {"title": "Top-2 Margin", "code": "Top-2 Margin = Top-1 Score − Top-2 Score"},
            {"title": "Validation", "code": "Validation = Predicted Class == Expected Class"},
        ],
        "interpretation_id": (
            f"Model memprediksi gambar sebagai {predicted_display} dengan confidence {confidence_pct:.2f}%."
        ),
        "confidence_badge": conf_cat,
        "known_class_status": analysis.get("known_class_status"),
        "steps": [],
    }
    return calc


def extract_algorithm_db_fields(calc: Dict[str, Any], result_row, image_meta: Optional[Dict] = None) -> Dict[str, Any]:
    """Extract persistable algorithm calculation fields for ClassificationResult."""
    softmax = calc.get("softmax_table") or []
    steps = calc.get("steps") or []
    preprocessing_steps = []
    for step in steps:
        if step.get("key") == "preprocessing":
            preprocessing_steps = step.get("steps_list") or []
            break

    top = softmax[0] if softmax else {}
    second = softmax[1] if len(softmax) > 1 else {}
    margin_pct = round(top.get("percent", 0) - second.get("percent", 0), 2) if top else 0

    image_input = next((s for s in steps if s.get("key") == "image_input"), {})
    input_fields = {f["label"]: f["value"] for f in image_input.get("fields", [])}

    khat_step = next((s for s in steps if s.get("key") == "khat_validation"), {})

    return {
        "input_source": (getattr(result_row, "input_source", None) or "upload"),
        "original_width": image_meta.get("width") if image_meta else getattr(result_row, "original_width", None),
        "original_height": image_meta.get("height") if image_meta else getattr(result_row, "original_height", None),
        "processed_width": 224,
        "processed_height": 224,
        "top1_class": result_row.predicted_class,
        "top1_score": top.get("softmax_score"),
        "top2_margin": margin_pct,
        "confidence_level": calc.get("confidence_badge", {}).get("label"),
        "expected_class_source": getattr(result_row, "source_type", None),
        "stage1_decision": khat_step.get("decision_status") or result_row.detection_decision,
        "softmax_scores_json": json.dumps(softmax),
        "preprocessing_steps_json": json.dumps(preprocessing_steps),
        "calculation_notes": calc.get("interpretation"),
        "known_class_status": calc.get("known_class_status"),
    }


def apply_algorithm_fields_to_row(row, calc: Dict[str, Any], image_meta: Optional[Dict] = None) -> None:
    fields = extract_algorithm_db_fields(calc, row, image_meta=image_meta)
    for key, value in fields.items():
        if hasattr(row, key) and value is not None:
            setattr(row, key, value)


def build_algorithm_report_summary(all_records: List) -> Dict[str, Any]:
    """Aggregate algorithm calculation metrics for the research report."""
    total = len(all_records)
    valid = [r for r in all_records if (r.input_status or "khat") == "khat" and r.predicted_class]

    def _decision_count(label: str) -> int:
        return sum(1 for r in valid if (r.final_decision or "") == label)

    margins = [_margin_pct(r.top2_margin) for r in valid if r.top2_margin is not None]
    confidences = [_confidence_pct(r.confidence_score or r.confidence) for r in valid if r.confidence is not None]

    unknown_expected = sum(
        1 for r in valid
        if not r.expected_class or r.validation_status == "No Expected Class Provided"
    )
    possible_unknown = sum(
        1 for r in valid
        if (r.known_class_status or "").startswith("Possible Unknown")
        or (r.known_class_status or "").startswith("Uncertain Class")
    )
    mismatch = sum(
        1 for r in valid
        if r.validation_status == "Prediction Mismatch"
        or (
            r.expected_class and r.predicted_class and r.expected_class != r.predicted_class
        )
    )

    return {
        "total_predictions": total,
        "accepted": _decision_count("Accepted"),
        "accepted_with_caution": _decision_count("Accepted with Caution"),
        "manual_review_recommended": _decision_count("Manual Review Recommended") + _decision_count("Low Confidence Review"),
        "manual_review_required": _decision_count("Manual Review Required"),
        "prediction_mismatch_count": mismatch,
        "avg_confidence": round(sum(confidences) / len(confidences), 2) if confidences else None,
        "avg_top2_margin": round(sum(margins) / len(margins), 2) if margins else None,
        "unknown_expected_class_count": unknown_expected,
        "possible_unknown_class_count": possible_unknown,
        "rejected_non_khat": sum(1 for r in all_records if (r.input_status or "") == "non_khat"),
    }
