import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
from flask import current_app

from services.class_mapping_service import (
    format_trained_class_list,
    load_saved_class_indices,
    resolve_model_class_indices,
    resolve_trained_class_labels,
    validate_class_mapping,
)
from services.dataset_readiness_service import is_valid_trained_model
from services.dataset_alignment_service import (
    align_prediction_with_dataset,
    apply_characteristic_shape_prior,
)
from services.ensemble_prediction_service import classify_with_ensemble, ensemble_enabled
from services.external_assets_service import should_use_keras_model
from services.dataset_similarity_service import check_trained_class_fit
from services.khat_recognition_messages import (
    MSG_LOW_DATASET_SIMILARITY,
    MSG_NOT_ARABIC_IMAGE,
    MSG_NOT_ARABIC_SCRIPT,
    MSG_OUTSIDE_TRAINED_CLASSES,
    REJECTION_LOW_SIMILARITY,
    REJECTION_NOT_ARABIC_IMAGE,
    REJECTION_NOT_ARABIC_SCRIPT,
    REJECTION_OUTSIDE_CLASSES,
    build_unrecognized_payload,
    reason_invalid_class,
    reason_low_confidence,
    reason_low_margin,
    should_treat_as_unrecognized,
)
from services.khat_detector_service import detect_khat
from services.model_builder_service import load_trained_classifier
from services.model_context_service import get_active_model_path, get_model_context, load_model_metadata
from services.teachable_machine_service import get_tm_display_info, is_teachable_machine_available, predict_tm_image
from services.preprocessing_service import (
    get_architecture_preprocess_fn,
    preprocess_calligraphy_image,
    preprocess_for_model,
    preprocess_pil_image,
)
from services.probability_refinement_service import (
    build_sorted_probability_list,
    refine_class_probabilities,
)
from services.validation_pipeline import (
    pipeline_enabled,
    to_flask_rejection,
    validate_before_cnn,
    validate_distribution,
)

logger = logging.getLogger(__name__)


def _load_class_indices(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict) and "class_indices" in data:
        return data["class_indices"]
    return data


def _invert_class_indices(class_indices: Dict[str, int]) -> Dict[int, str]:
    return {int(index): label for label, index in class_indices.items()}


def _resolve_architecture() -> str:
    try:
        config = current_app.config
    except RuntimeError:
        config = {}
    active_path = get_active_model_path(config) if config else ""
    if config.get("USE_EXTERNAL_MODEL") and active_path == config.get("EXTERNAL_MODEL_PATH"):
        return "keras_h5"
    ctx = get_model_context(config) if config else {}
    meta = load_model_metadata(config) if config else {}
    return (
        ctx.get("model_architecture_key")
        or meta.get("model_architecture_key")
        or meta.get("architecture")
        or "efficientnetb0"
    )


def _margin_from_scores(scores: Dict[str, float], pred_class: Optional[str]) -> float:
    if not scores or not pred_class:
        return 0.0
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) < 2:
        return round(ordered[0] * 100, 2) if ordered else 0.0
    top = float(scores.get(pred_class, ordered[0]))
    second = max((value for label, value in scores.items() if label != pred_class), default=0.0)
    return round((top - second) * 100, 2)


def _stage2_thresholds(config=None) -> Dict[str, float]:
    if config is None:
        config = current_app.config
    return {
        "min_confidence": float(config.get("STAGE2_MIN_CONFIDENCE", 0.70)),
        "min_margin": float(config.get("STAGE2_MIN_MARGIN", 0.15)),
        "min_dataset_similarity": float(config.get("STAGE2_MIN_DATASET_SIMILARITY", 45)),
        "min_class_similarity": float(config.get("STAGE2_MIN_CLASS_SIMILARITY", 50)),
    }


def _class_recognition_gate(
    type_result: Dict,
    *,
    image_path: Optional[str] = None,
    config=None,
    stage1_confirmed: bool = False,
    stage1_content: Optional[Dict] = None,
) -> Optional[Dict]:
    """Return rejection metadata when the image should not receive a class label.

    After Stage 1 confirms Arabic calligraphy, use a softer confidence floor so
    valid Naskhi/Diwani/Diwani Jali/Tsuluts (including textbook scans) are labeled.
    Non-Arabic photos/scripts remain hard-blocked.
    """
    if config is None:
        try:
            config = current_app.config
        except RuntimeError:
            config = {}

    thresholds = _stage2_thresholds(config)
    min_conf_pct = thresholds["min_confidence"] * 100
    min_margin_pct = thresholds["min_margin"] * 100
    # After Arabic calligraphy is confirmed, apply the softer Stage-1-aware floor
    # (do NOT raise it back up to the standalone global floor).
    if stage1_confirmed:
        after_conf = float(
            config.get("STAGE2_MIN_CONFIDENCE_AFTER_STAGE1", min(thresholds["min_confidence"], 0.35))
        )
        after_margin = float(
            config.get("STAGE2_MIN_MARGIN_AFTER_STAGE1", min(thresholds["min_margin"], 0.02))
        )
        min_conf_pct = after_conf * 100
        min_margin_pct = after_margin * 100

    class_labels = resolve_trained_class_labels(config)
    display_list = format_trained_class_list(config)

    pred_class = type_result.get("predicted_class")
    gate_scores = type_result.get("gate_scores") or type_result.get("scores") or {}
    raw_scores = type_result.get("raw_model_scores") or gate_scores
    sorted_probs = type_result.get("sorted_probabilities") or []
    margin_pct = float(type_result.get("top2_margin_pct") or 0)
    if type_result.get("gate_top2_margin_pct") is not None:
        margin_pct = float(type_result["gate_top2_margin_pct"])
    confidence = type_result.get("gate_confidence")
    if confidence is None:
        confidence = type_result.get("confidence")
    if confidence is not None:
        conf_pct = confidence * 100 if confidence <= 1 else float(confidence)
    elif pred_class and gate_scores:
        conf_pct = float(gate_scores.get(pred_class, 0)) * 100
    else:
        conf_pct = float(sorted_probs[0]["percent"]) if sorted_probs else 0.0

    # Raw model evidence is authoritative. Never max-merge confidence from
    # refinement, fusion, similarity, or another component.
    if pred_class and raw_scores:
        raw_pred = max(raw_scores, key=raw_scores.get)
        pred_class = raw_pred
        conf_pct = float(raw_scores[raw_pred]) * 100
        margin_pct = _margin_from_scores(raw_scores, raw_pred)

    reasons = []
    rejection_code = REJECTION_OUTSIDE_CLASSES

    # Safety net: Latin / photographic content must never receive a class label.
    if image_path and stage1_content:
        try:
            content = stage1_content
            if content.get("is_latin_script_non_khat"):
                payload = build_unrecognized_payload(
                    display_list=display_list,
                    reasons=[MSG_NOT_ARABIC_SCRIPT],
                    rejection_code=REJECTION_NOT_ARABIC_SCRIPT,
                    message=MSG_NOT_ARABIC_SCRIPT,
                )
                return {
                    "predicted_class": None,
                    "confidence": None,
                    "input_status": "non_khat",
                    "rejection_reason": MSG_NOT_ARABIC_SCRIPT,
                    "rejection_code": REJECTION_NOT_ARABIC_SCRIPT,
                    "stage2_message": payload["stage2_message"],
                    "message": MSG_NOT_ARABIC_SCRIPT,
                    "detail_message": payload["detail_message"],
                    "status_label": None,
                    "reliability": "Rejected",
                    "stage2_rejected": True,
                    "trained_classes": class_labels,
                    "scores": None,
                    "probabilities": None,
                    "sorted_probabilities": None,
                    "top2_margin": None,
                    "top2_margin_pct": None,
                }
            if content.get("is_photographic_non_khat"):
                return {
                    "predicted_class": None,
                    "confidence": None,
                    "input_status": "non_khat",
                    "rejection_reason": MSG_NOT_ARABIC_IMAGE,
                    "rejection_code": REJECTION_NOT_ARABIC_IMAGE,
                    "stage2_message": "Klasifikasi diblokir: gambar bukan tulisan Arab.",
                    "message": MSG_NOT_ARABIC_IMAGE,
                    "reliability": "Rejected",
                    "stage2_rejected": True,
                    "trained_classes": class_labels,
                    "scores": None,
                    "probabilities": None,
                    "sorted_probabilities": None,
                    "top2_margin": None,
                    "top2_margin_pct": None,
                }
        except Exception:
            pass

    if not pred_class or pred_class not in class_labels:
        reasons.append(reason_invalid_class(display_list))
        rejection_code = REJECTION_OUTSIDE_CLASSES
    else:
        if (
            stage1_confirmed
            and pred_class in class_labels
            and conf_pct >= min_conf_pct
            and margin_pct >= min_margin_pct
        ):
            return None
        if conf_pct < min_conf_pct:
            reasons.append(reason_low_confidence(conf_pct, min_conf_pct))
            rejection_code = REJECTION_LOW_SIMILARITY
        if margin_pct < min_margin_pct:
            reasons.append(reason_low_margin(margin_pct, min_margin_pct))
            rejection_code = REJECTION_LOW_SIMILARITY

    if not reasons:
        return None

    payload = build_unrecognized_payload(
        display_list=display_list,
        reasons=reasons,
        rejection_code=rejection_code,
        message=(
            MSG_LOW_DATASET_SIMILARITY
            if rejection_code == REJECTION_LOW_SIMILARITY
            else MSG_OUTSIDE_TRAINED_CLASSES
        ),
    )
    return {
        "predicted_class": None,
        "confidence": None,
        "input_status": payload["input_status"],
        "rejection_reason": payload["rejection_reason"],
        "rejection_code": payload["rejection_code"],
        "stage2_message": payload["stage2_message"],
        "message": payload["message"],
        "detail_message": payload["detail_message"],
        "status_label": payload["status_label"],
        "reliability": "Rejected",
        "stage2_rejected": True,
        "final_status": "uncertain_class",
        "final_class": None,
        "abstained": True,
        "decision_reason": "; ".join(reasons),
        "trained_classes": class_labels,
        "scores": None,
        "probabilities": None,
        "sorted_probabilities": None,
        "top2_margin": None,
        "top2_margin_pct": None,
    }


def _apply_class_recognition_gate(
    type_result: Dict,
    *,
    image_path: Optional[str] = None,
    config=None,
    stage1_confirmed: bool = False,
    stage1_content: Optional[Dict] = None,
) -> Dict:
    evaluated = dict(type_result)
    raw_scores = evaluated.get("raw_model_scores") or {}
    if raw_scores:
        raw_pred = max(raw_scores, key=raw_scores.get)
        evaluated["predicted_class"] = raw_pred
        evaluated["confidence"] = float(raw_scores[raw_pred])
        evaluated["gate_confidence"] = float(raw_scores[raw_pred])
        evaluated["top2_margin_pct"] = _margin_from_scores(raw_scores, raw_pred)
        evaluated["top2_margin"] = evaluated["top2_margin_pct"] / 100.0
    rejection = _class_recognition_gate(
        evaluated,
        image_path=image_path,
        config=config,
        stage1_confirmed=stage1_confirmed,
        stage1_content=stage1_content,
    )
    if rejection is None:
        return evaluated
    return {**evaluated, **rejection}


def _reliability_label(confidence_pct: float, margin_pct: float) -> str:
    if confidence_pct >= 85 and margin_pct >= 20:
        return "Strong Prediction"
    if confidence_pct >= 70 and margin_pct >= 15:
        return "Acceptable Prediction"
    if confidence_pct < 70 or margin_pct < 15:
        return "Needs Manual Review"
    return "Acceptable Prediction"


def _tta_variants(image_path: str, architecture: str, preprocessing_mode: str = "standard") -> List[np.ndarray]:
    from PIL import Image, ImageEnhance, ImageOps

    from services.preprocessing_service import (
        center_crop_content_area,
        normalize_background,
        trim_excessive_borders,
    )

    preprocess_fn = get_architecture_preprocess_fn(architecture)
    variants = []
    with Image.open(image_path) as img:
        base = ImageOps.exif_transpose(img).convert("RGB")
        if preprocessing_mode == "manuscript":
            base = trim_excessive_borders(base, threshold=235, margin=4)
            base = center_crop_content_area(base)
            base = normalize_background(base)
        candidates = [
            base,
            ImageEnhance.Brightness(base).enhance(1.05),
            ImageEnhance.Contrast(base).enhance(1.05),
            base.resize((210, 210), Image.LANCZOS),
            base.rotate(3, fillcolor=(255, 255, 255)),
        ]
        for candidate in candidates:
            processed = preprocess_pil_image(candidate, (224, 224))
            array = preprocess_fn(np.array(processed, dtype=np.float32))
            variants.append(np.expand_dims(array, axis=0))
    return variants


def _classify_khat_type(image_path: str, use_tta: bool = False, preprocessing_mode: str = "standard") -> Dict:
    if is_teachable_machine_available() and not should_use_keras_model():
        type_result = predict_tm_image(image_path, use_tta=use_tta)
        tm_info = get_tm_display_info()
        top_conf = type_result["sorted_probabilities"][0]["percent"]
        second_conf = (
            type_result["sorted_probabilities"][1]["percent"]
            if len(type_result["sorted_probabilities"]) > 1
            else 0.0
        )
        margin = round(top_conf - second_conf, 2)
        type_result["reliability"] = _reliability_label(top_conf, margin)
        type_result["model_source"] = tm_info["model_source"]
        type_result["model_name"] = tm_info["model_name"]
        type_result["model_runtime"] = tm_info["model_runtime"]
        type_result["model_input_size"] = tm_info["model_input_size"]
        type_result["model_architecture"] = tm_info["model_architecture"]
        type_result["model_architecture_key"] = "teachable_machine"
        return type_result

    model_path = get_active_model_path()
    if not is_valid_trained_model(model_path):
        raise FileNotFoundError("Trained model not found.")

    simple_mode = current_app.config.get("APP_SIMPLE_MODE", False)
    class_labels = resolve_trained_class_labels(current_app.config)

    if ensemble_enabled(current_app.config):
        ensemble = classify_with_ensemble(
            image_path,
            current_app.config,
            use_tta=use_tta,
            preprocessing_mode=preprocessing_mode,
        )
        raw_scores = dict(ensemble["scores"])
        pred_class = ensemble.get("predicted_class")
        gate_confidence = float(ensemble.get("confidence") or 0.0)
        gate_margin = float(ensemble.get("margin") or 0.0) * 100
        model_path = ensemble.get("model_path") or model_path
        ensemble_used = bool(ensemble.get("ensemble_used"))
        model_source = ensemble.get("model_source") or "ensemble"
        best_single_confidence = float(ensemble.get("best_single_confidence") or 0.0)
        active_mapping = dict(ensemble.get("class_indices") or {})
        preprocessing_diagnostics = None
    else:
        class_indices = resolve_model_class_indices(current_app.config)
        if not class_indices:
            class_indices = load_saved_class_indices(current_app.config)
        if not class_indices:
            class_indices = {label: index for index, label in enumerate(class_labels)}

        validate_class_mapping(
            saved_indices=class_indices,
            num_model_outputs=len(class_labels),
            raise_on_mismatch=not simple_mode,
        )

        architecture = _resolve_architecture()
        from services.model_cache_service import get_cached_classifier

        model = get_cached_classifier(model_path, architecture, len(class_labels))
        if architecture == "efficientnetb0":
            from services.class_mapping_service import validate_production_mapping

            metadata_mapping = None
            best_meta_path = current_app.config.get("BEST_MODEL_METADATA_PATH")
            if best_meta_path and os.path.isfile(best_meta_path):
                try:
                    with open(best_meta_path, "r", encoding="utf-8") as handle:
                        best_meta = json.load(handle)
                    metadata_mapping = best_meta.get("class_mapping")
                except (OSError, json.JSONDecodeError):
                    metadata_mapping = None
            validate_production_mapping(
                class_indices=class_indices,
                num_model_outputs=int(model.output_shape[-1]),
                metadata_class_mapping=metadata_mapping,
            )
        inv_map = _invert_class_indices(class_indices) if class_indices else {
            index: label for index, label in enumerate(class_labels)
        }
        active_mapping = dict(class_indices)
        preprocessing_diagnostics = None

        if use_tta:
            batches = _tta_variants(image_path, architecture, preprocessing_mode=preprocessing_mode)
            probs_list = [model.predict(batch, verbose=0)[0] for batch in batches]
            probs = np.mean(probs_list, axis=0)
        else:
            if preprocessing_mode == "standard" and architecture != "keras_h5":
                from services.preprocessing_service import prepare_local_classifier_input

                batch, preprocessing_diagnostics = prepare_local_classifier_input(
                    image_path,
                    architecture=architecture,
                    return_diagnostics=True,
                )
            else:
                batch = preprocess_for_model(
                    image_path,
                    architecture=architecture,
                    preprocessing_mode=preprocessing_mode,
                )
            probs = model.predict(batch, verbose=0)[0]

        from services.calibration_service import load_calibrator, apply_calibrator

        calibrator = load_calibrator(current_app.config)
        if calibrator is not None and len(probs) == len(calibrator.get("bias", [])):
            model_class_order = [
                inv_map[index]
                for index in range(len(probs))
                if index in inv_map
            ]
            if len(model_class_order) == len(probs):
                probs = apply_calibrator(
                    probs.reshape(1, -1),
                    calibrator,
                    model_class_order=model_class_order,
                )[0]
            else:
                probs = apply_calibrator(probs.reshape(1, -1), calibrator)[0]

        pred_idx = int(np.argmax(probs))
        raw_scores = {
            inv_map[index]: float(probs[index])
            for index in range(len(probs))
            if inv_map.get(index) in class_labels
        }
        pred_class = inv_map.get(pred_idx)
        if pred_class not in class_labels:
            pred_class = max(raw_scores, key=raw_scores.get) if raw_scores else None
        gate_confidence = float(raw_scores.get(pred_class, 0)) if pred_class else None
        gate_margin = _margin_from_scores(raw_scores, pred_class)
        ensemble_used = False
        model_source = "external" if current_app.config.get("USE_EXTERNAL_MODEL") else "local"
        best_single_confidence = gate_confidence or 0.0

    immutable_raw_scores = dict(raw_scores)
    from services.diwani_pair_service import apply_contested_style_pairs

    # Contested-pair only: Diwani ↔ Diwani Jali, then Diwani Jali ↔ Tsuluts.
    # Skip when CNN already points to Naskhi/Tsuluts — ornate book borders otherwise
    # inflate Diwani Jali ornament cues and flip the label.
    raw_top = max(raw_scores, key=raw_scores.get) if raw_scores else None
    raw_top_conf = float(raw_scores.get(raw_top, 0.0)) if raw_top else 0.0
    skip_contested = bool(raw_top in {"naskhi", "tsuluts"} and raw_top_conf >= 0.32)
    if not skip_contested:
        try:
            raw_scores = apply_contested_style_pairs(raw_scores, image_path, class_labels)
        except Exception:
            pass
    if raw_scores:
        pred_class = max(raw_scores, key=raw_scores.get)
        gate_confidence = float(raw_scores.get(pred_class, 0.0))
        gate_margin = _margin_from_scores(raw_scores, pred_class)

    try:
        alignment = align_prediction_with_dataset(raw_scores, class_labels, image_path)
    except Exception:
        alignment = {
            "predicted_class": pred_class,
            "fused_scores": raw_scores,
            "dataset_aligned": False,
            "similarity_scores": {},
            "model_predicted_class": pred_class,
            "model_confidence": gate_confidence,
        }
    fused_scores = alignment.get("fused_scores") or raw_scores
    if not skip_contested and not alignment.get("kept_model_decision"):
        try:
            fused_scores = apply_contested_style_pairs(fused_scores, image_path, class_labels)
        except Exception:
            pass
    # Dataset style traits (ornament / monumental / openness) for ambiguous races only.
    if fused_scores and not alignment.get("kept_model_decision"):
        try:
            fused_scores = apply_characteristic_shape_prior(
                fused_scores,
                image_path,
                class_labels,
            )
        except Exception:
            pass
    if fused_scores:
        pred_class = max(fused_scores, key=fused_scores.get)
    elif alignment.get("predicted_class") in class_labels:
        pred_class = alignment["predicted_class"]
    if pred_class not in class_labels and raw_scores:
        pred_class = max(raw_scores, key=raw_scores.get)

    # Prefer clear raw CNN winner over fusion/ornament corrections.
    if raw_scores:
        raw_pred = max(raw_scores, key=raw_scores.get)
        raw_conf = float(raw_scores.get(raw_pred, 0.0))
        raw_margin_pct = _margin_from_scores(raw_scores, raw_pred)
        if (
            raw_pred in class_labels
            and raw_conf >= 0.32
            and raw_margin_pct >= 8.0
            and (pred_class != raw_pred or float(fused_scores.get(pred_class, 0) if fused_scores else 0) < raw_conf)
        ):
            pred_class = raw_pred
            fused_scores = dict(raw_scores)
            gate_confidence = raw_conf
            gate_margin = raw_margin_pct

    gate_confidence = float(
        fused_scores.get(pred_class, 0)
        if fused_scores and pred_class
        else alignment.get("fused_confidence")
        or gate_confidence
        or 0
    )
    if ensemble_used and best_single_confidence > gate_confidence:
        gate_confidence = best_single_confidence
    gate_margin = _margin_from_scores(fused_scores, pred_class) if fused_scores else 0.0

    try:
        display_scores = refine_class_probabilities(
            fused_scores or raw_scores,
            class_labels,
            image_path=image_path,
        )
    except Exception:
        display_scores = dict(fused_scores or raw_scores or {})
        total = sum(display_scores.values()) or 1.0
        display_scores = {key: float(value) / total for key, value in display_scores.items()}
    sorted_probs = build_sorted_probability_list(display_scores)
    top_conf = sorted_probs[0]["percent"] if sorted_probs else 0.0
    second_conf = sorted_probs[1]["percent"] if len(sorted_probs) > 1 else 0.0
    margin = round(top_conf - second_conf, 2)

    return {
        "predicted_class": pred_class if pred_class in class_labels else None,
        "confidence": gate_confidence,
        "gate_confidence": gate_confidence,
        "gate_scores": fused_scores,
        "raw_model_scores": immutable_raw_scores,
        "refined_scores": display_scores,
        "refined_top2_margin_pct": margin,
        "dataset_aligned": bool(alignment.get("dataset_aligned")),
        "ensemble_used": ensemble_used,
        "best_single_confidence": best_single_confidence if ensemble_used else None,
        "model_predicted_class": alignment.get("model_predicted_class"),
        "similarity_scores": alignment.get("similarity_scores"),
        "scores": display_scores,
        "probabilities": display_scores,
        "sorted_probabilities": sorted_probs,
        "top2_margin": gate_margin / 100.0,
        "top2_margin_pct": gate_margin,
        "gate_top2_margin_pct": gate_margin,
        "display_top2_margin_pct": margin,
        "reliability": _reliability_label(gate_confidence * 100, gate_margin),
        "model_path": model_path,
        "model_source_key": model_source,
        "model_diagnostics": {
            "class_mapping": active_mapping,
            "model_path": model_path,
            "preprocessing": preprocessing_diagnostics,
            "raw_top1": max(immutable_raw_scores, key=immutable_raw_scores.get)
            if immutable_raw_scores else None,
            "raw_top2_margin_pct": _margin_from_scores(
                immutable_raw_scores,
                max(immutable_raw_scores, key=immutable_raw_scores.get),
            ) if immutable_raw_scores else None,
        },
    }


def predict_cnn_image(image_path: str, preprocessing_mode: str = "standard", use_tta: bool = False) -> Dict:
    """Prediksi langsung dengan model CNN (tanpa detektor Khat / Teachable Machine)."""
    model_ctx = get_model_context()
    try:
        type_result = _classify_khat_type(image_path, use_tta=use_tta, preprocessing_mode=preprocessing_mode)
    except FileNotFoundError as exc:
        return {
            "predicted_class": None,
            "confidence": None,
            "scores": None,
            "message": str(exc),
            "input_status": "error",
        }

    type_result = _apply_class_recognition_gate(type_result, image_path=image_path)
    if type_result.get("input_status") == "unrecognized":
        return {
            **type_result,
            "is_khat": False,
            "stage2_ran": False,
            "model_path": get_active_model_path(),
            "using_best_model": model_ctx.get("using_best_model", False),
            "preprocessing_mode": preprocessing_mode,
            "tta_used": use_tta,
            "model_source": "CNN Transfer Learning",
            "model_runtime": "TensorFlow Keras",
            "model_input_size": "224x224",
        }

    return {
        **type_result,
        "input_status": "khat",
        "is_khat": True,
        "stage2_ran": True,
        "model_path": get_active_model_path(),
        "using_best_model": model_ctx.get("using_best_model", False),
        "preprocessing_mode": preprocessing_mode,
        "tta_used": use_tta,
        "model_source": "CNN Transfer Learning",
        "model_runtime": "TensorFlow Keras",
        "model_input_size": "224x224",
    }


def predict_image(
    image_path: str,
    use_tta: bool = False,
    force_classify: bool = False,
    preprocessing_mode: str = "standard",
) -> Dict:
    # Pre-CNN validation: Arabic writing only, reject photos / non-Arabic scripts.
    if pipeline_enabled() and not force_classify:
        try:
            pre = validate_before_cnn(image_path)
            if not pre.get("accepted"):
                return to_flask_rejection(pre)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pre-CNN validation pipeline failed open: %s", exc)

    detection = detect_khat(image_path, force_classify=force_classify)
    model_ctx = get_model_context()

    base = {
        "input_status": detection["input_status"],
        "detection_status": detection.get("detection_status"),
        "detection_decision": detection.get("detection_decision"),
        "detection_status_label": detection.get("detection_status_label"),
        "detection_decision_label": detection.get("detection_decision_label"),
        "is_khat": detection["is_khat"],
        "khat_probability": detection["khat_probability"],
        "non_khat_probability": detection["non_khat_probability"],
        "detector_available": detection.get("detector_available", False),
        "detector_skipped": detection.get("skipped", False),
        "rejection_reason": detection.get("rejection_reason"),
        "stage1_message": detection.get("message"),
        "stage1_title": detection.get("title"),
        "manual_review_required": detection.get("manual_review_required", False),
        "moderate_confidence_note": detection.get("moderate_confidence_note"),
        "accept_threshold": detection.get("accept_threshold"),
        "reject_threshold": detection.get("reject_threshold"),
        "borderline_threshold": detection.get("borderline_threshold"),
        "stage2_permission": detection.get("stage2_permission"),
        "detection_explanation": detection.get("detection_explanation"),
        "force_classify": force_classify,
        "tta_used": use_tta,
        "preprocessing_mode": preprocessing_mode,
        "model_path": None,
        "using_best_model": model_ctx.get("using_best_model", False),
        "model_architecture": None,
        "model_metadata": model_ctx,
        "is_demo_model": model_ctx.get("is_demo_model", False),
    }

    if not detection.get("stage2_allowed", False):
        content_gate = detection.get("content_gate") or {}
        if content_gate.get("is_latin_script_non_khat"):
            msg = MSG_NOT_ARABIC_SCRIPT
            code = REJECTION_NOT_ARABIC_SCRIPT
        elif content_gate.get("is_photographic_non_khat"):
            msg = MSG_NOT_ARABIC_IMAGE
            code = REJECTION_NOT_ARABIC_IMAGE
        else:
            msg = detection.get("message") or MSG_NOT_ARABIC_IMAGE
            # Map legacy Stage 1 wording onto the business messages.
            lower = str(msg).lower()
            if "latin" in lower or "abjad" in lower:
                msg = MSG_NOT_ARABIC_SCRIPT
                code = REJECTION_NOT_ARABIC_SCRIPT
            else:
                msg = MSG_NOT_ARABIC_IMAGE
                code = REJECTION_NOT_ARABIC_IMAGE
        return {
            **base,
            "predicted_class": None,
            "confidence": None,
            "scores": None,
            "probabilities": None,
            "sorted_probabilities": None,
            "top2_margin": None,
            "top2_margin_pct": None,
            "reliability": "Rejected",
            "rejection_reason": msg,
            "rejection_code": code,
            "message": msg,
            "stage2_ran": False,
            "stage2_message": (
                "Klasifikasi 4 kelas diblokir karena citra gagal Stage 1 deteksi khat "
                "atau berada di luar domain tulisan Arab yang didukung."
            ),
        }

    # Safety net: never classify Latin alphabet / photo OOD even if detector skipped.
    content_gate = detection.get("content_gate") or {}
    if content_gate.get("is_latin_script_non_khat") or content_gate.get("is_photographic_non_khat"):
        if content_gate.get("is_latin_script_non_khat"):
            msg = MSG_NOT_ARABIC_SCRIPT
            code = REJECTION_NOT_ARABIC_SCRIPT
        else:
            msg = MSG_NOT_ARABIC_IMAGE
            code = REJECTION_NOT_ARABIC_IMAGE
        return {
            **base,
            "input_status": "non_khat",
            "is_khat": False,
            "predicted_class": None,
            "confidence": None,
            "scores": None,
            "probabilities": None,
            "sorted_probabilities": None,
            "top2_margin": None,
            "top2_margin_pct": None,
            "reliability": "Rejected",
            "rejection_reason": msg,
            "rejection_code": code,
            "message": msg,
            "stage2_ran": False,
            "stage2_message": "Klasifikasi diblokir oleh validasi konten (bukan tulisan Arab).",
        }

    type_result = _classify_khat_type(image_path, use_tta=use_tta, preprocessing_mode=preprocessing_mode)

    # Cosine similarity + MSP OOD — advisory after Stage 1.
    # Hard-reject only when CNN cannot pick any trained class with usable confidence.
    # Decorative covers / multi-line textbook pages often score low vs crop galleries.
    if pipeline_enabled() and not force_classify:
        try:
            from services.validation_pipeline.feature_extractor import extract_embedding

            scores = type_result.get("raw_model_scores") or type_result.get("gate_scores") or type_result.get("scores") or {}
            class_labels = resolve_trained_class_labels(current_app.config)
            probs = [float(scores.get(label, 0.0)) for label in class_labels] if scores else []
            if not probs and type_result.get("sorted_probabilities"):
                probs = [
                    float(item.get("percent", 0)) / 100.0
                    for item in type_result["sorted_probabilities"]
                ]
            embedding = extract_embedding(image_path)
            dist = validate_distribution(
                embedding=embedding,
                probabilities=probs,
                predicted_class=type_result.get("predicted_class"),
                config=current_app.config,
            )
            type_result["validation_distribution"] = dist
            if not dist.get("accepted"):
                pred = type_result.get("predicted_class")
                conf = type_result.get("gate_confidence")
                if conf is None:
                    conf = type_result.get("confidence") or 0.0
                conf = float(conf)
                if conf > 1:
                    conf = conf / 100.0
                raw_scores = type_result.get("raw_model_scores") or {}
                raw_best = max(raw_scores.values()) if raw_scores else 0.0
                soft_conf = float(
                    current_app.config.get("STAGE2_MIN_CONFIDENCE_AFTER_STAGE1", 0.35)
                )
                if (pred in class_labels and conf >= soft_conf) or float(raw_best) >= soft_conf:
                    type_result["validation_distribution_soft_pass"] = True
                else:
                    payload = build_unrecognized_payload(
                        display_list=format_trained_class_list(current_app.config),
                        reasons=[MSG_LOW_DATASET_SIMILARITY],
                        rejection_code=REJECTION_LOW_SIMILARITY,
                        message=MSG_LOW_DATASET_SIMILARITY,
                    )
                    return {
                        **base,
                        **payload,
                        "is_khat": False,
                        "predicted_class": None,
                        "confidence": None,
                        "scores": None,
                        "probabilities": None,
                        "sorted_probabilities": None,
                        "top2_margin": None,
                        "top2_margin_pct": None,
                        "reliability": "Rejected",
                        "stage2_ran": False,
                        "validation_pipeline": dist,
                    }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Distribution validation failed open: %s", exc)

    type_result = _apply_class_recognition_gate(
        type_result,
        image_path=image_path,
        stage1_confirmed=True,
        stage1_content=detection.get("content_gate") or {},
    )
    if type_result.get("input_status") in ("unrecognized", "non_khat"):
        return {
            **base,
            **type_result,
            "is_khat": False,
            "stage2_ran": False,
            "stage2_message": type_result.get("stage2_message"),
            "reliability": "Rejected",
        }

    reliability = type_result["reliability"]
    if detection.get("manual_review_required"):
        reliability = "Manual Review Required"

    stage2_message = (
        "Citra lolos validasi tulisan Arab dan diklasifikasikan ke salah satu "
        "kelas khat yang didukung (Naskhi, Diwani, Diwani Jali, Tsuluts)."
    )
    if detection.get("detection_explanation"):
        stage2_message = f"{detection['detection_explanation']} {stage2_message}"
    elif detection.get("moderate_confidence_note"):
        stage2_message = f"{detection['moderate_confidence_note']} {stage2_message}"
    elif detection.get("message") and detection.get("input_status") == "uncertain":
        stage2_message = f"{detection['message']} {stage2_message}"
    if force_classify and detection["input_status"] == "uncertain":
        stage2_message = (
            "Klasifikasi dilanjutkan atas permintaan pengguna meskipun deteksi khat tidak yakin. "
            "Tinjauan manual diperlukan."
        )

    return {
        **base,
        **type_result,
        "is_khat": True,
        "stage2_ran": True,
        "reliability": reliability,
        "manual_review_required": detection.get("manual_review_required", False) or force_classify,
        "model_path": get_active_model_path() if not is_teachable_machine_available() else current_app.config["TEACHABLE_MODEL_JSON"],
        "model_architecture": type_result.get("model_architecture") or _resolve_architecture(),
        "stage2_message": stage2_message,
        "message": reliability,
        "final_status": "confirmed_class" if reliability == "Strong Prediction" else "probable_class",
        "final_class": type_result.get("predicted_class"),
        "abstained": False,
        "decision_reason": "raw confidence and top-two margin passed Stage 2",
        "stage1_trace": detection.get("content_gate"),
    }
