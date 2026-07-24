"""Evaluate the holdout test set using the same ensemble pipeline as live classification."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from flask import current_app

from services.class_mapping_service import (
    canonical_class_order,
    resolve_model_class_indices,
    resolve_trained_class_labels,
)
from services.dataset_readiness_service import is_valid_trained_model
from services.ensemble_prediction_service import ensemble_enabled
from services.evaluation_dataset_service import (
    build_dataset_summary,
    iter_test_images,
    resolve_evaluation_test_dir,
    validate_test_holdout_before_eval,
)
from services.evaluation_metrics_service import compute_classification_metrics
from services.metrics_service import ACCURACY_TARGET, target_status, training_recommendations, weak_classes
from services.model_cache_service import get_cached_classifier
from services.model_context_service import get_active_model_path, get_model_context
from services.preprocessing_service import get_architecture_preprocess_fn, preprocess_pil_image


def _label_order_from_indices(class_indices: Dict[str, int]) -> List[str]:
    return sorted(class_indices.keys(), key=lambda key: int(class_indices[key]))


def _remap_probability_row(
    row: np.ndarray,
    source_order: List[str],
    target_order: List[str],
) -> np.ndarray:
    source_index = {label: index for index, label in enumerate(source_order)}
    remapped = np.zeros(len(target_order), dtype=np.float64)
    for target_idx, label in enumerate(target_order):
        source_idx = source_index.get(label)
        if source_idx is not None and source_idx < len(row):
            remapped[target_idx] = float(row[source_idx])
    total = remapped.sum()
    if total > 0:
        remapped /= total
    return remapped


def _local_tta_variant_fns():
    """TTA transforms aligned with live prediction (no horizontal flip — Arabic script)."""
    from PIL import Image, ImageEnhance

    return {
        "orig": lambda pil: pil,
        "bright": lambda pil: ImageEnhance.Brightness(pil).enhance(1.05),
        "contrast": lambda pil: ImageEnhance.Contrast(pil).enhance(1.05),
        "resize210": lambda pil: pil.resize((210, 210), Image.LANCZOS),
        "rot3": lambda pil: pil.rotate(3, expand=False, fillcolor=(255, 255, 255)),
    }


def _local_tta_variant_names(use_tta: bool) -> List[str]:
    if not use_tta:
        return ["orig"]
    # Same family as prediction_service._tta_variants (no hflip).
    return ["orig", "bright", "contrast", "resize210", "rot3"]


def _predict_local_probs_for_filenames(
    model,
    test_dir: str,
    filenames: List[str],
    *,
    use_tta: bool = False,
) -> np.ndarray:
    from PIL import Image, ImageOps

    preprocess_input = get_architecture_preprocess_fn("efficientnetb0")
    variant_fns = _local_tta_variant_fns()
    variant_names = _local_tta_variant_names(use_tta)
    rows: List[np.ndarray] = []
    for filename in filenames:
        image_path = os.path.join(test_dir, filename.replace("\\", "/"))
        pil = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        tensors = []
        for name in variant_names:
            frame = variant_fns[name](pil)
            processed = preprocess_pil_image(frame, (224, 224))
            tensors.append(preprocess_input(np.array(processed, dtype=np.float32)))
        batch = np.stack(tensors)
        rows.append(model.predict(batch, verbose=0).mean(axis=0))
    return np.vstack(rows)


def _batched_probs_efficientnet(
    model,
    test_dir: str,
    class_order: List[str],
    batch_size: int = 64,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    from PIL import Image
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    preprocess_input = get_architecture_preprocess_fn("efficientnetb0")

    def _preprocess(img_array):
        pil = Image.fromarray(img_array.astype("uint8"))
        processed = preprocess_pil_image(pil, (224, 224))
        return preprocess_input(np.array(processed, dtype=np.float32))

    generator = ImageDataGenerator(preprocessing_function=_preprocess)
    flow = generator.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        class_mode="categorical",
        classes=class_order,
        batch_size=batch_size,
        shuffle=False,
    )
    if flow.samples == 0:
        raise ValueError("Test dataset is empty.")
    probs = model.predict(flow, verbose=0)
    return probs, list(flow.filenames), np.array(flow.classes)


def _batched_probs_keras_h5(
    model,
    test_dir: str,
    class_order: List[str],
    batch_size: int = 64,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    from PIL import Image, ImageOps
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    def _preprocess(img_array):
        pil = Image.fromarray(img_array.astype("uint8"))
        rgb = ImageOps.exif_transpose(pil).convert("RGB")
        array = np.array(rgb.resize((224, 224)), dtype=np.float32)
        return (array / 127.5) - 1.0

    generator = ImageDataGenerator(preprocessing_function=_preprocess)
    flow = generator.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        class_mode="categorical",
        classes=class_order,
        batch_size=batch_size,
        shuffle=False,
    )
    if flow.samples == 0:
        raise ValueError("Test dataset is empty.")
    probs = model.predict(flow, verbose=0)
    return probs, list(flow.filenames), np.array(flow.classes)


def _fuse_ensemble_probabilities(
    external_probs: np.ndarray,
    external_order: List[str],
    local_probs: np.ndarray,
    local_order: List[str],
    target_order: List[str],
    *,
    external_weight: float,
    local_weight: float,
    adaptive: bool = True,
    weak_external_threshold: float = 0.55,
    strong_local_threshold: float = 0.72,
) -> np.ndarray:
    fused_rows = []
    for index in range(len(external_probs)):
        external_row = _remap_probability_row(external_probs[index], external_order, target_order)
        local_row = _remap_probability_row(local_probs[index], local_order, target_order)
        row_external_weight = external_weight
        if adaptive:
            if float(external_row.max()) < weak_external_threshold:
                row_external_weight = 0.0
            elif float(local_row.max()) >= strong_local_threshold:
                row_external_weight = min(row_external_weight, 0.10)
        row_local_weight = 1.0 - row_external_weight if row_external_weight < 1.0 else local_weight
        if row_external_weight + row_local_weight <= 0:
            row_local_weight = 1.0
            row_external_weight = 0.0
        total_w = row_external_weight + row_local_weight
        fused = (row_external_weight / total_w) * external_row + (row_local_weight / total_w) * local_row
        total = fused.sum()
        if total > 0:
            fused = fused / total
        fused_rows.append(fused)
    return np.vstack(fused_rows)


def should_use_production_eval_pipeline(config=None) -> bool:
    """Prefer the production-aligned pipeline whenever a usable local model exists."""
    if config is None:
        config = current_app.config
    if not config.get("EVAL_USE_INFERENCE_PIPELINE", True):
        return False
    if ensemble_enabled(config):
        return True
    local_candidates = [
        os.path.join(config.get("BASE_DIR", ""), "model", "khat_best.keras"),
        config.get("BEST_MODEL_PATH") or "",
        config.get("MODEL_PATH") or "",
    ]
    return any(is_valid_trained_model(path) for path in local_candidates if path)


def evaluate_with_production_pipeline():
    """Run holdout evaluation with external + local ensemble (matches classification UI)."""
    import matplotlib

    matplotlib.use("Agg")

    from services.evaluation_service import _plot_confusion_matrix, generate_evaluation_charts
    from services.training_utils import save_json

    cfg = current_app.config
    test_dir, test_source = resolve_evaluation_test_dir(cfg)
    eval_class_labels = list(resolve_trained_class_labels(cfg) or cfg.get("CLASS_LABELS") or [])
    test_items = iter_test_images(test_dir, eval_class_labels)
    validate_test_holdout_before_eval(test_items, cfg)
    dataset_summary = build_dataset_summary(test_dir, test_items, eval_class_labels, cfg)
    dataset_summary["test_source_key"] = test_source

    batch_size = int(cfg.get("EVAL_BATCH_SIZE", 32))
    plot_dpi = int(cfg.get("EVAL_PLOT_DPI", 100))
    defer_previews = bool(cfg.get("EVAL_DEFER_PREVIEWS", True))
    use_tta = bool(cfg.get("EVAL_USE_TTA", True))
    external_weight = float(
        cfg.get("EVAL_ENSEMBLE_EXTERNAL_WEIGHT", cfg.get("ENSEMBLE_EXTERNAL_WEIGHT", 0.15))
    )
    local_weight = float(
        cfg.get("EVAL_ENSEMBLE_LOCAL_WEIGHT", cfg.get("ENSEMBLE_LOCAL_WEIGHT", 0.85))
    )
    if external_weight > 0 and local_weight > 0:
        weight_total = external_weight + local_weight
        external_weight /= weight_total
        local_weight /= weight_total
    elif external_weight <= 0:
        external_weight = 0.0
        local_weight = 1.0
    else:
        local_weight = 0.0
        external_weight = 1.0

    external_path = cfg.get("EXTERNAL_MODEL_PATH") or ""
    local_path = os.path.join(cfg.get("BASE_DIR", ""), "model", "khat_best.keras")
    if not is_valid_trained_model(local_path):
        local_path = cfg.get("BEST_MODEL_PATH") or cfg.get("MODEL_PATH") or ""
    if not is_valid_trained_model(local_path):
        raise FileNotFoundError(
            "Model lokal untuk evaluasi tidak ditemukan. Latih model terlebih dahulu "
            "atau pastikan file model/khat_best.keras tersedia."
        )

    external_indices = resolve_model_class_indices(cfg) or {}
    if not external_indices:
        external_indices = dict(canonical_class_order(cfg))
    external_order = _label_order_from_indices(external_indices)
    local_indices = dict(canonical_class_order(cfg))
    best_meta_path = cfg.get("BEST_MODEL_METADATA_PATH") or os.path.join(
        cfg.get("BASE_DIR", ""), "model", "best_model_metadata.json"
    )
    if best_meta_path and os.path.isfile(best_meta_path):
        try:
            with open(best_meta_path, "r", encoding="utf-8") as handle:
                best_meta = json.load(handle)
            mapping = best_meta.get("class_mapping") or best_meta.get("class_labels") or []
            if isinstance(mapping, list) and len(mapping) == len(eval_class_labels):
                local_indices = {str(label): index for index, label in enumerate(mapping)}
            elif isinstance(mapping, dict) and mapping:
                local_indices = {str(k): int(v) for k, v in mapping.items()}
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    local_order = _label_order_from_indices(local_indices)

    local_model = get_cached_classifier(local_path, "efficientnetb0", len(local_order))

    use_external = external_weight > 0 and is_valid_trained_model(external_path)

    def _run_inference(active_batch_size: int, active_use_tta: bool):
        nonlocal use_external
        if use_external:
            external_model = get_cached_classifier(external_path, "keras_h5", len(external_order))
            ext_probs, names, truth = _batched_probs_keras_h5(
                external_model,
                test_dir,
                eval_class_labels,
                batch_size=active_batch_size,
            )
        else:
            _, names, truth = _batched_probs_efficientnet(
                local_model,
                test_dir,
                eval_class_labels,
                batch_size=active_batch_size,
            )
            ext_probs = None

        loc_probs = _predict_local_probs_for_filenames(
            local_model,
            test_dir,
            names,
            use_tta=active_use_tta,
        )
        return ext_probs, loc_probs, names, truth

    try:
        external_probs, local_probs, filenames, y_true = _run_inference(batch_size, use_tta)
    except Exception as exc:
        message = str(exc).lower()
        oom_like = any(
            token in message
            for token in ("resource_exhausted", "oom", "out of memory", "memoryerror", "allocation")
        )
        if not oom_like:
            raise
        # Retry leaner: smaller batches, no TTA — keeps evaluation finishing on low-RAM hosts.
        batch_size = min(batch_size, 8)
        use_tta = False
        external_probs, local_probs, filenames, y_true = _run_inference(batch_size, use_tta)

    labels = eval_class_labels
    if use_external and external_probs is not None:
        probs = _fuse_ensemble_probabilities(
            external_probs,
            external_order,
            local_probs,
            local_order,
            labels,
            external_weight=external_weight,
            local_weight=local_weight,
        )
    else:
        probs = np.vstack(
            [_remap_probability_row(local_probs[index], local_order, labels) for index in range(len(local_probs))]
        )

    # Align with live single-model calibration (label-aware remapping).
    from services.calibration_service import apply_calibrator, load_calibrator

    calibrator = load_calibrator(cfg)
    if calibrator is not None and probs.shape[1] == len(calibrator.get("bias", [])):
        probs = apply_calibrator(probs, calibrator, model_class_order=labels)

    from services.diwani_pair_service import adjust_probability_row_for_style_pairs

    for index, filename in enumerate(filenames):
        image_path = os.path.join(test_dir, filename.replace("\\", "/"))
        probs[index] = adjust_probability_row_for_style_pairs(probs[index], labels, image_path)

    y_pred = np.argmax(probs, axis=1)
    metrics = compute_classification_metrics(y_true, y_pred, labels)
    sample_count = int(len(y_true))
    accuracy = metrics["accuracy"]

    misclassified = []
    pair_counts = {}
    for index in range(sample_count):
        true_idx = int(y_true[index])
        pred_idx = int(y_pred[index])
        true_label = labels[true_idx]
        pred_label = labels[pred_idx]
        prob_row = probs[index]
        sorted_idx = np.argsort(prob_row)[::-1]
        top_conf = float(prob_row[sorted_idx[0]]) * 100
        second_conf = float(prob_row[sorted_idx[1]]) * 100 if len(sorted_idx) > 1 else 0.0
        margin = round(top_conf - second_conf, 2)
        if true_idx != pred_idx:
            from services.evaluation_image_service import attach_preview_metadata_only, attach_preview_to_record
            from services.evaluation_ui_service import classify_error_type

            error_meta = classify_error_type(top_conf, margin, true_label=true_label, predicted_label=pred_label)
            pair_key = (true_label, pred_label)
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
            preview_fn = attach_preview_metadata_only if defer_previews else attach_preview_to_record
            misclassified.append(
                preview_fn(
                    {
                        "image_path": os.path.join(test_dir, filenames[index]),
                        "filename": filenames[index],
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "confidence": float(prob_row[pred_idx]),
                        "confidence_pct": round(top_conf, 2),
                        "top2_class": labels[int(sorted_idx[1])] if len(sorted_idx) > 1 else None,
                        "top2_probability": float(prob_row[sorted_idx[1]]) if len(sorted_idx) > 1 else None,
                        "margin_pct": margin,
                        "error_type": error_meta["key"],
                        "error_type_label": error_meta["label_id"],
                        "error_hint": error_meta["hint"],
                        "is_high_confidence_error": error_meta["key"] == "high_confidence",
                        "probability_distribution": {
                            labels[j]: round(float(prob_row[j]), 4) for j in range(len(prob_row))
                        },
                    }
                )
            )

    confused_pairs = [
        {
            "true_class": key[0],
            "predicted_class": key[1],
            "count": value,
            "percentage": round(value / len(misclassified) * 100, 2) if misclassified else 0,
            "example_images": [
                item["image_path"]
                for item in misclassified
                if item["true_label"] == key[0] and item["predicted_label"] == key[1]
            ][:5],
        }
        for key, value in sorted(pair_counts.items(), key=lambda item: -item[1])
    ]

    history_file = {}
    history_path = cfg.get("TRAINING_HISTORY_PATH")
    if history_path and os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as handle:
                history_file = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            history_file = {}

    hist_series = history_file.get("history", history_file) if isinstance(history_file, dict) else {}
    if not isinstance(hist_series, dict):
        hist_series = {}

    eval_dir = cfg["EVALUATION_FOLDER"]
    os.makedirs(eval_dir, exist_ok=True)
    try:
        generate_evaluation_charts(
            eval_dir,
            history_file=history_file,
            labels=labels,
            per_class=metrics["per_class"],
            classification_report=metrics["classification_report"],
            plot_dpi=plot_dpi,
        )
        _plot_confusion_matrix(
            metrics["confusion_matrix"],
            labels,
            os.path.join(eval_dir, "confusion_matrix.png"),
            dpi=plot_dpi,
        )
    except Exception:
        # Metrics/results still saved even if chart rendering fails on headless hosts.
        pass

    csv_path = os.path.join(eval_dir, "classification_report.csv")
    report_dict = metrics["classification_report"]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["class", "precision", "recall", "f1-score", "support", "class_accuracy"])
        for label in labels:
            row = report_dict.get(label, {})
            writer.writerow(
                [
                    label,
                    row.get("precision", ""),
                    row.get("recall", ""),
                    row.get("f1-score", ""),
                    row.get("support", ""),
                    metrics["per_class_accuracy"].get(label, ""),
                ]
            )
        for avg_key in ("macro avg", "weighted avg"):
            row = report_dict.get(avg_key, {})
            writer.writerow(
                [
                    avg_key,
                    row.get("precision", ""),
                    row.get("recall", ""),
                    row.get("f1-score", ""),
                    row.get("support", ""),
                ]
            )

    ctx = get_model_context(cfg)
    target = target_status(accuracy)
    recommendations = training_recommendations(
        history_file.get("final_validation_accuracy")
        or history_file.get("config", {}).get("best_validation_accuracy")
        or (hist_series.get("val_accuracy") or [None])[-1],
        accuracy,
        history_file.get("config", {}).get("training_mode_key"),
    )
    if dataset_summary.get("warnings"):
        recommendations = list(dataset_summary["warnings"][:2]) + list(recommendations or [])

    result = {
        "accuracy": float(accuracy),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "precision_macro": float(metrics["precision_macro"]),
        "recall_macro": float(metrics["recall_macro"]),
        "f1_macro": float(metrics["f1_macro"]),
        "cohen_kappa": metrics["cohen_kappa"],
        "confusion_matrix": metrics["confusion_matrix"],
        "confusion_matrix_normalized": metrics["confusion_matrix_normalized"],
        "classification_report": report_dict,
        "history": history_file,
        "evaluated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "test_samples": sample_count,
        "accuracy_target": ACCURACY_TARGET,
        "target_achieved": target["achieved"],
        "target_status": target,
        "weak_classes": weak_classes(report_dict),
        "recommendations": recommendations,
        "per_class": metrics["per_class"],
        "per_class_accuracy": metrics["per_class_accuracy"],
        "misclassified_count": len(misclassified),
        "misclassified_images": misclassified[:500],
        "confused_pairs": confused_pairs,
        "correct_predictions": sample_count - len(misclassified),
        "incorrect_predictions": len(misclassified),
        "evaluation_split": "test_holdout",
        "dataset_summary": dataset_summary,
        "active_engine": "local_tta_pipeline" if not use_external else "ensemble_pipeline",
        "model_architecture": (
            "EfficientNetB0 (Local + TTA)"
            if not use_external
            else "Ensemble (External Keras H5 + EfficientNetB0)"
        ),
        "model_path": local_path if not use_external else get_active_model_path(cfg),
        "ensemble_models": {
            "external": external_path if use_external else None,
            "local": local_path,
            "external_weight": external_weight if use_external else 0.0,
            "local_weight": local_weight if use_external else 1.0,
            "tta_enabled": use_tta,
            "calibrator_applied": bool(calibrator is not None),
            "batch_size": batch_size,
        },
        "invalid_predictions": metrics["invalid_predictions"],
    }

    with open(cfg["EVALUATION_RESULT_PATH"], "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    save_json(
        cfg["MISCLASSIFICATION_REPORT_PATH"],
        {
            "generated_at": result["evaluated_at"],
            "split": "test",
            "total": sample_count,
            "misclassified_count": len(misclassified),
            "misclassified_images": misclassified[:500],
        },
    )
    save_json(
        cfg["CONFUSION_PAIR_REPORT_PATH"],
        {
            "generated_at": result["evaluated_at"],
            "split": "test",
            "pairs": confused_pairs,
        },
    )
    return result
