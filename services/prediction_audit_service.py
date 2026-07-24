"""Batch prediction audit on dataset splits with misclassification and confusion-pair analysis."""

import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from flask import current_app

from services.class_mapping_service import load_saved_class_indices, validate_class_mapping
from services.dataset_readiness_service import is_valid_trained_model
from services.model_builder_service import load_trained_classifier
from services.model_context_service import get_active_model_path, get_model_context
from services.preprocessing_service import preprocess_calligraphy_image
from services.result_interpretation_service import display_name
from services.training_utils import load_json, save_json

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def _entropy(probs: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def _reliability(confidence_pct: float, margin_pct: float) -> str:
    if confidence_pct >= 85 and margin_pct >= 20:
        return "strong"
    if confidence_pct >= 70 and margin_pct >= 15:
        return "acceptable"
    if confidence_pct < 60 or margin_pct < 10:
        return "unreliable"
    return "needs_review"


def _iter_split_images(split_dir: str, class_labels: List[str]) -> List[Dict]:
    rows = []
    if not os.path.isdir(split_dir):
        return rows
    for cls in class_labels:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for name in os.listdir(cls_dir):
            if "." not in name or name.rsplit(".", 1)[1].lower() not in ALLOWED_EXTENSIONS:
                continue
            path = os.path.join(cls_dir, name)
            if os.path.isfile(path):
                rows.append({"true_label": cls, "path": path, "filename": name})
    return rows


def audit_predictions_on_dataset(
    split: str = "test",
    use_tta: bool = False,
    config=None,
) -> Dict:
    if config is None:
        config = current_app.config

    split_key = split.lower()
    split_dirs = {
        "train": config["TRAIN_DIR"],
        "validation": config["VALIDATION_DIR"],
        "test": config["TEST_DIR"],
    }
    if split_key not in split_dirs:
        raise ValueError(f"Invalid split '{split}'. Use train, validation, or test.")

    model_path = get_active_model_path(config)
    if not is_valid_trained_model(model_path):
        raise FileNotFoundError("Trained model not found. Train model first.")

    ctx = get_model_context(config)
    architecture = ctx.get("model_architecture_key", "efficientnetb0")
    class_labels = list(config["CLASS_LABELS"])
    num_classes = len(class_labels)

    mapping = validate_class_mapping(config, raise_on_mismatch=True)
    class_indices = load_saved_class_indices(config) or mapping["canonical_order"]
    inv_map = {int(v): k for k, v in class_indices.items()}

    model = load_trained_classifier(model_path, architecture, num_classes)
    images = _iter_split_images(split_dirs[split_key], class_labels)
    if not images:
        raise ValueError(f"No images found in {split} split.")

    results: List[Dict] = []
    correct = 0
    wrong = 0
    low_confidence = 0
    ambiguous = 0
    wrong_per_class = Counter()
    pair_counts = Counter()

    for item in images:
        batch = preprocess_calligraphy_image(item["path"], architecture=architecture)
        if use_tta:
            from services.prediction_service import _tta_variants
            batches = _tta_variants(item["path"], architecture)
            probs_list = [model.predict(b, verbose=0)[0] for b in batches]
            probs = np.mean(probs_list, axis=0)
        else:
            probs = model.predict(batch, verbose=0)[0]

        pred_idx = int(np.argmax(probs))
        pred_label = inv_map.get(pred_idx, class_labels[pred_idx] if pred_idx < len(class_labels) else "unknown")
        sorted_idx = np.argsort(probs)[::-1]
        top2_idx = sorted_idx[:2]
        top_conf = float(probs[top2_idx[0]]) * 100
        second_conf = float(probs[top2_idx[1]]) * 100 if len(top2_idx) > 1 else 0.0
        margin = round(top_conf - second_conf, 2)
        rel = _reliability(top_conf, margin)
        is_correct = pred_label == item["true_label"]
        is_low = top_conf < 70
        is_ambiguous = margin < 15

        if is_correct:
            correct += 1
        else:
            wrong += 1
            wrong_per_class[item["true_label"]] += 1
            pair_counts[(item["true_label"], pred_label)] += 1

        if is_low:
            low_confidence += 1
        if is_ambiguous:
            ambiguous += 1

        top2 = [
            {
                "class": inv_map.get(int(i), class_labels[int(i)]),
                "display": display_name(inv_map.get(int(i), class_labels[int(i)])),
                "probability": float(probs[i]),
                "percent": round(float(probs[i]) * 100, 2),
            }
            for i in top2_idx
        ]

        entry = {
            "image_path": item["path"],
            "filename": item["filename"],
            "true_label": item["true_label"],
            "true_display": display_name(item["true_label"]),
            "predicted_label": pred_label,
            "predicted_display": display_name(pred_label),
            "confidence": float(probs[pred_idx]),
            "confidence_pct": round(top_conf, 2),
            "top2": top2,
            "margin_pct": margin,
            "entropy": round(_entropy(probs), 4),
            "correct": is_correct,
            "low_confidence": is_low,
            "ambiguous": is_ambiguous,
            "reliability": rel,
            "probability_distribution": {
                inv_map.get(i, class_labels[i]): round(float(probs[i]), 4) for i in range(len(probs))
            },
        }
        results.append(entry)

    total = len(results)
    accuracy = correct / total if total else 0.0

    confused_pairs = []
    for (true_cls, pred_cls), count in pair_counts.most_common():
        confused_pairs.append({
            "true_class": true_cls,
            "true_display": display_name(true_cls),
            "predicted_class": pred_cls,
            "predicted_display": display_name(pred_cls),
            "count": count,
            "percentage": round(count / wrong * 100, 2) if wrong else 0.0,
            "example_images": [
                r["image_path"] for r in results
                if not r["correct"] and r["true_label"] == true_cls and r["predicted_label"] == pred_cls
            ][:5],
        })

    misclassified = [r for r in results if not r["correct"]]

    report = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "split": split_key,
        "model_path": model_path,
        "architecture": architecture,
        "training_mode": ctx.get("training_mode_key"),
        "is_demo_model": ctx.get("is_demo_model", False),
        "tta_enabled": use_tta,
        "total_tested": total,
        "correct_predictions": correct,
        "wrong_predictions": wrong,
        "accuracy": round(accuracy, 4),
        "low_confidence_count": low_confidence,
        "ambiguous_count": ambiguous,
        "wrong_per_class": dict(wrong_per_class),
        "most_confused_pairs": confused_pairs,
        "wrong_images": misclassified[:500],
        "all_results": results,
    }
    save_json(config["PREDICTION_AUDIT_REPORT_PATH"], report)
    save_json(config["MISCLASSIFICATION_REPORT_PATH"], {
        "generated_at": report["generated_at"],
        "split": split_key,
        "total": total,
        "misclassified_count": wrong,
        "misclassified_images": misclassified[:500],
    })
    save_json(config["CONFUSION_PAIR_REPORT_PATH"], {
        "generated_at": report["generated_at"],
        "split": split_key,
        "pairs": confused_pairs,
        "recommendations": _pair_recommendations(confused_pairs),
    })
    return report


def _pair_recommendations(pairs: List[Dict]) -> List[str]:
    recs = []
    for pair in pairs[:5]:
        recs.append(
            f"Add more '{pair['true_display']}' samples or clean mislabeled images — "
            f"often confused with '{pair['predicted_display']}' ({pair['count']} times)."
        )
    if not recs:
        recs.append("No major confusion pairs detected in this audit.")
    return recs


def load_prediction_audit_report(config=None) -> Dict:
    if config is None:
        config = current_app.config
    return load_json(config.get("PREDICTION_AUDIT_REPORT_PATH"), {})


def load_misclassification_report(config=None) -> Dict:
    if config is None:
        config = current_app.config
    return load_json(config.get("MISCLASSIFICATION_REPORT_PATH"), {})


def load_confusion_pair_report(config=None) -> Dict:
    if config is None:
        config = current_app.config
    return load_json(config.get("CONFUSION_PAIR_REPORT_PATH"), {})
