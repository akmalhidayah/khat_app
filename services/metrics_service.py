"""Accuracy target helpers — all metrics must come from real evaluation/training."""

import os
from typing import Any, Dict, List, Optional

ACCURACY_TARGET = float(os.getenv("ACCURACY_TARGET", "0.85"))


def target_status(value: Optional[float], target: float = ACCURACY_TARGET) -> Dict[str, Any]:
    if value is None:
        return {"achieved": False, "label": "Not available", "tone": "muted", "percent": None}
    achieved = float(value) >= target
    return {
        "achieved": achieved,
        "label": "Target Achieved" if achieved else "Below Target",
        "tone": "success" if achieved else "warning",
        "percent": round(float(value) * 100, 2),
    }


def weak_classes(classification_report: Dict, target: float = ACCURACY_TARGET) -> List[Dict[str, Any]]:
    weak = []
    for class_name, metrics in classification_report.items():
        if not isinstance(metrics, dict):
            continue
        recall = metrics.get("recall")
        if recall is None:
            continue
        if float(recall) < target:
            weak.append({
                "class_name": class_name,
                "recall": float(recall),
                "precision": float(metrics.get("precision", 0)),
                "f1_score": float(metrics.get("f1-score", 0)),
                "message": f"Class {class_name} recall is below target ({float(recall)*100:.1f}% < {target*100:.0f}%).",
            })
    return weak


def training_recommendations(
    validation_accuracy: Optional[float],
    test_accuracy: Optional[float],
    training_mode_key: Optional[str] = None,
) -> List[str]:
    recs = []
    if training_mode_key in ("ultra_fast", "fast"):
        recs.append("Use Research Accuracy Mode with full dataset and two-phase fine-tuning for final results.")
    if validation_accuracy is not None and validation_accuracy < ACCURACY_TARGET:
        recs.extend([
            "Add more Naskhi and Diwani images to reduce class imbalance.",
            "Remove mislabeled or low-quality images from the dataset.",
            "Run dataset cleaning and balanced augmentation before retraining.",
            "Increase research training epochs and enable fine-tuning.",
            "Try EfficientNetB0 as an alternative architecture if VGG16 plateaus.",
            "Validate class labels manually, especially for visually similar styles.",
        ])
    if test_accuracy is not None and test_accuracy < ACCURACY_TARGET:
        recs.append("Testing accuracy is below 85%. The model is not yet ready for final research reporting.")
    return list(dict.fromkeys(recs))


def academic_improvement_recommendations(
    accuracy: Optional[float] = None,
    weak_classes_list: Optional[List[Dict]] = None,
    confused_pairs: Optional[List[Dict]] = None,
    is_teachable_machine: bool = False,
) -> List[str]:
    recs = []
    weak_classes_list = weak_classes_list or []
    confused_pairs = confused_pairs or []

    if accuracy is not None and accuracy < ACCURACY_TARGET:
        recs.append("Model accuracy is below the 85% research target. Do not rely on confidence alone — review validation status and per-class metrics.")

    for wc in weak_classes_list[:4]:
        name = wc.get("class_name", "").replace("_", " ").title()
        recs.append(f"Add more training images for {name} — recall is below target.")

    for pair in confused_pairs[:5]:
        true_d = pair.get("true_display") or pair.get("true_class", "").replace("_", " ").title()
        pred_d = pair.get("predicted_display") or (pair.get("predicted_class") or "").replace("_", " ").title()
        recs.append(f"Classes often confused: {true_d} predicted as {pred_d}. Add more visual variations and review mislabeled samples.")

    recs.extend([
        "Remove mislabeled images identified in dataset label audit.",
        "Balance the dataset between Diwani, Diwani Jali, Naskhi, and Tsuluts (minimum 300 images per class recommended).",
        "Add more visual variations (background, stroke thickness, layout) for visually similar classes.",
        "Evaluate using the separate 20% test holdout — never on training images.",
    ])

    if is_teachable_machine:
        recs.append("Retrain the Teachable Machine model with additional epochs and balanced classes, then replace static/model/ files.")
    else:
        recs.append("Retrain the model with higher epochs in Research Accuracy Mode.")

    return list(dict.fromkeys(recs))
