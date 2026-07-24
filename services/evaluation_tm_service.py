"""Evaluate Teachable Machine model on held-out test set only."""

import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from flask import current_app

from services.evaluation_dataset_service import (
    build_dataset_summary,
    iter_test_images,
    resolve_evaluation_test_dir,
    validate_test_holdout_before_eval,
    validate_tm_class_alignment,
)
from services.evaluation_metrics_service import compute_classification_metrics
from services.metrics_service import ACCURACY_TARGET, academic_improvement_recommendations, target_status, weak_classes
from services.teachable_machine_service import get_tm_display_info, predict_tm_image
from services.training_utils import save_json
from services.evaluation_image_service import attach_preview_to_record


def evaluate_tm_model():
    import matplotlib
    matplotlib.use("Agg")
    from services.evaluation_service import _plot_confusion_matrix

    config = current_app.config
    class_labels = config["CLASS_LABELS"]
    test_dir, test_source = resolve_evaluation_test_dir(config)
    test_items = iter_test_images(test_dir, class_labels)
    if not test_items:
        raise ValueError("Test dataset is empty. Split dataset before evaluation. Evaluation uses test images only.")

    validate_test_holdout_before_eval(test_items, config)

    alignment = validate_tm_class_alignment(class_labels, config)
    if not alignment.get("valid"):
        raise ValueError(
            "Mapping kelas model TM tidak sesuai dataset. "
            + " ".join(alignment.get("warnings") or [])
        )

    dataset_summary = build_dataset_summary(test_dir, test_items, class_labels, config)
    dataset_summary["test_source_key"] = test_source

    tm_info = get_tm_display_info()
    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    y_true, y_pred = [], []
    misclassified = []
    pair_counts = {}

    for abs_path, true_label in test_items:
        result = predict_tm_image(abs_path, use_tta=False)
        pred_label = result["predicted_class"]
        pred_idx = label_to_idx.get(pred_label, -1)
        y_true.append(label_to_idx[true_label])
        y_pred.append(pred_idx)

        scores = result.get("scores") or {}
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_conf = round(sorted_scores[0][1] * 100, 2) if sorted_scores else 0
        second_conf = round(sorted_scores[1][1] * 100, 2) if len(sorted_scores) > 1 else 0
        margin = round(top_conf - second_conf, 2)

        if pred_label != true_label:
            pair_key = (true_label, pred_label)
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
            from services.evaluation_ui_service import classify_error_type

            error_meta = classify_error_type(
                top_conf,
                margin,
                true_label=true_label,
                predicted_label=pred_label,
            )
            misclassified.append(attach_preview_to_record({
                "image_path": abs_path,
                "filename": f"{true_label}/{os.path.basename(abs_path)}",
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": float(result.get("confidence") or 0),
                "confidence_pct": top_conf,
                "top2_class": sorted_scores[1][0] if len(sorted_scores) > 1 else None,
                "top2_probability": sorted_scores[1][1] if len(sorted_scores) > 1 else None,
                "margin_pct": margin,
                "error_type": error_meta["key"],
                "error_type_label": error_meta["label_id"],
                "error_hint": error_meta["hint"],
                "is_high_confidence_error": error_meta["key"] == "high_confidence",
                "probability_distribution": {k: round(float(v), 4) for k, v in scores.items()},
            }))

    metrics = compute_classification_metrics(y_true, y_pred, class_labels)
    labels = class_labels
    cm = metrics["confusion_matrix"]
    report_dict = metrics["classification_report"]

    confused_pairs = [
        {
            "true_class": k[0],
            "predicted_class": k[1],
            "true_display": k[0].replace("_", " ").title(),
            "predicted_display": k[1].replace("_", " ").title(),
            "count": v,
            "percentage": round(v / len(misclassified) * 100, 2) if misclassified else 0,
            "example_images": [m["image_path"] for m in misclassified if m["true_label"] == k[0] and m["predicted_label"] == k[1]][:5],
        }
        for k, v in sorted(pair_counts.items(), key=lambda x: -x[1])
    ]

    eval_dir = config["EVALUATION_FOLDER"]
    os.makedirs(eval_dir, exist_ok=True)
    from services.evaluation_service import generate_evaluation_charts

    generate_evaluation_charts(
        eval_dir,
        labels=labels,
        per_class=metrics["per_class"],
        classification_report=report_dict,
        plot_dpi=int(config.get("EVAL_PLOT_DPI", 100)),
    )
    _plot_confusion_matrix(cm, labels, os.path.join(eval_dir, "confusion_matrix.png"))

    csv_path = os.path.join(eval_dir, "classification_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["class", "precision", "recall", "f1-score", "support", "class_accuracy"])
        for label in labels:
            row = report_dict.get(label, {})
            writer.writerow([
                label,
                row.get("precision", ""),
                row.get("recall", ""),
                row.get("f1-score", ""),
                row.get("support", ""),
                metrics["per_class_accuracy"].get(label, ""),
            ])
        for avg_key in ("macro avg", "weighted avg"):
            row = report_dict.get(avg_key, {})
            writer.writerow([
                avg_key,
                row.get("precision", ""),
                row.get("recall", ""),
                row.get("f1-score", ""),
                row.get("support", ""),
                "",
            ])

    weak = weak_classes(report_dict)
    target = target_status(metrics["accuracy"])
    recommendations = academic_improvement_recommendations(
        accuracy=metrics["accuracy"],
        weak_classes_list=weak,
        confused_pairs=confused_pairs,
        is_teachable_machine=True,
    )
    if dataset_summary.get("warnings"):
        recommendations = list(dataset_summary["warnings"][:2]) + recommendations

    correct_count = len(test_items) - len(misclassified)
    result = {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "cohen_kappa": metrics["cohen_kappa"],
        "confusion_matrix": cm,
        "confusion_matrix_normalized": metrics["confusion_matrix_normalized"],
        "classification_report": report_dict,
        "history": {},
        "evaluated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "test_samples": len(test_items),
        "correct_predictions": correct_count,
        "incorrect_predictions": len(misclassified),
        "evaluation_split": "test_holdout",
        "dataset_summary": dataset_summary,
        "class_alignment": alignment,
        "model_source": tm_info["model_source"],
        "model_name": tm_info["model_name"],
        "model_runtime": tm_info["model_runtime"],
        "active_engine": "teachable_machine",
        "accuracy_target": ACCURACY_TARGET,
        "target_achieved": target["achieved"],
        "target_status": target,
        "weak_classes": weak,
        "recommendations": recommendations,
        "per_class": metrics["per_class"],
        "per_class_accuracy": metrics["per_class_accuracy"],
        "misclassified_count": len(misclassified),
        "misclassified_images": misclassified[:500],
        "confused_pairs": confused_pairs,
        "invalid_predictions": metrics["invalid_predictions"],
    }

    with open(config["EVALUATION_RESULT_PATH"], "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    save_json(config["MISCLASSIFICATION_REPORT_PATH"], {
        "generated_at": result["evaluated_at"],
        "split": "test",
        "total": len(test_items),
        "misclassified_count": len(misclassified),
        "misclassified_images": misclassified[:500],
    })
    save_json(config["CONFUSION_PAIR_REPORT_PATH"], {
        "generated_at": result["evaluated_at"],
        "split": "test",
        "pairs": confused_pairs,
    })
    return result
