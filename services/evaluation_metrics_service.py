"""Shared sklearn metrics for Keras and Teachable Machine evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from services.metrics_service import ACCURACY_TARGET


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: List[str],
) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    label_indices = list(range(len(labels)))

    invalid_mask = (y_pred_arr < 0) | (y_pred_arr >= len(labels))
    invalid_count = int(invalid_mask.sum())

    accuracy = accuracy_score(y_true_arr, y_pred_arr)
    precision_w = precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0, labels=label_indices)
    recall_w = recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0, labels=label_indices)
    f1_w = f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0, labels=label_indices)
    precision_m = precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0, labels=label_indices)
    recall_m = recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0, labels=label_indices)
    f1_m = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0, labels=label_indices)

    cm_arr = confusion_matrix(y_true_arr, y_pred_arr, labels=label_indices)
    cm = cm_arr.tolist()
    report_dict = classification_report(
        y_true_arr,
        y_pred_arr,
        target_names=labels,
        labels=label_indices,
        output_dict=True,
        zero_division=0,
    )

    per_class_accuracy: Dict[str, float | None] = {}
    cm_normalized: List[List[float]] = []
    for index, label in enumerate(labels):
        row_sum = int(cm_arr[index].sum())
        correct = int(cm_arr[index, index])
        per_class_accuracy[label] = round(correct / row_sum, 4) if row_sum > 0 else None
        if row_sum > 0:
            cm_normalized.append([round(float(value) / row_sum * 100, 2) for value in cm_arr[index]])
        else:
            cm_normalized.append([0.0 for _ in label_indices])

    kappa = float(cohen_kappa_score(y_true_arr, y_pred_arr, labels=label_indices))

    per_class = {
        label: {
            "precision": float(report_dict[label]["precision"]),
            "recall": float(report_dict[label]["recall"]),
            "f1_score": float(report_dict[label]["f1-score"]),
            "support": int(report_dict[label]["support"]),
            "class_accuracy": per_class_accuracy.get(label),
            "below_target": float(report_dict[label]["recall"]) < ACCURACY_TARGET,
        }
        for label in labels
        if label in report_dict
    }

    return {
        "accuracy": float(accuracy),
        "precision": float(precision_w),
        "recall": float(recall_w),
        "f1_score": float(f1_w),
        "precision_macro": float(precision_m),
        "recall_macro": float(recall_m),
        "f1_macro": float(f1_m),
        "cohen_kappa": kappa,
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_normalized,
        "classification_report": report_dict,
        "per_class": per_class,
        "per_class_accuracy": per_class_accuracy,
        "invalid_predictions": invalid_count,
    }
