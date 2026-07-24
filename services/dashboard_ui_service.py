"""Dashboard UI helpers — dataset balance, insights, and display labels."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

CLASS_LABELS_ID = {
    "naskhi": "Naskhi",
    "diwani": "Diwani",
    "diwani_jali": "Diwani Jali",
    "tsuluts": "Tsuluts",
}


def _pct(count: int, total: int) -> float:
    return round((count / total) * 100, 1) if total else 0.0


def build_dataset_insights(
    distribution: Dict[str, int],
    class_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    labels = class_labels or list(CLASS_LABELS_ID.keys())
    total = sum(distribution.get(k, 0) for k in labels)
    if total <= 0:
        return {
            "total": 0,
            "active_classes": 0,
            "is_imbalanced": False,
            "percentages": {},
            "dominant_classes": [],
            "minority_classes": [],
            "warning_message": None,
            "recommendation": None,
        }

    percentages = {k: _pct(distribution.get(k, 0), total) for k in labels}
    sorted_items = sorted(percentages.items(), key=lambda x: x[1])
    minority = [k for k, p in sorted_items if p < 15]
    dominant = [k for k, p in sorted_items if p > 30]
    spread = sorted_items[-1][1] - sorted_items[0][1]
    is_imbalanced = spread >= 25 or len(minority) >= 2

    warning_message = None
    recommendation = None
    if is_imbalanced:
        dom_names = ", ".join(CLASS_LABELS_ID.get(k, k) for k in dominant[:2])
        min_names = ", ".join(CLASS_LABELS_ID.get(k, k) for k in minority[:2])
        warning_message = (
            f"Dataset belum seimbang. Kelas {dom_names} jauh lebih dominan "
            f"dibanding {min_names}."
        )
        if minority:
            rec_names = " dan ".join(CLASS_LABELS_ID.get(k, k) for k in minority[:2])
            recommendation = f"Tambahkan data {rec_names} agar distribusi lebih seimbang."

    return {
        "total": total,
        "active_classes": sum(1 for k in labels if distribution.get(k, 0) > 0),
        "is_imbalanced": is_imbalanced,
        "percentages": percentages,
        "dominant_classes": dominant,
        "minority_classes": minority,
        "warning_message": warning_message,
        "recommendation": recommendation,
    }


def build_dashboard_insights(
    *,
    test_accuracy: Optional[float],
    accuracy_target: float,
    target_achieved: bool,
    dataset_insights: Dict[str, Any],
    detector_available: bool,
    khat_thresholds: Dict[str, float],
) -> Dict[str, Any]:
    acc_pct = round(test_accuracy * 100, 1) if test_accuracy is not None else None
    target_pct = round(accuracy_target * 100, 1)
    gap_pct = None
    if acc_pct is not None and not target_achieved:
        gap_pct = round(max(0, target_pct - acc_pct), 1)

    accuracy_status = "Belum dievaluasi"
    accuracy_tone = "muted"
    if acc_pct is not None:
        if target_achieved:
            accuracy_status = "Target tercapai"
            accuracy_tone = "success"
        elif gap_pct is not None and gap_pct <= 2:
            accuracy_status = "Mendekati target"
            accuracy_tone = "info"
        else:
            accuracy_status = "Di bawah target"
            accuracy_tone = "warn"

    dataset_status = "Belum ada data" if dataset_insights["total"] <= 0 else (
        "Belum seimbang" if dataset_insights["is_imbalanced"] else "Seimbang"
    )
    dataset_tone = "muted" if dataset_insights["total"] <= 0 else (
        "warn" if dataset_insights["is_imbalanced"] else "success"
    )

    detector_status = "Aktif" if detector_available else "Nonaktif"
    detector_tone = "success" if detector_available else "warn"

    recommendations: List[str] = []
    if acc_pct is not None and not target_achieved:
        recommendations.append(
            "Akurasi uji masih di bawah target 85%. Disarankan menambah data dan melakukan evaluasi ulang."
        )
    if dataset_insights.get("recommendation"):
        recommendations.append(dataset_insights["recommendation"])
    if detector_available:
        accept = round(khat_thresholds.get("accept", 0.7) * 100)
        reject = round(khat_thresholds.get("reject", 0.5) * 100)
        recommendations.append(
            f"Khat detector aktif untuk memvalidasi input sebelum klasifikasi (Accept ≥{accept:.0f}%, Reject <{reject:.0f}%)."
        )
    elif not recommendations:
        recommendations.append("Latih atau aktifkan khat detector untuk validasi input sebelum klasifikasi.")

    primary_recommendation = recommendations[0] if recommendations else "Sistem siap digunakan."

    return {
        "accuracy_status": accuracy_status,
        "accuracy_tone": accuracy_tone,
        "accuracy_pct": acc_pct,
        "accuracy_gap_pct": gap_pct,
        "target_pct": round(target_pct),
        "dataset_status": dataset_status,
        "dataset_tone": dataset_tone,
        "detector_status": detector_status,
        "detector_tone": detector_tone,
        "recommendations": recommendations,
        "primary_recommendation": primary_recommendation,
    }


def workflow_steps(
    *,
    total_dataset: int,
    has_training: bool,
    has_evaluation: bool,
    total_predictions: int,
) -> List[Dict[str, Any]]:
    """Return workflow step metadata with completion hints."""
    steps = [
        {"num": 1, "label": "Unggah Dataset", "icon": "bi-upload", "done": total_dataset > 0},
        {"num": 2, "label": "Preprocessing", "icon": "bi-sliders", "done": total_dataset > 0},
        {"num": 3, "label": "Training", "icon": "bi-cpu", "done": has_training},
        {"num": 4, "label": "Evaluasi", "icon": "bi-graph-up", "done": has_evaluation},
        {"num": 5, "label": "Prediksi", "icon": "bi-easel2", "done": total_predictions > 0},
    ]
    current_idx = 0
    for i, step in enumerate(steps):
        if not step["done"]:
            current_idx = i
            break
    else:
        current_idx = len(steps) - 1

    for i, step in enumerate(steps):
        step["current"] = i == current_idx
    return steps


def format_metric_pct(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return f"{round(value * 100, 1)}%"
