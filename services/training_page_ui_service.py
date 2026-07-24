"""Model Management page UI helpers — diagnosis, safety checks, and display context."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

CLASS_LABELS_ID = {
    "naskhi": "Naskhi",
    "diwani": "Diwani",
    "diwani_jali": "Diwani Jali",
    "tsuluts": "Tsuluts",
}


def _display(key: str) -> str:
    return CLASS_LABELS_ID.get(key, key.replace("_", " ").title())


def _target_gap_text(
    percent: Optional[float],
    target_percent: float,
    label: str,
) -> Optional[str]:
    if percent is None:
        return None
    gap = round(target_percent - percent, 2)
    if gap <= 0:
        return f"{label} mencapai target {target_percent:.0f}%."
    return f"Kurang {gap:.2f}% dari target {label.lower()} {target_percent:.0f}%."


def build_training_safety(
    *,
    readiness: Dict[str, Any],
    validation_report: Dict[str, Any],
    classes_count: int,
    model_dir: str,
    quality_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    checks: List[Dict[str, str]] = []
    reasons: List[str] = []

    def add(name: str, ok: bool, fail_msg: str) -> None:
        checks.append({"name": name, "ok": ok, "message": "OK" if ok else fail_msg})
        if not ok:
            reasons.append(fail_msg)

    add("Dataset diproses", readiness.get("is_ready", False), "Dataset belum diproses dan dibagi.")
    add("Data training tersedia", readiness.get("train_count", 0) > 0, "Folder training kosong.")
    add("Data validasi tersedia", readiness.get("validation_count", 0) > 0, "Folder validasi kosong.")
    add("Data uji tersedia", readiness.get("test_count", 0) > 0, "Folder testing kosong.")
    add("Jumlah kelas = 4", classes_count == 4, "Konfigurasi kelas harus 4 (Diwani, Diwani Jali, Naskhi, Tsuluts).")
    writable = os.path.isdir(model_dir) and os.access(model_dir, os.W_OK)
    add("Model path dapat ditulis", writable, "Folder model tidak dapat ditulis.")
    qc = quality_report or {}
    distribution_ok = bool(qc.get("per_class_counts") or qc.get("class_counts"))
    add(
        "Distribusi kelas diperiksa",
        distribution_ok or readiness.get("raw_count", 0) > 0,
        "Jalankan quality check dataset sebelum training.",
    )

    if validation_report.get("errors"):
        reasons.append(validation_report["errors"][0])

    can_retrain = readiness.get("is_ready", False) and not validation_report.get("errors") and writable
    return {
        "can_retrain": can_retrain,
        "reasons": reasons,
        "checks": checks,
        "block_reason": reasons[0] if reasons and not can_retrain else None,
    }


def build_training_page_insights(
    *,
    readiness: Dict[str, Any],
    quality_report: Optional[Dict[str, Any]],
    target_validation: Dict[str, Any],
    target_test: Dict[str, Any],
    target_achieved: bool,
    accuracy_target: float,
    below_target: bool,
    eval_result: Optional[Dict[str, Any]],
    history: Optional[Dict[str, Any]],
    is_tm_active: bool,
    active_model_version: Optional[Any],
) -> Dict[str, Any]:
    qc = quality_report or {}
    per_class = qc.get("per_class_counts") or qc.get("class_counts") or {}
    imbalance_pct = float(qc.get("class_imbalance_percent") or qc.get("imbalance_ratio") or 0)

    sorted_classes = sorted(per_class.items(), key=lambda x: x[1]) if per_class else []
    minority = [_display(k) for k, _ in sorted_classes[:2]]
    dominant = [_display(k) for k, _ in sorted_classes[-2:]] if len(sorted_classes) >= 2 else []
    is_imbalanced = imbalance_pct > 25 or bool(qc.get("warnings"))

    target_pct = round(accuracy_target * 100, 1)
    val_pct = target_validation.get("percent")
    test_pct = target_test.get("percent")
    val_progress = min(100.0, (val_pct / target_pct * 100) if val_pct is not None else 0)
    test_progress = min(100.0, (test_pct / target_pct * 100) if test_pct is not None else 0)

    test_gap_text = _target_gap_text(test_pct, target_pct, "Akurasi uji")
    val_gap_text = _target_gap_text(val_pct, target_pct, "Akurasi validasi")

    diagnosis_items: List[Dict[str, str]] = []
    if is_imbalanced and minority and dominant:
        dom = " dan ".join(dominant)
        mino = " dan ".join(minority)
        diagnosis_items.append({
            "tone": "warn",
            "icon": "bi-pie-chart",
            "text": (
                f"Distribusi dataset belum seimbang. Kelas {dom} lebih dominan dibanding {mino}. "
                "Hal ini dapat menyebabkan model bias terhadap kelas mayoritas."
            ),
        })
    if val_pct is not None and val_pct < target_pct:
        diagnosis_items.append({
            "tone": "warn",
            "icon": "bi-graph-down",
            "text": "Validation accuracy masih perlu ditingkatkan.",
        })
    if below_target:
        diagnosis_items.append({
            "tone": "warn",
            "icon": "bi-bullseye",
            "text": "Akurasi model masih di bawah target riset 85%.",
        })
    if eval_result and eval_result.get("classification_report"):
        diagnosis_items.append({
            "tone": "info",
            "icon": "bi-shuffle",
            "text": "Periksa confusion matrix evaluasi untuk kemungkinan class confusion antar gaya Khat.",
        })
    if is_imbalanced:
        diagnosis_items.append({
            "tone": "info",
            "icon": "bi-sliders",
            "text": "Disarankan mengaktifkan class weights saat training eksperimental.",
        })
    if minority:
        diagnosis_items.append({
            "tone": "info",
            "icon": "bi-plus-circle",
            "text": f"Tambahkan lebih banyak gambar {', '.join(minority[:2])}.",
        })
    diagnosis_items.append({
        "tone": "info",
        "icon": "bi-arrow-repeat",
        "text": "Jalankan evaluasi ulang setelah retraining.",
    })

    train_acc = None
    val_acc_hist = None
    if history and history.get("accuracy") and history.get("val_accuracy"):
        train_acc = round(float(history["accuracy"][-1]) * 100, 2)
        val_acc_hist = round(float(history["val_accuracy"][-1]) * 100, 2)

    generalization_note = None
    if train_acc is not None and val_acc_hist is not None and val_acc_hist < train_acc - 5:
        generalization_note = (
            "Validation accuracy lebih rendah dari training accuracy. "
            "Kemungkinan terdapat generalization gap atau dataset belum seimbang."
        )

    has_eval = bool(eval_result and eval_result.get("accuracy") is not None)
    checklist = [
        {
            "text": "Gunakan dataset penuh",
            "status": "done" if readiness.get("train_count", 0) > 0 else "needs_attention",
        },
        {
            "text": "Aktifkan class weights",
            "status": "recommended" if is_imbalanced else "done",
        },
        {
            "text": "Terapkan augmentasi aman",
            "status": "recommended",
        },
        {
            "text": "Seimbangkan kelas minoritas",
            "status": "needs_attention" if is_imbalanced else "done",
        },
        {
            "text": "Pantau validation accuracy",
            "status": "needs_attention" if below_target else "done",
        },
        {
            "text": "Jalankan evaluasi confusion matrix",
            "status": "done" if has_eval else "recommended",
        },
        {
            "text": "Simpan versi model sebelum deployment",
            "status": "done" if active_model_version else "recommended",
        },
    ]

    return {
        "target_pct": target_pct,
        "val_pct": val_pct,
        "test_pct": test_pct,
        "val_progress": round(val_progress, 1),
        "test_progress": round(test_progress, 1),
        "test_gap_text": test_gap_text,
        "val_gap_text": val_gap_text,
        "val_below_target": val_pct is not None and val_pct < target_pct,
        "status_label": "Target Tercapai" if target_achieved else "Di Bawah Target",
        "status_tone": "success" if target_achieved else "warn",
        "diagnosis_items": diagnosis_items,
        "generalization_note": generalization_note,
        "checklist": checklist,
        "is_imbalanced": is_imbalanced,
        "minority_classes": minority,
        "dominant_classes": dominant,
        "is_tm_active": is_tm_active,
    }


def build_model_management_insights(
    *,
    readiness: Dict[str, Any],
    quality_report: Optional[Dict[str, Any]],
    eval_result: Optional[Dict[str, Any]],
    is_tm_active: bool,
    below_target: bool,
    target_achieved: bool = False,
) -> Dict[str, Any]:
    qc = quality_report or {}
    per_class = qc.get("per_class_counts") or qc.get("class_counts") or {}
    imbalance_pct = float(qc.get("class_imbalance_percent") or qc.get("imbalance_ratio") or 0)
    is_imbalanced = imbalance_pct > 25 or bool(qc.get("warnings"))

    sorted_classes = sorted(per_class.items(), key=lambda x: x[1]) if per_class else []
    minority = [_display(k) for k, _ in sorted_classes[:2]]
    has_eval = bool(eval_result and eval_result.get("accuracy") is not None)

    retraining_checklist: List[Dict[str, str]] = [
        {
            "text": "Akurasi masih di bawah target 85%",
            "status": "needs_attention" if below_target and not target_achieved else "done",
        },
        {
            "text": "Dataset belum seimbang",
            "status": "needs_attention" if is_imbalanced else "done",
        },
        {
            "text": f"Tambahkan gambar {' dan '.join(minority[:2]) if minority else 'kelas minoritas'}",
            "status": "needs_attention" if is_imbalanced and minority else ("recommended" if minority else "done"),
        },
        {
            "text": "Audit label dataset",
            "status": "recommended",
        },
        {
            "text": "Hapus gambar duplikat atau rusak",
            "status": "recommended",
        },
        {
            "text": "Training ulang di Teachable Machine",
            "status": "recommended" if below_target or is_imbalanced else "done",
        },
        {
            "text": "Export TensorFlow.js model",
            "status": "recommended",
        },
        {
            "text": "Upload model baru ke sistem",
            "status": "done" if is_tm_active else "recommended",
        },
    ]

    retraining_workflow = [
        {"step": 1, "title": "Perbaiki dataset", "icon": "bi-folder-check"},
        {"step": 2, "title": "Tambahkan gambar kelas minoritas", "icon": "bi-plus-square"},
        {"step": 3, "title": "Training ulang di Teachable Machine", "icon": "bi-mortarboard"},
        {"step": 4, "title": "Export TensorFlow.js model", "icon": "bi-box-arrow-up"},
        {"step": 5, "title": "Upload model baru ke sistem", "icon": "bi-cloud-upload"},
        {"step": 6, "title": "Jalankan evaluasi", "icon": "bi-graph-up"},
        {"step": 7, "title": "Aktifkan model baru", "icon": "bi-check-circle"},
    ]

    return {
        "notice": (
            "Model klasifikasi aktif adalah model Teachable Machine. "
            "Training dilakukan di luar aplikasi menggunakan Teachable Machine. "
            "Halaman ini mengelola, mengevaluasi, dan mengganti model yang diekspor."
        ),
        "retraining_checklist": retraining_checklist,
        "retraining_workflow": retraining_workflow,
        "is_imbalanced": is_imbalanced,
        "minority_classes": minority,
        "has_eval": has_eval,
        "dataset_ready": readiness.get("is_ready", False),
        "is_tm_active": is_tm_active,
    }


def build_eval_display(eval_result: Optional[Dict[str, Any]], class_labels: List[str]) -> Dict[str, Any]:
    if not eval_result or eval_result.get("accuracy") is None:
        return {"has_eval": False}

    accuracy = round(float(eval_result["accuracy"]) * 100, 2)
    precision = round(float(eval_result.get("precision", 0)) * 100, 2)
    recall = round(float(eval_result.get("recall", 0)) * 100, 2)
    f1 = round(float(eval_result.get("f1_score", 0)) * 100, 2)
    confusion = eval_result.get("confusion_matrix") or []
    per_class = eval_result.get("per_class") or {}

    per_class_rows = []
    for label in class_labels:
        stats = per_class.get(label) or {}
        if not stats and eval_result.get("classification_report"):
            stats = eval_result["classification_report"].get(label, {})
        per_class_rows.append({
            "class_name": _display(label),
            "class_key": label,
            "precision": round(float(stats.get("precision", 0)) * 100, 2),
            "recall": round(float(stats.get("recall", 0)) * 100, 2),
            "f1_score": round(float(stats.get("f1_score", stats.get("f1-score", 0))) * 100, 2),
            "support": int(stats.get("support", 0)),
            "below_target": bool(stats.get("below_target")),
        })

    return {
        "has_eval": True,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": confusion,
        "class_labels": [_display(label) for label in class_labels],
        "per_class_rows": per_class_rows,
        "test_samples": eval_result.get("test_samples"),
        "total_dataset_processed": (eval_result.get("dataset_summary") or {}).get("total_dataset_processed"),
        "holdout_percent": (eval_result.get("dataset_summary") or {}).get("holdout_percent"),
        "correct_predictions": eval_result.get("correct_predictions"),
        "incorrect_predictions": eval_result.get("incorrect_predictions"),
        "evaluated_at": eval_result.get("evaluated_at"),
        "model_source": eval_result.get("model_source") or "Teachable Machine",
    }
