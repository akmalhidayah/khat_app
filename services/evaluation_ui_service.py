"""UI helpers for the model evaluation dashboard — no metric manipulation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from services.metrics_service import ACCURACY_TARGET

CLASS_LABELS_ID = {
    "naskhi": "Naskhi",
    "diwani": "Diwani",
    "diwani_jali": "Diwani Jali",
    "tsuluts": "Tsuluts",
}

CLASS_CHART_COLORS = {
    "naskhi": "#047857",
    "diwani": "#4338ca",
    "diwani_jali": "#b45309",
    "tsuluts": "#0e7490",
}

VISUALLY_SIMILAR_PAIRS = {
    frozenset({"diwani_jali", "tsuluts"}),
    frozenset({"diwani", "diwani_jali"}),
    frozenset({"diwani", "tsuluts"}),
    frozenset({"naskhi", "diwani"}),
}

ERROR_TYPE_STYLES = {
    "high_confidence": {"badge": "err-high", "tone": "critical"},
    "label_issue": {"badge": "err-label", "tone": "warning"},
    "low_confidence": {"badge": "err-low", "tone": "orange"},
    "ambiguous": {"badge": "err-ambiguous", "tone": "info"},
    "class_confusion": {"badge": "err-confusion", "tone": "warning"},
}

DEFAULT_RECOMMENDATIONS = [
    "Tambahkan gambar Naskhi dan Diwani untuk mengurangi ketidakseimbangan kelas.",
    "Hapus atau pisahkan gambar dengan label salah dan kualitas rendah.",
    "Jalankan cleaning dataset dan augmentasi seimbang sebelum retraining.",
    "Tingkatkan epoch training riset dan aktifkan fine-tuning.",
    "Gunakan class weights untuk mengurangi bias terhadap kelas dominan.",
    "Tinjau gambar salah klasifikasi sebelum training ulang.",
    "Evaluasi ulang model setelah perbaikan dataset.",
]


def display_class(name: str) -> str:
    return CLASS_LABELS_ID.get(name, (name or "").replace("_", " ").title())


def format_metric_pct(value) -> str:
    """Match evaluation table display: two decimal places."""
    if value is None:
        return "—"
    return f"{round(float(value) * 100, 2)}%"


def resolve_per_class_chart_order(per_class: Dict, preferred_labels: Optional[List[str]] = None) -> List[str]:
    """Same class order as Kinerja Per Kelas table (sorted by F1 ascending)."""
    source = dict(per_class or {})
    if not source:
        return list(preferred_labels or [])

    labels = [label for label in (preferred_labels or []) if label in source]
    for label in source:
        if label not in labels:
            labels.append(label)

    def _f1_value(label: str) -> float:
        row = source.get(label, {})
        return float(row.get("f1_score") or row.get("f1-score") or 0)

    return sorted(labels, key=_f1_value)


def resolve_test_image_rel(row: Dict, test_dir: Optional[str] = None, optimized_test_dir: Optional[str] = None) -> str:
    """Build path relative to test split root, e.g. diwani_jali/uuid.jpg."""
    filename = (row.get("filename") or "").replace("\\", "/").strip()
    image_path = (row.get("image_path") or "").replace("\\", "/").strip()
    true_label = (row.get("true_label") or "").strip()

    if filename and "/" in filename:
        return filename

    bases = [b for b in (test_dir, optimized_test_dir) if b]
    if image_path:
        normalized = os.path.normpath(image_path)
        for base in bases:
            try:
                rel = os.path.relpath(normalized, base)
            except ValueError:
                continue
            if not rel.startswith("..") and os.path.isfile(normalized):
                return rel.replace("\\", "/")

        for marker in ("/dataset/test/", "/optimized/model/test/"):
            if marker in image_path:
                return image_path.split(marker, 1)[1]

    if true_label and filename:
        return f"{true_label}/{filename}"

    return filename


def metric_tone(value: Optional[float], target: float = ACCURACY_TARGET) -> str:
    if value is None:
        return "muted"
    pct = float(value) * 100 if value <= 1 else float(value)
    target_pct = target * 100 if target <= 1 else target
    if pct >= target_pct:
        return "success"
    if pct >= max(75.0, target_pct * 0.88):
        return "warning"
    return "critical"


METRIC_DESCRIPTIONS = {
    "accuracy": "Proporsi prediksi benar dari seluruh sampel test.",
    "precision": "Seberapa akurat prediksi positif model untuk setiap kelas.",
    "recall": "Proporsi sampel benar yang berhasil terdeteksi model.",
    "f1": "Rata-rata harmonik presisi dan recall — keseimbangan kinerja.",
}


def metric_description(name: str) -> str:
    return METRIC_DESCRIPTIONS.get(name, "")


def metric_card_hint(name: str, value: Optional[float], target: float = ACCURACY_TARGET) -> str:
    if value is None:
        return "Belum tersedia"
    pct = round(float(value) * 100, 2) if value <= 1 else round(float(value), 2)
    target_pct = target * 100 if target <= 1 else target
    gap = round(target_pct - pct, 2)
    if pct >= target_pct:
        return "Target tercapai"
    hints = {
        "accuracy": f"Di bawah target {gap}%",
        "precision": "Perlu peningkatan",
        "recall": "Banyak sampel terlewat",
        "f1": "Di bawah target",
    }
    return hints.get(name, f"Di bawah target {gap}%")


def find_test_image_path(image_rel: str, config: Optional[Dict] = None) -> Optional[Path]:
    """Resolve a test-set image to an absolute path using pathlib."""
    rel = (image_rel or "").replace("\\", "/").strip()
    if not rel or ".." in rel:
        return None

    if config is None:
        from flask import current_app
        config = current_app.config

    bases = [
        Path(config["TEST_DIR"]),
        Path(config["OPTIMIZED_MODEL_DIR"]) / "test",
    ]

    for base in bases:
        candidate = (base / rel).resolve()
        try:
            candidate.relative_to(base.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            return candidate

    if "/" not in rel:
        for base in bases:
            if not base.is_dir():
                continue
            for cls in config.get("CLASS_LABELS", []):
                candidate = (base / cls / rel).resolve()
                if candidate.is_file():
                    return candidate
    return None


def _is_similar_class_pair(true_label: Optional[str], predicted_label: Optional[str]) -> bool:
    if not true_label or not predicted_label:
        return False
    return frozenset({true_label, predicted_label}) in VISUALLY_SIMILAR_PAIRS


def classify_error_type(
    confidence_pct: Optional[float],
    margin_pct: Optional[float],
    true_label: Optional[str] = None,
    predicted_label: Optional[str] = None,
) -> Dict[str, str]:
    conf = float(confidence_pct or 0)
    margin = float(margin_pct or 0)

    if conf >= 85:
        return {
            "key": "high_confidence",
            "label": "High Confidence Error",
            "label_id": "High Confidence Error",
            "hint": "Review label and preprocessing immediately.",
            "recommendation": "Review label",
            "analysis": (
                "The model predicted the wrong class with high confidence. This may indicate mislabeled data, "
                "non-relevant visual patterns learned by the model, or strong overlap between the true and predicted classes."
            ),
        }

    if margin < 10:
        return {
            "key": "ambiguous",
            "label": "Ambiguous Prediction",
            "label_id": "Ambiguous Prediction",
            "hint": "Add more examples for both confusing classes.",
            "recommendation": "Inspect class confusion",
            "analysis": (
                "The model shows uncertainty because the probability gap between the top classes is small."
            ),
        }

    if _is_similar_class_pair(true_label, predicted_label):
        return {
            "key": "class_confusion",
            "label": "Class Confusion",
            "label_id": "Class Confusion",
            "hint": "Add distinguishing examples and improve class definition.",
            "recommendation": "Add similar training samples",
            "analysis": (
                "The true and predicted classes are visually similar. The model may be learning overlapping "
                "decorative or structural patterns between these Khat styles."
            ),
        }

    if conf >= 70:
        return {
            "key": "label_issue",
            "label": "Possible Label or Class Similarity Issue",
            "label_id": "Possible Label Issue",
            "hint": "Compare visual traits and inspect dataset quality.",
            "recommendation": "Check preprocessing",
            "analysis": (
                "The prediction result may indicate a label inconsistency or visual similarity between classes. "
                "Manual inspection is recommended."
            ),
        }

    if conf >= 60:
        return {
            "key": "low_confidence",
            "label": "Low Confidence Error",
            "label_id": "Low Confidence Error",
            "hint": "Add more training samples and improve preprocessing.",
            "recommendation": "Add similar training samples",
            "analysis": (
                "The model predicted the wrong class with moderate-to-low confidence. Additional training data "
                "and preprocessing review may improve separation."
            ),
        }

    return {
        "key": "low_confidence",
        "label": "Low Confidence Error",
        "label_id": "Low Confidence Error",
        "hint": "Add more training samples and improve preprocessing.",
        "recommendation": "Add similar training samples",
        "analysis": (
            "The model is uncertain about this prediction. The error may be caused by weak visual cues or insufficient training coverage."
        ),
    }


def build_modal_recommended_actions(row: Dict) -> List[str]:
    true_d = row.get("true_display") or display_class(row.get("true_label"))
    pred_d = row.get("predicted_display") or display_class(row.get("predicted_label"))
    actions = [
        "Review whether the true label is correct.",
        "Check whether the image contains excessive border, background, or decoration.",
        f"Compare this image with other {true_d} and {pred_d} samples.",
        f"Add more similar {true_d} samples to the dataset.",
        "Re-run preprocessing and evaluation after correction.",
    ]
    if row.get("error_type") == "high_confidence":
        actions.insert(0, "Treat this as a priority label audit case because confidence is ≥85%.")
    return actions


def build_confusion_insight(pairs: List[Dict], misc_summary: Dict) -> Dict[str, str]:
    top = misc_summary.get("top_pair") or (pairs[0] if pairs else {})
    if not top:
        return {
            "title": "Class Confusion Insight",
            "message": "No confusion pairs detected yet.",
            "action": "Run evaluation on the test holdout set.",
        }
    true_d = top.get("true_display") or display_class(top.get("true_class"))
    pred_d = top.get("predicted_display") or display_class(top.get("predicted_class"))
    count = top.get("count", 0)
    return {
        "title": "Class Confusion Insight",
        "message": (
            f"{true_d} is frequently predicted as {pred_d} ({count} error{'s' if count != 1 else ''}). "
            f"Add more {true_d} samples with diverse visual styles and review mislabeled {pred_d} images."
        ),
        "action": f"Inspect the {true_d} → {pred_d} confusion pair in the error table below.",
        "true_class": true_d,
        "predicted_class": pred_d,
        "count": count,
    }


def build_error_dashboard_cards(
    test_samples: Optional[int],
    correct_predictions: Optional[int],
    incorrect_predictions: Optional[int],
    accuracy_pct: Optional[float],
    misc_summary: Dict,
    misclassified_rows: List[Dict],
) -> List[Dict[str, Any]]:
    confidences = [r.get("confidence_pct") for r in misclassified_rows if r.get("confidence_pct") is not None]
    avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else None
    cards = [
        {
            "key": "total",
            "label": "Total Evaluated Images",
            "value": test_samples if test_samples is not None else "—",
            "icon": "bi-images",
            "tone": "neutral",
        },
        {
            "key": "correct",
            "label": "Correct Predictions",
            "value": correct_predictions if correct_predictions is not None else "—",
            "icon": "bi-check-circle",
            "tone": "success",
        },
        {
            "key": "wrong",
            "label": "Wrong Predictions",
            "value": incorrect_predictions if incorrect_predictions is not None else misc_summary.get("total", "—"),
            "icon": "bi-x-circle",
            "tone": "critical",
        },
        {
            "key": "accuracy",
            "label": "Accuracy",
            "value": f"{accuracy_pct}%" if accuracy_pct is not None else "—",
            "icon": "bi-bullseye",
            "tone": "info",
        },
        {
            "key": "high_conf",
            "label": "High Confidence Errors",
            "value": misc_summary.get("high_confidence_count", 0),
            "icon": "bi-exclamation-octagon",
            "tone": "critical",
        },
        {
            "key": "pair",
            "label": "Most Confused Pair",
            "value": misc_summary.get("top_pair_label") or "—",
            "icon": "bi-shuffle",
            "tone": "warning",
        },
        {
            "key": "weakest",
            "label": "Weakest Class",
            "value": misc_summary.get("weakest_class") or "—",
            "icon": "bi-bar-chart",
            "tone": "warning",
        },
        {
            "key": "avg_conf",
            "label": "Average Error Confidence",
            "value": f"{avg_conf}%" if avg_conf is not None else "—",
            "icon": "bi-speedometer2",
            "tone": "neutral",
        },
    ]
    return cards


def _image_meta_from_path(abs_path: Optional[Path]) -> Dict[str, Any]:
    meta = {"width": None, "height": None, "size_display": None, "format": None}
    if not abs_path or not abs_path.is_file():
        return meta
    size = abs_path.stat().st_size
    if size < 1024:
        meta["size_display"] = f"{size} B"
    elif size < 1024 * 1024:
        meta["size_display"] = f"{size / 1024:.1f} KB"
    else:
        meta["size_display"] = f"{size / (1024 * 1024):.2f} MB"
    try:
        from PIL import Image
        with Image.open(abs_path) as img:
            meta["width"], meta["height"] = img.size
            meta["format"] = img.format or abs_path.suffix.lstrip(".").upper()
    except Exception:
        meta["format"] = abs_path.suffix.lstrip(".").upper() or "Unknown"
    return meta


def enrich_misclassified_row(
    row: Dict,
    row_id: int = 0,
    test_dir: Optional[str] = None,
    optimized_test_dir: Optional[str] = None,
    image_url_builder=None,
    static_url_builder=None,
) -> Dict:
    conf_pct = row.get("confidence_pct")
    if conf_pct is None and row.get("confidence") is not None:
        conf_pct = round(float(row["confidence"]) * 100, 2)

    margin_pct = row.get("margin_pct")
    if margin_pct is None and row.get("top2_probability") is not None and conf_pct is not None:
        second_pct = float(row["top2_probability"]) * 100 if float(row["top2_probability"]) <= 1 else float(row["top2_probability"])
        margin_pct = round(conf_pct - second_pct, 2)

    true_label = row.get("true_label")
    predicted_label = row.get("predicted_label")
    error = classify_error_type(conf_pct, margin_pct, true_label=true_label, predicted_label=predicted_label)
    image_rel = resolve_test_image_rel(row, test_dir=test_dir, optimized_test_dir=optimized_test_dir)

    abs_path = None
    try:
        from flask import has_app_context
        if has_app_context():
            abs_path = find_test_image_path(image_rel)
    except Exception:
        pass
    if abs_path is None and test_dir:
        candidate = Path(test_dir) / image_rel
        if candidate.is_file():
            abs_path = candidate
    if abs_path is None and optimized_test_dir:
        candidate = Path(optimized_test_dir) / image_rel
        if candidate.is_file():
            abs_path = candidate

    image_meta = _image_meta_from_path(abs_path)
    from services.evaluation_image_service import resolve_evaluation_image

    image_info = resolve_evaluation_image(
        {**row, "image_rel": image_rel},
        static_url_builder=static_url_builder,
        test_image_url_builder=image_url_builder,
    )
    image_available = bool(image_info.get("image_available"))
    image_url = image_info.get("image_url")
    if image_info.get("preview_path"):
        abs_path = Path(image_info["preview_path"])
        image_meta = _image_meta_from_path(abs_path)
    elif not abs_path and image_info.get("image_exists"):
        located_path = image_info.get("preview_path")
        if located_path:
            abs_path = Path(located_path)
            image_meta = _image_meta_from_path(abs_path)

    top2_class = row.get("top2_class")
    top2_score = row.get("top2_probability")
    top2_pct = None
    if top2_score is not None:
        top2_pct = round(float(top2_score) * 100, 2) if float(top2_score) <= 1 else round(float(top2_score), 2)

    style = ERROR_TYPE_STYLES.get(error["key"], ERROR_TYPE_STYLES["label_issue"])
    enriched = {
        **row,
        "id": row_id,
        "confidence_pct": conf_pct,
        "margin_pct": margin_pct,
        "true_display": display_class(true_label),
        "predicted_display": display_class(predicted_label),
        "top2_class": top2_class,
        "top2_display": display_class(top2_class) if top2_class else None,
        "top2_score_pct": top2_pct,
        "error_type": error["key"],
        "error_type_label": error["label_id"],
        "error_badge_class": style["badge"],
        "error_tone": style["tone"],
        "error_hint": error["hint"],
        "recommendation": error["recommendation"],
        "error_analysis": error["analysis"],
        "is_high_confidence_error": error["key"] == "high_confidence",
        "image_rel": image_rel,
        "image_url": image_url,
        "image_available": image_available,
        "image_exists": image_info.get("image_exists", image_available),
        "preview_filename": image_info.get("preview_filename"),
        "preview_status": image_info.get("preview_status") or ("available" if image_available else "unavailable"),
        "source_folder": image_info.get("source_folder"),
        "input_source": "Test Holdout Dataset",
        "reviewed_status": row.get("reviewed_status") or "pending",
        "correction_label": row.get("correction_label"),
        "correction_note": row.get("correction_note"),
        **image_meta,
    }
    enriched["modal_actions"] = build_modal_recommended_actions(enriched)
    if conf_pct is not None and top2_pct is not None:
        enriched["margin_calculation"] = f"Top-2 Margin = {conf_pct:.2f}% − {top2_pct:.2f}% = {margin_pct:.2f}%"
    else:
        enriched["margin_calculation"] = f"Top-2 Margin = {margin_pct:.2f}%" if margin_pct is not None else "—"
    enriched["confidence_category"] = (
        "Strong Prediction" if (conf_pct or 0) >= 85 else
        "Moderate Prediction" if (conf_pct or 0) >= 70 else
        "Weak Prediction" if (conf_pct or 0) >= 60 else
        "Low Confidence"
    )
    return enriched


def per_class_row_status(f1: float, recall: float, target: float = ACCURACY_TARGET) -> Dict[str, Any]:
    f1_pct = float(f1) * 100
    recall_pct = float(recall) * 100
    target_pct = target * 100
    if f1_pct >= target_pct:
        return {
            "status": "Target achieved",
            "status_id": "Target tercapai",
            "badge": "ready",
            "reason": "",
            "recommendation": "Pertahankan kualitas data kelas ini.",
        }
    if recall_pct < 50:
        return {
            "status": "Critical",
            "status_id": "Kritis",
            "badge": "critical",
            "reason": f"Recall {recall_pct:.2f}%",
            "recommendation": "Tambah sampel dan audit label kelas ini.",
        }
    if f1_pct < 70:
        return {
            "status": "Below target",
            "status_id": "Di bawah target",
            "badge": "critical" if recall_pct < 50 else "warning",
            "reason": f"F1 {f1_pct:.2f}%",
            "recommendation": "Augmentasi dan class weights disarankan.",
        }
    return {
        "status": "Moderate",
        "status_id": "Sedang",
        "badge": "warning",
        "reason": f"F1 {f1_pct:.2f}% (target {target_pct:.0f}%)",
        "recommendation": "Tingkatkan sampel atau fine-tuning.",
    }


def backfill_class_accuracy(
    eval_json: dict,
    class_labels: List[str] | None = None,
) -> dict:
    """Ensure per_class and per_class_accuracy exist using confusion matrix."""
    if not eval_json:
        return eval_json

    labels = class_labels or eval_json.get("dataset_summary", {}).get("class_labels") or []
    cm = eval_json.get("confusion_matrix") or []
    if not labels and cm:
        labels = list((eval_json.get("per_class") or {}).keys())

    per_class = dict(eval_json.get("per_class") or {})
    per_class_accuracy = dict(eval_json.get("per_class_accuracy") or {})

    for index, label in enumerate(labels):
        if index >= len(cm):
            break
        row = cm[index]
        row_sum = sum(row) if row else 0
        accuracy = round(row[index] / row_sum, 6) if row_sum > 0 else None
        if accuracy is None:
            continue
        per_class_accuracy[label] = accuracy
        entry = dict(per_class.get(label) or {})
        if entry.get("class_accuracy") is None:
            entry["class_accuracy"] = accuracy
        per_class[label] = entry

    if not per_class and eval_json.get("classification_report"):
        for key, vals in eval_json["classification_report"].items():
            if not isinstance(vals, dict) or key in ("accuracy", "macro avg", "weighted avg"):
                continue
            per_class[key] = {
                "precision": vals.get("precision"),
                "recall": vals.get("recall"),
                "f1_score": vals.get("f1-score"),
                "support": vals.get("support"),
                "class_accuracy": per_class_accuracy.get(key, vals.get("recall")),
            }

    eval_json["per_class"] = per_class
    eval_json["per_class_accuracy"] = per_class_accuracy
    return eval_json


def _resolve_class_accuracy(
    class_key: str,
    vals: Dict[str, Any],
    confusion_matrix: Optional[List[List[int]]],
    class_labels: Optional[List[str]],
) -> Optional[float]:
    class_accuracy = vals.get("class_accuracy")
    if class_accuracy is not None:
        return float(class_accuracy)

    if confusion_matrix and class_labels and class_key in class_labels:
        index = class_labels.index(class_key)
        if index < len(confusion_matrix):
            row = confusion_matrix[index]
            row_sum = sum(row) if row else 0
            if row_sum > 0:
                return row[index] / row_sum

    recall = vals.get("recall")
    return float(recall) if recall is not None else None


def build_per_class_rows(
    per_class: Dict,
    report_rows: Dict,
    target: float = ACCURACY_TARGET,
    confusion_matrix: Optional[List[List[int]]] = None,
    class_labels: Optional[List[str]] = None,
) -> List[Dict]:
    rows = []
    source = dict(per_class or {})
    if not source and report_rows:
        for key, vals in report_rows.items():
            if not isinstance(vals, dict) or vals.get("precision") is None:
                continue
            if key in ("accuracy", "macro avg", "weighted avg"):
                continue
            source[key] = {
                "precision": vals.get("precision"),
                "recall": vals.get("recall"),
                "f1_score": vals.get("f1-score"),
                "support": vals.get("support"),
                "class_accuracy": vals.get("class_accuracy"),
            }

    label_order = class_labels or list(source.keys())
    for cls in label_order:
        vals = source.get(cls)
        if not vals:
            continue
        f1 = vals.get("f1_score", vals.get("f1-score", 0))
        recall = vals.get("recall", 0)
        class_accuracy = _resolve_class_accuracy(cls, vals, confusion_matrix, class_labels)
        status = per_class_row_status(f1, recall, target)
        rows.append(
            {
                "class_key": cls,
                "class_name": display_class(cls),
                "precision": vals.get("precision"),
                "recall": recall,
                "f1_score": f1,
                "support": vals.get("support"),
                "class_accuracy": class_accuracy,
                **status,
            }
        )
    for cls, vals in source.items():
        if cls in label_order:
            continue
        f1 = vals.get("f1_score", vals.get("f1-score", 0))
        recall = vals.get("recall", 0)
        class_accuracy = _resolve_class_accuracy(cls, vals, confusion_matrix, class_labels)
        status = per_class_row_status(f1, recall, target)
        rows.append(
            {
                "class_key": cls,
                "class_name": display_class(cls),
                "precision": vals.get("precision"),
                "recall": recall,
                "f1_score": f1,
                "support": vals.get("support"),
                "class_accuracy": class_accuracy,
                **status,
            }
        )
    rows.sort(key=lambda r: (r.get("f1_score") or 0))
    return rows


def resolve_eval_class_labels(config: Optional[Dict] = None) -> List[str]:
    """Class order aligned with model output indices (from class_indices.json)."""
    import json

    if config is None:
        from flask import current_app

        config = current_app.config

    labels = list(config.get("CLASS_LABELS") or [])
    path = config.get("CLASS_INDICES_PATH")
    if not path or not os.path.isfile(path):
        return labels
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        indices = data.get("class_indices", data) if isinstance(data, dict) else {}
        if isinstance(indices, dict) and indices:
            return sorted(indices.keys(), key=lambda key: int(indices[key]))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return labels


def build_active_model_eval_display(config: Optional[Dict] = None, eval_json: Optional[Dict] = None) -> Dict[str, Any]:
    if config is None:
        from flask import current_app

        config = current_app.config

    eval_json = eval_json or {}
    if eval_json.get("active_engine") == "ensemble_pipeline":
        ensemble = eval_json.get("ensemble_models") or {}
        return {
            "available": True,
            "engine": "ensemble_pipeline",
            "engine_label": "Ensemble (External + EfficientNetB0)",
            "model_name": "Ensemble Pipeline",
            "model_path": eval_json.get("model_path") or config.get("EXTERNAL_MODEL_PATH"),
            "runtime": "TensorFlow Keras",
            "architecture": eval_json.get("model_architecture") or "Ensemble (External Keras H5 + EfficientNetB0)",
            "ensemble_external": ensemble.get("external"),
            "ensemble_local": ensemble.get("local"),
            "ensemble_external_weight": ensemble.get("external_weight"),
            "ensemble_local_weight": ensemble.get("local_weight"),
            "dataset_source": config.get("EXTERNAL_DATASET_SOURCE") or config.get("EXTERNAL_DATASET_DIR"),
        }

    if config.get("USE_EXTERNAL_MODEL"):
        model_path = config.get("EXTERNAL_MODEL_PATH") or config.get("MODEL_PATH") or ""
        return {
            "available": True,
            "engine": "keras_h5",
            "engine_label": "TensorFlow Keras (External)",
            "model_name": os.path.basename(model_path) if model_path else "keras_model.h5",
            "model_path": model_path,
            "labels_path": config.get("EXTERNAL_LABELS_PATH"),
            "dataset_source": config.get("EXTERNAL_DATASET_SOURCE") or config.get("EXTERNAL_DATASET_DIR"),
            "runtime": "TensorFlow Keras",
            "architecture": eval_json.get("model_architecture") or "Keras H5 Transfer Learning",
        }

    from services.teachable_machine_service import is_teachable_machine_available, get_tm_display_info
    from services.external_assets_service import should_use_keras_model
    from services.model_context_service import get_active_model_path, get_model_context

    if is_teachable_machine_available() and not should_use_keras_model():
        info = get_tm_display_info()
        return {
            "available": True,
            "engine": "teachable_machine",
            "engine_label": "Teachable Machine",
            "model_name": info.get("model_name"),
            "model_path": config.get("TEACHABLE_MODEL_DIR"),
            "runtime": info.get("model_runtime"),
            "architecture": info.get("model_architecture"),
        }

    ctx = get_model_context(config)
    return {
        "available": True,
        "engine": "keras",
        "engine_label": "TensorFlow Keras",
        "model_name": os.path.basename(get_active_model_path(config)),
        "model_path": get_active_model_path(config),
        "runtime": "TensorFlow Keras",
        "architecture": ctx.get("model_architecture"),
    }


def build_accuracy_formula_display(
    correct: Optional[int],
    total: Optional[int],
    accuracy: Optional[float],
) -> Dict[str, Any]:
    if correct is None or not total:
        return {"available": False}
    pct = round(float(accuracy) * 100, 2) if accuracy is not None else round(correct / total * 100, 2)
    return {
        "available": True,
        "formula": f"Akurasi = benar / total = {correct} / {total} = {pct}%",
        "correct": correct,
        "total": total,
        "accuracy_pct": pct,
    }


def build_dataset_eval_display(dataset_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not dataset_summary:
        return {"available": False}

    rows = []
    for item in dataset_summary.get("per_class") or []:
        rows.append({
            "class_key": item.get("class_key"),
            "class_name": display_class(item.get("class_key")),
            "actual_count": item.get("actual_count", 0),
            "expected_count": item.get("expected_count"),
            "share_pct": item.get("share_pct", 0),
        })

    return {
        "available": True,
        "total_images": dataset_summary.get("total_images"),
        "total_dataset_raw": dataset_summary.get("total_dataset_raw"),
        "total_dataset_processed": dataset_summary.get("total_dataset_processed"),
        "holdout_percent": dataset_summary.get("holdout_percent"),
        "expected_total": dataset_summary.get("expected_total"),
        "train_images": dataset_summary.get("train_images"),
        "validation_images": dataset_summary.get("validation_images"),
        "processed_images": dataset_summary.get("processed_images"),
        "split_ratio": dataset_summary.get("split_ratio"),
        "effective_ratio": dataset_summary.get("effective_ratio"),
        "split_policy": dataset_summary.get("split_policy"),
        "stratified": dataset_summary.get("stratified"),
        "warnings": dataset_summary.get("warnings") or [],
        "per_class_rows": rows,
    }


def build_macro_metric_cards(eval_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    cards = []
    mapping = [
        ("precision_macro", "Presisi Macro", "bi-diagram-3"),
        ("recall_macro", "Recall Macro", "bi-bullseye"),
        ("f1_macro", "F1 Macro", "bi-graph-up"),
        ("cohen_kappa", "Cohen's Kappa", "bi-intersect"),
    ]
    for key, label, icon in mapping:
        value = eval_json.get(key)
        if value is None:
            continue
        pct = round(float(value) * 100, 2) if key != "cohen_kappa" else round(float(value) * 100, 2)
        display = f"{pct}%" if key != "cohen_kappa" else f"{round(float(value), 3)}"
        cards.append({
            "key": key,
            "label": label,
            "icon": icon,
            "value": display,
            "tone": metric_tone(value if key != "cohen_kappa" else min(float(value), 1.0)),
            "desc": "Rata-rata tidak tertimbang antar kelas." if key.endswith("_macro") else "Kesepakatan prediksi vs label di luar kebetulan.",
        })
    return cards


def merge_recommendations(existing: List[str]) -> List[str]:
    """Return a single consistent Indonesian recommendation block for the UI."""
    return DEFAULT_RECOMMENDATIONS[:7]


def normalize_training_history(raw: Dict) -> Dict:
    if not raw:
        return {}
    inner = raw.get("history")
    if isinstance(inner, dict) and ("accuracy" in inner or "val_accuracy" in inner):
        return inner
    return raw


def interpret_training_history(history: Dict) -> List[str]:
    history = normalize_training_history(history)
    if not history:
        return ["Riwayat training belum tersedia. Jalankan training untuk melihat kurva."]

    messages = []
    val_acc = history.get("val_accuracy") or []
    train_acc = history.get("accuracy") or []

    if len(val_acc) >= 2:
        spread = max(val_acc) - min(val_acc)
        if spread > 0.12:
            messages.append(
                "Akurasi validasi fluktuat — generalisasi model belum stabil."
            )

    if train_acc and val_acc:
        gap = float(train_acc[-1]) - float(val_acc[-1])
        if gap > 0.12:
            messages.append("Akurasi training jauh di atas validasi — indikasi overfitting.")

    if val_acc and float(val_acc[-1]) < 0.70:
        messages.append("Akurasi validasi masih rendah — perlu data lebih baik atau training lebih efektif.")

    if not messages:
        messages.append("Kurva training tersedia. Bandingkan training vs validation untuk stabilitas model.")

    return messages


def build_performance_insights(
    history: Dict,
    eval_summary: Dict,
    confusion_info: Dict,
) -> List[str]:
    """Concise bullet insights for the evaluation visuals panel."""
    insights = []
    history = normalize_training_history(history)
    val_acc = history.get("val_accuracy") or []
    train_acc = history.get("accuracy") or []

    if eval_summary.get("target_achieved"):
        insights.append("Model memenuhi target riset 85% — siap untuk pelaporan hasil akhir.")
    else:
        acc = eval_summary.get("accuracy_pct")
        gap = eval_summary.get("accuracy_gap")
        if acc is not None:
            insights.append(f"Akurasi test {acc}% — belum mencapai target 85% (gap {gap}%).")

    if confusion_info.get("message"):
        insights.append(confusion_info["message"])

    if train_acc and val_acc:
        gap = float(train_acc[-1]) - float(val_acc[-1])
        if gap > 0.12:
            insights.append("Indikasi overfitting: akurasi training lebih tinggi daripada validasi.")
        elif gap < 0.03 and val_acc:
            insights.append("Gap training–validation kecil — generalisasi relatif stabil.")
        if len(val_acc) >= 2 and val_acc[-1] > val_acc[0]:
            insights.append("Akurasi validasi cenderung meningkat sepanjang epoch.")
        elif val_acc and float(val_acc[-1]) < 0.70:
            insights.append("Akurasi validasi masih rendah — perlu perbaikan dataset atau konfigurasi training.")

    if len(val_acc) >= 2:
        spread = max(val_acc) - min(val_acc)
        if spread > 0.12:
            insights.append("Akurasi validasi fluktuatif — stabilitas model belum optimal.")

    if not insights:
        insights.append("Jalankan evaluasi dan tinjau confusion matrix serta kurva training untuk interpretasi lengkap.")

    return insights[:6]


def build_confusion_interpretation(confusion_matrix: List[List[int]]) -> Dict[str, Any]:
    if not confusion_matrix:
        return {"off_diagonal_high": False, "message": ""}
    total = sum(sum(row) for row in confusion_matrix)
    correct = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
    off_diag = total - correct
    ratio = off_diag / total if total else 0
    high = ratio > 0.25
    return {
        "off_diagonal_high": high,
        "off_diagonal_ratio": round(ratio * 100, 1),
        "message": (
            "Model masih sering membingungkan kelas yang mirip secara visual."
            if high
            else "Diagonal matrix relatif kuat — sebagian besar prediksi benar per kelas."
        ),
    }


def build_eval_summary(
    accuracy: Optional[float],
    f1: Optional[float],
    target: float = ACCURACY_TARGET,
) -> Dict[str, Any]:
    acc_status = metric_tone(accuracy, target)
    f1_status = metric_tone(f1, target)
    achieved = (
        accuracy is not None
        and f1 is not None
        and float(accuracy) >= target
        and float(f1) >= target
    )
    acc_pct = round(float(accuracy) * 100, 2) if accuracy is not None else None
    f1_pct = round(float(f1) * 100, 2) if f1 is not None else None
    target_pct = int(target * 100)
    return {
        "target_pct": target_pct,
        "accuracy_pct": acc_pct,
        "f1_pct": f1_pct,
        "accuracy_gap": round(target_pct - acc_pct, 2) if acc_pct is not None else None,
        "target_achieved": achieved,
        "readiness_label": "Siap laporan akhir" if achieved else "Belum layak laporan akhir",
        "readiness_message": (
            "Model memenuhi target riset 85% untuk akurasi dan F1-score pada test set."
            if achieved
            else "Model belum mencapai target riset 85%. Diperlukan pembersihan dataset, audit label, augmentasi seimbang, dan training Mode Riset Akurasi."
        ),
        "acc_tone": acc_status,
        "f1_tone": f1_status,
    }


def analyze_misclassification_summary(rows: List[Dict], pairs: List[Dict]) -> Dict[str, Any]:
    if not rows:
        return {}
    high_conf = [r for r in rows if r.get("is_high_confidence_error")]
    sorted_pairs = sorted(pairs or [], key=lambda p: p.get("count", 0), reverse=True)
    top_pair = sorted_pairs[0] if sorted_pairs else {}
    class_counts: Dict[str, int] = {}
    for row in rows:
        cls = row.get("true_label") or ""
        class_counts[cls] = class_counts.get(cls, 0) + 1
    weakest = max(class_counts, key=class_counts.get) if class_counts else None
    return {
        "total": len(rows),
        "high_confidence_count": len(high_conf),
        "top_pair": top_pair,
        "top_pair_label": (
            f"{display_class(top_pair.get('true_class'))} → {display_class(top_pair.get('predicted_class'))}"
            if top_pair
            else "—"
        ),
        "weakest_class": display_class(weakest) if weakest else "—",
        "weakest_class_key": weakest,
    }


def weak_class_summary(per_class_rows: List[Dict]) -> str:
    critical = [r for r in per_class_rows if r.get("badge") == "critical"]
    if len(critical) >= 2:
        names = ", ".join(r["class_name"] for r in critical[:2])
        return f"Kelas terlemah: {names}. Model sering salah klasifikasi ke kelas yang mirip (Diwani Jali / Tsuluts)."
    if critical:
        return f"Kelas terlemah: {critical[0]['class_name']}. Perlu lebih banyak sampel dan audit label."
    return "Semua kelas masih di bawah target riset — tingkatkan kualitas data dan training."
