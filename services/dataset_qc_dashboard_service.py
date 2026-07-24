"""Comprehensive dataset quality dashboard for academic dataset review."""

import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

from flask import current_app
from PIL import Image

from services.dataset_quality_service import load_quality_report
from services.dataset_readiness_service import count_images_per_class
from services.label_audit_service import load_label_audit_report
from services.label_mismatch_service import load_label_mismatch_report
from services.dataset_cleaning_service import load_duplicate_report

MIN_IMAGES_PER_CLASS = 300
IMBALANCE_RATIO_THRESHOLD = 0.35
CLASS_LABELS_DISPLAY = {
    "naskhi": "Naskhi",
    "diwani": "Diwani",
    "diwani_jali": "Diwani Jali",
    "tsuluts": "Tsuluts",
}


def _resolution_summary(raw_dir: str, class_labels: List[str], sample_limit: int = 40) -> Dict:
    widths, heights = [], []
    sampled = 0
    for label in class_labels:
        class_dir = os.path.join(raw_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for name in os.listdir(class_dir):
            if sampled >= sample_limit:
                break
            path = os.path.join(class_dir, name)
            if not os.path.isfile(path):
                continue
            try:
                with Image.open(path) as img:
                    w, h = img.size
                widths.append(w)
                heights.append(h)
                sampled += 1
            except Exception:
                continue
    if not widths:
        return {"sampled": 0, "min_width": None, "max_width": None, "min_height": None, "max_height": None, "avg_width": None, "avg_height": None}
    return {
        "sampled": len(widths),
        "min_width": min(widths),
        "max_width": max(widths),
        "min_height": min(heights),
        "max_height": max(heights),
        "avg_width": round(sum(widths) / len(widths)),
        "avg_height": round(sum(heights) / len(heights)),
    }


def _duplicate_filenames(raw_dir: str, class_labels: List[str]) -> List[Dict]:
    seen: Dict[str, str] = {}
    duplicates = []
    for label in class_labels:
        class_dir = os.path.join(raw_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for name in os.listdir(class_dir):
            key = name.lower()
            if key in seen and seen[key] != label:
                duplicates.append({"filename": name, "classes": [seen[key], label]})
            elif key not in seen:
                seen[key] = label
    return duplicates[:50]


def _balance_status(counts: Dict[str, int]) -> Tuple[str, List[Dict]]:
    values = [counts.get(c, 0) for c in counts]
    total = sum(values)
    if total == 0:
        return "empty", []

    max_count = max(values)
    min_count = min(values)
    spread = (max_count - min_count) / max_count if max_count else 0
    rows = []
    for label, count in counts.items():
        pct = round(count / total * 100, 1) if total else 0
        status = "ok"
        warnings = []
        if count < MIN_IMAGES_PER_CLASS:
            status = "low"
            warnings.append(f"Below recommended minimum ({MIN_IMAGES_PER_CLASS} images).")
        if max_count and count < max_count * (1 - IMBALANCE_RATIO_THRESHOLD):
            status = "imbalanced"
            warnings.append("Dataset imbalance detected. Add more images to this class.")
        rows.append({
            "class_key": label,
            "class_display": CLASS_LABELS_DISPLAY.get(label, label.replace("_", " ").title()),
            "count": count,
            "percent": pct,
            "status": status,
            "warnings": warnings,
            "meets_minimum": count >= MIN_IMAGES_PER_CLASS,
        })

    if spread >= IMBALANCE_RATIO_THRESHOLD:
        overall = "imbalanced"
    elif any(r["count"] < MIN_IMAGES_PER_CLASS for r in rows):
        overall = "needs_more_data"
    else:
        overall = "balanced"
    return overall, rows


def build_dataset_qc_report(config=None, db_rows=None) -> Dict:
    if config is None:
        config = current_app.config

    class_labels = config["CLASS_LABELS"]
    raw_dir = config["RAW_DATASET_DIR"]
    counts = count_images_per_class(raw_dir, class_labels)
    total = sum(counts.values())
    balance_overall, class_rows = _balance_status(counts)

    quality = load_quality_report(config) or {}
    label_audit = load_label_audit_report(config) or {}
    label_mismatch = load_label_mismatch_report(config) or {}
    duplicate_report = load_duplicate_report(config) or {}

    missing_files = []
    mislabeled_warnings = []
    if db_rows:
        for row in db_rows:
            if hasattr(row, "file_exists") and not row.get("file_exists", True):
                missing_files.append(row.get("filename") or row.get("original_filename"))
            elif isinstance(row, dict) and not row.get("file_exists", True):
                missing_files.append(row.get("filename") or row.get("original_filename"))

    for item in label_audit.get("issues", [])[:20]:
        mislabeled_warnings.append(item.get("message") or str(item))
    for item in label_mismatch.get("mismatches", [])[:20]:
        mislabeled_warnings.append(
            item.get("message")
            or f"Possible mislabel: {item.get('filename')} — folder {item.get('folder_class')} vs filename hint {item.get('filename_class')}"
        )

    resolution = _resolution_summary(raw_dir, class_labels)
    dup_names = _duplicate_filenames(raw_dir, class_labels)

    return {
        "total_images": total,
        "class_counts": counts,
        "class_rows": class_rows,
        "balance_status": balance_overall,
        "balance_label": {
            "balanced": "Balanced",
            "imbalanced": "Imbalanced",
            "needs_more_data": "Needs More Data",
            "empty": "Empty",
        }.get(balance_overall, balance_overall),
        "min_images_target": MIN_IMAGES_PER_CLASS,
        "resolution": resolution,
        "duplicate_filenames": dup_names,
        "duplicate_filename_count": len(dup_names),
        "missing_file_count": len(missing_files),
        "missing_files": missing_files[:30],
        "corrupted_count": quality.get("corrupted_removed", quality.get("corrupted", 0)),
        "duplicate_removed_count": quality.get("duplicates_removed", quality.get("duplicate_removed", 0)),
        "quality_report_available": bool(quality),
        "mislabeled_warnings": mislabeled_warnings[:15],
        "cross_split_leakage": duplicate_report.get("cross_split_leakage_detected", False),
        "recommendations": _qc_recommendations(class_rows, balance_overall, total, quality, mislabeled_warnings),
    }


def _qc_recommendations(class_rows, balance_overall, total, quality, mislabeled) -> List[str]:
    recs = []
    if total == 0:
        return ["Upload Arabic Khat images into the raw dataset folders before training or evaluation."]
    for row in class_rows:
        if row["count"] < MIN_IMAGES_PER_CLASS:
            recs.append(
                f"Add more images for {row['class_display']} — current count {row['count']}, recommended minimum {MIN_IMAGES_PER_CLASS}."
            )
        if "Dataset imbalance detected" in " ".join(row["warnings"]):
            recs.append(f"Balance dataset: increase {row['class_display']} samples to match other classes.")
    if balance_overall == "imbalanced":
        recs.append("Run dataset balancing or collect additional samples for underrepresented classes.")
    if quality.get("corrupted_removed", 0):
        recs.append("Review corrupted images removed during cleaning and replace them with valid calligraphy samples.")
    if mislabeled:
        recs.append("Review possible mislabeled images flagged by label audit before retraining.")
    recs.append("Use the 80/20 train/test split and evaluate only on the held-out test set.")
    return list(dict.fromkeys(recs))
