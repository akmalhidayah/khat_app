"""Audit dataset labels across raw, split folders, database, and filename heuristics."""

import os
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

from flask import current_app

from models import Dataset
from services.result_interpretation_service import display_name, parse_filename_class
from services.training_utils import load_json, save_json

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def _is_image(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _scan_folder_labels(root: str, class_labels: List[str]) -> List[Dict]:
    rows = []
    if not os.path.isdir(root):
        return rows
    for cls in class_labels:
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for name in os.listdir(cls_dir):
            if not _is_image(name):
                continue
            path = os.path.join(cls_dir, name)
            if not os.path.isfile(path):
                continue
            suggested, suggested_display = parse_filename_class(name)
            rows.append({
                "image_path": path,
                "filename": name,
                "folder_label": cls,
                "database_label": None,
                "filename_suggested_label": suggested,
                "filename_suggested_display": suggested_display,
                "source": os.path.basename(root),
            })
    return rows


def _match_db_label(image_path: str, base_dir: str) -> Optional[str]:
    rel = os.path.relpath(image_path, base_dir).replace("\\", "/")
    row = Dataset.query.filter_by(image_path=rel).first()
    if row:
        return row.class_name
    basename = os.path.basename(image_path)
    row = Dataset.query.filter(
        (Dataset.filename == basename) | (Dataset.original_filename == basename)
    ).first()
    return row.class_name if row else None


def run_label_audit(config=None) -> Dict:
    if config is None:
        config = current_app.config

    class_labels = config["CLASS_LABELS"]
    base_dir = config["BASE_DIR"]
    splits = {
        "raw": config["RAW_DATASET_DIR"],
        "train": config["TRAIN_DIR"],
        "validation": config["VALIDATION_DIR"],
        "test": config["TEST_DIR"],
    }

    all_rows: List[Dict] = []
    for split_name, root in splits.items():
        for row in _scan_folder_labels(root, class_labels):
            row["split"] = split_name
            row["database_label"] = _match_db_label(row["image_path"], base_dir)
            all_rows.append(row)

    for db_row in Dataset.query.all():
        abs_path = os.path.join(base_dir, db_row.image_path)
        if not os.path.isfile(abs_path):
            all_rows.append({
                "image_path": abs_path,
                "filename": db_row.original_filename or db_row.filename,
                "folder_label": None,
                "database_label": db_row.class_name,
                "filename_suggested_label": None,
                "filename_suggested_display": None,
                "source": "database_only",
                "split": "missing_file",
                "dataset_id": db_row.id,
            })

    consistent = 0
    possible_mismatches: List[Dict] = []
    missing_labels: List[Dict] = []
    suspicious: List[Dict] = []
    distribution = Counter()

    for row in all_rows:
        folder = row.get("folder_label")
        db = row.get("database_label")
        suggested = row.get("filename_suggested_label")
        if folder:
            distribution[folder] += 1

        issues = []
        if folder and db and folder != db:
            issues.append("folder_vs_database")
        if folder and suggested and folder != suggested:
            issues.append("folder_vs_filename")
        if db and suggested and db != suggested:
            issues.append("database_vs_filename")
        if not folder and not db:
            issues.append("missing_label")
            missing_labels.append(row)
            continue

        if issues:
            entry = dict(row)
            entry["issues"] = issues
            entry["current_label"] = folder or db
            entry["suggested_label"] = suggested
            entry["suggested_display"] = row.get("filename_suggested_display")
            possible_mismatches.append(entry)
        else:
            consistent += 1

        if suggested and not folder and not db:
            suspicious.append(row)

    recommendations = []
    if possible_mismatches:
        recommendations.append("Review mismatched images before research training.")
        recommendations.append("Verify Diwani vs Diwani Jali labels — they are often confused.")
    if missing_labels:
        recommendations.append("Fix or remove database records with missing image files.")
    if distribution:
        counts = list(distribution.values())
        if counts and max(counts) > 0:
            spread = (max(counts) - min(counts)) / max(counts) * 100
            if spread > 30:
                recommendations.append(
                    f"Class imbalance detected ({spread:.0f}% spread). Balance minority classes before training."
                )

    report = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_images_checked": len(all_rows),
        "consistent_labels": consistent,
        "possible_mismatches": possible_mismatches[:500],
        "mismatch_count": len(possible_mismatches),
        "missing_labels": missing_labels[:200],
        "missing_label_count": len(missing_labels),
        "suspicious_filenames": suspicious[:200],
        "class_distribution": dict(distribution),
        "class_distribution_display": {
            display_name(k): v for k, v in distribution.items()
        },
        "recommended_corrections": recommendations,
        "auto_correct_applied": False,
        "note": "Labels are not changed automatically. Review mismatches manually.",
    }
    save_json(config["LABEL_AUDIT_REPORT_PATH"], report)
    save_json(config["LABEL_MISMATCH_REPORT_PATH"], {
        "generated_at": report["generated_at"],
        "total_checked": report["total_images_checked"],
        "mismatch_count": report["mismatch_count"],
        "mismatches": possible_mismatches[:500],
        "warnings": recommendations,
        "recommendations": recommendations,
    })
    return report


def load_label_audit_report(config=None) -> Dict:
    if config is None:
        config = current_app.config
    return load_json(config.get("LABEL_AUDIT_REPORT_PATH"), {})
