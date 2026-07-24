"""Detect possible mislabeled dataset images from filename vs stored class."""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from flask import current_app

from models import Dataset
from services.result_interpretation_service import parse_filename_class
from services.training_utils import save_json


def detect_label_mismatch(filename: str, db_class: str, folder_class: Optional[str] = None) -> Optional[Dict]:
    expected_key, expected_display = parse_filename_class(filename or "")
    if not expected_key:
        return None
    if expected_key != db_class:
        return {
            "filename": filename,
            "expected_class": expected_key,
            "expected_display": expected_display,
            "database_class": db_class,
            "folder_class": folder_class or db_class,
            "message": f"Filename suggests {expected_display}, but database class is {db_class.replace('_', ' ').title()}.",
        }
    return None


def generate_label_mismatch_report(config=None) -> Dict:
    if config is None:
        config = current_app.config

    mismatches: List[Dict] = []
    for row in Dataset.query.all():
        folder_class = row.class_name
        hit = detect_label_mismatch(row.original_filename or row.filename, row.class_name, folder_class)
        if hit:
            hit["dataset_id"] = row.id
            hit["image_path"] = row.image_path
            mismatches.append(hit)

    report = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_checked": Dataset.query.count(),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:500],
        "warnings": [],
        "recommendations": [],
    }
    if mismatches:
        report["warnings"].append(
            f"Found {len(mismatches)} possible mislabeled image(s). Review before final research training."
        )
        report["recommendations"] = [
            "Manually verify filename-based class hints against expert labels.",
            "Move mislabeled images to the correct class folder and re-process the dataset.",
            "Remove decorative or ambiguous images that confuse similar classes.",
        ]

    save_json(config["LABEL_MISMATCH_REPORT_PATH"], report)
    return report


def load_label_mismatch_report(config=None) -> Dict:
    if config is None:
        config = current_app.config
    path = config.get("LABEL_MISMATCH_REPORT_PATH")
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
