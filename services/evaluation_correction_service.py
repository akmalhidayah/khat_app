"""Evaluation error corrections and review tracking."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from flask import current_app, session
from PIL import Image

from models import Dataset, db
from services.classification_correction_service import normalize_class_key
from services.dataset_service import is_duplicate_hash
from services.evaluation_ui_service import find_test_image_path
from services.preprocessing_service import preprocess_pil_image


def _corrections_path() -> Path:
    return Path(current_app.config["BASE_DIR"]) / "model" / "evaluation_corrections.json"


def load_evaluation_corrections() -> Dict[str, Any]:
    path = _corrections_path()
    if not path.is_file():
        return {"records": {}}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "records" in data:
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"records": {}}


def save_evaluation_corrections(data: Dict[str, Any]) -> None:
    path = _corrections_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _record_key(image_rel: str) -> str:
    return (image_rel or "").replace("\\", "/").strip()


def get_correction_record(image_rel: str) -> Dict[str, Any]:
    data = load_evaluation_corrections()
    return data.get("records", {}).get(_record_key(image_rel), {})


def merge_correction_into_row(row: Dict[str, Any]) -> Dict[str, Any]:
    rec = get_correction_record(row.get("image_rel") or "")
    if not rec:
        return row
    merged = dict(row)
    merged["reviewed_status"] = rec.get("reviewed_status") or merged.get("reviewed_status")
    merged["correction_label"] = rec.get("correction_label") or merged.get("correction_label")
    merged["correction_note"] = rec.get("correction_note") or merged.get("correction_note")
    merged["reviewed_by"] = rec.get("reviewed_by")
    merged["reviewed_at"] = rec.get("reviewed_at")
    return merged


def _upsert_record(image_rel: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    data = load_evaluation_corrections()
    key = _record_key(image_rel)
    existing = data["records"].get(key, {})
    existing.update(updates)
    existing["image_rel"] = key
    existing["updated_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    data["records"][key] = existing
    save_evaluation_corrections(data)
    return existing


def mark_as_reviewed(image_rel: str, note: str = "") -> Dict[str, Any]:
    return _upsert_record(
        image_rel,
        {
            "reviewed_status": "reviewed",
            "reviewed_by": session.get("username") or session.get("name") or "admin",
            "reviewed_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "correction_note": note or None,
        },
    )


def save_label_correction(image_rel: str, correct_class: str, note: str = "") -> Dict[str, Any]:
    class_key = normalize_class_key(correct_class)
    if not class_key:
        raise ValueError("Invalid correction class.")

    abs_path = find_test_image_path(image_rel)
    if not abs_path or not abs_path.is_file():
        raise FileNotFoundError("Test image file not found.")

    config = current_app.config
    ext = abs_path.suffix.lower() or ".jpg"
    raw_dir = Path(config["RAW_DATASET_DIR"]) / class_key
    processed_dir = Path(config["PROCESSED_DATASET_DIR"]) / class_key
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_name = f"{uuid.uuid4().hex}{ext}"
    raw_path = raw_dir / raw_name
    shutil.copy2(abs_path, raw_path)

    if is_duplicate_hash(str(raw_path)):
        raw_path.unlink(missing_ok=True)
        raise ValueError("Image already exists in dataset (duplicate hash).")

    with Image.open(raw_path) as img:
        image = img.convert("RGB")
        image_size = f"{image.width}x{image.height}"
        image_format = img.format or ext.lstrip(".").upper()
        processed = preprocess_pil_image(image)
        processed_name = f"{uuid.uuid4().hex}.jpg"
        processed_path = processed_dir / processed_name
        processed.save(processed_path, format="JPEG", quality=92)

    raw_rel = os.path.relpath(raw_path, config["BASE_DIR"]).replace("\\", "/")
    processed_rel = os.path.relpath(processed_path, config["BASE_DIR"]).replace("\\", "/")

    from services.db_compat import create_gambar_dataset

    dataset_row = create_gambar_dataset(
        class_name=class_key,
        filename=raw_name,
        original_filename=raw_path.name,
        image_path=raw_rel,
        data_type="raw",
        image_format=image_format,
        image_size=image_size,
    )
    db.session.flush()
    try:
        from services.relational_sync_service import (
            sync_correction_for_evaluation,
            sync_dataset_image_from_legacy,
        )
        img_row = sync_dataset_image_from_legacy(dataset_row, class_key)
        sync_correction_for_evaluation(
            image_rel,
            class_key,
            note,
            dataset_image_id=img_row.id if img_row else None,
        )
    except Exception:
        pass
    db.session.commit()

    return _upsert_record(
        image_rel,
        {
            "reviewed_status": "corrected",
            "correction_label": class_key,
            "correction_note": note or None,
            "reviewed_by": session.get("username") or session.get("name") or "admin",
            "reviewed_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "correction_action": "save_to_dataset",
        },
    )


def move_image_to_class(image_rel: str, correct_class: str, note: str = "") -> Dict[str, Any]:
    class_key = normalize_class_key(correct_class)
    if not class_key:
        raise ValueError("Invalid correction class.")

    abs_path = find_test_image_path(image_rel)
    if not abs_path or not abs_path.is_file():
        raise FileNotFoundError("Test image file not found.")

    config = current_app.config
    bases = [
        Path(config["TEST_DIR"]),
        Path(config["OPTIMIZED_MODEL_DIR"]) / "test",
    ]
    dest_base = None
    for base in bases:
        try:
            rel = abs_path.relative_to(base)
            dest_base = base
            break
        except ValueError:
            continue

    if dest_base is None:
        raise FileNotFoundError("Could not resolve test split root for image.")

    dest_dir = dest_base / class_key
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / abs_path.name
    if dest_path.resolve() != abs_path.resolve():
        if dest_path.exists():
            dest_path = dest_dir / f"{uuid.uuid4().hex}{abs_path.suffix.lower() or '.jpg'}"
        shutil.move(str(abs_path), str(dest_path))

    new_rel = str(dest_path.relative_to(dest_base)).replace("\\", "/")
    return _upsert_record(
        new_rel,
        {
            "reviewed_status": "moved",
            "correction_label": class_key,
            "correction_note": note or None,
            "reviewed_by": session.get("username") or session.get("name") or "admin",
            "reviewed_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "correction_action": "move_to_class",
            "previous_image_rel": _record_key(image_rel),
        },
    )
