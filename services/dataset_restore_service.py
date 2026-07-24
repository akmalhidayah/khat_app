"""Restore raw class folders to the original image count by merging rejected copies back."""

from __future__ import annotations

import os
import shutil
import uuid
from typing import Dict, List, Optional, Tuple

from services.dataset_import_service import normalize_class_name
from services.dataset_path_service import (
    DATASET_NON_SOURCE_SUBDIRS,
    count_images_in_folder,
    resolve_class_folder,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLIT_FOLDERS = ("train", "validation", "test")
TARGET_PER_CLASS = 700


def _is_image_file(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS


def count_class_images(class_dir: str) -> int:
    return count_images_in_folder(class_dir, IMAGE_EXTENSIONS)


def _unique_destination(dest_dir: str, filename: str) -> str:
    candidate = os.path.join(dest_dir, filename)
    if not os.path.exists(candidate):
        return candidate
    base, ext = os.path.splitext(filename)
    while True:
        alt = os.path.join(dest_dir, f"{base}_{uuid.uuid4().hex[:8]}{ext}")
        if not os.path.exists(alt):
            return alt


def merge_rejected_into_class(
    class_dir: str,
    rejected_class_root: str,
    *,
    target_count: int = TARGET_PER_CLASS,
) -> Dict:
    """Move images from rejected/{class}/** into the class folder until target_count is reached."""
    os.makedirs(class_dir, exist_ok=True)
    moved = 0
    skipped = 0
    sources: List[str] = []

    if not os.path.isdir(rejected_class_root):
        return {"moved": 0, "skipped": 0, "before": count_class_images(class_dir), "after": count_class_images(class_dir)}

    before = count_class_images(class_dir)
    candidates: List[str] = []
    for dirpath, _, files in os.walk(rejected_class_root):
        for name in files:
            if not _is_image_file(name):
                continue
            candidates.append(os.path.join(dirpath, name))

    candidates.sort(key=lambda path: os.path.basename(path).lower())

    for source_path in candidates:
        if count_class_images(class_dir) >= target_count:
            break
        dest_path = _unique_destination(class_dir, os.path.basename(source_path))
        shutil.move(source_path, dest_path)
        moved += 1
        sources.append(dest_path)

    after = count_class_images(class_dir)
    skipped = max(len(candidates) - moved, 0)
    return {
        "moved": moved,
        "skipped": skipped,
        "before": before,
        "after": after,
        "target": target_count,
    }


def clear_split_folders(dataset_root: str) -> Dict[str, int]:
    removed = {}
    for split_name in SPLIT_FOLDERS:
        split_root = os.path.join(dataset_root, split_name)
        if not os.path.isdir(split_root):
            removed[split_name] = 0
            continue
        count = 0
        for dirpath, _, files in os.walk(split_root):
            count += sum(1 for name in files if _is_image_file(name))
        shutil.rmtree(split_root, ignore_errors=True)
        removed[split_name] = count
    return removed


def restore_dataset_to_target(config: Dict, *, target_per_class: int = TARGET_PER_CLASS, resplit: bool = True) -> Dict:
    dataset_root = config.get("DATASET_DIR") or config.get("RAW_DATASET_DIR") or ""
    rejected_root = config.get("REJECTED_DATASET_DIR") or os.path.join(dataset_root, "rejected")
    class_labels = list(config.get("CLASS_LABELS") or [])

    if not dataset_root or not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    merge_report: Dict[str, Dict] = {}
    for class_slug in class_labels:
        class_dir = resolve_class_folder(dataset_root, class_slug)
        rejected_class_root = os.path.join(rejected_root, class_slug)
        merge_report[class_slug] = {
            "class_dir": class_dir,
            **merge_rejected_into_class(class_dir, rejected_class_root, target_count=target_per_class),
        }

    per_class = {
        class_slug: count_class_images(resolve_class_folder(dataset_root, class_slug))
        for class_slug in class_labels
    }
    total_raw = sum(per_class.values())

    split_cleared = clear_split_folders(dataset_root)

    db_sync = None
    split_result = None
    if config.get("USE_EXTERNAL_DATASET"):
        from services.external_assets_service import rebuild_db_from_raw

        registered, skipped = rebuild_db_from_raw(config)
        db_sync = {"registered": registered, "skipped": skipped}

    if resplit and total_raw > 0:
        from app import create_app

        app = create_app()
        with app.app_context():
            from services.dataset_service import split_and_copy_dataset

            split_result = split_and_copy_dataset(clean_first=False, deduplicate_hashes=False, phash_threshold=-1)

    rename_result = None
    if total_raw > 0:
        from services.dataset_rename_service import rename_dataset_images

        rename_result = rename_dataset_images(config)

    return {
        "dataset_root": dataset_root,
        "rejected_root": rejected_root,
        "target_per_class": target_per_class,
        "per_class": per_class,
        "total_raw": total_raw,
        "merge": merge_report,
        "split_cleared": split_cleared,
        "db_sync": db_sync,
        "split": split_result,
        "rename": {
            "total_renamed": (rename_result or {}).get("total_renamed", 0),
            "per_class": (rename_result or {}).get("per_class", {}),
        },
    }
