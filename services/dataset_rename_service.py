"""Rename dataset images to a consistent class-based pattern, e.g. diwani-1.jpg."""

from __future__ import annotations

import os
import re
import uuid
from typing import Dict, List, Tuple

from services.dataset_import_service import normalize_class_name
from services.dataset_path_service import DATASET_SPLIT_SUBDIRS

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SKIP_ROOT_ENTRIES = DATASET_SPLIT_SUBDIRS | {
    "rejected",
    "non_khat",
    "khat_detector",
    "display",
}
SPLIT_FOLDERS = ("train", "validation", "test")


def _file_prefix(class_slug: str) -> str:
    return class_slug.replace("_", "-")


def _is_image_file(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS


def _natural_sort_key(name: str) -> tuple:
    parts = re.split(r"(\d+)", name.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def rename_images_in_class_dir(class_dir: str, class_slug: str) -> List[Tuple[str, str]]:
    """Rename images inside one class folder. Returns list of (old_name, new_name)."""
    if not os.path.isdir(class_dir):
        return []

    prefix = _file_prefix(class_slug)
    files = sorted(
        [name for name in os.listdir(class_dir) if _is_image_file(name)],
        key=_natural_sort_key,
    )
    if not files:
        return []

    staged: List[Tuple[str, str]] = []
    for index, filename in enumerate(files, start=1):
        ext = os.path.splitext(filename)[1].lower()
        final_name = f"{prefix}-{index}{ext}"
        if filename == final_name:
            continue
        temp_name = f"__rename_{uuid.uuid4().hex}{ext}"
        old_path = os.path.join(class_dir, filename)
        temp_path = os.path.join(class_dir, temp_name)
        os.rename(old_path, temp_path)
        staged.append((temp_path, os.path.join(class_dir, final_name), filename, final_name))

    changes: List[Tuple[str, str]] = []
    for temp_path, final_path, old_name, new_name in staged:
        os.rename(temp_path, final_path)
        changes.append((old_name, new_name))
    return changes


def iter_class_directories(dataset_root: str, class_labels: List[str]) -> List[Tuple[str, str]]:
    """Yield (absolute_class_dir, class_slug) for source and split folders."""
    labels = set(class_labels)
    found: List[Tuple[str, str]] = []
    seen_dirs: set[str] = set()

    if not os.path.isdir(dataset_root):
        return found

    for entry in sorted(os.listdir(dataset_root)):
        if entry.lower() in SKIP_ROOT_ENTRIES:
            continue
        entry_path = os.path.join(dataset_root, entry)
        if not os.path.isdir(entry_path):
            continue
        slug = normalize_class_name(entry)
        if slug in labels:
            norm = os.path.normcase(os.path.normpath(entry_path))
            if norm not in seen_dirs:
                seen_dirs.add(norm)
                found.append((entry_path, slug))

    for split_name in SPLIT_FOLDERS:
        split_root = os.path.join(dataset_root, split_name)
        if not os.path.isdir(split_root):
            continue
        for entry in sorted(os.listdir(split_root)):
            entry_path = os.path.join(split_root, entry)
            if not os.path.isdir(entry_path):
                continue
            slug = normalize_class_name(entry) or entry
            if slug not in labels:
                continue
            norm = os.path.normcase(os.path.normpath(entry_path))
            if norm not in seen_dirs:
                seen_dirs.add(norm)
                found.append((entry_path, slug))

    return found


def rename_dataset_images(config: Dict) -> Dict:
    """Rename all class image files under the configured external dataset root."""
    dataset_root = config.get("DATASET_DIR") or config.get("RAW_DATASET_DIR") or ""
    class_labels = list(config.get("CLASS_LABELS") or [])
    if not dataset_root or not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    per_class: Dict[str, int] = {label: 0 for label in class_labels}
    per_folder: List[Dict] = []
    total_renamed = 0

    for class_dir, class_slug in iter_class_directories(dataset_root, class_labels):
        changes = rename_images_in_class_dir(class_dir, class_slug)
        if not changes:
            continue
        per_class[class_slug] = per_class.get(class_slug, 0) + len(changes)
        total_renamed += len(changes)
        per_folder.append(
            {
                "folder": class_dir,
                "class": class_slug,
                "renamed": len(changes),
                "sample": changes[:3],
            }
        )

    db_sync = None
    if total_renamed > 0 and config.get("USE_EXTERNAL_DATASET"):
        from services.external_assets_service import rebuild_db_from_raw

        registered, skipped = rebuild_db_from_raw(config)
        db_sync = {"registered": registered, "skipped": skipped}

    return {
        "dataset_root": dataset_root,
        "total_renamed": total_renamed,
        "per_class": per_class,
        "folders": per_folder,
        "db_sync": db_sync,
    }
