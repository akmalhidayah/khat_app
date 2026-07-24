"""Resolve dataset image paths safely for serving and preview."""

import os
import re
import time
from typing import Dict, List, Optional, Tuple

from flask import current_app

from models import Dataset
from services.image_optimization_service import resolve_optimized_display_path

DATASET_SPLIT_SUBDIRS = frozenset({
    "train",
    "validation",
    "test",
    "processed",
    "optimized",
    "model_ready",
    "tmp",
})
DATASET_NON_SOURCE_SUBDIRS = DATASET_SPLIT_SUBDIRS | {
    "rejected",
    "non_khat",
    "khat_detector",
    "display",
}


def _is_allowed_image(filename: str, allowed_extensions: set) -> bool:
    if "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in allowed_extensions


def count_images_in_folder(folder: str, allowed_extensions: Optional[set] = None, *, recursive: bool = False) -> int:
    allowed = allowed_extensions or {"jpg", "jpeg", "png", "webp"}
    if not folder or not os.path.isdir(folder):
        return 0
    if not recursive:
        return sum(
            1
            for name in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, name)) and _is_allowed_image(name, allowed)
        )
    total = 0
    for _, _, files in os.walk(folder):
        total += sum(1 for name in files if _is_allowed_image(name, allowed))
    return total


def resolve_class_folder(dataset_root: str, class_slug: str) -> str:
    """Return the on-disk folder for a class slug (handles aliases like 'diwani jali')."""
    from services.dataset_import_service import normalize_class_name

    matches: List[str] = []
    if dataset_root and os.path.isdir(dataset_root):
        for entry in os.listdir(dataset_root):
            entry_path = os.path.join(dataset_root, entry)
            if not os.path.isdir(entry_path):
                continue
            if entry.lower() in DATASET_NON_SOURCE_SUBDIRS:
                continue
            if normalize_class_name(entry) == class_slug:
                matches.append(entry_path)

    if matches:
        return max(matches, key=count_images_in_folder)

    preferred = os.path.join(dataset_root, class_slug)
    os.makedirs(preferred, exist_ok=True)
    return preferred


def count_source_class_images(config: Optional[Dict] = None) -> Dict[str, int]:
    """Count images in each raw class folder under the configured dataset root."""
    if config is None:
        config = current_app.config

    root = config.get("DATASET_DIR") or config.get("RAW_DATASET_DIR") or ""
    class_labels: List[str] = list(config.get("CLASS_LABELS") or [])
    allowed = set(config.get("ALLOWED_EXTENSIONS") or {"jpg", "jpeg", "png", "webp"})
    counts = {label: 0 for label in class_labels}
    if not root or not os.path.isdir(root):
        return counts

    for label in class_labels:
        class_dir = resolve_class_folder(root, label)
        counts[label] = count_images_in_folder(class_dir, allowed)
    return counts


def count_split_images(config: Optional[Dict] = None) -> Dict[str, int]:
    if config is None:
        config = current_app.config

    class_labels: List[str] = list(config.get("CLASS_LABELS") or [])

    def _split_total(dir_key: str) -> int:
        folder = config.get(dir_key, "")
        if not folder or not os.path.isdir(folder):
            return 0
        return sum(count_images_per_class_split(folder, class_labels).values())

    return {
        "train": _split_total("TRAIN_DIR"),
        "validation": _split_total("VALIDATION_DIR"),
        "test": _split_total("TEST_DIR"),
    }


def count_images_per_class_split(root_dir: str, class_labels: List[str]) -> Dict[str, int]:
    """Count images per class folder (non-recursive) under a split directory."""
    allowed = {"jpg", "jpeg", "png", "webp"}
    counts = {label: 0 for label in class_labels}
    if not root_dir or not os.path.isdir(root_dir):
        return counts
    for label in class_labels:
        class_dir = resolve_class_folder(root_dir, label)
        counts[label] = count_images_in_folder(class_dir, allowed)
    return counts


_INVENTORY_CACHE: Optional[Tuple[str, float, Dict]] = None


def get_dataset_inventory(config: Optional[Dict] = None, *, force_refresh: bool = False) -> Dict:
    """Authoritative dataset totals for UI pages (filesystem under DATASET_DIR)."""
    global _INVENTORY_CACHE
    if config is None:
        config = current_app.config

    root = config.get("DATASET_DIR") or config.get("RAW_DATASET_DIR") or ""
    ttl = float(config.get("MODEL_PAGE_CACHE_TTL", 30))
    now = time.monotonic()
    if (
        not force_refresh
        and _INVENTORY_CACHE is not None
        and _INVENTORY_CACHE[0] == root
        and now - _INVENTORY_CACHE[1] < ttl
    ):
        return _INVENTORY_CACHE[2]

    class_labels: List[str] = list(config.get("CLASS_LABELS") or [])
    per_class = count_source_class_images(config)
    raw_total = sum(per_class.values())
    split_counts = count_split_images(config)
    processed_total = split_counts["train"] + split_counts["validation"] + split_counts["test"]
    active_classes = len([label for label in class_labels if per_class.get(label, 0) > 0])

    result = {
        "dataset_root": root,
        "per_class": per_class,
        "raw_total": raw_total,
        "train": split_counts["train"],
        "validation": split_counts["validation"],
        "test": split_counts["test"],
        "processed_total": processed_total,
        "active_classes": active_classes or len(class_labels),
        "expected_classes": len(class_labels),
        "is_balanced": active_classes == len(class_labels) and len(set(per_class.values())) <= 1,
    }
    _INVENTORY_CACHE = (root, now, result)
    return result


def count_dataset_images_on_disk(config: Optional[Dict] = None, *, include_splits: bool = False) -> int:
    """
    Count image files under the configured dataset directory.
    By default, split/preprocessed folders are excluded so totals reflect source inventory.
    """
    if config is None:
        config = current_app.config

    root = config.get("DATASET_DIR") or config.get("RAW_DATASET_DIR") or ""
    allowed = set(config.get("ALLOWED_EXTENSIONS") or {"jpg", "jpeg", "png", "webp"})
    if not root or not os.path.isdir(root):
        return 0

    total = 0
    for dirpath, dirnames, files in os.walk(root):
        if not include_splits:
            dirnames[:] = [
                name for name in dirnames if name.lower() not in DATASET_NON_SOURCE_SUBDIRS
            ]
        total += sum(1 for name in files if _is_allowed_image(name, allowed))
    return total


def resolve_stored_path(stored_path: str, base_dir: str) -> str:
    if not stored_path:
        return ""
    raw = stored_path.strip().replace("\\", "/")
    if os.path.isabs(raw):
        return os.path.normpath(raw)
    return os.path.normpath(os.path.join(base_dir, raw))


def safe_relpath(abs_path: str, base_dir: str) -> str:
    """Return a path relative to base_dir, or a normalized absolute path if drives differ."""
    if not abs_path:
        return ""
    abs_norm = os.path.normpath(abs_path)
    base_norm = os.path.normpath(base_dir) if base_dir else ""
    if base_norm:
        try:
            rel = os.path.relpath(abs_norm, base_norm)
            if not rel.startswith(".."):
                return rel.replace("\\", "/")
        except ValueError:
            pass
    return abs_norm.replace("\\", "/")


def store_path(abs_path: str, base_dir: str) -> str:
    return safe_relpath(abs_path, base_dir)


def infer_image_source(item: Dataset) -> str:
    if re.match(r"^[a-f0-9]{32}\.", item.filename or ""):
        return "Uploaded"
    return "Imported"


def _normalize_candidates(item: Dataset, config: Dict) -> list:
    base_dir = config["BASE_DIR"]
    candidates = []

    if item.image_path:
        raw = item.image_path.strip().replace("\\", "/")
        candidates.append(raw)
        if not os.path.isabs(raw):
            candidates.append(os.path.join(base_dir, raw))

    if item.class_name:
        class_dir = os.path.join(config["RAW_DATASET_DIR"], item.class_name)
        for name in (item.filename, item.original_filename):
            if name:
                candidates.append(os.path.join(class_dir, name))

    for split_key in ("TRAIN_DIR", "VALIDATION_DIR", "TEST_DIR"):
        split_root = config[split_key]
        if item.class_name:
            for name in (item.filename, item.original_filename):
                if name:
                    candidates.append(os.path.join(split_root, item.class_name, name))

    seen = set()
    unique = []
    for path in candidates:
        if not path:
            continue
        norm = os.path.normpath(path)
        if norm not in seen:
            seen.add(norm)
            unique.append(norm)
    return unique


def _find_by_filename_in_dir(root_dir: str, filenames: set) -> Optional[str]:
    if not os.path.isdir(root_dir) or not filenames:
        return None
    lowered = {n.lower() for n in filenames if n}
    for dirpath, _, files in os.walk(root_dir):
        for fname in files:
            if fname.lower() in lowered:
                return os.path.normpath(os.path.join(dirpath, fname))
    return None


def resolve_dataset_image_path(item: Dataset, config=None) -> Optional[str]:
    if config is None:
        config = current_app.config

    for path in _normalize_candidates(item, config):
        if os.path.isfile(path):
            return path

    lookup_names = {n for n in (item.filename, item.original_filename) if n}
    for split_key in ("TRAIN_DIR", "VALIDATION_DIR", "TEST_DIR", "RAW_DATASET_DIR"):
        found = _find_by_filename_in_dir(config[split_key], lookup_names)
        if found:
            return found

    class_dir = os.path.join(config["RAW_DATASET_DIR"], item.class_name or "")
    lookup_lower = {n.lower() for n in lookup_names}
    if os.path.isdir(class_dir) and lookup_lower:
        for fname in os.listdir(class_dir):
            if fname.lower() in lookup_lower:
                return os.path.normpath(os.path.join(class_dir, fname))

    return None


def resolve_static_thumbnail_path(item: Dataset, config=None) -> Optional[str]:
    if config is None:
        config = current_app.config

    preview_root = config.get("DATASET_PREVIEW_FOLDER")
    if not preview_root or not os.path.isdir(preview_root):
        return None

    names = []
    if item.id:
        names.extend([f"{item.id}.jpg", f"{item.id}.jpeg", f"{item.id}.png", f"{item.id}.webp"])
    for name in (item.filename, item.original_filename):
        if name:
            names.append(name)

    for name in names:
        path = os.path.join(preview_root, name)
        if os.path.isfile(path):
            return os.path.normpath(path)
    return None


def resolve_optimized_from_stored_path(item: Dataset, config=None) -> Optional[str]:
    """Find optimized display copy using stored relative path even if original is gone."""
    if config is None:
        config = current_app.config
    if not item.image_path:
        return None

    dataset_dir = os.path.normpath(config["DATASET_DIR"])
    display_root = config["OPTIMIZED_DISPLAY_DIR"]
    rel = item.image_path.strip().replace("\\", "/")

    if rel.startswith("dataset/"):
        rel = rel[len("dataset/") :]

    base, _ = os.path.splitext(rel)
    candidates = [
        os.path.join(display_root, f"{base}.jpg"),
        os.path.join(display_root, f"{base}.jpeg"),
        os.path.join(display_root, rel),
    ]
    if item.class_name and item.filename:
        stem, _ = os.path.splitext(item.filename)
        candidates.extend(
            [
                os.path.join(display_root, "raw", item.class_name, f"{stem}.jpg"),
                os.path.join(display_root, item.class_name, f"{stem}.jpg"),
            ]
        )

    for candidate in candidates:
        norm = os.path.normpath(candidate)
        if os.path.isfile(norm):
            return norm
    return None


def resolve_dataset_serving_path(item: Dataset, config=None) -> Tuple[Optional[str], str]:
    """
    Return (absolute_path, source) where source is one of:
    optimized, original, split, static, none
    """
    if config is None:
        config = current_app.config

    original = resolve_dataset_image_path(item, config)
    if original:
        optimized = resolve_optimized_display_path(original, config)
        if optimized and os.path.isfile(optimized):
            return optimized, "optimized"
        return original, "original"

    optimized_only = resolve_optimized_from_stored_path(item, config)
    if optimized_only:
        return optimized_only, "optimized"

    static_thumb = resolve_static_thumbnail_path(item, config)
    if static_thumb:
        return static_thumb, "static"

    return None, "none"


def resolve_dataset_image_url(item: Dataset, config=None, placeholder_url: str = "") -> Dict:
    """
    Resolve display URL metadata for templates.
    1. Optimized display image
    2. Original / split copy
    3. Static thumbnail
    4. Placeholder
    """
    if config is None:
        config = current_app.config

    serving_path, source = resolve_dataset_serving_path(item, config)
    original_path = resolve_dataset_image_path(item, config)
    optimized_path = None
    if original_path:
        optimized_path = resolve_optimized_display_path(original_path, config)
    elif item.image_path:
        optimized_path = resolve_optimized_from_stored_path(item, config)

    can_preview = serving_path is not None and os.path.isfile(serving_path)

    return {
        "can_preview": can_preview,
        "serving_path": serving_path,
        "original_path": original_path,
        "optimized_path": optimized_path if optimized_path and os.path.isfile(optimized_path) else None,
        "preview_source": source,
        "is_orphan": original_path is None and not can_preview,
        "stored_path": item.image_path,
        "placeholder_url": placeholder_url,
    }


def enrich_dataset_item(item: Dataset, config=None) -> Dict:
    if config is None:
        config = current_app.config

    url_meta = resolve_dataset_image_url(item, config)
    abs_path = url_meta["original_path"]
    serving_path = url_meta["serving_path"]
    can_preview = url_meta["can_preview"]
    optimized_available = url_meta["preview_source"] == "optimized"

    file_size_display = None
    size_path = serving_path or abs_path
    if size_path and os.path.isfile(size_path):
        size = os.path.getsize(size_path)
        if size < 1024:
            file_size_display = f"{size} B"
        elif size < 1024 * 1024:
            file_size_display = f"{round(size / 1024, 1)} KB"
        else:
            file_size_display = f"{round(size / (1024 * 1024), 2)} MB"

    debug_info = None
    if config.get("DEBUG") or (hasattr(current_app, "debug") and current_app.debug):
        debug_info = {
            "stored_path": item.image_path,
            "resolved_path": abs_path,
            "serving_path": serving_path,
            "preview_source": url_meta["preview_source"],
            "file_exists": abs_path is not None,
            "optimized_exists": url_meta["optimized_path"] is not None,
            "can_preview": can_preview,
        }

    return {
        "id": item.id,
        "item": item,
        "class_name": item.class_name,
        "filename": item.original_filename or item.filename,
        "stored_filename": item.filename,
        "format": (item.image_format or "-").upper(),
        "dimensions": item.image_size or "-",
        "source": infer_image_source(item),
        "created_at": item.created_at,
        "image_path": item.image_path,
        "resolved_path": abs_path,
        "serving_path": serving_path,
        "optimized_available": optimized_available,
        "file_exists": can_preview,
        "original_exists": abs_path is not None,
        "is_orphan": url_meta["is_orphan"],
        "preview_source": url_meta["preview_source"],
        "file_size_display": file_size_display,
        "debug_info": debug_info,
    }
