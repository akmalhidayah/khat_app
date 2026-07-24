"""Resolve and cache browser-accessible URLs for evaluation test images."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_evaluation_preview_dir(config: Optional[Dict] = None) -> Path:
    if config is None:
        from flask import current_app
        config = current_app.config
    preview_dir = Path(
        config.get("EVALUATION_PREVIEW_DIR")
        or Path(config["BASE_DIR"]) / "static" / "evaluation_previews"
    )
    preview_dir.mkdir(parents=True, exist_ok=True)
    return preview_dir


def _basename(filename: Optional[str]) -> str:
    if not filename:
        return ""
    return Path(str(filename).replace("\\", "/")).name


def preview_filename_for(true_label: Optional[str], filename: Optional[str]) -> str:
    label = (true_label or "unknown").strip().replace("/", "_")
    base = _basename(filename)
    return f"{label}__{base}" if base else ""


def _is_allowed_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def _search_roots(config: Dict) -> List[Tuple[str, Path]]:
    roots: List[Tuple[str, Path]] = []
    preview_dir = get_evaluation_preview_dir(config)
    roots.append(("evaluation_previews", preview_dir))

    ordered = [
        ("test", Path(config["TEST_DIR"])),
        ("optimized_test", Path(config["OPTIMIZED_MODEL_DIR"]) / "test"),
        ("raw", Path(config["RAW_DATASET_DIR"])),
        ("processed", Path(config["PROCESSED_DATASET_DIR"])),
        ("model_ready", Path(config["MODEL_READY_DIR"])),
        ("train", Path(config["TRAIN_DIR"])),
        ("validation", Path(config["VALIDATION_DIR"])),
        ("uploads", Path(config["UPLOAD_FOLDER"])),
        ("evaluation_static", Path(config["EVALUATION_FOLDER"])),
    ]
    for label, path in ordered:
        if path.is_dir():
            roots.append((label, path))
    return roots


def _candidate_paths(
    basename: str,
    true_label: Optional[str],
    image_rel: Optional[str],
    image_path: Optional[str],
    config: Dict,
) -> List[Tuple[Path, str]]:
    """Build ordered candidate paths to check (no recursive scan yet)."""
    if not basename and not image_rel and not image_path:
        return []

    candidates: List[Tuple[Path, str]] = []
    preview_dir = get_evaluation_preview_dir(config)
    preview_name = preview_filename_for(true_label, basename or image_rel)
    if preview_name:
        preview_file = preview_dir / preview_name
        candidates.append((preview_file, "evaluation_previews"))

    if image_path:
        abs_path = Path(os.path.normpath(image_path))
        if _is_allowed_image(abs_path):
            candidates.append((abs_path, "stored_path"))

    rel = (image_rel or "").replace("\\", "/").strip()
    if not rel and true_label and basename:
        rel = f"{true_label}/{basename}"

    for folder_label, root in _search_roots(config):
        if folder_label == "evaluation_previews":
            continue
        if rel:
            direct = (root / rel).resolve()
            try:
                direct.relative_to(root.resolve())
            except ValueError:
                pass
            else:
                candidates.append((direct, folder_label))
        if basename and true_label:
            candidates.append((root / true_label / basename, folder_label))
        if basename:
            candidates.append((root / basename, folder_label))

    seen = set()
    unique: List[Tuple[Path, str]] = []
    for path, label in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append((path, label))
    return unique


def _recursive_search(basename: str, config: Dict) -> Optional[Tuple[Path, str]]:
    if not basename:
        return None
    for folder_label, root in _search_roots(config):
        if folder_label == "evaluation_previews":
            continue
        try:
            for path in root.rglob(basename):
                if _is_allowed_image(path):
                    return path, folder_label
        except OSError:
            continue
    return None


def locate_evaluation_image_file(row: Dict, config: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """Find the best on-disk image file for an evaluation record."""
    if config is None:
        from flask import current_app
        config = current_app.config

    filename = row.get("filename") or ""
    basename = _basename(filename)
    true_label = row.get("true_label")
    image_path = row.get("image_path")
    image_rel = row.get("image_rel")

    if not image_rel:
        from services.evaluation_ui_service import resolve_test_image_rel
        image_rel = resolve_test_image_rel(
            row,
            test_dir=config.get("TEST_DIR"),
            optimized_test_dir=os.path.join(config.get("OPTIMIZED_MODEL_DIR", ""), "test"),
        )

    for path, source_folder in _candidate_paths(basename, true_label, image_rel, image_path, config):
        if _is_allowed_image(path):
            return {
                "path": path,
                "source_folder": source_folder,
                "basename": basename or path.name,
                "image_rel": image_rel or f"{true_label}/{path.name}" if true_label else path.name,
            }

    recursive = _recursive_search(basename, config)
    if recursive:
        path, source_folder = recursive
        return {
            "path": path,
            "source_folder": source_folder,
            "basename": basename or path.name,
            "image_rel": image_rel or f"{true_label}/{path.name}" if true_label else path.name,
        }

    logger.debug(
        "Evaluation image not found: id=%s filename=%s true_class=%s rel=%s",
        row.get("id"),
        basename,
        true_label,
        image_rel,
    )
    return None


def copy_evaluation_preview(
    source_path: Path,
    true_label: Optional[str],
    filename: Optional[str],
    config: Optional[Dict] = None,
) -> Optional[str]:
    """Copy source image into static/evaluation_previews; return preview filename."""
    if not _is_allowed_image(source_path):
        return None
    preview_dir = get_evaluation_preview_dir(config)
    dest_name = preview_filename_for(true_label, filename or source_path.name)
    if not dest_name:
        return None
    dest_path = preview_dir / dest_name
    if not dest_path.is_file():
        try:
            shutil.copy2(source_path, dest_path)
        except OSError as exc:
            logger.warning("Failed to copy evaluation preview %s: %s", source_path, exc)
            return None
    return dest_name if dest_path.is_file() else None


def attach_preview_metadata_only(row: Dict, config: Optional[Dict] = None) -> Dict:
    """Lightweight misclassification record — no file copy (preview built on demand in UI)."""
    located = locate_evaluation_image_file(row, config=config)
    updated = dict(row)
    if not located:
        updated.update(
            {
                "image_exists": False,
                "preview_filename": None,
                "preview_status": "deferred",
                "source_folder": None,
            }
        )
        return updated

    updated.update(
        {
            "image_path": str(located["path"]),
            "image_rel": located.get("image_rel") or updated.get("image_rel"),
            "filename": updated.get("filename") or located["basename"],
            "preview_filename": None,
            "image_exists": True,
            "preview_status": "deferred",
            "source_folder": located["source_folder"],
        }
    )
    return updated


def attach_preview_to_record(row: Dict, config: Optional[Dict] = None) -> Dict:
    """During evaluation — copy preview and attach metadata to misclassified record."""
    located = locate_evaluation_image_file(row, config=config)
    updated = dict(row)
    if not located:
        updated.update(
            {
                "image_exists": False,
                "preview_filename": None,
                "preview_status": "unavailable",
                "source_folder": None,
            }
        )
        return updated

    preview_name = copy_evaluation_preview(
        located["path"],
        row.get("true_label"),
        located["basename"],
        config=config,
    )
    updated.update(
        {
            "image_path": str(located["path"]),
            "image_rel": located.get("image_rel") or updated.get("image_rel"),
            "filename": updated.get("filename") or located["basename"],
            "preview_filename": preview_name,
            "image_exists": bool(preview_name),
            "preview_status": "available" if preview_name else "unavailable",
            "source_folder": located["source_folder"],
        }
    )
    return updated


def resolve_evaluation_image(
    row: Dict,
    config: Optional[Dict] = None,
    static_url_builder: Optional[Callable[[str], str]] = None,
    test_image_url_builder: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """Resolve browser URL and availability flags for UI rendering."""
    if config is None:
        from flask import current_app
        config = current_app.config

    preview_dir = get_evaluation_preview_dir(config)
    true_label = row.get("true_label")
    basename = _basename(row.get("filename"))
    preview_name = row.get("preview_filename") or preview_filename_for(true_label, basename)
    preview_path = preview_dir / preview_name if preview_name else None

    if preview_path and preview_path.is_file():
        static_rel = f"evaluation_previews/{preview_name}"
        image_url = static_url_builder(static_rel) if static_url_builder else f"/static/{static_rel}"
        return {
            "image_exists": True,
            "image_available": True,
            "image_url": image_url,
            "preview_filename": preview_name,
            "preview_status": "available",
            "preview_path": str(preview_path),
            "source_folder": row.get("source_folder") or "evaluation_previews",
            "image_rel": row.get("image_rel"),
        }

    located = locate_evaluation_image_file(row, config=config)
    if located:
        preview_name = copy_evaluation_preview(
            located["path"],
            true_label,
            located["basename"],
            config=config,
        )
        if preview_name:
            static_rel = f"evaluation_previews/{preview_name}"
            image_url = static_url_builder(static_rel) if static_url_builder else f"/static/{static_rel}"
            return {
                "image_exists": True,
                "image_available": True,
                "image_url": image_url,
                "preview_filename": preview_name,
                "preview_status": "available",
                "preview_path": str(preview_dir / preview_name),
                "source_folder": located["source_folder"],
                "image_rel": located.get("image_rel"),
            }

        image_rel = located.get("image_rel") or ""
        if test_image_url_builder and image_rel:
            return {
                "image_exists": True,
                "image_available": True,
                "image_url": test_image_url_builder(image_rel),
                "preview_filename": None,
                "preview_status": "available",
                "preview_path": str(located["path"]),
                "source_folder": located["source_folder"],
                "image_rel": image_rel,
            }

    return {
        "image_exists": False,
        "image_available": False,
        "image_url": None,
        "preview_filename": None,
        "preview_status": "unavailable",
        "preview_path": None,
        "source_folder": row.get("source_folder"),
        "image_rel": row.get("image_rel"),
    }


def rebuild_evaluation_previews(
    records: List[Dict],
    config: Optional[Dict] = None,
) -> Dict[str, int]:
    """Rebuild preview copies for existing evaluation records."""
    fixed = 0
    unavailable = 0
    updated_records: List[Dict] = []
    for row in records:
        enriched = attach_preview_to_record(row, config=config)
        updated_records.append(enriched)
        if enriched.get("image_exists"):
            fixed += 1
        else:
            unavailable += 1
    return {
        "fixed": fixed,
        "unavailable": unavailable,
        "total": len(records),
        "records": updated_records,
    }


def persist_rebuilt_previews(records: List[Dict], config: Optional[Dict] = None) -> None:
    """Write preview metadata back into evaluation JSON reports."""
    if config is None:
        from flask import current_app
        config = current_app.config

    paths = [
        config.get("MISCLASSIFICATION_REPORT_PATH"),
        config.get("EVALUATION_RESULT_PATH"),
    ]
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        if path == config.get("EVALUATION_RESULT_PATH"):
            payload["misclassified_images"] = records
        else:
            payload["misclassified_images"] = records

        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except OSError as exc:
            logger.warning("Could not persist rebuilt previews to %s: %s", path, exc)


def serve_evaluation_image_path(image_rel: str, config: Optional[Dict] = None) -> Optional[Path]:
    """Resolve a safe absolute path for Flask send_file."""
    if config is None:
        from flask import current_app
        config = current_app.config

    rel = (image_rel or "").replace("\\", "/").strip()
    if not rel or ".." in rel:
        return None

    preview_name = Path(rel).name
    preview_dir = get_evaluation_preview_dir(config)
    preview_candidate = preview_dir / preview_name
    if _is_allowed_image(preview_candidate):
        return preview_candidate

    if rel.startswith("evaluation_previews/"):
        preview_candidate = preview_dir / rel.split("/", 1)[1]
        if _is_allowed_image(preview_candidate):
            return preview_candidate

    located = locate_evaluation_image_file(
        {"filename": rel, "true_label": rel.split("/")[0] if "/" in rel else None, "image_rel": rel},
        config=config,
    )
    if located and _is_allowed_image(located["path"]):
        return located["path"]

    from services.evaluation_ui_service import find_test_image_path
    legacy = find_test_image_path(rel, config=config)
    if legacy and legacy.is_file():
        return legacy
    return None
