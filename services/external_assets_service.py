"""Import external dataset, sync class indices from labels.txt, and wire external Keras model."""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from services.dataset_import_service import (
    CLASS_ALIASES,
    copy_dataset_from_source,
    normalize_class_name,
    scan_raw_dataset,
    sync_raw_dataset_to_db,
)
from services.dataset_readiness_service import count_images_per_class, is_valid_trained_model
from services.dataset_path_service import resolve_stored_path, store_path
from services.teachable_machine_service import normalize_tm_label


def parse_labels_txt(path: str) -> List[Tuple[int, str]]:
    """Parse labels.txt lines like '0 Diwani Jali' into (index, slug) pairs."""
    if not path or not os.path.isfile(path):
        return []

    rows: List[Tuple[int, str]] = []
    pattern = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.match(line.strip())
            if not match:
                continue
            index = int(match.group(1))
            raw_label = match.group(2).strip()
            slug = normalize_tm_label(raw_label)
            rows.append((index, slug))
    rows.sort(key=lambda item: item[0])
    return rows


def build_class_indices_from_labels(path: str, class_labels: List[str]) -> Dict[str, int]:
    parsed = parse_labels_txt(path)
    if not parsed:
        return {label: idx for idx, label in enumerate(class_labels)}

    indices = {slug: index for index, slug in parsed}
    for label in class_labels:
        indices.setdefault(label, class_labels.index(label))
    return indices


def write_class_indices_file(path: str, class_indices: Dict[str, int], class_labels: List[str]) -> None:
    display_names = {
        "naskhi": "Naskhi",
        "diwani": "Diwani",
        "diwani_jali": "Diwani Jali",
        "tsuluts": "Tsuluts",
    }
    ordered_names = sorted(class_indices.keys(), key=lambda k: class_indices[k])
    payload = {
        "class_indices": class_indices,
        "class_names": ordered_names,
        "display_names": {k: display_names.get(k, k) for k in ordered_names},
        "source": "external_labels_txt",
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def ensure_external_model_ready(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Sync class_indices and metadata when an external Keras model is configured."""
    if not config.get("USE_EXTERNAL_MODEL"):
        return None
    model_path = config.get("EXTERNAL_MODEL_PATH") or config.get("MODEL_PATH")
    if not is_valid_trained_model(model_path or ""):
        return None
    return write_external_model_metadata(config)


def write_external_model_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    labels_path = config.get("EXTERNAL_LABELS_PATH") or ""
    class_labels = list(config.get("CLASS_LABELS") or [])
    class_indices = build_class_indices_from_labels(labels_path, class_labels)
    model_path = config.get("EXTERNAL_MODEL_PATH") or config.get("MODEL_PATH")

    meta = {
        "architecture": "keras_h5",
        "architecture_label": "Keras H5 (External Training)",
        "model_architecture_key": "keras_h5",
        "training_mode": "external",
        "training_mode_label": "External TensorFlow Model",
        "training_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "model_path": model_path,
        "external_model_dir": config.get("EXTERNAL_MODEL_DIR"),
        "external_labels_path": labels_path,
        "class_indices": class_indices,
        "class_labels": class_labels,
        "preprocessing": "teachable_machine",
        "input_size": 224,
        "source": "external",
    }
    meta_path = config.get("MODEL_METADATA_PATH")
    if meta_path:
        os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False)

    indices_path = config.get("CLASS_INDICES_PATH")
    if indices_path:
        write_class_indices_file(indices_path, class_indices, class_labels)

    return meta


def count_external_source_dataset(source_dir: str, class_labels: List[str]) -> Dict[str, int]:
    """Count images in external flat class folders (supports 'diwani jali' alias)."""
    counts = {label: 0 for label in class_labels}
    if not source_dir or not os.path.isdir(source_dir):
        return counts

    for entry in os.listdir(source_dir):
        entry_path = os.path.join(source_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        slug = normalize_class_name(entry)
        if slug not in counts:
            continue
        counts[slug] = sum(
            1
            for name in os.listdir(entry_path)
            if os.path.isfile(os.path.join(entry_path, name))
            and name.lower().rsplit(".", 1)[-1] in {"jpg", "jpeg", "png", "webp"}
        )
    return counts


def clear_raw_dataset_dirs(config: Dict[str, Any]) -> None:
    raw_dir = config["RAW_DATASET_DIR"]
    os.makedirs(raw_dir, exist_ok=True)
    for label in config["CLASS_LABELS"]:
        class_dir = os.path.join(raw_dir, label)
        if os.path.isdir(class_dir):
            shutil.rmtree(class_dir)
        os.makedirs(class_dir, exist_ok=True)


def rebuild_db_from_raw(config: Dict[str, Any]) -> Tuple[int, int]:
    from app import create_app
    from models import GambarDataset, db
    from services.db_compat import create_gambar_dataset
    from services.dataset_import_service import get_extension, is_ignored_file, is_supported_image

    app = create_app()
    registered = 0
    skipped = 0
    with app.app_context():
        GambarDataset.query.delete()
        db.session.commit()

        raw_dir = config["RAW_DATASET_DIR"]
        from services.dataset_import_service import normalize_class_name, get_extension, is_ignored_file, is_supported_image

        for entry in sorted(os.listdir(raw_dir)):
            class_dir = os.path.join(raw_dir, entry)
            if not os.path.isdir(class_dir):
                continue
            class_name = normalize_class_name(entry)
            if not class_name:
                continue
            for filename in sorted(os.listdir(class_dir)):
                if is_ignored_file(filename) or not is_supported_image(filename):
                    skipped += 1
                    continue
                abs_path = os.path.join(class_dir, filename)
                if not os.path.isfile(abs_path):
                    continue
                try:
                    from PIL import Image

                    image = Image.open(abs_path)
                    image_size = f"{image.width}x{image.height}"
                    ext = get_extension(filename)
                    rel_path = store_path(abs_path, config["BASE_DIR"])
                    create_gambar_dataset(
                        class_name=class_name,
                        filename=filename,
                        original_filename=filename,
                        image_path=rel_path,
                        data_type="raw",
                        image_format=(image.format or ext).upper(),
                        image_size=image_size,
                    )
                    registered += 1
                except Exception:
                    skipped += 1
        db.session.commit()
    return registered, skipped


def import_external_dataset(config: Dict[str, Any], sync_db: bool = True, replace_raw: bool = True) -> Dict[str, Any]:
    source = config.get("EXTERNAL_DATASET_SOURCE") or config.get("EXTERNAL_DATASET_DIR")
    if not source or not os.path.isdir(source):
        raise FileNotFoundError(f"External dataset folder not found: {source}")

    raw_dir = config["RAW_DATASET_DIR"]
    use_direct_source = os.path.normcase(os.path.normpath(source)) == os.path.normcase(os.path.normpath(raw_dir))

    if replace_raw and not use_direct_source:
        clear_raw_dataset_dirs(config)

    before = count_images_per_class(raw_dir, config["CLASS_LABELS"]) if not replace_raw or use_direct_source else {k: 0 for k in config["CLASS_LABELS"]}
    if use_direct_source:
        summary_per_class = count_external_source_dataset(source, config["CLASS_LABELS"])
        from services.dataset_import_service import ImportSummary

        summary = ImportSummary(
            source=source,
            destination=raw_dir,
            copied=sum(summary_per_class.values()),
            skipped_duplicates=0,
            per_class=summary_per_class,
        )
    else:
        summary = copy_dataset_from_source(source, raw_dir)
    after = count_images_per_class(raw_dir, config["CLASS_LABELS"])

    db_registered = 0
    db_skipped = 0
    if sync_db:
        db_registered, db_skipped = rebuild_db_from_raw(config)

    return {
        "source": source,
        "destination": raw_dir,
        "copied": summary.copied,
        "skipped_duplicates": summary.skipped_duplicates,
        "per_class_copied": summary.per_class,
        "raw_before": before,
        "raw_after": after,
        "db_registered": db_registered,
        "db_skipped": db_skipped,
        "source_counts": count_external_source_dataset(source, config["CLASS_LABELS"]),
    }


def split_external_dataset(config: Dict[str, Any], clean_first: bool = False) -> Dict[str, Any]:
    from app import create_app

    app = create_app()
    with app.app_context():
        from services.dataset_service import split_and_copy_dataset

        counts = split_and_copy_dataset(
            clean_first=clean_first,
            deduplicate_hashes=False,
            phash_threshold=-1,
        )
    return counts


def apply_external_assets(
    config: Dict[str, Any],
    import_dataset: bool = True,
    split_dataset: bool = True,
    clean_before_split: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": True}

    if config.get("USE_EXTERNAL_MODEL"):
        model_path = config.get("EXTERNAL_MODEL_PATH") or config.get("MODEL_PATH")
        if not is_valid_trained_model(model_path or ""):
            raise FileNotFoundError(f"External model not found or invalid: {model_path}")
        result["model"] = write_external_model_metadata(config)
        result["model_path"] = model_path

    if config.get("USE_EXTERNAL_DATASET") and import_dataset:
        result["import"] = import_external_dataset(config, sync_db=True)

    if config.get("USE_EXTERNAL_DATASET") and split_dataset:
        result["split"] = split_external_dataset(config, clean_first=clean_before_split)

    if config.get("USE_EXTERNAL_DATASET"):
        result["dataset_counts"] = {
            "raw": count_images_per_class(config["RAW_DATASET_DIR"], config["CLASS_LABELS"]),
            "train": count_images_per_class(config["TRAIN_DIR"], config["CLASS_LABELS"]),
            "validation": count_images_per_class(config["VALIDATION_DIR"], config["CLASS_LABELS"]),
            "test": count_images_per_class(config["TEST_DIR"], config["CLASS_LABELS"]),
        }

    return result


def should_use_keras_model(config: Optional[Dict[str, Any]] = None) -> bool:
    if config is None:
        from flask import current_app

        config = current_app.config
    if not config.get("USE_TEACHABLE_MACHINE", True):
        return True
    if config.get("USE_EXTERNAL_MODEL") and is_valid_trained_model(
        config.get("EXTERNAL_MODEL_PATH") or config.get("MODEL_PATH") or ""
    ):
        return True
    if config.get("APP_SIMPLE_MODE") and is_valid_trained_model(config.get("MODEL_PATH") or ""):
        return True
    return False
