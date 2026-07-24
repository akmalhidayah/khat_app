import os
import re
import shutil
import uuid
import zipfile
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image
from flask import current_app
from werkzeug.utils import secure_filename

from models import Dataset, db
from services.db_compat import count_dataset_by_class, create_gambar_dataset, filter_dataset_by_class
from services.dataset_path_service import resolve_stored_path, store_path
from .preprocessing_service import split_dataset_stratified, image_hash, copy_image_for_training, save_split_report
from .dataset_zip_import_service import ZipImportSummary

__all__ = ["DuplicateImageError", "UploadResult", "ZipImportSummary", "allowed_file", "import_dataset_zip"]


class DuplicateImageError(Exception):
    """Raised when an uploaded image already exists in the dataset."""


@dataclass
class UploadResult:
    added: int = 0
    duplicates: int = 0
    invalid: int = 0

    @property
    def total_attempted(self) -> int:
        return self.added + self.duplicates + self.invalid


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in current_app.config["ALLOWED_EXTENSIONS"]


def _existing_image_hashes(class_name: Optional[str] = None) -> set:
    """Build a set of content hashes for existing dataset images (optionally one class)."""
    hashes = set()
    base_dir = current_app.config["BASE_DIR"]
    query = Dataset.query
    if class_name:
        query = filter_dataset_by_class(query, class_name)
    for item in query.all():
        stored = getattr(item, "image_path", None) or ""
        abs_path = resolve_stored_path(stored, base_dir)
        if abs_path and os.path.exists(abs_path):
            try:
                hashes.add(image_hash(abs_path))
            except Exception:
                continue
    return hashes


def is_duplicate_hash(file_path: str, known_hashes: Optional[set] = None, class_name: Optional[str] = None) -> bool:
    img_hash = image_hash(file_path)
    if known_hashes is not None:
        return img_hash in known_hashes
    return img_hash in _existing_image_hashes(class_name=class_name)


def infer_image_source(item: Dataset) -> str:
    if re.match(r"^[a-f0-9]{32}\.", item.filename or ""):
        return "Uploaded"
    return "Imported"


def save_dataset_file(file, class_name: str) -> Dataset:
    result = save_dataset_files([file], class_name)
    if result.added == 0 and result.duplicates > 0:
        raise DuplicateImageError("The selected image already exists in the dataset.")
    if result.added == 0 and result.invalid > 0:
        raise ValueError("The selected file is not a valid supported image.")
    if result.added == 0:
        raise ValueError("Upload failed. No image was added.")
    return Dataset.query.order_by(Dataset.dibuat_pada.desc()).first()


def save_dataset_files(files, class_name: str) -> UploadResult:
    import logging

    logger = logging.getLogger(__name__)
    result = UploadResult()
    dest_dir = os.path.join(current_app.config["RAW_DATASET_DIR"], class_name)
    os.makedirs(dest_dir, exist_ok=True)
    known_hashes = _existing_image_hashes(class_name=class_name)

    for file in files:
        if not file or not getattr(file, "filename", None):
            continue
        if not allowed_file(file.filename):
            result.invalid += 1
            continue

        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        safe_original = secure_filename(file.filename) or file.filename
        path = os.path.join(dest_dir, unique_name)

        try:
            file.save(path)
            image = Image.open(path)
            image.verify()
            image = Image.open(path)
            image_size = f"{image.width}x{image.height}"

            if is_duplicate_hash(path, known_hashes, class_name=class_name):
                os.remove(path)
                result.duplicates += 1
                continue

            create_gambar_dataset(
                class_name=class_name,
                filename=unique_name,
                original_filename=safe_original,
                image_path=store_path(path, current_app.config["BASE_DIR"]),
                data_type="raw",
                image_format=image.format or ext.upper(),
                image_size=image_size,
            )
            known_hashes.add(image_hash(path))
            db.session.flush()
            result.added += 1
        except DuplicateImageError:
            if os.path.exists(path):
                os.remove(path)
            result.duplicates += 1
        except Exception as exc:
            logger.warning("Dataset image upload failed for %s: %s", safe_original, exc)
            if os.path.exists(path):
                os.remove(path)
            result.invalid += 1

    db.session.commit()
    return result


def _collect_raw_filesystem_items(config=None) -> List[dict]:
    """Build split payload directly from raw class folders (no DB required)."""
    from flask import current_app
    from services.dataset_path_service import resolve_class_folder

    if config is None:
        config = current_app.config

    raw_dir = config["RAW_DATASET_DIR"]
    class_labels = list(config["CLASS_LABELS"])
    allowed = set(config.get("ALLOWED_EXTENSIONS") or {"jpg", "jpeg", "png", "webp"})
    payload: List[dict] = []
    item_id = 1
    for label in class_labels:
        class_dir = resolve_class_folder(raw_dir, label)
        if not class_dir or not os.path.isdir(class_dir):
            continue
        for name in sorted(os.listdir(class_dir)):
            if "." not in name or name.rsplit(".", 1)[1].lower() not in allowed:
                continue
            path = os.path.join(class_dir, name)
            if os.path.isfile(path):
                payload.append({"id": item_id, "class_name": label, "image_path": path})
                item_id += 1
    return payload


def split_and_copy_dataset_from_filesystem(
    config=None,
    *,
    deduplicate_hashes: bool = False,
    phash_threshold: int = -1,
) -> Dict[str, int]:
    """Stratified split using raw filesystem folders only (safe for automated pipelines)."""
    from flask import current_app

    if config is None:
        config = current_app.config

    payload = _collect_raw_filesystem_items(config)
    if not payload:
        raise ValueError("No images found under raw dataset class folders.")

    split = split_dataset_stratified(
        payload,
        deduplicate_hashes=deduplicate_hashes,
        phash_threshold=phash_threshold,
    )
    save_split_report(
        split,
        config["DATASET_SPLIT_REPORT_PATH"],
        config["CLASS_LABELS"],
        deduplication_applied=deduplicate_hashes or phash_threshold >= 0,
    )

    base_dir = config["BASE_DIR"]
    for split_name, split_items in split.items():
        split_root = config[f"{split_name.upper()}_DIR"]
        for cls in config["CLASS_LABELS"]:
            cls_dir = os.path.join(split_root, cls)
            if os.path.exists(cls_dir):
                shutil.rmtree(cls_dir)
            os.makedirs(cls_dir, exist_ok=True)
        for item in split_items:
            target_dir = os.path.join(config[f"{split_name.upper()}_DIR"], item["class_name"])
            os.makedirs(target_dir, exist_ok=True)
            unique_name = f"{uuid.uuid4().hex}.jpg"
            target_file = os.path.join(target_dir, unique_name)
            copy_image_for_training(item["image_path"], target_file)

    return {k: len(v) for k, v in split.items()}


def split_and_copy_dataset(
    clean_first: bool = True,
    *,
    deduplicate_hashes: bool = True,
    phash_threshold: int = 5,
) -> Dict[str, int]:
    from services.dataset_cleaning_service import run_duplicate_audit
    from services.dataset_quality_service import clean_raw_dataset
    from services.label_audit_service import run_label_audit

    if clean_first:
        clean_raw_dataset()

    items = Dataset.query.all()
    if not items:
        raise ValueError("No dataset records found.")

    base_dir = current_app.config["BASE_DIR"]
    payload = []
    missing_paths = 0
    for item in items:
        abs_path = resolve_stored_path(item.image_path, base_dir)
        if not os.path.isfile(abs_path):
            missing_paths += 1
            continue
        payload.append(
            {"id": item.id, "class_name": item.class_name, "image_path": abs_path}
        )
    if not payload:
        raise ValueError("No readable dataset images found on disk.")
    if missing_paths:
        from services.external_assets_service import rebuild_db_from_raw

        rebuild_db_from_raw(current_app.config)
        items = Dataset.query.all()
        payload = [
            {
                "id": i.id,
                "class_name": i.class_name,
                "image_path": resolve_stored_path(i.image_path, base_dir),
            }
            for i in items
            if os.path.isfile(resolve_stored_path(i.image_path, base_dir))
        ]
        if not payload:
            raise ValueError("Dataset records exist but image files are missing. Re-import the dataset.")
    split = split_dataset_stratified(
        payload,
        deduplicate_hashes=deduplicate_hashes,
        phash_threshold=phash_threshold,
    )
    save_split_report(
        split,
        current_app.config["DATASET_SPLIT_REPORT_PATH"],
        current_app.config["CLASS_LABELS"],
        deduplication_applied=deduplicate_hashes or phash_threshold >= 0,
    )

    base_dir = current_app.config["BASE_DIR"]
    train_processed = []
    for split_name, split_items in split.items():
        split_root = current_app.config[f"{split_name.upper()}_DIR"]
        for cls in current_app.config["CLASS_LABELS"]:
            cls_dir = os.path.join(split_root, cls)
            if os.path.exists(cls_dir):
                shutil.rmtree(cls_dir)
            os.makedirs(cls_dir, exist_ok=True)
        for item in split_items:
            target_dir = os.path.join(current_app.config[f"{split_name.upper()}_DIR"], item["class_name"])
            os.makedirs(target_dir, exist_ok=True)
            source_file = item["image_path"]
            unique_name = f"{uuid.uuid4().hex}.jpg"
            target_file = os.path.join(target_dir, unique_name)
            try:
                final_path = copy_image_for_training(source_file, target_file)
                if split_name == "train":
                    train_processed.append(
                        {
                            "id": item.get("id"),
                            "class_name": item["class_name"],
                            "processed_path": store_path(final_path, base_dir),
                        }
                    )
            except Exception as exc:
                raise ValueError(f"Failed to prepare image for training: {source_file} ({exc})") from exc

    try:
        from services.dataset_similarity_service import rebuild_hash_index_after_split

        rebuild_hash_index_after_split(current_app.config, train_processed)
    except Exception:
        pass

    try:
        run_label_audit()
        run_duplicate_audit()
    except Exception:
        pass

    return {k: len(v) for k, v in split.items()}


def class_distribution():
    from models import GambarDataset, KelasKhat

    result = {}
    for kelas in KelasKhat.query.order_by(KelasKhat.id.asc()).all():
        result[kelas.slug] = GambarDataset.query.filter_by(id_kelas=kelas.id).count()
    return result


def import_dataset_zip(zip_file_path: str) -> ZipImportSummary:
    from services.dataset_zip_pipeline_service import process_zip_dataset
    from services.dataset_upload_settings_service import load_upload_settings

    return process_zip_dataset(zip_file_path, load_upload_settings(current_app.config))
