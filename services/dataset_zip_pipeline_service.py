"""ZIP upload pipeline with safe extraction and per-image optimization."""

from __future__ import annotations

import json
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from flask import current_app

from models import Dataset, db
from services.db_compat import create_gambar_dataset
from services.dataset_path_service import resolve_stored_path, store_path
from services.dataset_upload_settings_service import load_upload_settings
from services.dataset_zip_import_service import (
    ZipImportSummary,
    zip_is_ignored,
    zip_normalize_class,
    zip_safe_extract,
)
from services.image_optimization_service import create_model_ready_image, optimize_image
from services.preprocessing_service import image_hash


def _ensure_class_dirs(base_dir: str, class_labels) -> None:
    os.makedirs(base_dir, exist_ok=True)
    for label in class_labels:
        os.makedirs(os.path.join(base_dir, label), exist_ok=True)


def _cleanup_temp(path: str) -> None:
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _save_upload_report(summary: ZipImportSummary, config) -> None:
    report_path = config.get("DATASET_UPLOAD_REPORT_PATH")
    if not report_path:
        return
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    payload = summary.to_dict()
    payload["saved_mb"] = round(summary.saved_bytes / (1024 * 1024), 2)
    payload["original_total_mb"] = round(summary.original_total_bytes / (1024 * 1024), 2)
    payload["optimized_total_mb"] = round(summary.optimized_total_bytes / (1024 * 1024), 2)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _existing_image_hashes() -> set:
    hashes = set()
    base_dir = current_app.config["BASE_DIR"]
    for item in Dataset.query.all():
        stored = getattr(item, "image_path", None) or ""
        abs_path = resolve_stored_path(stored, base_dir)
        if abs_path and os.path.exists(abs_path):
            try:
                hashes.add(image_hash(abs_path))
            except Exception:
                continue
    return hashes


def _is_duplicate_hash(file_path: str, known_hashes: Optional[set] = None) -> bool:
    img_hash = image_hash(file_path)
    if known_hashes is not None:
        return img_hash in known_hashes
    return img_hash in _existing_image_hashes()


def process_zip_dataset(
    zip_file_path: str,
    settings: Optional[Dict[str, Any]] = None,
) -> ZipImportSummary:
    """Extract ZIP safely and optimize images one file at a time."""
    import logging

    logger = logging.getLogger(__name__)
    config = current_app.config
    opts = settings or load_upload_settings(config)
    summary = ZipImportSummary()
    summary.processing_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    tmp_root = config.get("DATASET_TMP_DIR") or os.path.join(config["DATASET_DIR"], "tmp")
    temp_extract = os.path.join(tmp_root, f"zip_extract_{uuid.uuid4().hex}")
    _cleanup_temp(temp_extract)
    os.makedirs(temp_extract, exist_ok=True)

    class_labels = config["CLASS_LABELS"]
    raw_root = config["RAW_DATASET_DIR"]
    processed_root = config.get("PROCESSED_DATASET_DIR") or os.path.join(config["DATASET_DIR"], "processed")
    model_ready_root = config.get("MODEL_READY_DIR") or os.path.join(config["DATASET_DIR"], "model_ready")

    if opts.get("keep_raw_files", True):
        _ensure_class_dirs(raw_root, class_labels)
    if opts.get("auto_optimize_on_upload", True):
        _ensure_class_dirs(processed_root, class_labels)
    if opts.get("generate_model_ready_images", True):
        _ensure_class_dirs(model_ready_root, class_labels)

    try:
        with zipfile.ZipFile(zip_file_path, "r") as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"ZIP archive contains corrupted entry: {bad}")
            zip_safe_extract(archive, temp_extract, summary)
    except zipfile.BadZipFile as exc:
        raise ValueError("File ZIP tidak valid atau rusak.") from exc

    commit_every = 20
    pending_commits = 0
    max_dim = int(opts.get("max_image_dimension", 512))
    quality = int(opts.get("jpeg_quality", 80))
    output_format = str(opts.get("output_format", "JPEG"))
    model_size = int(opts.get("model_ready_size", 224))
    keep_raw = bool(opts.get("keep_raw_files", True))
    auto_optimize = bool(opts.get("auto_optimize_on_upload", True))
    make_model_ready = bool(opts.get("generate_model_ready_images", True))
    known_hashes = _existing_image_hashes()

    for root, _, files in os.walk(temp_extract):
        folder = os.path.basename(root).strip().lower()
        class_name = zip_normalize_class(folder)
        if class_name not in class_labels:
            continue

        for name in files:
            if zip_is_ignored(name):
                summary.skipped_ignored += 1
                continue

            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if ext not in config["ALLOWED_EXTENSIONS"]:
                summary.skipped_unsupported += 1
                summary.skipped_files.append(os.path.join(root, name))
                continue

            summary.total_found += 1
            source = os.path.join(root, name)
            stem = uuid.uuid4().hex

            try:
                source_bytes = os.path.getsize(source)
            except OSError:
                source_bytes = 0

            try:
                if _is_duplicate_hash(source, known_hashes):
                    summary.skipped_duplicates += 1
                    continue
            except Exception:
                pass

            working_source = source
            raw_path = None

            try:
                if keep_raw:
                    raw_ext = ext
                    raw_name = f"{stem}.{raw_ext}"
                    raw_path = os.path.join(raw_root, class_name, raw_name)
                    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
                    shutil.copy2(source, raw_path)
                    working_source = raw_path

                if auto_optimize:
                    processed_out = os.path.join(processed_root, class_name, stem)
                    opt_result = optimize_image(
                        working_source,
                        processed_out,
                        max_size=max_dim,
                        quality=quality,
                        output_format=output_format,
                    )
                    if opt_result.get("status") != "optimized":
                        raise ValueError(opt_result.get("error") or "Optimization failed")
                    summary.optimized += 1
                    summary.original_total_bytes += opt_result.get("original_size", source_bytes)
                    summary.optimized_total_bytes += opt_result.get("optimized_size", 0)
                    working_source = opt_result["output_path"]
                elif not keep_raw:
                    raise ValueError("Either keep_raw_files or auto_optimize_on_upload must be enabled.")

                model_path = None
                if make_model_ready:
                    model_out = os.path.join(model_ready_root, class_name, stem)
                    model_result = create_model_ready_image(
                        working_source,
                        model_out,
                        image_size=model_size,
                        quality=quality,
                    )
                    if model_result.get("status") in ("optimized", "skipped_exists"):
                        summary.model_ready += 1
                        model_path = model_result["output_path"]

                db_source = raw_path if keep_raw and raw_path else working_source
                from PIL import Image

                with Image.open(db_source) as image:
                    image_size = f"{image.width}x{image.height}"
                    image_format = image.format or ext.upper()
                rel_path = store_path(db_source, config["BASE_DIR"])
                unique_name = os.path.basename(db_source)

                create_gambar_dataset(
                    class_name=class_name,
                    filename=unique_name,
                    original_filename=name,
                    image_path=rel_path,
                    data_type="raw",
                    image_format=image_format,
                    image_size=image_size,
                )
                try:
                    known_hashes.add(image_hash(db_source))
                except Exception:
                    pass
                summary.imported += 1
                summary.per_class[class_name] = summary.per_class.get(class_name, 0) + 1
                pending_commits += 1

                if pending_commits >= commit_every:
                    db.session.commit()
                    pending_commits = 0

            except Exception as exc:
                logger.warning("ZIP image import failed for %s (%s): %s", name, class_name, exc)
                if raw_path and os.path.exists(raw_path):
                    try:
                        os.remove(raw_path)
                    except OSError:
                        pass
                summary.skipped_corrupted += 1
                summary.skipped_files.append(source)

    if pending_commits:
        db.session.commit()

    summary.saved_bytes = max(summary.original_total_bytes - summary.optimized_total_bytes, 0)
    if summary.original_total_bytes > 0:
        summary.compression_ratio = round(
            (summary.saved_bytes / summary.original_total_bytes) * 100,
            2,
        )

    if summary.skipped_unsafe:
        summary.processing_notes.append(f"{summary.skipped_unsafe} entri ZIP diblokir karena path tidak aman.")
    if summary.skipped_ignored:
        summary.processing_notes.append(f"{summary.skipped_ignored} file sistem/hidden dilewati.")
    if summary.skipped_unsupported:
        summary.processing_notes.append(f"{summary.skipped_unsupported} file format tidak didukung dilewati.")
    if summary.skipped_corrupted:
        summary.processing_notes.append(f"{summary.skipped_corrupted} gambar rusak atau gagal diproses.")
    if summary.skipped_duplicates:
        summary.processing_notes.append(f"{summary.skipped_duplicates} duplikat dilewati.")
    if summary.optimized:
        summary.processing_notes.append(
            f"{summary.optimized} gambar dioptimasi (max {max_dim}px, quality {quality})."
        )
    if summary.model_ready:
        summary.processing_notes.append(f"{summary.model_ready} gambar model-ready {model_size}x{model_size} dibuat.")
    if summary.total_found == 0:
        summary.processing_notes.append(
            "Tidak ada gambar kelas yang dikenali. Pastikan ZIP berisi folder: "
            + ", ".join(class_labels)
            + "."
        )

    _save_upload_report(summary, config)
    _cleanup_temp(temp_extract)
    return summary
