"""Validate, backup, and install Teachable Machine TensorFlow.js model exports."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from typing import Dict, Optional

from flask import current_app

from services.model_version_service import load_latest_evaluation_accuracy, register_model_version, update_version_notes
from services.teachable_machine_service import clear_tm_model_cache, is_teachable_machine_available
from services.tm_model_validation_service import (
    REQUIRED_FILES,
    TmModelValidationError,
    validate_tm_files,
)

__all__ = ["TmModelValidationError", "process_tm_upload", "validate_tm_files"]


def _find_model_files(root: str) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name in REQUIRED_FILES and name not in found:
                found[name] = os.path.join(dirpath, name)
    return found


def backup_active_model(model_dir: str) -> Optional[str]:
    if not all(os.path.isfile(os.path.join(model_dir, name)) for name in REQUIRED_FILES):
        return None

    backup_root = os.path.join(model_dir, "backups")
    os.makedirs(backup_root, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(backup_root, stamp)
    os.makedirs(backup_dir, exist_ok=True)
    for name in REQUIRED_FILES:
        shutil.copy2(os.path.join(model_dir, name), os.path.join(backup_dir, name))
    return backup_dir


def _snapshot_installed_model(model_dir: str, version_id: int) -> str:
    snapshot_dir = os.path.join(model_dir, "backups", f"v_{version_id}")
    if os.path.isdir(snapshot_dir):
        shutil.rmtree(snapshot_dir, ignore_errors=True)
    os.makedirs(snapshot_dir, exist_ok=True)
    for name in REQUIRED_FILES:
        shutil.copy2(os.path.join(model_dir, name), os.path.join(snapshot_dir, name))
    return snapshot_dir


def install_tm_model_files(file_map: Dict[str, str], validated: Dict) -> Dict:
    config = current_app.config
    model_dir = config["TEACHABLE_MODEL_DIR"]
    os.makedirs(model_dir, exist_ok=True)

    previous_backup = None
    if is_teachable_machine_available(config):
        previous_backup = backup_active_model(model_dir)

    for name in REQUIRED_FILES:
        shutil.copy2(file_map[name], os.path.join(model_dir, name))

    clear_tm_model_cache()

    version = register_model_version(
        model_name=validated["model_name"],
        model_source="Teachable Machine",
        model_runtime="TensorFlow.js",
        evaluation_accuracy=load_latest_evaluation_accuracy(),
        notes=json.dumps(
            {
                "labels": validated["labels_raw"],
                "labels_normalized": validated["labels"],
                "input_size": validated["image_size"],
                "previous_backup": previous_backup,
            }
        ),
        set_active=True,
    )

    snapshot_dir = _snapshot_installed_model(model_dir, version.id)
    update_version_notes(
        version.id,
        {
            "labels": validated["labels_raw"],
            "labels_normalized": validated["labels"],
            "input_size": validated["image_size"],
            "previous_backup": previous_backup,
            "snapshot_dir": snapshot_dir,
            "archived": False,
        },
    )

    return {
        "previous_backup": previous_backup,
        "snapshot_dir": snapshot_dir,
        "version": version,
        "model_name": validated["model_name"],
        "labels": validated["labels_raw"],
        "image_size": validated["image_size"],
    }


def process_tm_upload(
    *,
    model_json=None,
    metadata_json=None,
    weights_bin=None,
    zip_file=None,
) -> Dict:
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_map: Dict[str, str] = {}

        if zip_file and getattr(zip_file, "filename", None):
            zip_path = os.path.join(tmp_dir, "upload.zip")
            zip_file.save(zip_path)
            extract_dir = os.path.join(tmp_dir, "extract")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(extract_dir)
            file_map = _find_model_files(extract_dir)
        else:
            uploads = {
                "model.json": model_json,
                "metadata.json": metadata_json,
                "weights.bin": weights_bin,
            }
            for name, handle in uploads.items():
                if handle and getattr(handle, "filename", None):
                    dest = os.path.join(tmp_dir, name)
                    handle.save(dest)
                    file_map[name] = dest

            if len(file_map) != len(REQUIRED_FILES):
                raise TmModelValidationError(
                    "Unggah ketiga file (model.json, metadata.json, weights.bin) "
                    "atau satu arsip ZIP yang berisi ketiga file tersebut."
                )

        validated = validate_tm_files(file_map)
        return install_tm_model_files(file_map, validated)
