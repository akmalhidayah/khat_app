"""Admin-configurable settings for dataset ZIP upload and image optimization."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, Optional

DEFAULT_UPLOAD_SETTINGS: Dict[str, Any] = {
    "max_upload_size_mb": 1024,
    "max_image_dimension": 512,
    "jpeg_quality": 80,
    "output_format": "JPEG",
    "keep_raw_files": True,
    "auto_optimize_on_upload": True,
    "generate_model_ready_images": True,
    "model_ready_size": 224,
    "duplicate_strategy": "skip",
}


def _settings_path(config=None) -> str:
    if config is None:
        from flask import current_app
        config = current_app.config
    return config.get("DATASET_UPLOAD_SETTINGS_PATH", "")


def load_upload_settings(config=None) -> Dict[str, Any]:
    path = _settings_path(config)
    settings = deepcopy(DEFAULT_UPLOAD_SETTINGS)
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                stored = json.load(handle)
            if isinstance(stored, dict):
                settings.update(stored)
        except (json.JSONDecodeError, OSError):
            pass
    return settings


def save_upload_settings(settings: Dict[str, Any], config=None) -> Dict[str, Any]:
    merged = deepcopy(DEFAULT_UPLOAD_SETTINGS)
    merged.update(settings or {})
    path = _settings_path(config)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(merged, handle, indent=2)
    return merged


def apply_upload_settings_to_config(config) -> None:
    settings = load_upload_settings(config)
    max_mb = int(settings.get("max_upload_size_mb", 1024))
    config["UPLOAD_MAX_SIZE_MB"] = max_mb
    config["MAX_CONTENT_LENGTH"] = max_mb * 1024 * 1024
