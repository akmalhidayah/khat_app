"""Validate Teachable Machine export files without TensorFlow dependencies."""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

CANONICAL_CLASSES = ["naskhi", "diwani", "diwani_jali", "tsuluts"]
REQUIRED_FILES = ("model.json", "metadata.json", "weights.bin")


class TmModelValidationError(ValueError):
    """Raised when uploaded Teachable Machine files fail validation."""


def normalize_tm_label(raw: str) -> str:
    text = (raw or "").strip().lower().replace("-", "_")
    compact = text.replace(" ", "_")

    if compact in ("diwani_jali", "diwanijali") or ("diwani" in compact and "jali" in compact):
        return "diwani_jali"
    if compact in CANONICAL_CLASSES:
        return compact
    if "naskhi" in compact or "naski" in compact:
        return "naskhi"
    if "diwani" in compact:
        return "diwani"
    if any(token in compact for token in ("tsuluts", "tsuluth", "thuluth", "sulus")):
        return "tsuluts"
    return compact


def get_tm_model_name(metadata: Dict) -> str:
    return metadata.get("modelName") or metadata.get("name") or "tm-my-image-model"


def validate_tm_metadata_dict(metadata: dict) -> Dict:
    errors: List[str] = []
    raw_labels = metadata.get("labels") or []
    if not raw_labels:
        errors.append("metadata.json tidak berisi labels.")

    normalized = [normalize_tm_label(label) for label in raw_labels]
    missing = [label for label in CANONICAL_CLASSES if label not in normalized]
    if missing:
        display = ", ".join(label.replace("_", " ") for label in missing)
        errors.append(f"Label wajib tidak lengkap: {display}.")

    image_size = int(metadata.get("imageSize") or 224)
    if image_size != 224:
        errors.append(f"Input size harus 224 × 224 px (ditemukan {image_size}).")

    if errors:
        raise TmModelValidationError(" ".join(errors))

    return {
        "metadata": metadata,
        "labels": normalized,
        "labels_raw": raw_labels,
        "image_size": image_size,
        "model_name": get_tm_model_name(metadata),
    }


def validate_tm_files(file_map: Dict[str, str]) -> Dict:
    missing = [name for name in REQUIRED_FILES if name not in file_map]
    if missing:
        raise TmModelValidationError(
            f"File wajib tidak ditemukan: {', '.join(missing)}. "
            "Pastikan model.json, metadata.json, dan weights.bin tersedia."
        )

    with open(file_map["metadata.json"], "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return validate_tm_metadata_dict(metadata)
