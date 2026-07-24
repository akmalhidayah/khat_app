"""Step 1 — basic image file / resolution validation."""

from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

from PIL import Image, UnidentifiedImageError

from services.validation_pipeline.utils import rejected

logger = logging.getLogger(__name__)

MIN_SIDE = 32
MAX_SIDE = 8192


def validate_image_file(image_path: str) -> Dict:
    """Reject unreadable, corrupted, empty, or invalid-resolution images."""
    if not image_path or not os.path.isfile(image_path):
        return rejected(
            "The uploaded image file is missing or unreadable.",
            stage="image_validation",
        )
    if os.path.getsize(image_path) <= 0:
        return rejected(
            "The uploaded image is empty.",
            stage="image_validation",
        )
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.load()
            width, height = img.size
            mode = img.mode
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        logger.warning("Image validation failed for %s: %s", image_path, exc)
        return rejected(
            "The uploaded image is corrupted or unreadable.",
            stage="image_validation",
            details={"error": str(exc)},
        )

    if width < MIN_SIDE or height < MIN_SIDE:
        return rejected(
            f"Image resolution is too small ({width}x{height}).",
            stage="image_validation",
            details={"width": width, "height": height},
        )
    if width > MAX_SIDE or height > MAX_SIDE:
        return rejected(
            f"Image resolution is too large ({width}x{height}).",
            stage="image_validation",
            details={"width": width, "height": height},
        )
    if mode not in {"RGB", "RGBA", "L", "P", "CMYK"}:
        return rejected(
            f"Unsupported image mode: {mode}.",
            stage="image_validation",
            details={"mode": mode},
        )
    return {
        "status": "Accepted",
        "accepted": True,
        "stage": "image_validation",
        "width": width,
        "height": height,
        "mode": mode,
    }


def load_rgb_size(image_path: str) -> Tuple[Image.Image, int, int]:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        return rgb.copy(), width, height
