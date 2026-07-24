"""Save user corrections from classification results into the dataset."""

from __future__ import annotations

import os
import uuid
from typing import Dict, Optional

from flask import current_app
from PIL import Image
from werkzeug.utils import secure_filename

from models import Dataset, db
from services.dataset_service import is_duplicate_hash
from services.preprocessing_service import preprocess_pil_image

VALID_CLASSES = ("naskhi", "diwani", "diwani_jali", "tsuluts")

EXTERNAL_WARNING = (
    "Gambar ini tidak memiliki label pembanding. Walaupun model memberikan confidence tinggi, "
    "hasil tetap perlu divalidasi manual karena gambar eksternal dapat memiliki gaya visual "
    "yang berbeda dari dataset training."
)

NO_COMPARATOR_MESSAGE = (
    "Tidak ada label pembanding. Hasil prediksi perlu divalidasi secara manual."
)


def normalize_class_key(value: Optional[str]) -> Optional[str]:
    key = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not key or key in ("unknown", "none", "-"):
        return None
    if key in VALID_CLASSES:
        return key
    aliases = {
        "diwanijali": "diwani_jali",
        "diwani_jali": "diwani_jali",
        "tsuluth": "tsuluts",
        "thuluth": "tsuluts",
    }
    return aliases.get(key)


def save_correction_to_dataset(result_row, correct_class: str, notes: str = "") -> Dict:
    """Copy classified upload into raw + processed dataset folders."""
    class_key = normalize_class_key(correct_class)
    if not class_key:
        raise ValueError("Invalid correction class.")

    config = current_app.config
    base_dir = config["BASE_DIR"]
    abs_upload = os.path.join(base_dir, result_row.image_path)
    if not os.path.isfile(abs_upload):
        raise FileNotFoundError("Uploaded image file is missing.")

    ext = os.path.splitext(abs_upload)[1].lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        ext = ".jpg"

    raw_dir = os.path.join(config["RAW_DATASET_DIR"], class_key)
    processed_dir = os.path.join(config["PROCESSED_DATASET_DIR"], class_key)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    raw_name = f"{uuid.uuid4().hex}{ext}"
    raw_path = os.path.join(raw_dir, raw_name)
    with open(abs_upload, "rb") as src, open(raw_path, "wb") as dst:
        dst.write(src.read())

    if is_duplicate_hash(raw_path):
        os.remove(raw_path)
        raise ValueError("Image already exists in dataset (duplicate hash).")

    with Image.open(raw_path) as img:
        img.verify()
    with Image.open(raw_path) as img:
        image = img.convert("RGB")
        image_size = f"{image.width}x{image.height}"
        image_format = image.format or ext.replace(".", "").upper()

    raw_rel = os.path.relpath(raw_path, base_dir).replace("\\", "/")

    from services.db_compat import create_gambar_dataset

    dataset = create_gambar_dataset(
        class_name=class_key,
        filename=raw_name,
        original_filename=secure_filename(result_row.uploaded_filename or result_row.filename or raw_name),
        image_path=raw_rel,
        data_type="raw",
        image_format=image_format,
        image_size=image_size,
    )

    processed_name = f"{uuid.uuid4().hex}.jpg"
    processed_path = os.path.join(processed_dir, processed_name)
    with Image.open(raw_path) as img:
        processed_img = preprocess_pil_image(img.convert("RGB"))
        processed_img.save(processed_path, "JPEG", quality=92, optimize=True)

    processed_rel = os.path.relpath(processed_path, base_dir).replace("\\", "/")

    result_row.correction_label = class_key
    result_row.correction_notes = (notes or "").strip() or None
    result_row.validation_status = "User Corrected Label"
    result_row.review_status = "Correction Saved"
    result_row.final_decision = "Corrected by User"
    result_row.manual_review_required = False
    db.session.flush()
    try:
        from services.relational_sync_service import (
            sync_correction_for_classification,
            sync_dataset_image_from_legacy,
        )
        img_row = sync_dataset_image_from_legacy(dataset, class_key)
        sync_correction_for_classification(
            result_row,
            class_key,
            notes,
            dataset_image_id=img_row.id if img_row else None,
        )
    except Exception:
        pass
    db.session.commit()

    return {
        "correct_class": class_key,
        "raw_path": raw_rel,
        "processed_path": processed_rel,
        "dataset_id": dataset.id,
    }


def mark_misclassification(result_row, notes: str = "") -> None:
    note = (notes or "").strip()
    result_row.validation_status = "Marked Misclassification"
    result_row.review_status = "Manual Review Required"
    result_row.final_decision = "Manual Review Required"
    result_row.manual_review_required = True
    if note:
        existing = (result_row.correction_notes or "").strip()
        result_row.correction_notes = f"{existing}\n{note}".strip() if existing else note
    db.session.commit()
