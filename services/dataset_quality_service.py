"""Dataset cleaning, validation, and quality reporting."""

import hashlib
import os
import shutil
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image, ImageOps

from services.preprocessing_service import image_hash, perceptual_hash, resize_with_padding
from services.training_utils import save_json

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def _hamming_distance(hash_a: str, hash_b: str) -> int:
    ia, ib = int(hash_a, 16), int(hash_b, 16)
    return bin(ia ^ ib).count("1")


def _is_image_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _count_per_class(root: str, class_labels: List[str]) -> Dict[str, int]:
    counts = {}
    for cls in class_labels:
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            counts[cls] = 0
            continue
        counts[cls] = sum(
            1 for f in os.listdir(cls_dir)
            if _is_image_file(f) and os.path.isfile(os.path.join(cls_dir, f))
        )
    return counts


def clean_raw_dataset(config=None, phash_threshold: int = 5, *, remove_duplicates: bool = True) -> Dict:
    from flask import current_app

    if config is None:
        config = current_app.config

    raw_dir = config["RAW_DATASET_DIR"]
    rejected_root = config.get("REJECTED_DATASET_DIR", os.path.join(config["DATASET_DIR"], "rejected"))
    class_labels = config["CLASS_LABELS"]
    report_path = config["DATASET_QUALITY_REPORT_PATH"]

    os.makedirs(rejected_root, exist_ok=True)

    def _reject_file(path: str, cls: str, reason: str) -> None:
        dest_dir = os.path.join(rejected_root, cls, reason)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(path))
        if os.path.exists(dest):
            base, ext = os.path.splitext(os.path.basename(path))
            dest = os.path.join(dest_dir, f"{base}_{reason}{ext}")
        try:
            shutil.move(path, dest)
        except OSError:
            if os.path.exists(path):
                os.remove(path)

    before_total = 0
    after_total = 0
    duplicates_removed = 0
    near_duplicates_removed = 0
    corrupted_removed = 0
    non_image_removed = 0
    webp_converted = 0
    resized_count = 0

    seen_file_hashes: Set[str] = set()
    seen_phashes: List[Tuple[str, str]] = []
    warnings: List[str] = []
    recommendations: List[str] = []

    for cls in class_labels:
        cls_dir = os.path.join(raw_dir, cls)
        if not os.path.isdir(cls_dir):
            os.makedirs(cls_dir, exist_ok=True)
            continue

        for filename in list(os.listdir(cls_dir)):
            path = os.path.join(cls_dir, filename)
            if not os.path.isfile(path):
                continue
            before_total += 1

            if not _is_image_file(filename):
                _reject_file(path, cls, "unsupported")
                non_image_removed += 1
                continue

            try:
                with Image.open(path) as img:
                    img.verify()
                with Image.open(path) as img:
                    rgb = img.convert("RGB")
                    ext = filename.rsplit(".", 1)[1].lower()
                    fhash = image_hash(path)
                    if remove_duplicates and fhash in seen_file_hashes:
                        _reject_file(path, cls, "duplicate")
                        duplicates_removed += 1
                        continue

                    try:
                        phash = perceptual_hash(path)
                        is_near_dup = False
                        if phash_threshold >= 0:
                            for existing_path, existing_phash in seen_phashes:
                                if _hamming_distance(phash, existing_phash) <= phash_threshold:
                                    _reject_file(path, cls, "near_duplicate")
                                    near_duplicates_removed += 1
                                    is_near_dup = True
                                    break
                        if is_near_dup:
                            continue
                        seen_phashes.append((path, phash))
                    except Exception:
                        pass

                    seen_file_hashes.add(fhash)
                    processed = resize_with_padding(rgb, (224, 224))

                    if ext == "webp":
                        new_name = f"{os.path.splitext(filename)[0]}.jpg"
                        new_path = os.path.join(cls_dir, new_name)
                        if new_path != path and os.path.exists(new_path):
                            os.remove(path)
                        else:
                            processed.save(new_path, format="JPEG", quality=95)
                            if new_path != path:
                                os.remove(path)
                            webp_converted += 1
                            resized_count += 1
                            after_total += 1
                        continue

                    if rgb.size != (224, 224):
                        processed.save(path, format="JPEG", quality=95)
                        resized_count += 1
                    after_total += 1
            except Exception:
                if os.path.exists(path):
                    _reject_file(path, cls, "corrupted")
                corrupted_removed += 1

    per_class = _count_per_class(raw_dir, class_labels)
    total_after = sum(per_class.values())
    if total_after == 0:
        warnings.append("No valid images remain after cleaning.")

    counts = list(per_class.values())
    imbalance_pct = 0.0
    if counts and max(counts) > 0:
        imbalance_pct = round((max(counts) - min(counts)) / max(counts) * 100, 2)

    if imbalance_pct > 30:
        warnings.append(f"Class imbalance detected ({imbalance_pct}% spread). Naskhi and Diwani are minority classes.")
        recommendations.append("Run balanced augmentation before research training.")
        recommendations.append("Use class weights during training.")

    report = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_scanned": before_total,
        "total_before_cleaning": before_total,
        "valid_images": total_after,
        "total_after_cleaning": total_after,
        "duplicate_images": duplicates_removed,
        "duplicates_removed": duplicates_removed,
        "near_duplicates_removed": near_duplicates_removed,
        "corrupted_images": corrupted_removed,
        "corrupted_removed": corrupted_removed,
        "converted_images": webp_converted,
        "non_image_removed": non_image_removed,
        "webp_converted": webp_converted,
        "resized_to_224": resized_count,
        "per_class_counts": per_class,
        "class_counts": per_class,
        "class_imbalance_percent": imbalance_pct,
        "imbalance_ratio": imbalance_pct,
        "rejected_folder": rejected_root,
        "supported_formats": sorted(ALLOWED_EXTENSIONS),
        "warnings": warnings,
        "recommendations": recommendations,
    }
    os.makedirs(config["MODEL_DIR"], exist_ok=True)
    save_json(report_path, report)
    image_quality_path = config.get("IMAGE_QUALITY_REPORT_PATH")
    if image_quality_path:
        save_json(image_quality_path, report)
    return report


def load_quality_report(config=None) -> Dict:
    from flask import current_app

    if config is None:
        config = current_app.config
    path = config.get("DATASET_QUALITY_REPORT_PATH")
    if not path or not os.path.isfile(path):
        return {}
    import json
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
