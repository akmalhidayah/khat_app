import os
from typing import Dict, List, Tuple

from flask import current_app
from PIL import Image

from services.dataset_readiness_service import count_images_in_dir, count_images_per_class

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def _is_image_file(filename: str) -> bool:
    if "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_training_dataset(config=None, *, fast: bool = False) -> Dict:
    if config is None:
        config = current_app.config

    class_labels: List[str] = config["CLASS_LABELS"]
    splits = {
        "train": config["TRAIN_DIR"],
        "validation": config["VALIDATION_DIR"],
        "test": config["TEST_DIR"],
    }

    errors: List[str] = []
    corrupt_files: List[str] = []
    detected_classes: Dict[str, List[str]] = {}

    for split_name, split_dir in splits.items():
        detected_classes[split_name] = []
        if not os.path.isdir(split_dir):
            errors.append(f"missing dataset folder {split_dir}")
            continue

        for class_name in class_labels:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                errors.append(f"missing dataset folder {class_dir}")
                continue

            images = [f for f in os.listdir(class_dir) if _is_image_file(f)]
            if not images:
                errors.append(f"class folder is empty: {class_dir}")
                continue

            detected_classes[split_name].append(class_name)
            if fast:
                continue
            for filename in images:
                path = os.path.join(class_dir, filename)
                try:
                    with Image.open(path) as img:
                        img.verify()
                    with Image.open(path) as img:
                        img.convert("RGB")
                except Exception:
                    corrupt_files.append(path)

    train_class_counts = count_images_per_class(config["TRAIN_DIR"], class_labels)
    validation_class_counts = count_images_per_class(config["VALIDATION_DIR"], class_labels)
    train_count = sum(train_class_counts.values())
    validation_count = sum(validation_class_counts.values())
    test_count = sum(count_images_per_class(config["TEST_DIR"], class_labels).values())

    return {
        "valid": not errors and train_count > 0 and validation_count > 0,
        "errors": errors,
        "corrupt_files": corrupt_files,
        "train_count": train_count,
        "validation_count": validation_count,
        "test_count": test_count,
        "train_class_counts": train_class_counts,
        "validation_class_counts": validation_class_counts,
        "detected_classes": detected_classes,
        "class_labels": class_labels,
    }


def ensure_training_dataset_ready(config=None) -> Dict:
    report = validate_training_dataset(config)
    if report["errors"]:
        raise ValueError(f"Training failed: {report['errors'][0]}")
    if report["train_count"] == 0 or report["validation_count"] == 0:
        raise ValueError(
            "Training failed: training or validation folder has no valid images. "
            "Please process and split the dataset first."
        )
    if report["corrupt_files"]:
        sample = report["corrupt_files"][:3]
        raise ValueError(
            f"Training failed: found {len(report['corrupt_files'])} unreadable image(s). "
            f"Examples: {', '.join(sample)}"
        )
    return report
