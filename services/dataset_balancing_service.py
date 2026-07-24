"""Balanced dataset generation via augmented oversampling of minority classes."""

import os
import random
import shutil
import uuid
from collections import Counter
from datetime import datetime
from typing import Dict, List

from PIL import Image, ImageEnhance, ImageOps

from services.preprocessing_service import copy_image_for_training, resize_with_padding
from services.training_utils import save_json


def _augment_image(image: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = image.copy()
    if rng.random() > 0.5:
        img = ImageOps.mirror(img)
    angle = rng.uniform(-8, 8)
    img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=(0, 0, 0))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.92, 1.08))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.92, 1.08))
    zoom = rng.uniform(0.95, 1.05)
    w, h = img.size
    nw, nh = int(w * zoom), int(h * zoom)
    img = img.resize((max(nw, 1), max(nh, 1)), Image.LANCZOS)
    return resize_with_padding(img, (224, 224))


def build_balanced_train_set(config=None, target_per_class: int = None) -> Dict:
    from flask import current_app

    if config is None:
        config = current_app.config

    train_dir = config["TRAIN_DIR"]
    balanced_root = config["PROCESSED_BALANCED_DIR"]
    balanced_train = os.path.join(balanced_root, "train")
    class_labels = config["CLASS_LABELS"]

    if os.path.exists(balanced_root):
        shutil.rmtree(balanced_root)
    os.makedirs(balanced_train, exist_ok=True)

    original_counts = {}
    all_images: Dict[str, List[str]] = {}

    for cls in class_labels:
        cls_dir = os.path.join(train_dir, cls)
        os.makedirs(os.path.join(balanced_train, cls), exist_ok=True)
        files = []
        if os.path.isdir(cls_dir):
            files = [
                os.path.join(cls_dir, f)
                for f in os.listdir(cls_dir)
                if os.path.isfile(os.path.join(cls_dir, f))
            ]
        all_images[cls] = files
        original_counts[cls] = len(files)

    if not any(original_counts.values()):
        raise ValueError("Training folder is empty. Split dataset before balancing.")

    max_count = max(original_counts.values())
    if target_per_class is None:
        target_per_class = max_count

    augmented_counts = {cls: 0 for cls in class_labels}
    final_counts = {}

    for cls in class_labels:
        dest_cls = os.path.join(balanced_train, cls)
        sources = all_images[cls]
        for src in sources:
            name = f"{uuid.uuid4().hex}.jpg"
            copy_image_for_training(src, os.path.join(dest_cls, name))

        needed = max(0, target_per_class - len(sources))
        if needed > 0 and sources:
            for i in range(needed):
                src = sources[i % len(sources)]
                with Image.open(src) as img:
                    aug = _augment_image(img.convert("RGB"), seed=i * 17 + hash(cls) % 1000)
                aug_name = f"aug_{uuid.uuid4().hex}.jpg"
                aug.save(os.path.join(dest_cls, aug_name), format="JPEG", quality=92)
                augmented_counts[cls] += 1

        final_counts[cls] = len(os.listdir(dest_cls))

    report = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "target_per_class": target_per_class,
        "original_counts": original_counts,
        "augmented_counts": augmented_counts,
        "final_counts": final_counts,
        "balanced_train_dir": balanced_train,
    }
    save_json(config["DATASET_BALANCE_REPORT_PATH"], report)
    return report
