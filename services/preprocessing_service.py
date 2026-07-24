import hashlib
import json
import os
from typing import Dict, List, Set, Tuple

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split


def image_hash(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as image_file:
        for chunk in iter(lambda: image_file.read(4096), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(image_path: str) -> str:
    with Image.open(image_path) as img:
        gray = ImageOps.grayscale(img.convert("RGB"))
        small = gray.resize((16, 16), Image.LANCZOS)
        pixels = list(small.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p >= avg else "0" for p in pixels)
    return format(int(bits, 2), "064x")


def is_valid_image(file_stream) -> bool:
    try:
        image = Image.open(file_stream)
        image.verify()
        file_stream.seek(0)
        return True
    except Exception:
        return False


def resize_with_padding(image: Image.Image, target_size=(224, 224), fill=(255, 255, 255)) -> Image.Image:
    image = image.convert("RGB")
    tw, th = target_size
    w, h = image.size
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = image.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", target_size, fill)
    canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return canvas


def get_architecture_preprocess_fn(architecture: str = "efficientnetb0"):
    """Return TensorFlow preprocessing function for the given architecture."""
    architecture = (architecture or "efficientnetb0").lower()
    if architecture in ("keras_h5", "teachable_machine_h5", "external_h5"):
        return lambda array: (array / 127.5) - 1.0
    if architecture == "vgg16":
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input
    if architecture == "mobilenetv2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input


def trim_excessive_borders(image: Image.Image, threshold: int = 245, margin: int = 8) -> Image.Image:
    """Trim near-white borders while keeping a small margin around content."""
    rgb = image.convert("RGB")
    gray = ImageOps.grayscale(rgb)
    mask = gray.point(lambda p: 255 if p < threshold else 0)
    bbox = mask.getbbox()
    if not bbox:
        return rgb
    left = max(0, bbox[0] - margin)
    top = max(0, bbox[1] - margin)
    right = min(rgb.width, bbox[2] + margin)
    bottom = min(rgb.height, bbox[3] + margin)
    if right - left < 10 or bottom - top < 10:
        return rgb
    return rgb.crop((left, top, right, bottom))


def preprocess_external_keras_image(image_path: str, target_size=(224, 224)) -> np.ndarray:
    """Preprocessing for external Keras H5 models (Teachable Machine / Colab export style)."""
    with Image.open(image_path) as img:
        rgb = ImageOps.exif_transpose(img).convert("RGB")
        resized = rgb.resize(target_size, Image.Resampling.LANCZOS)
        array = np.array(resized, dtype=np.float32)
    array = (array / 127.5) - 1.0
    return np.expand_dims(array, axis=0)


def preprocess_calligraphy_image(
    image_path: str,
    target_size=(224, 224),
    architecture: str = "efficientnetb0",
    fill=(255, 255, 255),
) -> np.ndarray:
    """
    Shared calligraphy preprocessing for training and prediction.
    EXIF → RGB → trim borders → aspect-preserving pad → resize → architecture preprocess.
    """
    with Image.open(image_path) as img:
        image = ImageOps.exif_transpose(img).convert("RGB")
        image = trim_excessive_borders(image)
        processed = resize_with_padding(image, target_size, fill)
    array = np.array(processed, dtype=np.float32)
    preprocess_fn = get_architecture_preprocess_fn(architecture)
    array = preprocess_fn(array)
    return np.expand_dims(array, axis=0)


def preprocess_for_model(
    image_path: str,
    architecture: str = "efficientnetb0",
    preprocessing_mode: str = "standard",
) -> np.ndarray:
    arch = (architecture or "").lower()
    if arch in ("keras_h5", "teachable_machine_h5", "external_h5"):
        return preprocess_external_keras_image(image_path)
    if preprocessing_mode == "manuscript":
        return preprocess_manuscript_image(image_path, architecture=architecture)
    return preprocess_calligraphy_image(image_path, architecture=architecture)


def preprocess_pil_image(image: Image.Image, target_size=(224, 224), fill=(255, 255, 255)) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = trim_excessive_borders(image)
    return resize_with_padding(image, target_size, fill)


def normalize_background(image: Image.Image, threshold: int = 238) -> Image.Image:
    """Lighten near-white pixels to reduce frame/background noise."""
    rgb = image.convert("RGB")
    pixels = rgb.load()
    width, height = rgb.size
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if r >= threshold and g >= threshold and b >= threshold:
                pixels[x, y] = (255, 255, 255)
    return rgb


def center_crop_content_area(image: Image.Image, padding_ratio: float = 0.04) -> Image.Image:
    """Tight crop around ink/content after border trim."""
    rgb = image.convert("RGB")
    gray = ImageOps.grayscale(rgb)
    mask = gray.point(lambda p: 255 if p < 242 else 0)
    bbox = mask.getbbox()
    if not bbox:
        return rgb
    left, top, right, bottom = bbox
    bw, bh = right - left, bottom - top
    pad_w = int(bw * padding_ratio)
    pad_h = int(bh * padding_ratio)
    left = max(0, left - pad_w)
    top = max(0, top - pad_h)
    right = min(rgb.width, right + pad_w)
    bottom = min(rgb.height, bottom + pad_h)
    cropped = rgb.crop((left, top, right, bottom))
    if cropped.width < 10 or cropped.height < 10:
        return rgb
    return cropped


def preprocess_manuscript_image(
    image_path: str,
    target_size=(224, 224),
    architecture: str = "efficientnetb0",
    fill=(255, 255, 255),
) -> np.ndarray:
    """Document/manuscript mode: stronger border removal and content focus."""
    with Image.open(image_path) as img:
        image = ImageOps.exif_transpose(img).convert("RGB")
        image = trim_excessive_borders(image, threshold=235, margin=4)
        image = center_crop_content_area(image)
        image = normalize_background(image)
        processed = resize_with_padding(image, target_size, fill)
    array = np.array(processed, dtype=np.float32)
    preprocess_fn = get_architecture_preprocess_fn(architecture)
    array = preprocess_fn(array)
    return np.expand_dims(array, axis=0)


def preprocess_image_for_model(
    image_path: str,
    target_size=(224, 224),
    architecture: str = "efficientnetb0",
    preprocessing_mode: str = "standard",
) -> np.ndarray:
    """Load, pad-resize, and apply architecture-specific preprocessing."""
    if preprocessing_mode == "manuscript":
        return preprocess_manuscript_image(image_path, target_size, architecture)
    return preprocess_calligraphy_image(image_path, target_size, architecture)


def load_split_report(report_path: str) -> Dict:
    if not report_path or not os.path.isfile(report_path):
        return {}
    try:
        with open(report_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


def copy_image_for_training(source: str, destination: str) -> str:
    """Copy image into split folders as padded RGB JPEG 224x224."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    base_name = os.path.splitext(os.path.basename(destination))[0]
    target_path = os.path.join(os.path.dirname(destination), f"{base_name}.jpg")
    with Image.open(source) as image:
        processed = resize_with_padding(image, (224, 224))
        processed.save(target_path, format="JPEG", quality=95)
    return target_path


def _item_fingerprint(item: dict) -> Tuple[str, str]:
    path = item["image_path"]
    return image_hash(path), perceptual_hash(path)


def split_dataset_stratified(
    dataset_items: List[dict],
    seed: int = 42,
    phash_threshold: int = 5,
    deduplicate_hashes: bool = True,
) -> Dict[str, List[dict]]:
    if not dataset_items:
        raise ValueError("Dataset is empty.")

    unique_items = []
    seen_hashes: Set[str] = set()
    for item in dataset_items:
        if deduplicate_hashes:
            try:
                fhash, phash = _item_fingerprint(item)
            except Exception:
                continue
            if fhash in seen_hashes:
                continue
            seen_hashes.add(fhash)
            item = dict(item)
            item["_file_hash"] = fhash
            item["_phash"] = phash
        else:
            item = dict(item)
            try:
                item["_file_hash"], item["_phash"] = _item_fingerprint(item)
            except Exception:
                item["_file_hash"] = ""
                item["_phash"] = ""
        unique_items.append(item)

    labels = [item["class_name"] for item in unique_items]
    train_val_items, test_items = train_test_split(
        unique_items, test_size=0.20, random_state=seed, stratify=labels
    )
    train_val_labels = [item["class_name"] for item in train_val_items]
    train_items, val_items = train_test_split(
        train_val_items, test_size=0.15, random_state=seed, stratify=train_val_labels
    )

    def _dedup_across_splits(splits: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
        used_phashes: List[str] = []
        cleaned = {}
        for split_name, items in splits.items():
            kept = []
            for item in items:
                phash = item.get("_phash", "")
                conflict = False
                for existing in used_phashes:
                    ia, ib = int(phash, 16), int(existing, 16)
                    if bin(ia ^ ib).count("1") <= phash_threshold:
                        conflict = True
                        break
                if conflict:
                    continue
                used_phashes.append(phash)
                kept.append(item)
            cleaned[split_name] = kept
        return cleaned

    if phash_threshold < 0:
        splits = {
            "train": train_items,
            "validation": val_items,
            "test": test_items,
        }
    else:
        splits = _dedup_across_splits({
            "train": train_items,
            "validation": val_items,
            "test": test_items,
        })
    return splits


def save_split_report(
    splits: Dict[str, List[dict]],
    report_path: str,
    class_labels: List[str],
    *,
    deduplication_applied: bool = False,
) -> Dict:
    per_split = {}
    for split_name, items in splits.items():
        counts = {cls: 0 for cls in class_labels}
        for item in items:
            cls = item.get("class_name")
            if cls in counts:
                counts[cls] += 1
        per_split[split_name] = {
            "total": len(items),
            "per_class": counts,
        }

    report = {
        "split_ratio": "80/20 (train+validation / test holdout)",
        "effective_ratio": "68/12/20 train/validation/test",
        "train_holdout_percent": 80,
        "test_holdout_percent": 20,
        "stratified": True,
        "deduplication_applied": deduplication_applied,
        "evaluation_policy": "Evaluate only on test holdout — never on training images.",
        "splits": per_split,
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def ensure_directories(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)
