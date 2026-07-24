"""Login page UI helpers — decorative dataset artwork selection."""

import os
from typing import Optional, Tuple

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG")
PREFERRED_CLASSES = ("tsuluts", "naskhi", "diwani_jali", "diwani")
PREFERRED_SAMPLE = "tsuluts_110.jpg"
STATIC_LOGIN_HERO = "images/login-calligraphy-hero.jpg"
STATIC_LOGIN_BG = "images/login-calligraphy-bg.jpg"
STATIC_LOGIN_IMAGE = "images/login-calligraphy.jpg"
HERO_TARGET_WIDTH = 1600
MIN_HERO_WIDTH = 800


def _is_image_file(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(ext.lower()) for ext in IMAGE_EXTENSIONS)


def _image_dimensions(path: str) -> Tuple[int, int]:
    try:
        with Image.open(path) as img:
            return img.size
    except OSError:
        return (0, 0)


def _pick_best_from_class_dir(class_dir: str) -> Optional[str]:
    if not os.path.isdir(class_dir):
        return None
    preferred = os.path.join(class_dir, PREFERRED_SAMPLE)
    if os.path.isfile(preferred):
        return preferred

    best_path = None
    best_score = -1
    for name in os.listdir(class_dir):
        if name.startswith("aug_") or not _is_image_file(name):
            continue
        path = os.path.join(class_dir, name)
        if not os.path.isfile(path):
            continue
        width, height = _image_dimensions(path)
        score = width * height
        if score > best_score:
            best_score = score
            best_path = path
    return best_path


def _pick_source_image(config) -> Optional[str]:
    base_dir = config["BASE_DIR"]
    # Prefer dedicated dark cinematic login background when present.
    for relative in (STATIC_LOGIN_BG, STATIC_LOGIN_HERO, STATIC_LOGIN_IMAGE):
        candidate = os.path.join(base_dir, "static", relative)
        if os.path.isfile(candidate):
            width, _ = _image_dimensions(candidate)
            if width >= MIN_HERO_WIDTH or relative == STATIC_LOGIN_BG:
                return candidate

    raw_root = config.get("RAW_DATASET_DIR")
    if raw_root:
        for class_name in PREFERRED_CLASSES:
            picked = _pick_best_from_class_dir(os.path.join(raw_root, class_name))
            if picked:
                return picked

    for split_key in ("TRAIN_DIR", "VALIDATION_DIR", "TEST_DIR"):
        split_dir = config.get(split_key)
        if not split_dir:
            continue
        for class_name in PREFERRED_CLASSES:
            picked = _pick_best_from_class_dir(os.path.join(split_dir, class_name))
            if picked:
                return picked

    legacy = os.path.join(base_dir, "static", STATIC_LOGIN_IMAGE)
    if os.path.isfile(legacy):
        return legacy
    return None


def build_login_hero_image(source_path: str, dest_path: str, target_width: int = HERO_TARGET_WIDTH) -> bool:
    """Build a high-quality login hero JPEG from a source artwork."""
    if not source_path or not os.path.isfile(source_path):
        return False
    try:
        with Image.open(source_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            width, height = img.size
            if width <= 0 or height <= 0:
                return False

            if width != target_width:
                scale = target_width / float(width)
                new_height = max(1, int(round(height * scale)))
                img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)

            img = img.filter(ImageFilter.UnsharpMask(radius=1.15, percent=118, threshold=3))
            img = ImageEnhance.Contrast(img).enhance(1.1)
            img = ImageEnhance.Color(img).enhance(1.08)
            img = ImageEnhance.Brightness(img).enhance(1.0)

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            img.save(dest_path, "JPEG", quality=92, optimize=True, subsampling=0)
        return os.path.isfile(dest_path)
    except OSError:
        return False


def ensure_login_hero_image(config) -> Optional[str]:
    """Return path to HD hero image, regenerating from preferred BG when needed."""
    base_dir = config["BASE_DIR"]
    hero_path = os.path.join(base_dir, "static", STATIC_LOGIN_HERO)
    bg_path = os.path.join(base_dir, "static", STATIC_LOGIN_BG)

    if os.path.isfile(bg_path):
        # Keep hero in sync with dedicated background asset.
        bg_mtime = os.path.getmtime(bg_path)
        hero_mtime = os.path.getmtime(hero_path) if os.path.isfile(hero_path) else 0
        if (not os.path.isfile(hero_path)) or bg_mtime > hero_mtime:
            if build_login_hero_image(bg_path, hero_path):
                return hero_path
        if os.path.isfile(hero_path):
            return hero_path
        return bg_path

    if os.path.isfile(hero_path):
        width, _ = _image_dimensions(hero_path)
        if width >= MIN_HERO_WIDTH:
            return hero_path

    source = _pick_source_image(config)
    if not source:
        return None
    if build_login_hero_image(source, hero_path):
        return hero_path
    return source if os.path.isfile(source) else None


def resolve_login_decorative_image(config) -> Tuple[Optional[str], str]:
    """
    Return (absolute_path, source_label).
    Prefers generated HD hero artwork for crisp login display.
    """
    hero = ensure_login_hero_image(config)
    if hero and os.path.isfile(hero):
        width, height = _image_dimensions(hero)
        if width >= MIN_HERO_WIDTH or hero.endswith(STATIC_LOGIN_HERO) or hero.endswith(STATIC_LOGIN_BG):
            return hero, "hero_hd"
        return hero, "source_fallback"

    source = _pick_source_image(config)
    if source:
        return source, "dataset"
    return None, "none"
