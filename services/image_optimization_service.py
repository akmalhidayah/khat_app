"""Image compression and optimization for Arabic Khat dataset images."""

import json
import os
import threading
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DATASET_SOURCE_FOLDERS = [
    "raw",
    "train",
    "validation",
    "test",
    "processed",
    "processed_balanced",
]
DEFAULT_FILL = (255, 255, 255)
STATUS_FILENAME = "image_optimization_status.json"
REPORT_FILENAME = "image_optimization_report.json"

_optimization_lock = threading.Lock()
_optimization_thread: Optional[threading.Thread] = None


def _is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _jpg_output_path(output_path: str) -> str:
    base, _ = os.path.splitext(output_path)
    return f"{base}.jpg"


def _bytes_to_mb(num_bytes: int) -> float:
    return round(num_bytes / (1024 * 1024), 2)


def _open_image_safe(input_path: str) -> Image.Image:
    with Image.open(input_path) as img:
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")


def optimize_image(
    input_path: str,
    output_path: str,
    max_size: int = 512,
    quality: int = 80,
    output_format: str = "JPEG",
) -> Dict:
    """Resize and compress a single image with EXIF correction."""
    original_size = 0
    try:
        original_size = os.path.getsize(input_path)
    except OSError:
        pass

    result = {
        "input_path": input_path,
        "output_path": output_path,
        "original_size": original_size,
        "optimized_size": 0,
        "width": 0,
        "height": 0,
        "saved_bytes": 0,
        "compression_ratio": 0.0,
        "status": "failed",
        "error": None,
    }

    fmt = (output_format or "JPEG").upper()
    if fmt in ("JPG", "JPEG"):
        output_path = _jpg_output_path(output_path)
        save_format = "JPEG"
        save_kwargs = {"quality": quality, "optimize": True, "progressive": True}
    elif fmt == "WEBP":
        base, _ = os.path.splitext(output_path)
        output_path = f"{base}.webp"
        save_format = "WEBP"
        save_kwargs = {"quality": quality, "method": 6}
    else:
        save_format = fmt
        save_kwargs = {"quality": quality, "optimize": True}

    result["output_path"] = output_path

    try:
        image = _open_image_safe(input_path)
        image.thumbnail((max_size, max_size), Image.LANCZOS)
        result["width"], result["height"] = image.size
        _ensure_parent_dir(output_path)
        image.save(output_path, format=save_format, **save_kwargs)
        optimized_size = os.path.getsize(output_path)
        result["optimized_size"] = optimized_size
        result["saved_bytes"] = max(original_size - optimized_size, 0)
        result["compression_ratio"] = (
            round((result["saved_bytes"] / original_size) * 100, 2) if original_size > 0 else 0.0
        )
        result["status"] = "optimized"
    except Exception as exc:
        result["error"] = str(exc)

    return result


def trim_near_white_border(image: Image.Image, threshold: int = 245, margin: int = 10) -> Image.Image:
    """Crop excessive near-white borders while keeping margin around calligraphy strokes."""
    rgb = image.convert("RGB")
    gray = ImageOps.grayscale(rgb)
    bw = gray.point(lambda p: 255 if p >= threshold else 0, mode="1")
    bbox = ImageOps.invert(bw.convert("L")).getbbox()
    if not bbox:
        return rgb

    left, top, right, bottom = bbox
    width, height = rgb.size
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(width, right + margin)
    bottom = min(height, bottom + margin)

    cropped_w = right - left
    cropped_h = bottom - top
    if cropped_w < width * 0.35 or cropped_h < height * 0.35:
        return rgb

    return rgb.crop((left, top, right, bottom))


def resize_with_padding(
    image: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    fill: Tuple[int, int, int] = DEFAULT_FILL,
) -> Image.Image:
    image = image.convert("RGB")
    tw, th = target_size
    w, h = image.size
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = image.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", target_size, fill)
    canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return canvas


def optimize_single_image(
    input_path: str,
    output_path: str,
    max_size: int = 1024,
    quality: int = 85,
    convert_to_jpg: bool = True,
    force: bool = False,
) -> Dict:
    """Create an optimized display copy (max dimension, JPEG, progressive)."""
    result = {
        "input_path": input_path,
        "output_path": output_path,
        "status": "skipped",
        "original_bytes": 0,
        "optimized_bytes": 0,
        "error": None,
    }

    if not os.path.isfile(input_path):
        result["status"] = "failed"
        result["error"] = "Source file not found"
        return result

    if convert_to_jpg:
        output_path = _jpg_output_path(output_path)

    result["output_path"] = output_path

    if os.path.isfile(output_path) and not force:
        result["status"] = "skipped_exists"
        result["original_bytes"] = os.path.getsize(input_path)
        result["optimized_bytes"] = os.path.getsize(output_path)
        return result

    try:
        result["original_bytes"] = os.path.getsize(input_path)
        image = _open_image_safe(input_path)
        image.thumbnail((max_size, max_size), Image.LANCZOS)
        _ensure_parent_dir(output_path)
        image.save(
            output_path,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
        )
        result["optimized_bytes"] = os.path.getsize(output_path)
        result["status"] = "optimized"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)

    return result


def create_model_ready_image(
    input_path: str,
    output_path: str,
    image_size: int = 224,
    quality: int = 90,
    trim_border: bool = False,
    force: bool = False,
) -> Dict:
    """Create a padded square model-ready JPEG (224x224 by default)."""
    result = {
        "input_path": input_path,
        "output_path": output_path,
        "status": "skipped",
        "original_bytes": 0,
        "optimized_bytes": 0,
        "error": None,
    }

    if not os.path.isfile(input_path):
        result["status"] = "failed"
        result["error"] = "Source file not found"
        return result

    output_path = _jpg_output_path(output_path)
    result["output_path"] = output_path

    if os.path.isfile(output_path) and not force:
        result["status"] = "skipped_exists"
        result["original_bytes"] = os.path.getsize(input_path)
        result["optimized_bytes"] = os.path.getsize(output_path)
        return result

    try:
        result["original_bytes"] = os.path.getsize(input_path)
        image = _open_image_safe(input_path)
        if trim_border:
            image = trim_near_white_border(image)
        processed = resize_with_padding(image, (image_size, image_size))
        _ensure_parent_dir(output_path)
        processed.save(
            output_path,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
        )
        result["optimized_bytes"] = os.path.getsize(output_path)
        result["status"] = "optimized"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)

    return result


def optimize_dataset_images(
    source_dir: str,
    output_dir: str,
    max_size: int = 1024,
    quality: int = 85,
    model_size: int = 224,
    model_quality: int = 90,
    convert_to_jpg: bool = True,
    trim_border: bool = False,
    force: bool = False,
    create_display: bool = True,
    create_model: bool = True,
    display_output_dir: Optional[str] = None,
    model_output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict:
    """Scan source_dir recursively and write optimized display/model copies."""
    if not os.path.isdir(source_dir):
        raise ValueError(f"Image optimization failed: source folder not found ({source_dir}).")

    display_root = display_output_dir or os.path.join(output_dir, "display")
    model_root = model_output_dir or os.path.join(output_dir, "model")
    source_dir = os.path.normpath(source_dir)

    image_files: List[str] = []
    for root, _, files in os.walk(source_dir):
        for fname in files:
            if _is_image_file(fname):
                image_files.append(os.path.join(root, fname))

    stats = {
        "total_scanned": len(image_files),
        "optimized": 0,
        "skipped": 0,
        "skipped_exists": 0,
        "corrupted": 0,
        "unsupported": 0,
        "failed": [],
        "original_bytes": 0,
        "optimized_bytes": 0,
        "per_class": {},
        "per_format": {},
        "display_output_dir": display_root,
        "model_output_dir": model_root,
        "source_dir": source_dir,
    }

    total = len(image_files)
    for index, input_path in enumerate(image_files, start=1):
        rel_path = os.path.relpath(input_path, source_dir)
        rel_dir = os.path.dirname(rel_path)
        base_name = os.path.basename(rel_path)

        parts = rel_path.replace("\\", "/").split("/")
        class_name = None
        for part in parts:
            if part in ("naskhi", "diwani", "diwani_jali", "tsuluts"):
                class_name = part
                break

        ext = os.path.splitext(base_name)[1].lower()
        stats["per_format"][ext] = stats["per_format"].get(ext, 0) + 1
        if class_name:
            stats["per_class"][class_name] = stats["per_class"].get(class_name, 0) + 1

        if progress_callback:
            progress_callback(index, total, rel_path)

        display_result = None
        model_result = None

        if create_display:
            display_out = os.path.join(display_root, rel_dir, base_name)
            display_result = optimize_single_image(
                input_path,
                display_out,
                max_size=max_size,
                quality=quality,
                convert_to_jpg=convert_to_jpg,
                force=force,
            )

        if create_model:
            model_out = os.path.join(model_root, rel_dir, base_name)
            model_result = create_model_ready_image(
                input_path,
                model_out,
                image_size=model_size,
                quality=model_quality,
                trim_border=trim_border,
                force=force,
            )

        results = [r for r in (display_result, model_result) if r is not None]
        if not results:
            stats["skipped"] += 1
            continue

        stats["original_bytes"] += results[0].get("original_bytes", 0)
        any_success = False
        any_corrupt = False
        for res in results:
            stats["optimized_bytes"] += res.get("optimized_bytes", 0)
            status = res.get("status")
            if status == "optimized":
                any_success = True
            elif status == "skipped_exists":
                stats["skipped_exists"] += 1
            elif status == "failed":
                any_corrupt = True
                stats["failed"].append({"path": input_path, "error": res.get("error")})

        if any_success:
            stats["optimized"] += 1
        elif any_corrupt:
            stats["corrupted"] += 1
        else:
            stats["skipped"] += 1

    return stats


def _status_path(config: Dict) -> str:
    return os.path.join(config["MODEL_DIR"], STATUS_FILENAME)


def _report_path(config: Dict) -> str:
    return os.path.join(config["MODEL_DIR"], REPORT_FILENAME)


def load_optimization_status(config: Dict) -> Dict:
    path = _status_path(config)
    if not os.path.isfile(path):
        return {"status": "idle", "progress": 0, "processed": 0, "total": 0, "message": "No optimization run yet."}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {"status": "idle", "progress": 0, "processed": 0, "total": 0, "message": "Status unavailable."}


def load_optimization_report(config: Dict) -> Dict:
    path = _report_path(config)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _update_status(config: Dict, **fields) -> Dict:
    current = load_optimization_status(config)
    current.update(fields)
    if current.get("total"):
        current["progress"] = round((current.get("processed", 0) / current["total"]) * 100, 1)
    _save_json(_status_path(config), current)
    return current


def generate_optimization_report(stats_list: List[Dict], config: Dict, output_base: str) -> Dict:
    """Aggregate stats from multiple source folders into a final report."""
    total_scanned = sum(s.get("total_scanned", 0) for s in stats_list)
    optimized = sum(s.get("optimized", 0) for s in stats_list)
    skipped = sum(s.get("skipped", 0) for s in stats_list)
    skipped_exists = sum(s.get("skipped_exists", 0) for s in stats_list)
    corrupted = sum(s.get("corrupted", 0) for s in stats_list)
    original_bytes = sum(s.get("original_bytes", 0) for s in stats_list)
    optimized_bytes = sum(s.get("optimized_bytes", 0) for s in stats_list)

    per_class: Dict[str, int] = {}
    per_format: Dict[str, int] = {}
    failed: List[Dict] = []
    for stats in stats_list:
        for cls, count in stats.get("per_class", {}).items():
            per_class[cls] = per_class.get(cls, 0) + count
        for fmt, count in stats.get("per_format", {}).items():
            per_format[fmt] = per_format.get(fmt, 0) + count
        failed.extend(stats.get("failed", []))

    saved_bytes = max(original_bytes - optimized_bytes, 0)
    compression_percent = round((saved_bytes / original_bytes) * 100, 1) if original_bytes else 0.0

    report = {
        "total_scanned": total_scanned,
        "optimized": optimized,
        "skipped": skipped + skipped_exists,
        "skipped_exists": skipped_exists,
        "corrupted": corrupted,
        "unsupported": 0,
        "failed_files": failed[:100],
        "failed_count": len(failed),
        "original_size_mb": _bytes_to_mb(original_bytes),
        "optimized_size_mb": _bytes_to_mb(optimized_bytes),
        "saved_mb": _bytes_to_mb(saved_bytes),
        "compression_percent": compression_percent,
        "per_class": per_class,
        "per_format": per_format,
        "output_folder": output_base,
        "display_folder": os.path.join(output_base, "display"),
        "model_folder": os.path.join(output_base, "model"),
        "processing_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source_folders": [s.get("source_dir") for s in stats_list],
    }
    _save_json(_report_path(config), report)
    return report


def _count_images_recursive(root_dir: str) -> int:
    if not os.path.isdir(root_dir):
        return 0
    total = 0
    for _, _, files in os.walk(root_dir):
        total += sum(1 for f in files if _is_image_file(f))
    return total


def optimize_all_dataset_folders(
    config: Dict,
    quality: int = 85,
    max_size: int = 1024,
    model_size: int = 224,
    model_quality: int = 90,
    trim_border: bool = False,
    force: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict:
    """Optimize all known dataset folders into dataset/optimized/."""
    dataset_dir = config["DATASET_DIR"]
    output_base = config["OPTIMIZED_DIR"]
    display_root = config["OPTIMIZED_DISPLAY_DIR"]
    model_root = config["OPTIMIZED_MODEL_DIR"]

    existing_sources = []
    for folder in DATASET_SOURCE_FOLDERS:
        source = os.path.join(dataset_dir, folder)
        if os.path.isdir(source) and _count_images_recursive(source) > 0:
            existing_sources.append(source)

    if not existing_sources:
        raise ValueError("Image optimization failed: no dataset image folders found.")

    all_stats: List[Dict] = []
    processed_global = 0
    total_global = sum(_count_images_recursive(src) for src in existing_sources)

    _update_status(
        config,
        status="running",
        progress=0,
        processed=0,
        total=total_global,
        message="Optimizing images...",
        started_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

    for source_dir in existing_sources:
        folder_name = os.path.basename(source_dir)
        folder_display = os.path.join(display_root, folder_name)
        folder_model = os.path.join(model_root, folder_name)

        def _folder_progress(local_index: int, local_total: int, rel_path: str) -> None:
            nonlocal processed_global
            processed_global += 1
            message = f"Optimizing {folder_name}: {rel_path}"
            if progress_callback:
                progress_callback(processed_global, total_global, message)
            _update_status(
                config,
                status="running",
                processed=processed_global,
                total=total_global,
                message=message,
            )

        stats = optimize_dataset_images(
            source_dir=source_dir,
            output_dir=output_base,
            max_size=max_size,
            quality=quality,
            model_size=model_size,
            model_quality=model_quality,
            trim_border=trim_border,
            force=force,
            display_output_dir=folder_display,
            model_output_dir=folder_model,
            progress_callback=_folder_progress,
        )
        all_stats.append(stats)

    report = generate_optimization_report(all_stats, config, output_base)
    _update_status(
        config,
        status="completed",
        processed=total_global,
        total=total_global,
        progress=100,
        message="Image optimization completed successfully.",
        finished_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        report_summary={
            "optimized": report["optimized"],
            "skipped": report["skipped"],
            "original_size_mb": report["original_size_mb"],
            "optimized_size_mb": report["optimized_size_mb"],
            "saved_mb": report["saved_mb"],
            "compression_percent": report["compression_percent"],
        },
    )
    return report


def resolve_optimized_display_path(original_abs_path: str, config: Dict) -> Optional[str]:
    """Map an original dataset file to its optimized display copy if it exists."""
    if not original_abs_path or not os.path.isfile(original_abs_path):
        return None

    dataset_dir = os.path.normpath(config["DATASET_DIR"])
    abs_path = os.path.normpath(original_abs_path)
    if not abs_path.startswith(dataset_dir):
        return None

    rel_path = os.path.relpath(abs_path, dataset_dir)
    display_root = config["OPTIMIZED_DISPLAY_DIR"]
    base, _ = os.path.splitext(rel_path)
    candidates = [
        os.path.join(display_root, f"{base}.jpg"),
        os.path.join(display_root, f"{base}.jpeg"),
        os.path.join(display_root, rel_path),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def is_optimization_running(config: Dict) -> bool:
    status = load_optimization_status(config)
    return status.get("status") == "running"


def start_optimization_background(app, **kwargs) -> bool:
    """Start optimization in a background thread. Returns False if already running."""
    global _optimization_thread

    with _optimization_lock:
        config = app.config
        if is_optimization_running(config):
            return False
        if _optimization_thread and _optimization_thread.is_alive():
            return False

        def _worker():
            with app.app_context():
                try:
                    optimize_all_dataset_folders(config, **kwargs)
                except Exception as exc:
                    _update_status(
                        config,
                        status="failed",
                        message=f"Image optimization failed: {exc}",
                        finished_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                    )

        _optimization_thread = threading.Thread(target=_worker, daemon=True)
        _optimization_thread.start()
        return True
