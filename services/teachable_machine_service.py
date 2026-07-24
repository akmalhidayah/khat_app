"""Load and run Teachable Machine TensorFlow.js models from static/model/."""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from flask import current_app
from PIL import Image, ImageOps

from services.class_mapping_service import resolve_trained_class_labels

_MODEL_CACHE: Dict[str, object] = {
    "model": None,
    "labels": None,
    "image_size": 224,
}


def _paths(config=None) -> Dict[str, str]:
    if config is None:
        config = current_app.config
    model_dir = config["TEACHABLE_MODEL_DIR"]
    return {
        "dir": model_dir,
        "json": config["TEACHABLE_MODEL_JSON"],
        "metadata": config["TEACHABLE_MODEL_METADATA"],
        "weights": config["TEACHABLE_MODEL_WEIGHTS"],
    }


def normalize_tm_label(raw: str) -> str:
    text = (raw or "").strip().lower().replace("-", "_")
    compact = text.replace(" ", "_")

    try:
        trained = resolve_trained_class_labels()
    except RuntimeError:
        trained = ["naskhi", "diwani", "diwani_jali", "tsuluts"]

    if compact in ("diwani_jali", "diwanijali") or ("diwani" in compact and "jali" in compact):
        return "diwani_jali" if "diwani_jali" in trained else compact
    if compact in trained:
        return compact
    if "naskhi" in compact or "naski" in compact:
        return "naskhi" if "naskhi" in trained else compact
    if "diwani" in compact:
        return "diwani" if "diwani" in trained else compact
    if any(token in compact for token in ("tsuluts", "tsuluth", "thuluth", "sulus", "thusuth")):
        return "tsuluts" if "tsuluts" in trained else compact
    return compact


def is_teachable_machine_available(config=None) -> bool:
    if config is None:
        from flask import current_app

        config = current_app.config
    if not config.get("USE_TEACHABLE_MACHINE", True):
        return False
    paths = _paths(config)
    return all(os.path.isfile(paths[key]) for key in ("json", "metadata", "weights"))


def load_tm_metadata(config=None) -> Dict:
    paths = _paths(config)
    with open(paths["metadata"], "r", encoding="utf-8") as handle:
        return json.load(handle)


def clear_tm_model_cache() -> None:
    _MODEL_CACHE["model"] = None
    _MODEL_CACHE["labels"] = None
    _MODEL_CACHE["image_size"] = 224


def load_tm_model(config=None):
    if _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"], _MODEL_CACHE["labels"], _MODEL_CACHE["image_size"]

    try:
        import tensorflow  # noqa: F401
    except ImportError as exc:
        from services.eval_dependency_service import INSTALL_HINT

        raise ImportError(
            "TensorFlow belum terpasang. " + INSTALL_HINT.replace("\n", " ")
        ) from exc

    try:
        from services.tfjs_import_bootstrap import load_keras_model
    except ImportError as exc:
        from services.eval_dependency_service import INSTALL_HINT

        raise ImportError(INSTALL_HINT) from exc

    paths = _paths(config)
    metadata = load_tm_metadata(config)
    raw_labels = metadata.get("labels") or []
    labels = [normalize_tm_label(label) for label in raw_labels]
    image_size = int(metadata.get("imageSize") or 224)

    model = load_keras_model(paths["json"])
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["labels"] = labels
    _MODEL_CACHE["image_size"] = image_size
    return model, labels, image_size


def preprocess_tm_image(image_path: str, image_size: int = 224) -> np.ndarray:
    with Image.open(image_path) as img:
        rgb = ImageOps.exif_transpose(img).convert("RGB")
        resized = rgb.resize((image_size, image_size), Image.Resampling.LANCZOS)
        array = np.array(resized, dtype=np.float32)
    array = (array / 127.5) - 1.0
    return np.expand_dims(array, axis=0)


def _tta_batches(image_path: str, image_size: int) -> List[np.ndarray]:
    from PIL import ImageEnhance

    batches: List[np.ndarray] = []
    with Image.open(image_path) as img:
        base = ImageOps.exif_transpose(img).convert("RGB")
        candidates = [
            base,
            ImageEnhance.Brightness(base).enhance(1.05),
            ImageEnhance.Contrast(base).enhance(1.05),
            base.resize((int(image_size * 0.94), int(image_size * 0.94)), Image.Resampling.LANCZOS),
            base.rotate(3, fillcolor=(255, 255, 255)),
        ]
        for candidate in candidates:
            resized = candidate.resize((image_size, image_size), Image.Resampling.LANCZOS)
            array = np.array(resized, dtype=np.float32)
            array = (array / 127.5) - 1.0
            batches.append(np.expand_dims(array, axis=0))
    return batches


def _scores_from_probs(probs: np.ndarray, labels: List[str], trained_classes: List[str]) -> Dict[str, float]:
    scores = {key: 0.0 for key in trained_classes}
    for index, label in enumerate(labels):
        if index >= len(probs):
            break
        key = normalize_tm_label(label)
        if key in scores:
            scores[key] = float(probs[index])
    return scores


def predict_tm_image(image_path: str, use_tta: bool = False) -> Dict:
    model, labels, image_size = load_tm_model()

    if use_tta:
        batches = _tta_batches(image_path, image_size)
        probs_list = [model.predict(batch, verbose=0)[0] for batch in batches]
        probs = np.mean(probs_list, axis=0)
    else:
        batch = preprocess_tm_image(image_path, image_size)
        probs = model.predict(batch, verbose=0)[0]

    trained_classes = resolve_trained_class_labels()
    raw_scores = _scores_from_probs(probs, labels, trained_classes)
    from services.diwani_pair_service import apply_contested_style_pairs
    from services.probability_refinement_service import (
        build_sorted_probability_list,
        refine_class_probabilities,
    )

    # Same contested-pair cues as Keras path (Diwani/Jali, Jali/Tsuluts).
    raw_scores = apply_contested_style_pairs(raw_scores, image_path, trained_classes)
    scores = refine_class_probabilities(
        raw_scores,
        trained_classes,
        image_path=image_path,
    )
    sorted_probs = build_sorted_probability_list(scores)
    pred_class = sorted_probs[0]["class"] if sorted_probs else None
    confidence = float(sorted_probs[0]["probability"]) if sorted_probs else 0.0
    top_conf = sorted_probs[0]["percent"] if sorted_probs else 0.0
    second_conf = sorted_probs[1]["percent"] if len(sorted_probs) > 1 else 0.0
    margin = round(top_conf - second_conf, 2)

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "scores": scores,
        "probabilities": scores,
        "sorted_probabilities": sorted_probs,
        "top2_margin": margin / 100.0,
        "top2_margin_pct": margin,
        "model_source": "teachable_machine",
        "image_size": image_size,
        "labels": labels,
    }


def get_tm_model_name(metadata: Optional[Dict] = None, config=None) -> str:
    if metadata is None:
        metadata = load_tm_metadata(config) if is_teachable_machine_available(config) else {}
    return (
        metadata.get("modelName")
        or metadata.get("name")
        or "tm-my-image-model"
    )


def get_tm_display_info(config=None) -> Dict:
    paths = _paths(config)
    available = is_teachable_machine_available(config)
    metadata = load_tm_metadata(config) if available else {}
    raw_labels = metadata.get("labels") or []
    labels = [normalize_tm_label(label) for label in raw_labels]
    image_size = int(metadata.get("imageSize") or 224)
    model_name = get_tm_model_name(metadata, config)

    return {
        "available": available,
        "model_source": "Teachable Machine",
        "model_name": model_name,
        "model_runtime": "TensorFlow.js",
        "model_input_size": f"{image_size}x{image_size}",
        "model_input_size_px": image_size,
        "model_type": "Teachable Machine TensorFlow.js",
        "model_architecture": "Teachable Machine Image Model",
        "model_json_path": paths["json"],
        "model_metadata_path": paths["metadata"],
        "model_weights_path": paths["weights"],
        "model_json_url": "/static/model/model.json",
        "model_metadata_url": "/static/model/metadata.json",
        "model_weights_url": "/static/model/weights.bin",
        "labels": labels,
        "labels_display": ", ".join(raw_labels) if raw_labels else "diwani, diwani jali, naskhi, tsuluts",
        "labels_raw": raw_labels,
        "training_date": metadata.get("timeStamp"),
        "tfjs_version": metadata.get("tfjsVersion"),
        "tm_version": metadata.get("tmVersion"),
    }


def get_tm_model_context(config=None) -> Dict:
    info = get_tm_display_info(config)
    paths = _paths(config)
    return {
        "model_exists": info["available"],
        "active_model_path": paths["json"],
        "model_json_url": info["model_json_url"],
        "model_metadata_url": info["model_metadata_url"],
        "model_weights_url": info["model_weights_url"],
        "using_best_model": False,
        "training_mode_key": "teachable_machine",
        "training_mode_label": "Teachable Machine Model",
        "is_demo_model": False,
        "is_research_model": True,
        "model_architecture": info["model_architecture"],
        "model_architecture_key": "teachable_machine",
        "model_source": info["model_source"],
        "model_name": info["model_name"],
        "model_runtime": info["model_runtime"],
        "model_input_size": info["model_input_size"],
        "validation_accuracy": None,
        "test_accuracy": None,
        "training_date": info["training_date"],
        "teachable_machine_labels": info["labels"],
        "teachable_machine_labels_display": info["labels_display"],
        "teachable_machine_image_size": info["model_input_size_px"],
        "tm_display_info": info,
    }
