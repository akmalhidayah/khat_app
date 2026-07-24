"""Step 4 — CNN / VGG16 penultimate-layer feature extraction."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_feature_model = None
_feature_lock = threading.Lock()
_feature_source = None


def _build_feature_model(classifier) -> Any:
    """Return a model that outputs the penultimate layer embedding."""
    from tensorflow.keras.models import Model

    # Prefer explicit dense/global pooling before softmax.
    for layer in reversed(classifier.layers[:-1]):
        name = (layer.name or "").lower()
        if any(key in name for key in ("dense", "dropout", "global", "flatten", "gap")):
            try:
                return Model(inputs=classifier.inputs, outputs=layer.output)
            except Exception:  # noqa: BLE001
                continue
    try:
        return Model(inputs=classifier.input, outputs=classifier.layers[-2].output)
    except Exception:  # noqa: BLE001
        # Fallback: flatten last conv activation.
        return Model(inputs=classifier.inputs, outputs=classifier.layers[-2].output)


def get_feature_model(classifier=None):
    """Lazy singleton feature extractor bound to the active classifier."""
    global _feature_model, _feature_source
    if classifier is None:
        from flask import current_app

        from services.class_mapping_service import resolve_trained_class_labels
        from services.model_cache_service import get_cached_classifier
        from services.model_context_service import get_active_model_path, get_model_context, load_model_metadata

        config = current_app.config
        model_path = get_active_model_path(config)
        class_labels = resolve_trained_class_labels(config)
        ctx = get_model_context(config) if config else {}
        meta = load_model_metadata(config) if config else {}
        architecture = (
            ctx.get("model_architecture_key")
            or meta.get("model_architecture_key")
            or meta.get("architecture")
            or "efficientnetb0"
        )
        if config.get("USE_EXTERNAL_MODEL") and model_path == config.get("EXTERNAL_MODEL_PATH"):
            architecture = "keras_h5"
        classifier = get_cached_classifier(model_path, architecture, len(class_labels))
    src = id(classifier)
    if _feature_model is not None and _feature_source == src:
        return _feature_model
    with _feature_lock:
        if _feature_model is not None and _feature_source == src:
            return _feature_model
        _feature_model = _build_feature_model(classifier)
        _feature_source = src
        logger.info("Feature extractor ready from layer near classifier head")
        return _feature_model


def preprocess_image_array(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    from tensorflow.keras.applications.vgg16 import preprocess_input

    img = load_img(image_path, target_size=target_size)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    # External Teachable Machine / custom H5 models often expect 0..1; try VGG preprocess first.
    try:
        return preprocess_input(arr.copy())
    except Exception:  # noqa: BLE001
        return arr / 255.0


def extract_embedding(image_path: str, classifier=None) -> np.ndarray:
    """Return L2-normalized embedding vector for an image."""
    model = get_feature_model(classifier)
    # Match classifier input size when available.
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        h = int(shape[1]) if shape and shape[1] else 224
        w = int(shape[2]) if shape and shape[2] else 224
    except Exception:  # noqa: BLE001
        h = w = 224

    # Prefer the app's existing preprocess when possible.
    batch = None
    try:
        from services.preprocessing_service import preprocess_for_model

        batch = preprocess_for_model(image_path)
    except Exception:  # noqa: BLE001
        batch = preprocess_image_array(image_path, target_size=(h, w))

    feats = model.predict(batch, verbose=0)
    vec = np.asarray(feats, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec)) + 1e-8
    return vec / norm
