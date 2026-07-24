"""In-memory cache for loaded Keras classifiers."""

from __future__ import annotations

from typing import Dict, Tuple

_CACHE: Dict[str, object] = {}


def cache_key(model_path: str, architecture: str, num_classes: int) -> str:
    return f"{model_path}|{architecture}|{num_classes}"


def get_cached_classifier(model_path: str, architecture: str, num_classes: int):
    key = cache_key(model_path, architecture, num_classes)
    if key not in _CACHE:
        from services.model_builder_service import load_trained_classifier

        model = load_trained_classifier(model_path, architecture, num_classes)
        output_shape = getattr(model, "output_shape", None)
        if isinstance(output_shape, list):
            output_shape = output_shape[0]
        actual_outputs = int(output_shape[-1]) if output_shape and output_shape[-1] else None
        if actual_outputs is not None and actual_outputs != int(num_classes):
            raise ValueError(
                f"Model output size is {actual_outputs}; expected {num_classes}. "
                "Prediction stopped to prevent an invalid class-index mapping."
            )
        _CACHE[key] = model
    return _CACHE[key]


def get_cached_keras_model(model_path: str):
    """Load external/local Keras H5 with default 4-class head (evaluation compat)."""
    return get_cached_classifier(model_path, "keras_h5", 4)


def clear_model_cache() -> None:
    _CACHE.clear()
