"""Validate class index mapping consistency across training, prediction, and config."""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from flask import current_app

logger = logging.getLogger(__name__)

CLASS_NAMES = ["naskhi", "diwani", "diwani_jali", "tsuluts"]


def validate_production_mapping(
    *,
    class_indices: Dict[str, int],
    num_model_outputs: int,
    metadata_class_mapping: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate the immutable local production classifier output contract."""
    expected_indices = {name: index for index, name in enumerate(CLASS_NAMES)}
    errors: List[str] = []
    normalized = {str(name): int(index) for name, index in (class_indices or {}).items()}
    if int(num_model_outputs) != len(CLASS_NAMES):
        errors.append(
            f"model output size is {num_model_outputs}; expected {len(CLASS_NAMES)}"
        )
    if normalized != expected_indices:
        errors.append(
            f"class_indices must be exactly {expected_indices}; received {normalized}"
        )
    values = list(normalized.values())
    if len(values) != len(set(values)) or sorted(values) != list(range(len(CLASS_NAMES))):
        errors.append("class_indices must contain unique contiguous indexes 0..3")
    if metadata_class_mapping is not None and list(metadata_class_mapping) != CLASS_NAMES:
        errors.append(
            f"metadata class_mapping must be exactly {CLASS_NAMES}; "
            f"received {list(metadata_class_mapping)}"
        )
    if errors:
        raise ValueError("Invalid production class mapping: " + "; ".join(errors))
    return {
        "valid": True,
        "class_names": list(CLASS_NAMES),
        "class_indices": expected_indices,
        "num_model_outputs": int(num_model_outputs),
    }


def canonical_class_order(config=None) -> Dict[str, int]:
    if config is None:
        config = current_app.config
    return dict(config.get("CANONICAL_CLASS_ORDER") or {
        "naskhi": 0,
        "diwani": 1,
        "diwani_jali": 2,
        "tsuluts": 3,
    })


def load_saved_class_indices(config=None) -> Dict[str, int]:
    if config is None:
        config = current_app.config
    path = config.get("CLASS_INDICES_PATH")
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        indices = data.get("class_indices", data) if isinstance(data, dict) else {}
        return {str(k): int(v) for k, v in indices.items()}
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return {}


def validate_class_mapping(
    config=None,
    saved_indices: Optional[Dict[str, int]] = None,
    loader_class_names: Optional[List[str]] = None,
    num_model_outputs: Optional[int] = None,
    raise_on_mismatch: bool = False,
) -> Dict[str, Any]:
    """Compare config CLASS_LABELS, canonical order, saved indices, and optional loader order."""
    if config is None:
        config = current_app.config

    class_labels = list(config["CLASS_LABELS"])
    canonical = canonical_class_order(config)
    saved = saved_indices if saved_indices is not None else load_saved_class_indices(config)

    mismatches: List[str] = []
    details: Dict[str, Any] = {
        "class_labels": class_labels,
        "canonical_order": canonical,
        "saved_class_indices": saved,
        "loader_class_names": loader_class_names,
        "num_model_outputs": num_model_outputs,
        "valid": True,
        "mismatches": mismatches,
    }

    for label in class_labels:
        if label not in canonical:
            mismatches.append(f"Class '{label}' missing from canonical order.")
        elif canonical[label] != class_labels.index(label):
            mismatches.append(
                f"CLASS_LABELS order for '{label}' ({class_labels.index(label)}) "
                f"differs from canonical ({canonical[label]})."
            )

    use_external = False
    if config is not None:
        use_external = bool(config.get("USE_EXTERNAL_MODEL"))
    else:
        try:
            use_external = bool(current_app.config.get("USE_EXTERNAL_MODEL"))
        except RuntimeError:
            use_external = False

    if saved and not use_external:
        for label in class_labels:
            if label not in saved:
                mismatches.append(f"Saved class_indices missing '{label}'.")
            elif saved[label] != canonical.get(label):
                mismatches.append(
                    f"Saved index for '{label}' is {saved[label]}, expected {canonical.get(label)}."
                )
    elif saved:
        for label in class_labels:
            if label not in saved:
                mismatches.append(f"Saved class_indices missing '{label}'.")
        for label, idx in saved.items():
            if label not in class_labels:
                mismatches.append(f"Saved class_indices contains unknown class '{label}'.")

    if loader_class_names:
        if list(loader_class_names) != class_labels:
            mismatches.append(
                f"Training loader class order {loader_class_names} != config CLASS_LABELS {class_labels}."
            )

    if num_model_outputs is not None and num_model_outputs != len(class_labels):
        mismatches.append(
            f"Model output size ({num_model_outputs}) != number of classes ({len(class_labels)})."
        )

    details["valid"] = len(mismatches) == 0
    if mismatches:
        logger.warning("Class mapping mismatch: %s", "; ".join(mismatches))
        if raise_on_mismatch:
            raise ValueError(
                "Class mapping mismatch detected. Please retrain the model. "
                + "; ".join(mismatches)
            )
    else:
        logger.info(
            "Class mapping OK — labels=%s saved=%s loader=%s outputs=%s",
            class_labels,
            saved or "none",
            loader_class_names or "n/a",
            num_model_outputs,
        )
    return details


def ensure_canonical_class_indices(path: str, class_labels: List[str], config=None) -> Dict[str, int]:
    """Write class_indices.json in canonical order."""
    canonical = canonical_class_order(config)
    ordered = {label: int(canonical.get(label, class_labels.index(label))) for label in class_labels}
    display = (config or current_app.config).get("CLASS_DISPLAY_NAMES", {})
    payload = {
        "class_indices": ordered,
        "class_names": class_labels,
        "display_names": {label: display.get(label, label.replace("_", " ").title()) for label in class_labels},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return ordered


def invert_class_indices(class_indices: Dict[str, int]) -> Dict[int, str]:
    return {int(v): k for k, v in class_indices.items()}


def load_trained_class_metadata(config=None) -> Dict[str, Any]:
    """Load class names and display labels saved with the trained model."""
    if config is None:
        config = current_app.config
    path = config.get("CLASS_INDICES_PATH")
    payload: Dict[str, Any] = {
        "class_indices": load_saved_class_indices(config),
        "class_names": list(config.get("CLASS_LABELS", [])),
        "display_names": dict(config.get("CLASS_DISPLAY_NAMES", {})),
    }
    if not path or not os.path.isfile(path):
        return payload
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            if data.get("class_indices"):
                payload["class_indices"] = {
                    str(k): int(v) for k, v in data["class_indices"].items()
                }
            if data.get("class_names"):
                payload["class_names"] = [str(name) for name in data["class_names"]]
            if data.get("display_names"):
                payload["display_names"] = {
                    str(k): str(v) for k, v in data["display_names"].items()
                }
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        pass
    return payload


def resolve_model_class_indices(config=None) -> Dict[str, int]:
    """Return class index map aligned with the active model output order."""
    if config is None:
        config = current_app.config

    class_labels = list(config.get("CLASS_LABELS") or [])
    if config.get("USE_EXTERNAL_MODEL"):
        labels_path = config.get("EXTERNAL_LABELS_PATH") or ""
        if labels_path and os.path.isfile(labels_path):
            from services.external_assets_service import build_class_indices_from_labels

            external = build_class_indices_from_labels(labels_path, class_labels)
            if external:
                return external

        meta_path = config.get("MODEL_METADATA_PATH")
        if meta_path and os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as handle:
                    meta = json.load(handle)
                indices = meta.get("class_indices") if isinstance(meta, dict) else None
                if indices:
                    return {str(k): int(v) for k, v in indices.items()}
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                pass

    saved = load_saved_class_indices(config)
    if saved:
        return saved
    canonical = canonical_class_order(config)
    return {label: int(canonical.get(label, index)) for index, label in enumerate(class_labels)}


def resolve_trained_class_labels(config=None) -> List[str]:
    """Return class slugs the active model was trained on (with dataset samples when available)."""
    if config is None:
        config = current_app.config

    meta = load_trained_class_metadata(config)
    indices = meta.get("class_indices") or {}
    if indices:
        ordered = [label for label, _ in sorted(indices.items(), key=lambda item: item[1])]
    else:
        ordered = list(meta.get("class_names") or config.get("CLASS_LABELS", []))

    from services.dataset_readiness_service import count_images_per_class

    for root_key in ("TRAIN_DIR", "RAW_DATASET_DIR", "DATASET_DIR"):
        root_dir = config.get(root_key)
        if not root_dir or not os.path.isdir(root_dir):
            continue
        counts = count_images_per_class(root_dir, ordered)
        with_samples = [label for label in ordered if counts.get(label, 0) > 0]
        if with_samples:
            return with_samples
    return ordered


def trained_class_display_names(config=None) -> Dict[str, str]:
    """Human-readable labels for classes in the trained model."""
    if config is None:
        config = current_app.config
    meta = load_trained_class_metadata(config)
    display = dict(meta.get("display_names") or {})
    fallback = dict(config.get("CLASS_DISPLAY_NAMES", {}))
    labels = resolve_trained_class_labels(config)
    return {
        label: display.get(label) or fallback.get(label) or label.replace("_", " ").title()
        for label in labels
    }


def format_trained_class_list(config=None) -> str:
    names = trained_class_display_names(config)
    return ", ".join(names[label] for label in resolve_trained_class_labels(config))
