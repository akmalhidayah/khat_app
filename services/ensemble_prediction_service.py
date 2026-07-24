"""Fuse predictions from multiple trained classifiers for higher accuracy."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from services.class_mapping_service import (
    canonical_class_order,
    resolve_model_class_indices,
    resolve_trained_class_labels,
)
from services.dataset_readiness_service import is_valid_trained_model
from services.model_cache_service import get_cached_classifier
from services.preprocessing_service import preprocess_for_model


def _invert_class_indices(class_indices: Dict[str, int]) -> Dict[int, str]:
    return {int(index): label for label, index in class_indices.items()}


def _scores_from_probs(probs: np.ndarray, inv_map: Dict[int, str], class_labels: List[str]) -> Dict[str, float]:
    scores = {
        inv_map[index]: float(probs[index])
        for index in range(len(probs))
        if inv_map.get(index) in class_labels
    }
    for label in class_labels:
        scores.setdefault(label, 0.0)
    return scores


def _prediction_strength(scores: Dict[str, float]) -> Tuple[str, float, float]:
    if not scores:
        return "", 0.0, 0.0
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    pred = ordered[0][0]
    top = float(ordered[0][1])
    second = float(ordered[1][1]) if len(ordered) > 1 else 0.0
    return pred, top, top - second


def _local_model_paths(config) -> List[str]:
    base = config.get("BASE_DIR", "")
    paths = [
        os.path.join(base, "model", "khat_best.keras"),
        os.path.join(base, "model", "khat_latest.keras"),
        config.get("BEST_MODEL_PATH"),
        config.get("LATEST_MODEL_PATH"),
    ]
    unique: List[str] = []
    for path in paths:
        norm = os.path.normcase(os.path.normpath(path or ""))
        if path and norm not in {os.path.normcase(os.path.normpath(p)) for p in unique}:
            unique.append(path)
    return unique


def _predict_with_model(
    image_path: str,
    model_path: str,
    architecture: str,
    class_indices: Dict[str, int],
    class_labels: List[str],
    *,
    use_tta: bool = False,
    preprocessing_mode: str = "standard",
) -> Dict[str, float]:
    from services.prediction_service import _tta_variants

    model = get_cached_classifier(model_path, architecture, len(class_labels))
    inv_map = _invert_class_indices(class_indices)

    if use_tta:
        batches = _tta_variants(image_path, architecture, preprocessing_mode=preprocessing_mode)
        probs_list = [model.predict(batch, verbose=0)[0] for batch in batches]
        probs = np.mean(probs_list, axis=0)
    else:
        batch = preprocess_for_model(
            image_path,
            architecture=architecture,
            preprocessing_mode=preprocessing_mode,
        )
        probs = model.predict(batch, verbose=0)[0]

    return _scores_from_probs(probs, inv_map, class_labels)


def ensemble_enabled(config) -> bool:
    if not config.get("USE_MODEL_ENSEMBLE", True):
        return False
    local_ok = any(is_valid_trained_model(path) for path in _local_model_paths(config))
    external_ok = bool(
        config.get("USE_EXTERNAL_MODEL")
        and is_valid_trained_model(config.get("EXTERNAL_MODEL_PATH") or "")
    )
    return local_ok and external_ok


def classify_with_ensemble(
    image_path: str,
    config=None,
    *,
    use_tta: bool = False,
    preprocessing_mode: str = "standard",
) -> Dict:
    if config is None:
        from flask import current_app

        config = current_app.config

    class_labels = resolve_trained_class_labels(config)
    candidates: List[Tuple[str, str, Dict[str, int], Dict[str, float], float]] = []

    external_path = config.get("EXTERNAL_MODEL_PATH") or ""
    if config.get("USE_EXTERNAL_MODEL") and is_valid_trained_model(external_path):
        external_indices = resolve_model_class_indices(config)
        scores = _predict_with_model(
            image_path,
            external_path,
            "keras_h5",
            external_indices,
            class_labels,
            use_tta=use_tta,
            preprocessing_mode=preprocessing_mode,
        )
        _, conf, margin = _prediction_strength(scores)
        strength = conf * max(margin, 0.05)
        candidates.append(("external", external_path, external_indices, scores, strength))

    local_weight = float(config.get("ENSEMBLE_LOCAL_WEIGHT", 0.65))
    external_weight = float(config.get("ENSEMBLE_EXTERNAL_WEIGHT", 0.35))
    canonical = canonical_class_order(config)

    for local_path in _local_model_paths(config):
        if not is_valid_trained_model(local_path):
            continue
        if os.path.normcase(local_path) == os.path.normcase(external_path or ""):
            continue
        scores = _predict_with_model(
            image_path,
            local_path,
            "efficientnetb0",
            canonical,
            class_labels,
            use_tta=use_tta,
            preprocessing_mode=preprocessing_mode,
        )
        _, conf, margin = _prediction_strength(scores)
        strength = conf * max(margin, 0.05)
        candidates.append(("local", local_path, canonical, scores, strength))
        break

    if not candidates:
        raise FileNotFoundError("No trained model available for ensemble classification.")

    if len(candidates) == 1:
        source, path, indices, scores, strength = candidates[0]
        pred, conf, margin = _prediction_strength(scores)
        return {
            "scores": scores,
            "predicted_class": pred,
            "confidence": conf,
            "margin": margin,
            "ensemble_used": False,
            "model_source": source,
            "model_path": path,
            "class_indices": indices,
            "component_scores": {source: scores},
        }

    fused = {label: 0.0 for label in class_labels}
    total_weight = 0.0
    component_scores: Dict[str, Dict[str, float]] = {}
    for source, path, indices, scores, strength in candidates:
        weight = external_weight if source == "external" else local_weight
        if strength < 0.08:
            weight *= 0.5
        for label in class_labels:
            fused[label] += weight * float(scores.get(label, 0.0))
        total_weight += weight
        component_scores[source] = scores

    if total_weight > 0:
        fused = {label: value / total_weight for label, value in fused.items()}

    best_single = max(candidates, key=lambda item: item[4])
    single_pred, single_conf, single_margin = _prediction_strength(best_single[3])
    fused_pred, fused_conf, fused_margin = _prediction_strength(fused)

    use_single = (
        single_conf >= float(config.get("ENSEMBLE_SINGLE_MIN_CONFIDENCE", 0.72))
        and single_margin >= float(config.get("ENSEMBLE_SINGLE_MIN_MARGIN", 0.18))
        and single_conf >= fused_conf + 0.08
    )
    if use_single:
        pred, conf, margin = single_pred, single_conf, single_margin
        final_scores = dict(best_single[3])
        model_source = best_single[0]
        model_path = best_single[1]
        class_indices = best_single[2]
    else:
        pred, conf, margin = fused_pred, fused_conf, fused_margin
        final_scores = fused
        model_source = "ensemble"
        model_path = best_single[1]
        class_indices = resolve_model_class_indices(config)

    return {
        "scores": final_scores,
        "predicted_class": pred,
        "confidence": conf,
        "margin": margin,
        "ensemble_used": True,
        "model_source": model_source,
        "model_path": model_path,
        "class_indices": class_indices,
        "component_scores": component_scores,
        "best_single_source": best_single[0],
        "best_single_confidence": single_conf,
    }
