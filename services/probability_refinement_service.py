"""Refine raw model softmax into similarity-aware class probabilities for display and gating."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def temperature_softmax(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Soften over-confident softmax outputs (e.g. 100/0/0/0) using temperature scaling."""
    temperature = max(float(temperature), 0.05)
    clipped = np.clip(probs.astype(np.float64), 1e-8, 1.0)
    logits = np.log(clipped) / temperature
    logits -= np.max(logits)
    exp = np.exp(logits)
    total = float(exp.sum())
    if total <= 0:
        return np.ones_like(exp) / len(exp)
    return exp / total


def refine_class_probabilities(
    model_scores: Dict[str, float],
    class_labels: List[str],
    *,
    image_path: Optional[str] = None,
    config=None,
) -> Dict[str, float]:
    """
    Combine softened model probabilities with per-class visual similarity to dataset training.
    Produces a smoother distribution that reflects kemiripan antar kelas khat.
    """
    if config is None:
        from flask import current_app

        config = current_app.config

    temperature = float(config.get("PROB_SOFTMAX_TEMPERATURE", 2.5))
    blend_sim = float(config.get("PROB_SIMILARITY_BLEND", 0.35))
    min_floor = float(config.get("PROB_MIN_FLOOR", 0.03))

    labels = [label for label in class_labels if label in model_scores] or list(class_labels)
    model_arr = np.array([float(model_scores.get(label, 0.0)) for label in labels], dtype=np.float64)
    if model_arr.sum() <= 0:
        model_arr = np.ones(len(labels), dtype=np.float64) / len(labels)
    else:
        model_arr = model_arr / model_arr.sum()

    softened = temperature_softmax(model_arr, temperature)

    sim_arr = None
    if image_path and blend_sim > 0:
        from services.dataset_similarity_service import compute_per_class_similarity_scores

        sim_pct = compute_per_class_similarity_scores(image_path, labels, config)
        if sim_pct:
            sim_arr = np.array(
                [float(sim_pct.get(label, 0.0)) / 100.0 for label in labels],
                dtype=np.float64,
            )
            if sim_arr.sum() > 0:
                sim_arr = sim_arr / sim_arr.sum()

    if sim_arr is not None:
        blend = (1.0 - blend_sim) * softened + blend_sim * sim_arr
    else:
        blend = softened

    blend = np.maximum(blend, min_floor)
    blend = blend / blend.sum()

    return {labels[index]: float(blend[index]) for index in range(len(labels))}


def build_sorted_probability_list(scores: Dict[str, float]) -> List[Dict]:
    return [
        {
            "class": label,
            "probability": value,
            "percent": round(value * 100, 2),
        }
        for label, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]
