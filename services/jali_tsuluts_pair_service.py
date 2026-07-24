"""Disambiguate Diwani Jali vs Tsuluts using style characteristics.

Diwani Jali: dense fill, many decorative titik/ornaments, little empty paper.
Tsuluts: monumental long strokes and large curves, decorative but less dense.

Only redistributes probability mass between those two labels when they are the
contested top pair — other classes are left unchanged.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from services.diwani_pair_service import compute_jali_density_features, sanitize_score_dict

PAIR = ("diwani_jali", "tsuluts")

# Prefer Tsuluts when strokes look monumental and ornament density is moderate.
# Midpoint slightly lower than earlier 0.42 to recover Tsuluts→Jali holdout errors.
_MONUMENTAL_MIDPOINT = 0.39
_MONUMENTAL_SCALE = 8.2
_CONTESTED_MARGIN = 0.30
_PAIR_MASS_MIN = 0.48


def _top_two(scores: Dict[str, float]) -> Tuple[Optional[str], Optional[str], float, float]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ordered:
        return None, None, 0.0, 0.0
    top_label, top_score = ordered[0][0], float(ordered[0][1])
    if len(ordered) < 2:
        return top_label, None, top_score, 0.0
    second_label, second_score = ordered[1][0], float(ordered[1][1])
    return top_label, second_label, top_score, second_score


def _pair_is_contested(scores: Dict[str, float]) -> bool:
    top, second, top_score, second_score = _top_two(scores)
    if not top or not second:
        return False
    if {top, second} != set(PAIR):
        return False
    pair_mass = float(scores.get("diwani_jali", 0.0)) + float(scores.get("tsuluts", 0.0))
    if pair_mass < _PAIR_MASS_MIN:
        return False
    margin = top_score - second_score
    # Intervene on close contests; also when both classes are strong runners-up.
    return margin <= _CONTESTED_MARGIN or min(top_score, second_score) >= 0.20


def _monumental_stroke_score(features: Dict[str, float]) -> float:
    """Higher => more Tsuluts-like (open monumental strokes, less dense ornament)."""
    ornament = float(features.get("ornament_index", 0.0))
    ink = float(features.get("ink_ratio", 0.0))
    paper = float(features.get("paper_ratio", 0.0))
    edge = float(features.get("ink_edge_ratio", 0.0))
    dots = float(features.get("dot_norm", 0.0))
    holes = float(features.get("hole_norm", 0.0))
    openness_feat = float(features.get("openness", max(0.0, paper - ink * 0.65)))

    # Tsuluts tends to keep more paper and longer edge structure without dense fill.
    openness = max(0.0, paper - ink * 0.55, openness_feat)
    stroke_presence = min(max(edge * 1.40, 0.0), 1.0)
    low_ornament = max(0.0, 0.58 - ornament)
    low_titik = max(0.0, 0.58 - 0.55 * dots - 0.45 * holes)

    monumental = (
        0.34 * stroke_presence
        + 0.30 * openness
        + 0.22 * low_ornament
        + 0.14 * low_titik
    )
    return float(monumental)


def disambiguate_jali_tsuluts_pair(
    scores: Dict[str, float],
    image_path: Optional[str],
) -> Dict[str, float]:
    """Redistribute only diwani_jali / tsuluts mass using style cues."""
    if not scores or "diwani_jali" not in scores or "tsuluts" not in scores:
        return scores
    if not _pair_is_contested(scores):
        return scores

    features = compute_jali_density_features(image_path or "")
    if not features:
        return scores

    jali = float(scores.get("diwani_jali", 0.0))
    tsuluts = float(scores.get("tsuluts", 0.0))
    pair_mass = jali + tsuluts
    if pair_mass <= 1e-9:
        return scores

    model_jali_share = jali / pair_mass
    monumental = _monumental_stroke_score(features)
    # Positive preference => Tsuluts; negative => Diwani Jali.
    tsuluts_preference = (monumental - _MONUMENTAL_MIDPOINT) * _MONUMENTAL_SCALE
    ornament = float(features.get("ornament_index", 0.0))
    if ornament >= 0.40:
        tsuluts_preference -= min((ornament - 0.40) * 11.0, 2.0)
    elif ornament <= 0.32:
        # Recover Tsuluts often mislabeled as dense Jali when ornament is moderate/low.
        tsuluts_preference += min((0.32 - ornament) * 9.5, 1.8)

    visual_tsuluts_share = 1.0 / (1.0 + math.exp(-1.35 * tsuluts_preference))
    margin = abs(jali - tsuluts)
    visual_weight = 0.62 * (1.0 - min(margin / _CONTESTED_MARGIN, 1.0))
    if abs(tsuluts_preference) >= 1.10:
        visual_weight = max(visual_weight, 0.44)
    if abs(tsuluts_preference) >= 1.70:
        visual_weight = max(visual_weight, 0.55)
    if abs(tsuluts_preference) < 0.20:
        visual_weight *= 0.35

    if visual_weight <= 0.02:
        return scores

    new_tsuluts_share = (1.0 - visual_weight) * (1.0 - model_jali_share) + visual_weight * visual_tsuluts_share
    new_tsuluts_share = float(min(max(new_tsuluts_share, 0.02), 0.98))

    updated = {key: float(value) for key, value in scores.items()}
    updated["tsuluts"] = pair_mass * new_tsuluts_share
    updated["diwani_jali"] = pair_mass * (1.0 - new_tsuluts_share)
    total = sum(max(value, 0.0) for value in updated.values())
    if total > 0:
        updated = {key: value / total for key, value in updated.items()}
    return updated


def adjust_probability_row_for_jali_tsuluts_pair(
    row: np.ndarray,
    class_order: List[str],
    image_path: str,
) -> np.ndarray:
    """Evaluation helper: adjust a softmax row in class_order layout."""
    if "diwani_jali" not in class_order or "tsuluts" not in class_order:
        return row
    scores = {label: float(row[index]) for index, label in enumerate(class_order)}
    adjusted = sanitize_score_dict(disambiguate_jali_tsuluts_pair(scores, image_path), class_order)
    return np.array([adjusted.get(label, 0.0) for label in class_order], dtype=np.float64)
