"""Disambiguate Diwani vs Diwani Jali using ornament / titik visual cues.

Diwani Jali typically has denser ink, more decorative dots (titik hias), and
more ornamental fill (enclosed gaps) with less empty paper. Diwani is more
open and flowing with fewer ornaments.

This module only redistributes probability mass between those two labels when
they are the contested top pair — other classes are left unchanged.

Feature midpoints were calibrated on D:\\dataset_kaligrafi train/validation samples.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

PAIR = ("diwani", "diwani_jali")

# Midpoint of ornament_index between Diwani / Diwani Jali validation means
# (~0.28 / ~0.38). Tuned for balanced separation on local dataset.
_ORNAMENT_MIDPOINT = 0.33
_ORNAMENT_SCALE = 9.5
_CONTESTED_MARGIN = 0.26
_PAIR_MASS_MIN = 0.50


def _ornament_dot_and_hole_scores(ink_u8: np.ndarray) -> Tuple[float, float, int, int]:
    """Score decorative titik and enclosed ornamental pockets.

    Returns (dot_norm, hole_norm, n_dots, n_holes) in roughly [0, 1] for norms.
    """
    try:
        import cv2

        ink = (ink_u8 > 0).astype(np.uint8)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Titik hias: small dark structures removed by morphological opening.
        opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k5)
        residual = cv2.subtract(ink, opened)
        residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, k3)
        n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            residual, connectivity=8
        )
        n_dots = 0
        for index in range(1, n_labels):
            area = int(stats[index, cv2.CC_STAT_AREA])
            width = int(stats[index, cv2.CC_STAT_WIDTH])
            height = int(stats[index, cv2.CC_STAT_HEIGHT])
            if 2 <= area <= 80 and width <= 16 and height <= 16:
                n_dots += 1

        # Ornamen fill: enclosed white pockets inside ink (not border paper).
        inv = (1 - ink).astype(np.uint8)
        n_bg, _lb, bg_stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        height_img, width_img = ink.shape
        n_holes = 0
        for index in range(1, n_bg):
            x = int(bg_stats[index, cv2.CC_STAT_LEFT])
            y = int(bg_stats[index, cv2.CC_STAT_TOP])
            w = int(bg_stats[index, cv2.CC_STAT_WIDTH])
            h = int(bg_stats[index, cv2.CC_STAT_HEIGHT])
            area = int(bg_stats[index, cv2.CC_STAT_AREA])
            touches_border = x == 0 or y == 0 or (x + w) >= width_img or (y + h) >= height_img
            if touches_border:
                continue
            if 3 <= area <= 400:
                n_holes += 1

        # Dataset medians ~46 dots (Diwani) / ~78 (Jali); holes ~7 / ~50.
        dot_norm = min(n_dots / 80.0, 1.0)
        hole_norm = min(n_holes / 40.0, 1.0)
        return dot_norm, hole_norm, n_dots, n_holes
    except Exception:
        from PIL import Image, ImageFilter

        speck = np.asarray(
            Image.fromarray((ink_u8 * 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES),
            dtype=np.float32,
        )
        speck_ratio = float((speck > 40).mean())
        approx_dots = int(round(speck_ratio * 400.0))
        return min(approx_dots / 80.0, 1.0), 0.0, approx_dots, 0


def _adaptive_ink_mask(gray: np.ndarray) -> np.ndarray:
    """Ink mask with adaptive threshold so faded / high-contrast scans stay comparable."""
    # Paper-aware threshold: keep between classic 90–160 band used in prior tuning.
    thresh = float(np.percentile(gray, 42))
    thresh = min(max(thresh, 90.0), 160.0)
    # Soft blend toward fixed 130 so prior validation midpoints stay meaningful.
    thresh = 0.55 * thresh + 0.45 * 130.0
    return (gray < thresh).astype(np.uint8)


def compute_jali_density_features(image_path: str) -> Optional[Dict[str, float]]:
    """Extract ornament/titik features that separate Diwani from Diwani Jali."""
    if not image_path or not os.path.isfile(image_path):
        return None
    try:
        from PIL import Image, ImageFilter, ImageOps

        with Image.open(image_path) as img:
            # Slightly higher working size improves titik/hole morphology without
            # changing the public feature scale (still normalized ratios).
            small = ImageOps.exif_transpose(img).convert("RGB").resize((256, 256), Image.LANCZOS)
        gray = np.asarray(ImageOps.grayscale(small), dtype=np.float32)
        ink = _adaptive_ink_mask(gray)
        ink_ratio = float(ink.mean())
        paper_ratio = float((gray > 200).mean())
        edges = np.asarray(ImageOps.grayscale(small).filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        edge_ratio = float((edges > 40).mean())
        ink_edges = np.asarray(Image.fromarray(ink * 255).filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        ink_edge_ratio = float((ink_edges > 40).mean())
        midtone_ratio = float(((gray >= 80) & (gray <= 180)).mean())

        # Open flowing Diwani keeps more contiguous paper corridors than dense Jali.
        openness = max(0.0, paper_ratio - ink_ratio * 0.65)
        openness = min(openness, 1.0)

        dot_norm, hole_norm, n_dots, n_holes = _ornament_dot_and_hole_scores(ink)

        # Higher => more Diwani Jali-like (titik + ornamen + denser fill).
        ornament_index = (
            0.26 * ink_ratio
            + 0.16 * (1.0 - paper_ratio)
            + 0.10 * ink_edge_ratio
            + 0.05 * edge_ratio
            + 0.06 * midtone_ratio
            + 0.20 * dot_norm
            + 0.15 * hole_norm
            + 0.02 * (1.0 - openness)
        )
        # Keep density_index as alias for callers/tests expecting the old key.
        density_index = ornament_index
        jali_preference = (ornament_index - _ORNAMENT_MIDPOINT) * _ORNAMENT_SCALE
        # Soft openness prior: open/flowing pages lean Diwani.
        if openness >= 0.22 and ornament_index < 0.36:
            jali_preference -= min((openness - 0.18) * 4.5, 1.1)
        return {
            "ink_ratio": round(ink_ratio, 4),
            "paper_ratio": round(paper_ratio, 4),
            "edge_ratio": round(edge_ratio, 4),
            "ink_edge_ratio": round(ink_edge_ratio, 4),
            "midtone_ratio": round(midtone_ratio, 4),
            "openness": round(openness, 4),
            "dot_count": float(n_dots),
            "hole_count": float(n_holes),
            "dot_norm": round(dot_norm, 4),
            "hole_norm": round(hole_norm, 4),
            "ornament_index": round(ornament_index, 4),
            "density_index": round(density_index, 4),
            "jali_preference": round(float(jali_preference), 4),
        }
    except OSError:
        return None


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
    contested = {top, second} == set(PAIR)
    if not contested:
        return False
    pair_mass = float(scores.get("diwani", 0.0)) + float(scores.get("diwani_jali", 0.0))
    if pair_mass < _PAIR_MASS_MIN:
        return False
    # Intervene when the pair is close, or either class is clearly leading the pair.
    margin = top_score - second_score
    return margin <= _CONTESTED_MARGIN or min(top_score, second_score) >= 0.18


def disambiguate_diwani_pair(
    scores: Dict[str, float],
    image_path: Optional[str],
) -> Dict[str, float]:
    """Redistribute only diwani / diwani_jali mass using ornament / titik cues."""
    if not scores or "diwani" not in scores or "diwani_jali" not in scores:
        return scores
    if not _pair_is_contested(scores):
        return scores

    features = compute_jali_density_features(image_path or "")
    if not features:
        return scores

    diwani = float(scores.get("diwani", 0.0))
    jali = float(scores.get("diwani_jali", 0.0))
    pair_mass = diwani + jali
    if pair_mass <= 1e-9:
        return scores

    model_jali_share = jali / pair_mass
    visual = float(features["jali_preference"])
    visual_jali_share = 1.0 / (1.0 + math.exp(-1.45 * visual))

    margin = abs(jali - diwani)
    visual_weight = 0.66 * (1.0 - min(margin / _CONTESTED_MARGIN, 1.0))
    if abs(visual) >= 1.10:
        # Strong characteristic cue: trust style features more.
        visual_weight = max(visual_weight, 0.48)
    if abs(visual) >= 1.80:
        visual_weight = max(visual_weight, 0.58)
    if abs(visual) < 0.18:
        visual_weight *= 0.40

    if visual_weight <= 0.02:
        return scores

    new_jali_share = (1.0 - visual_weight) * model_jali_share + visual_weight * visual_jali_share
    new_jali_share = float(min(max(new_jali_share, 0.02), 0.98))

    updated = {key: float(value) for key, value in scores.items()}
    updated["diwani_jali"] = pair_mass * new_jali_share
    updated["diwani"] = pair_mass * (1.0 - new_jali_share)
    total = sum(max(value, 0.0) for value in updated.values())
    if total > 0:
        updated = {key: value / total for key, value in updated.items()}
    return updated


def sanitize_score_dict(scores: Dict[str, float], class_labels: Optional[List[str]] = None) -> Dict[str, float]:
    """Drop internal debug keys and keep only class probabilities."""
    cleaned = {
        key: float(value)
        for key, value in scores.items()
        if not str(key).startswith("_") and isinstance(value, (int, float))
    }
    if class_labels:
        cleaned = {label: float(cleaned.get(label, 0.0)) for label in class_labels}
        total = sum(cleaned.values())
        if total > 0:
            cleaned = {label: value / total for label, value in cleaned.items()}
    return cleaned


def apply_contested_style_pairs(
    scores: Dict[str, float],
    image_path: Optional[str],
    class_labels: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Apply Diwani↔Jali then Jali↔Tsuluts contested-pair corrections."""
    from services.jali_tsuluts_pair_service import disambiguate_jali_tsuluts_pair

    updated = sanitize_score_dict(disambiguate_diwani_pair(scores, image_path), class_labels)
    updated = sanitize_score_dict(disambiguate_jali_tsuluts_pair(updated, image_path), class_labels)
    return updated


def adjust_probability_row_for_diwani_pair(
    row: np.ndarray,
    class_order: List[str],
    image_path: str,
) -> np.ndarray:
    """Evaluation helper: adjust a softmax row in class_order layout."""
    if "diwani" not in class_order or "diwani_jali" not in class_order:
        return row
    scores = {label: float(row[index]) for index, label in enumerate(class_order)}
    adjusted = sanitize_score_dict(disambiguate_diwani_pair(scores, image_path), class_order)
    return np.array([adjusted.get(label, 0.0) for label in class_order], dtype=np.float64)


def adjust_probability_row_for_style_pairs(
    row: np.ndarray,
    class_order: List[str],
    image_path: str,
) -> np.ndarray:
    """Evaluation helper: apply both contested style-pair corrections."""
    scores = {label: float(row[index]) for index, label in enumerate(class_order)}
    adjusted = apply_contested_style_pairs(scores, image_path, class_order)
    return np.array([adjusted.get(label, 0.0) for label in class_order], dtype=np.float64)
