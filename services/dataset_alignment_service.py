"""Align CNN predictions with trained dataset visual similarity and style traits."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.dataset_similarity_service import compute_per_class_similarity_scores


def _normalize_scores(scores: Dict[str, float], labels: List[str]) -> Dict[str, float]:
    total = sum(float(scores.get(label, 0.0)) for label in labels)
    if total <= 0:
        uniform = 1.0 / len(labels) if labels else 0.0
        return {label: uniform for label in labels}
    return {label: float(scores.get(label, 0.0)) / total for label in labels}


def _prediction_strength(scores: Dict[str, float]) -> tuple:
    if not scores:
        return None, 0.0, 0.0
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    pred = ordered[0][0]
    top = float(ordered[0][1])
    second = float(ordered[1][1]) if len(ordered) > 1 else 0.0
    return pred, top, top - second


def apply_characteristic_shape_prior(
    scores: Dict[str, float],
    image_path: Optional[str],
    class_labels: List[str],
    *,
    max_boost: float = 0.12,
) -> Dict[str, float]:
    """Gently nudge class scores using dataset style shape cues.

    Naskhi: neat, open, low ornament.
    Diwani: flowing, moderate openness, moderate ornament.
    Diwani Jali: dense ornament / titik.
    Tsuluts: monumental strokes, less dense than Jali.

    Does not touch non-khat rejection gates. Clear CNN winners (high conf+margin)
    are left unchanged.
    """
    if not scores or not image_path:
        return scores
    labels = [label for label in class_labels if label in scores] or list(scores.keys())
    if not labels:
        return scores

    pred, conf, margin = _prediction_strength(scores)
    # Preserve strong CNN decisions — prior only helps ambiguous cases.
    if conf >= 0.62 and margin >= 0.18:
        return scores

    try:
        from services.diwani_pair_service import compute_jali_density_features
        from services.jali_tsuluts_pair_service import _monumental_stroke_score
    except Exception:
        return scores

    features = compute_jali_density_features(image_path)
    if not features:
        return scores

    ornament = float(features.get("ornament_index") or 0.0)
    openness = float(features.get("openness") or 0.0)
    monumental = float(_monumental_stroke_score(features))

    # Soft targets from documented / dataset-calibrated traits.
    prior = {label: 0.0 for label in labels}
    if "naskhi" in prior:
        # Low ornament + readable open layout → Naskhi textbook pages.
        prior["naskhi"] = max(0.0, 0.34 - ornament) * 1.6 + max(0.0, openness - 0.16) * 0.8
    if "diwani" in prior:
        prior["diwani"] = max(0.0, 0.12 - abs(ornament - 0.30)) * 2.0 + max(0.0, openness - 0.18) * 0.5
    if "diwani_jali" in prior:
        prior["diwani_jali"] = max(0.0, ornament - 0.34) * 2.2 + max(0.0, 0.18 - openness) * 0.7
    if "tsuluts" in prior:
        prior["tsuluts"] = max(0.0, monumental - 0.36) * 2.0 + max(0.0, 0.42 - ornament) * 0.6

    prior_total = sum(prior.values())
    if prior_total <= 1e-6:
        return scores

    prior = {label: value / prior_total for label, value in prior.items()}
    # Scale boost by ambiguity: closer races get more help from shape cues.
    ambiguity = max(0.0, 1.0 - conf) * max(0.0, 1.0 - min(margin / 0.20, 1.0))
    blend = min(max_boost, 0.05 + 0.20 * ambiguity)

    base = _normalize_scores(scores, labels)
    fused = {
        label: (1.0 - blend) * base.get(label, 0.0) + blend * prior.get(label, 0.0)
        for label in labels
    }
    return _normalize_scores(fused, labels)


def fuse_model_and_dataset_scores(
    model_scores: Dict[str, float],
    class_labels: List[str],
    image_path: Optional[str],
    config=None,
) -> Dict[str, Any]:
    """Blend model softmax with dataset similarity; only override when model is uncertain."""
    if config is None:
        from flask import current_app

        config = current_app.config

    labels = [label for label in class_labels if label in model_scores] or list(class_labels)
    model_norm = _normalize_scores(model_scores, labels)
    model_pred, model_conf, model_margin = _prediction_strength(model_norm)

    if not image_path or not labels:
        return {
            "predicted_class": model_pred,
            "fused_scores": model_norm,
            "dataset_aligned": False,
            "similarity_scores": {},
            "model_predicted_class": model_pred,
            "model_confidence": model_conf,
            "kept_model_decision": False,
        }

    # Keep clear CNN decisions before pHash — ornate borders confuse similarity.
    clear_model = bool(
        model_pred
        and model_conf >= float(config.get("FUSION_KEEP_MODEL_MIN_CONFIDENCE", 0.38))
        and model_margin >= float(config.get("FUSION_KEEP_MODEL_MIN_MARGIN", 0.08))
    )
    if clear_model:
        return {
            "predicted_class": model_pred,
            "fused_scores": model_norm,
            "dataset_aligned": False,
            "similarity_scores": {},
            "model_predicted_class": model_pred,
            "model_confidence": model_conf,
            "fused_confidence": model_conf,
            "fused_margin": model_margin,
            "kept_model_decision": True,
        }

    stage2_min_conf = float(config.get("STAGE2_MIN_CONFIDENCE", 0.70))
    stage2_min_margin = float(config.get("STAGE2_MIN_MARGIN", 0.15))
    if model_pred and model_conf >= stage2_min_conf and model_margin >= stage2_min_margin:
        return {
            "predicted_class": model_pred,
            "fused_scores": model_norm,
            "dataset_aligned": False,
            "similarity_scores": {},
            "model_predicted_class": model_pred,
            "model_confidence": model_conf,
            "fused_confidence": model_conf,
            "fused_margin": model_margin,
            "kept_model_decision": True,
        }

    sim_pct = compute_per_class_similarity_scores(image_path, labels, config)
    if not sim_pct or max(sim_pct.values()) <= 0:
        return {
            "predicted_class": model_pred,
            "fused_scores": model_norm,
            "dataset_aligned": False,
            "similarity_scores": sim_pct or {},
            "model_predicted_class": model_pred,
            "model_confidence": model_conf,
            "kept_model_decision": False,
        }

    sim_norm = _normalize_scores({k: v / 100.0 for k, v in sim_pct.items()}, labels)
    sim_pred, sim_conf, sim_margin = _prediction_strength(sim_norm)

    model_weight = float(config.get("FUSION_MODEL_WEIGHT", 0.72))
    uncertain_conf = float(config.get("FUSION_UNCERTAIN_CONFIDENCE", 0.58))
    uncertain_margin = float(config.get("FUSION_UNCERTAIN_MARGIN", 0.12))

    if model_conf < uncertain_conf or model_margin < uncertain_margin:
        model_weight = float(config.get("FUSION_UNCERTAIN_MODEL_WEIGHT", 0.45))
    elif model_conf >= 0.80 and model_margin >= 0.20:
        model_weight = float(config.get("FUSION_STRONG_MODEL_WEIGHT", 0.88))

    sim_weight = 1.0 - model_weight
    fused = {
        label: model_weight * model_norm.get(label, 0.0) + sim_weight * sim_norm.get(label, 0.0)
        for label in labels
    }
    fused = _normalize_scores(fused, labels)
    fused_pred, fused_conf, fused_margin = _prediction_strength(fused)

    override_gap = float(config.get("DATASET_ALIGN_OVERRIDE_GAP", 12)) / 100.0
    strong_sim = float(config.get("DATASET_ALIGN_STRONG_SIMILARITY", 75))
    sim_beats_model = bool(
        sim_pred
        and model_pred
        and (sim_norm.get(sim_pred, 0.0) - model_norm.get(model_pred, 0.0)) >= override_gap
    )
    strong_sim_override = bool(
        sim_conf >= (strong_sim / 100.0) and sim_margin >= 0.08 and sim_beats_model
    )

    dataset_aligned = fused_pred != model_pred and (
        (model_conf < uncertain_conf and sim_beats_model) or strong_sim_override
    )

    # If fused label flipped but similarity did not clearly beat the CNN, keep model.
    if fused_pred != model_pred and not dataset_aligned:
        fused = model_norm
        fused_pred, fused_conf, fused_margin = model_pred, model_conf, model_margin

    # Style-characteristic veto: pHash must not force ornate-looking false labels
    # against low-ornament Naskhi or monumental Tsuluts pages.
    if fused_pred != model_pred and model_pred in {"naskhi", "tsuluts"}:
        try:
            from services.diwani_pair_service import compute_jali_density_features
            from services.jali_tsuluts_pair_service import _monumental_stroke_score

            features = compute_jali_density_features(image_path)
        except Exception:
            features = None
        if features:
            ornament = float(features.get("ornament_index") or 0.0)
            if model_pred == "naskhi" and fused_pred == "diwani_jali" and ornament < 0.34:
                fused = model_norm
                fused_pred, fused_conf, fused_margin = model_pred, model_conf, model_margin
                dataset_aligned = False
            elif model_pred == "tsuluts" and fused_pred == "diwani_jali":
                monumental = float(_monumental_stroke_score(features))
                if monumental >= 0.40 and ornament < 0.42:
                    fused = model_norm
                    fused_pred, fused_conf, fused_margin = model_pred, model_conf, model_margin
                    dataset_aligned = False

    return {
        "predicted_class": fused_pred,
        "fused_scores": fused,
        "dataset_aligned": dataset_aligned,
        "similarity_scores": sim_pct,
        "model_predicted_class": model_pred,
        "model_confidence": model_conf,
        "similarity_predicted_class": sim_pred,
        "max_similarity": max(sim_pct.values()),
        "fused_confidence": fused_conf,
        "fused_margin": fused_margin,
        "kept_model_decision": False,
    }


def align_prediction_with_dataset(
    raw_scores: Dict[str, float],
    class_labels: List[str],
    image_path: Optional[str],
    config=None,
) -> Dict[str, Any]:
    """Backward-compatible wrapper around score fusion."""
    result = fuse_model_and_dataset_scores(raw_scores, class_labels, image_path, config)
    return {
        "predicted_class": result.get("predicted_class"),
        "dataset_aligned": result.get("dataset_aligned", False),
        "similarity_scores": result.get("similarity_scores") or {},
        "model_predicted_class": result.get("model_predicted_class"),
        "model_confidence": result.get("model_confidence"),
        "similarity_predicted_class": result.get("similarity_predicted_class"),
        "max_similarity": result.get("max_similarity"),
        "fused_scores": result.get("fused_scores"),
        "fused_confidence": result.get("fused_confidence"),
        "fused_margin": result.get("fused_margin"),
        "kept_model_decision": bool(result.get("kept_model_decision")),
    }
