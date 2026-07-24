"""Training dataset similarity check for classification results."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from services.dataset_path_service import safe_relpath
from services.khat_recognition_messages import should_treat_as_unrecognized
from services.preprocessing_service import image_hash, perceptual_hash

CLASS_LABELS = {
    "naskhi": "Naskhi",
    "diwani": "Diwani",
    "diwani_jali": "Diwani Jali",
    "tsuluts": "Tsuluts",
}

ACADEMIC_EVAL_NOTE = (
    "Untuk evaluasi akademik, sebaiknya gunakan gambar uji yang tidak pernah masuk dataset training. "
    "Jika gambar terlalu mirip dengan data training, hasil akurasi dapat terlihat tinggi tetapi "
    "tidak mencerminkan kemampuan generalisasi model."
)

PHASH_BITS = 64


def hamming_distance(hash_a: str, hash_b: str) -> int:
    try:
        ia, ib = int(hash_a, 16), int(hash_b, 16)
    except (TypeError, ValueError):
        return PHASH_BITS
    return bin(ia ^ ib).count("1")


def phash_similarity_percent(hash_a: str, hash_b: str) -> float:
    dist = hamming_distance(hash_a, hash_b)
    return round((1 - dist / PHASH_BITS) * 100, 2)


def _status_from_score(score: float) -> Dict[str, str]:
    if score >= 95:
        return {
            "similarity_status": "Very Similar to Training Data",
            "risk_level": "Possible Duplicate / Data Leakage",
            "message": (
                "Gambar uji sangat mirip dengan gambar pada dataset training. "
                "Hasil prediksi mungkin dipengaruhi oleh kemiripan atau duplikasi data."
            ),
            "recommendation": (
                "Gunakan gambar uji yang benar-benar baru. Pertimbangkan untuk mengecualikan "
                "hasil ini dari evaluasi generalisasi model."
            ),
            "tone": "danger",
        }
    if score >= 85:
        return {
            "similarity_status": "Similar to Training Data",
            "risk_level": "Review Recommended",
            "message": (
                "Gambar uji memiliki kemiripan visual tinggi dengan dataset. "
                "Pastikan gambar ini bukan bagian dari data training."
            ),
            "recommendation": "Verifikasi sumber gambar dan pertimbangkan tinjauan manual sebelum evaluasi.",
            "tone": "warn",
        }
    if score >= 70:
        return {
            "similarity_status": "Moderately Similar",
            "risk_level": "Acceptable with Caution",
            "message": "Gambar uji memiliki kemiripan moderat dengan beberapa sampel training.",
            "recommendation": "Hasil dapat diterima, namun tetap waspada terhadap kemungkinan overlap data.",
            "tone": "info",
        }
    return {
        "similarity_status": "New / Low Similarity",
        "risk_level": "No obvious training duplicate detected",
        "message": "Tidak terdeteksi duplikasi atau kemiripan kuat dengan data training.",
        "recommendation": "Gambar uji tampak cukup baru untuk evaluasi generalisasi.",
        "tone": "ok",
    }


def save_hash_index(entries: List[dict], index_path: str) -> None:
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    payload = {
        "version": 1,
        "count": len(entries),
        "entries": entries,
    }
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_hash_index(index_path: str) -> List[dict]:
    if not index_path or not os.path.isfile(index_path):
        return []
    try:
        with open(index_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data.get("entries") or []
    except (json.JSONDecodeError, OSError):
        return []


def _resolve_dataset_scan_roots(config) -> List[str]:
    """Return dataset folders to scan for perceptual-hash index (train split preferred)."""
    roots: List[str] = []
    for key in ("TRAIN_DIR", "RAW_DATASET_DIR", "DATASET_DIR"):
        root = config.get(key)
        if root and os.path.isdir(root) and root not in roots:
            roots.append(root)
    return roots


def build_hash_index_from_train_dir(config) -> List[dict]:
    """Scan training folders and build perceptual hash index."""
    base_dir = config["BASE_DIR"]
    from services.class_mapping_service import resolve_trained_class_labels

    class_labels = resolve_trained_class_labels(config) or list(config.get("CLASS_LABELS") or [])
    scan_roots = _resolve_dataset_scan_roots(config)
    if not scan_roots or not class_labels:
        return []

    entries: List[dict] = []
    seen_paths = set()
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for root_dir in scan_roots:
        for class_name in class_labels:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for name in os.listdir(class_dir):
                if os.path.splitext(name)[1].lower() not in exts:
                    continue
                path = os.path.join(class_dir, name)
                norm_path = os.path.normcase(os.path.normpath(path))
                if not os.path.isfile(path) or norm_path in seen_paths:
                    continue
                seen_paths.add(norm_path)
                try:
                    entries.append(
                        {
                            "image_id": None,
                            "class_name": class_name,
                            "filename": name,
                            "image_hash": image_hash(path),
                            "perceptual_hash": perceptual_hash(path),
                            "processed_path": safe_relpath(path, base_dir),
                            "split": "train" if root_dir == config.get("TRAIN_DIR") else "dataset",
                        }
                    )
                except OSError:
                    continue
    return entries


def ensure_hash_index(config) -> List[dict]:
    index_path = config.get("DATASET_HASH_INDEX_PATH")
    entries = load_hash_index(index_path)
    if not entries:
        try:
            entries = build_hash_index_from_train_dir(config)
        except ValueError as exc:
            if should_treat_as_unrecognized(exc):
                return []
            raise
        if entries and index_path:
            save_hash_index(entries, index_path)
    return entries


def rebuild_hash_index_after_split(config, train_items: List[dict]) -> int:
    """Persist hash index from train split items after dataset processing."""
    base_dir = config["BASE_DIR"]
    entries: List[dict] = []
    for item in train_items:
        path = item.get("processed_path") or item.get("image_path")
        if not path:
            continue
        abs_path = path if os.path.isabs(path) else os.path.join(base_dir, path)
        if not os.path.isfile(abs_path):
            continue
        try:
            entries.append(
                {
                    "image_id": item.get("id"),
                    "class_name": item.get("class_name"),
                    "filename": os.path.basename(abs_path),
                    "image_hash": image_hash(abs_path),
                    "perceptual_hash": perceptual_hash(abs_path),
                    "processed_path": safe_relpath(abs_path, base_dir),
                    "split": "train",
                }
            )
        except OSError:
            continue
    index_path = config.get("DATASET_HASH_INDEX_PATH")
    if index_path:
        save_hash_index(entries, index_path)
    return len(entries)


def check_dataset_similarity(image_path: str, config) -> Dict[str, Any]:
    """Compare test image against indexed training images."""
    if not image_path or not os.path.isfile(image_path):
        return _empty_similarity("Index or test image unavailable.")

    try:
        query_phash = perceptual_hash(image_path)
        query_fhash = image_hash(image_path)
    except OSError as exc:
        return _empty_similarity(f"Could not hash test image: {exc}")

    entries = ensure_hash_index(config)
    if not entries:
        return _empty_similarity("Training hash index is empty. Process the dataset first.")

    best: Optional[Tuple[float, dict, int]] = None
    for entry in entries:
        score = phash_similarity_percent(query_phash, entry.get("perceptual_hash", ""))
        dist = hamming_distance(query_phash, entry.get("perceptual_hash", ""))
        if entry.get("image_hash") == query_fhash:
            score = 100.0
            dist = 0
        if best is None or score > best[0]:
            best = (score, entry, dist)

    if not best:
        return _empty_similarity("No comparable training images found.")

    score, match, dist = best
    meta = _status_from_score(score)
    class_key = match.get("class_name") or ""
    return {
        "similarity_score": score,
        "similarity_status": meta["similarity_status"],
        "risk_level": meta["risk_level"],
        "similarity_message": meta["message"],
        "similarity_recommendation": meta["recommendation"],
        "similarity_tone": meta["tone"],
        "nearest_dataset_image": match.get("filename") or "—",
        "nearest_dataset_class": CLASS_LABELS.get(class_key, class_key.replace("_", " ").title() if class_key else "—"),
        "nearest_dataset_class_key": class_key,
        "nearest_dataset_path": match.get("processed_path") or "",
        "hamming_distance": dist,
        "academic_note": ACADEMIC_EVAL_NOTE if score >= 85 else None,
        "index_available": True,
    }


def compute_per_class_similarity_scores(
    image_path: str,
    trained_classes: List[str],
    config,
) -> Dict[str, float]:
    """Return best perceptual-hash similarity (0–100) per trained class vs dataset index."""
    scores = {cls: 0.0 for cls in trained_classes}
    if not image_path or not os.path.isfile(image_path):
        return scores

    entries = ensure_hash_index(config)
    if not entries:
        return scores

    try:
        query_phash = perceptual_hash(image_path)
        query_fhash = image_hash(image_path)
    except OSError:
        return scores

    for entry in entries:
        class_key = entry.get("class_name")
        if class_key not in scores:
            continue
        value = phash_similarity_percent(query_phash, entry.get("perceptual_hash", ""))
        if entry.get("image_hash") == query_fhash:
            value = 100.0
        scores[class_key] = max(scores[class_key], value)
    return scores


def check_trained_class_fit(
    image_path: str,
    predicted_class: Optional[str],
    config,
    trained_classes: Optional[List[str]] = None,
    model_confidence_pct: Optional[float] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Check whether an upload visually fits the trained dataset classes.

    When ``strict=True``, model confidence must not relax the minimum similarity
    thresholds — used to block photos / OOD inputs that the CNN still labels.
    """
    classes = trained_classes or list(config.get("CLASS_LABELS") or [])
    if not image_path or not os.path.isfile(image_path) or not predicted_class:
        return {"fits": True, "available": False}

    entries = ensure_hash_index(config)
    if not entries:
        return {
            "fits": True,
            "available": False,
            "message": "Indeks dataset training belum tersedia.",
        }

    per_class_best = compute_per_class_similarity_scores(image_path, classes, config)
    global_best = max(per_class_best.values()) if per_class_best else 0.0
    global_best_class = max(per_class_best, key=per_class_best.get) if per_class_best else None
    pred_score = float(per_class_best.get(predicted_class, 0.0))

    min_global = float(config.get("STAGE2_MIN_DATASET_SIMILARITY", 45))
    min_class = float(config.get("STAGE2_MIN_CLASS_SIMILARITY", 50))
    model_conf = float(model_confidence_pct or 0.0)

    if not strict and model_conf >= 60 and predicted_class == global_best_class and global_best >= max(min_global - 10, 35):
        return {
            "fits": True,
            "available": True,
            "per_class_scores": per_class_best,
            "global_best_score": global_best,
            "global_best_class": global_best_class,
            "predicted_class_score": pred_score,
            "dataset_aligned": True,
        }

    if not strict and model_conf >= 55:
        min_global = max(min_global - 12, 32)
        min_class = max(min_class - 12, 38)

    if global_best < min_global:
        return {
            "fits": False,
            "available": True,
            "per_class_scores": per_class_best,
            "global_best_score": global_best,
            "global_best_class": global_best_class,
            "predicted_class_score": pred_score,
            "reason": (
                f"Khat tidak dikenali. Citra kurang mirip dengan sampel dataset training "
                f"(kemiripan tertinggi {global_best:.1f}%, minimum {min_global:.0f}%). "
                f"Sistem hanya menerima citra yang mirip kelas Naskhi, Diwani, Diwani Jali, atau Tsuluts."
            ),
        }

    if pred_score < min_class:
        nearest = max(per_class_best, key=per_class_best.get)
        nearest_score = per_class_best[nearest]
        return {
            "fits": False,
            "available": True,
            "per_class_scores": per_class_best,
            "global_best_score": global_best,
            "global_best_class": global_best_class,
            "predicted_class_score": pred_score,
            "reason": (
                f"Khat tidak dikenali. Citra tidak cukup mirip dengan sampel training kelas "
                f"{CLASS_LABELS.get(predicted_class, predicted_class)} "
                f"({pred_score:.1f}%, minimum {min_class:.0f}%)."
            ),
            "nearest_dataset_class_key": nearest,
            "nearest_dataset_score": nearest_score,
        }

    nearest = max(per_class_best, key=per_class_best.get)
    nearest_score = per_class_best[nearest]
    if nearest != predicted_class and nearest_score - pred_score >= 12:
        return {
            "fits": False,
            "available": True,
            "per_class_scores": per_class_best,
            "global_best_score": global_best,
            "global_best_class": global_best_class,
            "predicted_class_score": pred_score,
            "reason": (
                f"Khat tidak dikenali. Citra lebih mirip kelas "
                f"{CLASS_LABELS.get(nearest, nearest)} ({nearest_score:.1f}%) "
                f"daripada kelas yang diprediksi "
                f"{CLASS_LABELS.get(predicted_class, predicted_class)} ({pred_score:.1f}%)."
            ),
            "nearest_dataset_class_key": nearest,
            "nearest_dataset_score": nearest_score,
        }

    return {
        "fits": True,
        "available": True,
        "per_class_scores": per_class_best,
        "global_best_score": global_best,
        "global_best_class": global_best_class,
        "predicted_class_score": pred_score,
    }


def _empty_similarity(reason: str) -> Dict[str, Any]:
    return {
        "similarity_score": None,
        "similarity_status": "Not Available",
        "risk_level": "—",
        "similarity_message": reason,
        "similarity_recommendation": "Proses dataset training terlebih dahulu untuk mengaktifkan cek kemiripan.",
        "similarity_tone": "neutral",
        "nearest_dataset_image": None,
        "nearest_dataset_class": None,
        "nearest_dataset_class_key": None,
        "nearest_dataset_path": None,
        "hamming_distance": None,
        "academic_note": None,
        "index_available": False,
    }


def merge_similarity_review(
    validation_fields: Dict[str, Any],
    similarity: Optional[Dict[str, Any]],
    confidence_pct: Optional[float],
) -> Dict[str, Any]:
    """Adjust review/final decision based on similarity without changing prediction."""
    out = dict(validation_fields)
    if not similarity or similarity.get("similarity_score") is None:
        return out

    score = float(similarity["similarity_score"])
    conf = float(confidence_pct or 0)

    if score >= 95:
        out["manual_review_required"] = True
        out["review_status"] = "Similarity Review Recommended"
        if conf >= 70:
            out["final_decision"] = "Prediction Accepted, but Similarity Review Recommended"
        elif conf < 70:
            out["final_decision"] = "Manual Review Required"
    elif score >= 85:
        out["manual_review_required"] = True
        if out.get("review_status") in (None, "", "Review Optional", "No Review Required"):
            out["review_status"] = "Similarity Review Recommended"
        if conf >= 85 and out.get("final_decision") in (None, "", "Accepted", "Accepted with Caution"):
            out["final_decision"] = "Prediction Accepted, but Similarity Review Recommended"
    elif score >= 70:
        if out.get("review_status") in (None, "", "No Review Required"):
            out["review_status"] = "Review Optional"
        if conf < 70:
            out["final_decision"] = "Manual Review Required"
    else:
        if conf < 70:
            out["final_decision"] = "Manual Review Required"

    return out


def build_similarity_context(result_row) -> Dict[str, Any]:
    """Build template-friendly similarity dict from a ClassificationResult row."""
    score = getattr(result_row, "similarity_score", None)
    if score is None and not getattr(result_row, "similarity_status", None):
        return {"available": False}

    tone = "neutral"
    if score is not None:
        tone = _status_from_score(float(score)).get("tone", "neutral")

    return {
        "available": score is not None,
        "similarity_status": getattr(result_row, "similarity_status", None) or "—",
        "similarity_score": score,
        "similarity_score_pct": round(score, 1) if score is not None else None,
        "nearest_dataset_image": getattr(result_row, "nearest_dataset_image", None) or "—",
        "nearest_dataset_class": getattr(result_row, "nearest_dataset_class", None) or "—",
        "risk_level": getattr(result_row, "similarity_risk_level", None) or "—",
        "recommendation": getattr(result_row, "similarity_recommendation", None) or "",
        "message": getattr(result_row, "similarity_message", None) or "",
        "academic_note": ACADEMIC_EVAL_NOTE if score is not None and score >= 85 else None,
        "tone": tone,
    }
