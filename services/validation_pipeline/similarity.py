"""Step 5 — cosine similarity against training-set embeddings."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from services.khat_recognition_messages import MSG_LOW_DATASET_SIMILARITY, REJECTION_LOW_SIMILARITY
from services.validation_pipeline.utils import env_float, rejected

logger = logging.getLogger(__name__)

_gallery_lock = threading.Lock()
_gallery_vectors: Optional[np.ndarray] = None
_gallery_labels: Optional[List[str]] = None
_gallery_key: Optional[str] = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _dataset_roots(config=None) -> List[Path]:
    roots: List[Path] = []
    if config is None:
        try:
            from flask import current_app

            config = current_app.config
        except Exception:  # noqa: BLE001
            config = {}
    for key in ("TRAIN_DIR", "RAW_DATASET_DIR", "DATASET_DIR"):
        value = config.get(key) if hasattr(config, "get") else None
        if value and os.path.isdir(value):
            roots.append(Path(value))
    external = os.getenv("EXTERNAL_DATASET_DIR")
    if external and os.path.isdir(external):
        roots.append(Path(external))
    # Deduplicate
    uniq: List[Path] = []
    seen = set()
    for root in roots:
        resolved = str(root.resolve())
        if resolved not in seen:
            seen.add(resolved)
            uniq.append(root)
    return uniq


def _iter_training_images(roots: List[Path], max_per_class: int = 40) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    class_names = {"naskhi", "diwani", "diwani_jali", "diwani jali", "tsuluts", "thusuth", "thuluth"}
    for root in roots:
        # Class subfolders directly under root or under train/
        candidates = [root]
        train = root / "train"
        if train.is_dir():
            candidates.append(train)
        for base in candidates:
            for child in sorted(base.iterdir() if base.is_dir() else []):
                if not child.is_dir():
                    continue
                label = child.name.strip().lower().replace(" ", "_")
                if label not in class_names and child.name.lower() not in class_names:
                    # Still accept known display folders
                    if label not in {"naskhi", "diwani", "diwani_jali", "tsuluts"}:
                        continue
                files = [
                    p for p in child.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                ]
                files = files[:max_per_class]
                for path in files:
                    pairs.append((str(path), label))
    return pairs


def _cache_path(config=None) -> Path:
    base = Path(__file__).resolve().parents[2] / "model"
    base.mkdir(parents=True, exist_ok=True)
    return base / "validation_embedding_gallery.npz"


def build_or_load_gallery(classifier=None, config=None, max_per_class: int = 30) -> Tuple[np.ndarray, List[str]]:
    global _gallery_vectors, _gallery_labels, _gallery_key
    roots = _dataset_roots(config)
    key = "|".join(str(r) for r in roots) + f"|{max_per_class}"
    if _gallery_vectors is not None and _gallery_key == key:
        return _gallery_vectors, _gallery_labels or []

    with _gallery_lock:
        if _gallery_vectors is not None and _gallery_key == key:
            return _gallery_vectors, _gallery_labels or []

        cache = _cache_path(config)
        if cache.is_file():
            try:
                data = np.load(cache, allow_pickle=True)
                vectors = np.asarray(data["vectors"], dtype=np.float32)
                labels = [str(x) for x in data["labels"].tolist()]
                if vectors.ndim == 2 and len(labels) == len(vectors) and len(vectors) >= 8:
                    _gallery_vectors = vectors
                    _gallery_labels = labels
                    _gallery_key = key
                    logger.info("Loaded embedding gallery: %s vectors", len(vectors))
                    return vectors, labels
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load embedding gallery cache: %s", exc)

        from services.validation_pipeline.feature_extractor import extract_embedding

        pairs = _iter_training_images(roots, max_per_class=max_per_class)
        vectors_list: List[np.ndarray] = []
        labels_list: List[str] = []
        for path, label in pairs:
            try:
                vec = extract_embedding(path, classifier=classifier)
                vectors_list.append(vec)
                labels_list.append(label)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skip gallery image %s: %s", path, exc)
                continue

        if not vectors_list:
            empty = np.zeros((0, 1), dtype=np.float32)
            _gallery_vectors = empty
            _gallery_labels = []
            _gallery_key = key
            return empty, []

        vectors = np.stack(vectors_list, axis=0)
        # L2 normalize rows
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        vectors = vectors / norms
        try:
            np.savez_compressed(cache, vectors=vectors, labels=np.array(labels_list, dtype=object))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save embedding gallery: %s", exc)

        _gallery_vectors = vectors
        _gallery_labels = labels_list
        _gallery_key = key
        logger.info("Built embedding gallery: %s vectors from %s roots", len(labels_list), len(roots))
        return vectors, labels_list


def cosine_similarity_to_dataset(
    embedding: np.ndarray,
    *,
    classifier=None,
    config=None,
) -> Dict:
    """Compare query embedding to training gallery; reject below threshold."""
    min_sim = env_float("VALIDATION_SIMILARITY_MIN", 0.45)
    vectors, labels = build_or_load_gallery(classifier=classifier, config=config)
    if vectors is None or len(vectors) == 0:
        return {
            "status": "Skipped",
            "accepted": True,
            "stage": "similarity",
            "skipped": True,
            "reason": "Embedding gallery unavailable",
            "similarity": 0.0,
        }

    query = np.asarray(embedding, dtype=np.float32).reshape(-1)
    qn = float(np.linalg.norm(query)) + 1e-8
    query = query / qn
    if query.shape[0] != vectors.shape[1]:
        # Dimension mismatch — rebuild gallery next time; skip hard reject.
        return {
            "status": "Skipped",
            "accepted": True,
            "stage": "similarity",
            "skipped": True,
            "reason": "Embedding dimension mismatch",
            "similarity": 0.0,
        }

    sims = vectors @ query
    best_idx = int(np.argmax(sims))
    best = float(sims[best_idx])
    best_label = labels[best_idx] if labels else None
    top_k = min(5, len(sims))
    top_idx = np.argpartition(-sims, top_k - 1)[:top_k]
    top = sorted(
        [{"label": labels[i], "similarity": float(sims[i])} for i in top_idx],
        key=lambda x: -x["similarity"],
    )

    if best < min_sim:
        payload = rejected(
            MSG_LOW_DATASET_SIMILARITY,
            stage="similarity",
            details={
                "similarity": round(best, 4),
                "min_similarity": min_sim,
                "nearest_class": best_label,
                "top": top,
            },
        )
        payload["rejection_code"] = REJECTION_LOW_SIMILARITY
        payload["similarity"] = round(best, 4)
        return payload

    return {
        "status": "Accepted",
        "accepted": True,
        "stage": "similarity",
        "similarity": round(best, 4),
        "min_similarity": min_sim,
        "nearest_class": best_label,
        "top": top,
    }
