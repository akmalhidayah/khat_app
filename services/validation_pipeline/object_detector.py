"""Step 3 — non-Arabic object detection (YOLOv8 with graceful fallback)."""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Set

from services.khat_recognition_messages import MSG_NOT_ARABIC_IMAGE, REJECTION_NOT_ARABIC_IMAGE
from services.validation_pipeline.utils import env_flag, env_float, rejected

logger = logging.getLogger(__name__)


def _object_rejected(details: Optional[Dict] = None) -> Dict:
    payload = rejected(
        MSG_NOT_ARABIC_IMAGE,
        stage="object_detection",
        details=details or {},
    )
    payload["rejection_code"] = REJECTION_NOT_ARABIC_IMAGE
    return payload

# COCO labels that must never enter calligraphy classification.
FORBIDDEN_LABELS: Set[str] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
}

_yolo = None
_yolo_lock = threading.Lock()
_yolo_failed = False


def _get_yolo():
    global _yolo, _yolo_failed
    if _yolo_failed:
        return None
    if _yolo is not None:
        return _yolo
    with _yolo_lock:
        if _yolo is not None or _yolo_failed:
            return _yolo
        try:
            from ultralytics import YOLO  # type: ignore

            model_name = __import__("os").getenv("VALIDATION_YOLO_MODEL", "yolov8n.pt")
            _yolo = YOLO(model_name)
            logger.info("YOLOv8 loaded: %s", model_name)
        except Exception as exc:  # noqa: BLE001
            _yolo_failed = True
            logger.warning("YOLOv8 unavailable: %s", exc)
            _yolo = None
        return _yolo


def detect_forbidden_objects(image_path: str) -> Dict:
    """Reject photos dominated by COCO objects (people, animals, vehicles, food, …)."""
    if not env_flag("VALIDATION_USE_YOLO", "1"):
        return {
            "status": "Skipped",
            "accepted": True,
            "stage": "object_detection",
            "skipped": True,
        }

    model = _get_yolo()
    if model is None:
        # Fallback to existing photographic content gate.
        try:
            from services.khat_detector_service import assess_calligraphy_content

            content = assess_calligraphy_content(image_path)
            if content.get("is_photographic_non_khat"):
                return _object_rejected(
                    {"backend": "heuristic_photo_gate", "reason": content.get("rejection_reason")}
                )
            return {
                "status": "Accepted",
                "accepted": True,
                "stage": "object_detection",
                "backend": "heuristic_fallback",
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Object heuristic fallback failed: %s", exc)
            return {
                "status": "Skipped",
                "accepted": True,
                "stage": "object_detection",
                "skipped": True,
                "reason": str(exc),
            }

    conf_min = env_float("VALIDATION_YOLO_CONF", 0.45)
    try:
        results = model.predict(image_path, conf=conf_min, verbose=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("YOLO predict failed: %s", exc)
        return {
            "status": "Skipped",
            "accepted": True,
            "stage": "object_detection",
            "skipped": True,
            "reason": str(exc),
        }

    hits: List[Dict] = []
    for result in results or []:
        names = result.names or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls[0])
            score = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf[0])
            label = str(names.get(cls_id, cls_id)).lower()
            if label in FORBIDDEN_LABELS and score >= conf_min:
                hits.append({"label": label, "confidence": round(score, 4)})

    # Allow a couple of weak incidental detections (e.g. "book" near manuscript).
    strong = [h for h in hits if h["confidence"] >= max(conf_min, 0.55)]
    dominant = [h for h in strong if h["label"] in {
        "person", "dog", "cat", "horse", "bird", "cow", "car", "bus", "truck",
        "boat", "airplane", "potted plant", "dining table", "couch", "bed",
        "laptop", "cell phone", "tv", "pizza", "banana", "sandwich",
    }]
    if dominant or len(strong) >= 2:
        return _object_rejected({"backend": "yolov8", "objects": hits[:12]})

    return {
        "status": "Accepted",
        "accepted": True,
        "stage": "object_detection",
        "backend": "yolov8",
        "objects": hits[:12],
    }
