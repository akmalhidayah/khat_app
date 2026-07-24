"""Step 2 — Arabic script detection (EasyOCR + Unicode fallback)."""

from __future__ import annotations

import logging
import re
import threading
from typing import Dict, List, Optional, Tuple

from services.khat_recognition_messages import (
    MSG_NOT_ARABIC_IMAGE,
    MSG_NOT_ARABIC_SCRIPT,
    REJECTION_NOT_ARABIC_IMAGE,
    REJECTION_NOT_ARABIC_SCRIPT,
)
from services.validation_pipeline.utils import env_flag, rejected

logger = logging.getLogger(__name__)

_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_CJK_RE = re.compile(r"[\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF\u0E00-\u0E7F]")
_DIGIT_RE = re.compile(r"[0-9]")
_SYMBOL_RE = re.compile(r"[∑∫√≈≠≤≥±∞π∆∂∈∉⊂⊃∪∩∀∃∇°‰€£¥©®™#@$%&*+=<>^_|~\\`]")

_reader = None
_reader_lock = threading.Lock()
_reader_failed = False


def _script_rejected(details: Optional[Dict] = None) -> Dict:
    payload = rejected(
        MSG_NOT_ARABIC_SCRIPT,
        stage="arabic_script",
        details=details or {},
    )
    payload["rejection_code"] = REJECTION_NOT_ARABIC_SCRIPT
    return payload


def _detect_qr_or_barcode(image_path: str) -> bool:
    """Reject QR codes / barcodes via OpenCV when available.

    Conservative by design: dense multi-line Arabic calligraphy (harakat, red
    correction marks, connected strokes) creates mid-band edge columns that
    look vaguely like barcode stripes. Only reject clear QR payloads or strong
    1D barcode alternation that does not look like multi-line writing.
    """
    try:
        import cv2  # type: ignore
        import numpy as np
        from PIL import Image

        with Image.open(image_path) as img:
            rgb = np.array(img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(bgr)
        # Require decoded payload (or a large, well-formed finder quad).
        if data:
            return True
        if points is not None:
            pts = np.asarray(points).reshape(-1, 2)
            if pts.shape[0] >= 4:
                xs = pts[:, 0]
                ys = pts[:, 1]
                area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
                img_area = float(bgr.shape[0] * bgr.shape[1]) or 1.0
                if area / img_area >= 0.04:
                    return True

        # Dense high-contrast grid heuristic for barcode-like strips.
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        h, w = edges.shape
        if h < 40 or w < 40:
            return False
        mid = edges[h // 3 : 2 * h // 3, :]
        col_density = (mid > 0).mean(axis=0)
        row_density = (mid > 0).mean(axis=1)
        stripe_ratio = float((col_density > 0.25).mean())
        col_mean = float(col_density.mean())
        row_std = float(np.std(row_density)) if row_density.size else 1.0
        binary_cols = (col_density > 0.20).astype(np.float32)
        transitions = (
            float(np.mean(np.abs(np.diff(binary_cols)))) if binary_cols.size > 1 else 0.0
        )

        # Multi-line calligraphy / manuscripts: several separated horizontal ink peaks.
        full_row = (edges > 0).mean(axis=1)
        win = max(3, h // 40)
        kernel = np.ones(win, dtype=np.float32) / float(win)
        smooth = np.convolve(full_row, kernel, mode="same")
        peaks = 0
        last_peak = -h
        min_gap = max(3, int(h * 0.04))
        for i in range(2, len(smooth) - 2):
            if (
                smooth[i] > 0.06
                and smooth[i] >= smooth[i - 1]
                and smooth[i] >= smooth[i + 1]
                and smooth[i] >= smooth[i - 2]
                and smooth[i] >= smooth[i + 2]
                and (i - last_peak) >= min_gap
            ):
                peaks += 1
                last_peak = i
        # Multi-line writing: several ink baselines with weak column alternation.
        # Strong barcode alternation can create noisy horizontal edge peaks — ignore those.
        if peaks >= 5 and transitions < 0.40:
            return False

        # Classic 1D barcode: rapid column alternation + vertically uniform strip.
        uniform_barcode = (
            transitions >= 0.45
            and row_std <= 0.06
            and col_mean >= 0.12
            and stripe_ratio >= 0.18
        )
        dense_stripe_barcode = (
            stripe_ratio > 0.65 and col_mean > 0.15 and transitions >= 0.35
        )
        return bool(uniform_barcode or dense_stripe_barcode)
    except Exception as exc:  # noqa: BLE001
        logger.debug("QR/barcode probe skipped: %s", exc)
        return False


def _get_reader():
    global _reader, _reader_failed
    if _reader_failed:
        return None
    if _reader is not None:
        return _reader
    with _reader_lock:
        if _reader is not None or _reader_failed:
            return _reader
        try:
            import easyocr  # type: ignore

            # Arabic + English so Latin-only pages are detectable.
            _reader = easyocr.Reader(["ar", "en"], gpu=False, verbose=False)
            logger.info("EasyOCR reader initialized (ar+en)")
        except Exception as exc:  # noqa: BLE001
            _reader_failed = True
            logger.warning("EasyOCR unavailable: %s", exc)
            _reader = None
        return _reader


def _classify_text(texts: List[str]) -> Tuple[bool, bool, bool, bool, bool, str]:
    joined = " ".join(texts)
    has_arabic = bool(_ARABIC_RE.search(joined))
    has_latin = bool(_LATIN_RE.search(joined))
    has_cjk = bool(_CJK_RE.search(joined))
    has_digit = bool(_DIGIT_RE.search(joined))
    has_symbol = bool(_SYMBOL_RE.search(joined))
    return has_arabic, has_latin, has_cjk, has_digit, has_symbol, joined[:240]


def detect_arabic_script(image_path: str) -> Dict:
    """Detect Arabic writing; reject Latin/CJK/digit/QR pages when OCR works."""
    if _detect_qr_or_barcode(image_path):
        return _script_rejected({"backend": "qr_barcode", "kind": "qr_or_barcode"})

    if not env_flag("VALIDATION_USE_EASYOCR", "1"):
        return {
            "status": "Skipped",
            "accepted": True,
            "stage": "arabic_script",
            "skipped": True,
            "reason": "EasyOCR disabled by configuration",
        }

    reader = _get_reader()
    if reader is None:
        # Fallback: reuse heuristic Latin gate already in the app.
        try:
            from services.khat_detector_service import assess_calligraphy_content

            content = assess_calligraphy_content(image_path)
            if content.get("is_latin_script_non_khat"):
                return _script_rejected(
                    {"backend": "heuristic_latin_gate", "latin_gate": content.get("latin_gate")}
                )
            if content.get("is_photographic_non_khat"):
                payload = rejected(
                    MSG_NOT_ARABIC_IMAGE,
                    stage="arabic_script",
                    details={"backend": "heuristic_photo_gate"},
                )
                payload["rejection_code"] = REJECTION_NOT_ARABIC_IMAGE
                return payload
            return {
                "status": "Accepted",
                "accepted": True,
                "stage": "arabic_script",
                "backend": "heuristic_fallback",
                "has_arabic": True,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Arabic heuristic fallback failed: %s", exc)
            return {
                "status": "Skipped",
                "accepted": True,
                "stage": "arabic_script",
                "skipped": True,
                "reason": f"OCR unavailable: {exc}",
            }

    try:
        results = reader.readtext(image_path, detail=0, paragraph=False)
        texts = [str(t).strip() for t in (results or []) if str(t).strip()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("EasyOCR read failed for %s: %s", image_path, exc)
        return {
            "status": "Skipped",
            "accepted": True,
            "stage": "arabic_script",
            "skipped": True,
            "reason": str(exc),
        }

    has_arabic, has_latin, has_cjk, has_digit, has_symbol, sample = _classify_text(texts)

    if has_arabic and not has_cjk:
        return {
            "status": "Accepted",
            "accepted": True,
            "stage": "arabic_script",
            "backend": "easyocr",
            "has_arabic": True,
            "text_sample": sample,
            "detections": len(texts),
        }

    if has_cjk:
        return _script_rejected({"backend": "easyocr", "kind": "cjk", "text_sample": sample})

    if has_latin or has_digit or has_symbol:
        return _script_rejected(
            {
                "backend": "easyocr",
                "has_latin": has_latin,
                "has_digit": has_digit,
                "has_symbol": has_symbol,
                "text_sample": sample,
            }
        )

    if not texts:
        # No OCR text — may still be stylized hijaiyah strokes; defer to similarity/OOD.
        return {
            "status": "Accepted",
            "accepted": True,
            "stage": "arabic_script",
            "backend": "easyocr",
            "has_arabic": False,
            "uncertain_empty_ocr": True,
            "detections": 0,
        }

    return _script_rejected({"backend": "easyocr", "text_sample": sample})
