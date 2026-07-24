"""Binary Khat vs Non-Khat detector for out-of-distribution rejection."""

import os
import random
import shutil
from functools import lru_cache
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from flask import current_app

from services.preprocessing_service import preprocess_calligraphy_image
from services.font_resolver import load_latin_font
from services.training_utils import get_keras, load_json, save_json

DETECTOR_CLASSES = ["khat", "non_khat"]
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
DETECTION_STATUS_LABELS = {
    "confirmed_khat": "Confirmed Khat",
    "moderate_khat": "Khat Detected — Moderate Confidence",
    "borderline_khat": "Borderline Khat Input",
    "uncertain_khat": "Uncertain Khat",
    "uncertain": "Uncertain Khat",
    "non_khat": "Rejected Non-Khat",
}
DETECTION_DECISION_LABELS = {
    "continue": "Continue to classification",
    "continue_caution": "Continue with caution",
    "manual_review": "Manual review",
    "rejected": "Classification blocked",
}


def is_detector_available(config=None) -> bool:
    if config is None:
        config = current_app.config
    path = config.get("KHAT_DETECTOR_PATH")
    return bool(path and os.path.isfile(path) and os.path.getsize(path) > 10_000)


def _is_image_file(name: str) -> bool:
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _detect_latin_alphabet_script(gray: np.ndarray) -> Dict:
    """Reject clear Latin alphabet writing; keep Arabic khat.

    Conservative by design: Arabic calligraphy is often fragmented by thresholding
    (titik, harakat, ornament). Only reject when Latin evidence is strong AND
    Arabic-like cursive connectivity (before morphological merge) is weak.
    """
    empty = {
        "is_latin_script": False,
        "glyph_count": 0,
        "largest_component_frac": None,
        "tall_glyph_frac": None,
        "height_consistency": None,
        "row_alignment": None,
        "arabic_cursive_score": None,
        "arabic_baseline_score": None,
        "baseline_continuity": None,
        "text_rows": 0,
        "dense_arabic_page": False,
        "latin_specimen_candidate": False,
        "latin_template_hits": 0,
        "latin_template_backend": "template_backend_unavailable",
        "latin_template_font_count": 0,
        "latin_template_available": False,
        "latin_score": 0.0,
        "decision_reason": "insufficient_evidence",
    }
    if gray is None or gray.size == 0:
        return empty

    try:
        import cv2
    except ImportError:
        return empty

    h, w = gray.shape[:2]
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(gray_u8, (3, 3), 0)
    binary_adapt = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 12
    )
    # Gold/brown ink on parchment is often midtone — also take pixels darker than local paper.
    mean_g = float(gray_u8.mean())
    binary_rel = ((gray_u8 < max(mean_g - 18.0, 90.0)) & (gray_u8 < 200)).astype(np.uint8) * 255
    binary = cv2.bitwise_or(binary_adapt, binary_rel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    area_img = float(h * w)
    # Score Arabic connectivity on raw strokes (do NOT horizontally close — that
    # falsely joins spaced Latin letters into one cursive band).
    n_raw, _lr, st_raw, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    long_cursive = 0
    raw_ink_areas = []
    max_raw_width = 0
    for index in range(1, n_raw):
        area = int(st_raw[index, cv2.CC_STAT_AREA])
        cw = int(st_raw[index, cv2.CC_STAT_WIDTH])
        ch = int(st_raw[index, cv2.CC_STAT_HEIGHT])
        if area < 30:
            continue
        raw_ink_areas.append(area)
        max_raw_width = max(max_raw_width, cw)
        aspect = cw / max(ch, 1)
        # True Arabic cursive: already wide/connected without joining Latin gaps.
        if cw >= 0.40 * w and aspect >= 1.8:
            long_cursive += 1
        elif cw >= 0.55 * w and aspect >= 1.5:
            long_cursive += 1
    raw_total = float(sum(raw_ink_areas)) or 1.0
    raw_largest_frac = float(max(raw_ink_areas) / raw_total) if raw_ink_areas else 0.0

    # Horizontal baseline continuity: Arabic strokes stay inked along a baseline;
    # Latin words have large gaps between discrete letters.
    ink_mask = binary > 0
    best_span = 0.0
    best_continuity = 0.0
    step = max(1, h // 50)
    for y in range(0, h, step):
        row = ink_mask[y]
        xs = np.flatnonzero(row)
        if xs.size < 8:
            continue
        span = float(xs[-1] - xs[0] + 1)
        span_frac = span / max(w, 1)
        if span_frac < 0.35:
            continue
        continuity = float(xs.size) / max(span, 1.0)
        if span_frac > best_span or (abs(span_frac - best_span) < 0.05 and continuity > best_continuity):
            best_span = span_frac
            best_continuity = continuity
    arabic_baseline_score = 0.0
    if best_span >= 0.50 and best_continuity >= 0.38:
        arabic_baseline_score = min(1.0, 0.55 * best_span + 0.45 * min(best_continuity / 0.60, 1.0))

    arabic_cursive_score = min(
        1.0,
        0.45 * min(long_cursive / 2.0, 1.0)
        + 0.20 * (raw_largest_frac if long_cursive >= 1 else raw_largest_frac * 0.35)
        + 0.15 * (min(max_raw_width / max(w * 0.50, 1.0), 1.0) if long_cursive >= 1 else 0.0)
        + 0.20 * arabic_baseline_score,
    )

    # Glyphs from the same binary so Latin letter gaps stay discrete.
    n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    glyphs = []
    total_ink_area = 0
    for index in range(1, n_labels):
        area = int(stats[index, cv2.CC_STAT_AREA])
        gw = int(stats[index, cv2.CC_STAT_WIDTH])
        gh = int(stats[index, cv2.CC_STAT_HEIGHT])
        if area < 20 or area > 0.45 * area_img:
            continue
        if gh < 8 or gw < 3:
            continue
        # Skip tiny diacritic-like dots typical in Arabic.
        if area < 45 and max(gw, gh) <= 12:
            continue
        total_ink_area += area
        aspect = gw / max(gh, 1)
        if gw >= 0.45 * w and aspect >= 1.8:
            continue
        glyphs.append(
            {
                "area": area,
                "w": gw,
                "h": gh,
                "aspect": aspect,
                "cx": float(centroids[index][0]),
                "cy": float(centroids[index][1]),
                "tall": gh >= gw * 1.05,
            }
        )

    glyph_count = len(glyphs)
    template_diagnostic = _latin_template_match_diagnostics(gray_u8)
    template_hits = int(template_diagnostic["hits"])

    # Protect Arabic calligraphy / hijaiyah strokes from Latin false positives.
    # Connected wide strokes are strong Arabic evidence; baseline continuity alone is weaker
    # because dense Latin typography can also create a long horizontal span.
    strong_arabic = long_cursive >= 1 or arabic_cursive_score >= 0.40
    mild_arabic = arabic_baseline_score >= 0.62 and best_continuity >= 0.42

    # Connected Latin wordmarks (e.g. cursive "Lettering") can look wide/cursive.
    # Keep evaluating them when Latin templates already match.
    latin_wordmark_candidate = bool(
        template_hits >= 2
        and glyph_count <= 8
        and arabic_cursive_score < 0.55
    )

    # Safe fast path for short, clearly connected Arabic compositions. Dense pages are
    # evaluated below so a Latin specimen sheet cannot bypass the layout checks.
    if (
        not latin_wordmark_candidate
        and strong_arabic
        and template_hits <= 1
        and glyph_count < 18
    ):
        return {
            **empty,
            "glyph_count": glyph_count,
            "arabic_cursive_score": round(arabic_cursive_score, 4),
            "arabic_baseline_score": round(arabic_baseline_score, 4),
            "baseline_continuity": round(best_continuity, 4),
            "latin_template_hits": int(template_hits),
            "latin_template_backend": template_diagnostic["backend"],
            "latin_template_font_count": template_diagnostic["font_count"],
            "latin_template_available": template_diagnostic["available"],
            "is_latin_script": False,
            "latin_score": 0.0,
            "decision_reason": "strong_connected_arabic",
        }

    if glyph_count <= 1 or total_ink_area <= 0:
        is_single = template_hits >= 1 and long_cursive == 0 and arabic_cursive_score < 0.40
        return {
            **empty,
            "glyph_count": glyph_count,
            "arabic_cursive_score": round(arabic_cursive_score, 4),
            "arabic_baseline_score": round(arabic_baseline_score, 4),
            "baseline_continuity": round(best_continuity, 4),
            "latin_template_hits": int(template_hits),
            "latin_template_backend": template_diagnostic["backend"],
            "latin_template_font_count": template_diagnostic["font_count"],
            "latin_template_available": template_diagnostic["available"],
            "is_latin_script": bool(is_single),
            "latin_score": round(0.75 if is_single else 0.0, 4),
            "decision_reason": "single_latin_template" if is_single else "insufficient_glyphs",
        }

    areas = np.array([g["area"] for g in glyphs], dtype=np.float32)
    heights = np.array([g["h"] for g in glyphs], dtype=np.float32)
    largest_frac = float(areas.max() / max(areas.sum(), 1.0))
    tall_frac = float(sum(1 for g in glyphs if g["tall"]) / glyph_count)
    height_mean = float(heights.mean())
    height_cv = float(heights.std() / height_mean) if height_mean > 1e-6 else 1.0
    height_consistency = max(0.0, 1.0 - min(height_cv, 1.0))

    ys = np.array([g["cy"] for g in glyphs], dtype=np.float32)
    order = np.argsort(ys)
    rows: List[List[int]] = []
    row_tol = max(6.0, float(np.median(heights)) * 0.50)
    for idx in order:
        y = float(ys[idx])
        placed = False
        for row in rows:
            if abs(y - float(np.mean([ys[j] for j in row]))) <= row_tol:
                row.append(int(idx))
                placed = True
                break
        if not placed:
            rows.append([int(idx)])
    multi_glyph_rows = [row for row in rows if len(row) >= 3]
    row_alignment = 0.0
    if multi_glyph_rows:
        covered = sum(len(row) for row in multi_glyph_rows) / glyph_count
        best_row = max(multi_glyph_rows, key=len)
        xs = sorted(glyphs[i]["cx"] for i in best_row)
        gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        gap_cv = float(np.std(gaps) / max(float(np.mean(gaps)), 1e-6)) if gaps else 1.0
        spacing_score = max(0.0, 1.0 - min(gap_cv, 1.0))
        row_alignment = float(0.55 * covered + 0.45 * spacing_score)

    latin_score = (
        0.18 * min(glyph_count / 10.0, 1.0)
        + 0.18 * max(0.0, 1.0 - largest_frac / 0.40)
        + 0.16 * tall_frac
        + 0.14 * height_consistency
        + 0.14 * row_alignment
        + 0.20 * min(template_hits / 3.0, 1.0)
    )
    if long_cursive >= 1:
        latin_score *= 0.55

    is_latin = False
    n_text_rows = len(multi_glyph_rows)

    # Dense multi-line Naskh/Quran pages: strong baseline continuity, many fragmented
    # glyphs, and only a small amount of Latin-template noise. Requiring at least
    # three text rows prevents a single wide Latin heading from taking this path.
    dense_arabic_page = bool(
        mild_arabic
        and glyph_count >= 35
        and n_text_rows >= 3
        and template_hits <= 2
        and arabic_baseline_score >= 0.70
        and best_continuity >= 0.42
    )

    # Multi-line Latin font specimen/poster. One decorative word may appear cursive,
    # but the overall sheet has repeated tall, aligned glyphs and weak baselines.
    latin_specimen_candidate = bool(
        template_hits >= 3
        and glyph_count >= 18
        and n_text_rows >= 4
        and tall_frac >= 0.45
        and height_consistency >= 0.55
        and row_alignment >= 0.50
        and largest_frac <= 0.30
        and arabic_baseline_score < 0.55
        and best_continuity < 0.36
        and latin_score >= 0.40
    )
    decorative_latin_specimen = bool(
        template_hits >= 4
        and glyph_count >= 25
        and n_text_rows >= 3
        and tall_frac >= 0.50
        and height_consistency >= 0.60
        and row_alignment >= 0.55
        and largest_frac <= 0.25
        and arabic_baseline_score < 0.60
        and best_continuity < 0.40
    )

    # A verified dense Arabic page must win over weak template noise. A verified
    # specimen sheet must be rejected before the generic strong-cursive protection.
    if dense_arabic_page:
        is_latin = False
    elif latin_specimen_candidate or decorative_latin_specimen:
        is_latin = True
    # Never treat clear Arabic cursive/baseline as Latin unless templates are overwhelming.
    elif strong_arabic and template_hits < 4:
        is_latin = False
    elif mild_arabic and template_hits < 2:
        is_latin = False
    # Strong templates + discrete Latin-like layout.
    elif (
        template_hits >= 3
        and long_cursive == 0
        and not strong_arabic
        and tall_frac >= 0.42
        and height_consistency >= 0.40
        and row_alignment >= 0.35
        and largest_frac <= 0.40
        and latin_score >= 0.52
    ):
        is_latin = True
    elif (
        template_hits >= 2
        and long_cursive == 0
        and not strong_arabic
        and arabic_cursive_score < 0.35
        and glyph_count >= 4
        and largest_frac <= 0.40
        and tall_frac >= 0.42
        and height_consistency >= 0.50
        and row_alignment >= 0.42
        and latin_score >= 0.55
    ):
        is_latin = True
    # Clear Latin word/line: many tall even glyphs + at least one letter template.
    elif (
        template_hits >= 1
        and long_cursive == 0
        and not strong_arabic
        and arabic_cursive_score < 0.25
        and arabic_baseline_score < 0.55
        and glyph_count >= 4
        and tall_frac >= 0.65
        and height_consistency >= 0.70
        and row_alignment >= 0.65
        and largest_frac <= 0.40
        and latin_score >= 0.58
    ):
        is_latin = True
    # Multi-line Latin typography / poster.
    elif (
        long_cursive == 0
        and not strong_arabic
        and arabic_cursive_score < 0.25
        and arabic_baseline_score < 0.50
        and glyph_count >= 8
        and n_text_rows >= 2
        and tall_frac >= 0.38
        and height_consistency >= 0.40
        and largest_frac <= 0.42
        and row_alignment >= 0.28
        and template_hits >= 1
        and latin_score >= 0.50
    ):
        is_latin = True
    elif (
        long_cursive == 0
        and not strong_arabic
        and arabic_cursive_score < 0.22
        and arabic_baseline_score < 0.45
        and glyph_count >= 12
        and n_text_rows >= 3
        and tall_frac >= 0.35
        and height_consistency >= 0.35
        and largest_frac <= 0.45
        and row_alignment >= 0.25
        and template_hits >= 2
        and latin_score >= 0.50
    ):
        is_latin = True
    # Strong Latin letter templates override weak "cursive" from props/tools in the photo.
    elif (
        template_hits >= 3
        and arabic_cursive_score < 0.32
        and glyph_count >= 8
        and tall_frac >= 0.45
        and latin_score >= 0.55
        and largest_frac <= 0.35
    ):
        is_latin = True
    elif (
        template_hits >= 4
        and arabic_cursive_score < 0.35
        and glyph_count >= 6
        and latin_score >= 0.58
    ):
        is_latin = True
    # Decorative connected Latin lettering / wordmark (e.g. cursive "Lettering"):
    # few fused components, Latin templates hit, but not strong Arabic structure.
    elif (
        template_hits >= 2
        and arabic_cursive_score < 0.55
        and arabic_baseline_score < 0.60
        and glyph_count <= 8
        and largest_frac >= 0.40
        and height_consistency >= 0.35
        and long_cursive <= 1
    ):
        is_latin = True
    elif (
        template_hits >= 1
        and arabic_cursive_score < 0.22
        and arabic_baseline_score < 0.45
        and 2 <= glyph_count <= 5
        and largest_frac >= 0.55
        and height_consistency >= 0.70
        and tall_frac >= 0.35
        and long_cursive == 0
        and not strong_arabic
    ):
        is_latin = True

    return {
        "is_latin_script": bool(is_latin),
        "glyph_count": glyph_count,
        "largest_component_frac": round(largest_frac, 4),
        "tall_glyph_frac": round(tall_frac, 4),
        "height_consistency": round(height_consistency, 4),
        "row_alignment": round(row_alignment, 4),
        "arabic_cursive_score": round(arabic_cursive_score, 4),
        "arabic_baseline_score": round(arabic_baseline_score, 4),
        "baseline_continuity": round(best_continuity, 4),
        "text_rows": int(n_text_rows),
        "dense_arabic_page": bool(dense_arabic_page),
        "latin_specimen_candidate": bool(
            latin_specimen_candidate or decorative_latin_specimen
        ),
        "latin_template_hits": int(template_hits),
        "latin_template_backend": template_diagnostic["backend"],
        "latin_template_font_count": template_diagnostic["font_count"],
        "latin_template_available": template_diagnostic["available"],
        "latin_score": round(float(latin_score), 4),
        "decision_reason": (
            "dense_arabic_page"
            if dense_arabic_page
            else "latin_specimen_layout"
            if latin_specimen_candidate or decorative_latin_specimen
            else "latin_evidence"
            if is_latin
            else "arabic_or_insufficient_latin_evidence"
        ),
    }


@lru_cache(maxsize=12)
def _latin_pillow_templates(font_size: int):
    """Build templates once per font size for at most one regular and one bold face."""
    from PIL import Image, ImageDraw

    letters = (
        "A", "B", "E", "F", "H", "K", "L", "M", "N", "P", "R", "T", "W", "Y",
        "a", "e", "m", "n", "r", "t",
    )
    fonts = [load_latin_font(font_size, False), load_latin_font(font_size, True)]
    fonts = [font for index, font in enumerate(fonts) if font and font not in fonts[:index]]
    templates = []
    for font_index, font in enumerate(fonts):
        for ch in letters:
            pad = max(8, font_size // 4)
            canvas = Image.new("L", (font_size + pad * 2, font_size + pad * 2), 255)
            ImageDraw.Draw(canvas).text((pad, pad // 2), ch, fill=0, font=font)
            templates.append((f"pillow_{font_index}", ch, np.asarray(canvas, dtype=np.uint8)))
    return tuple(templates)


@lru_cache(maxsize=12)
def _latin_hershey_templates(font_size: int):
    """OpenCV fallback templates when no TrueType font exists."""
    import cv2

    letters = ("A", "B", "E", "F", "H", "K", "L", "M", "N", "P", "R", "T", "W", "Y")
    templates = []
    scale = max(float(font_size) / 32.0, 0.6)
    thickness = max(1, int(round(scale * 1.5)))
    for ch in letters:
        (width, height), baseline = cv2.getTextSize(
            ch, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        canvas = np.full(
            (height + baseline + 12, width + 12), 255, dtype=np.uint8
        )
        cv2.putText(
            canvas, ch, (6, height + 4), cv2.FONT_HERSHEY_SIMPLEX,
            scale, 0, thickness, cv2.LINE_AA,
        )
        templates.append(("opencv_hershey", ch, canvas))
    return tuple(templates)


def _latin_template_font_count() -> int:
    return sum(load_latin_font(28, bold) is not None for bold in (False, True))


def _latin_template_match_diagnostics(gray_u8: np.ndarray) -> Dict:
    """Count strict Latin template matches and report backend availability."""
    try:
        import cv2
    except ImportError:
        return {
            "hits": 0,
            "backend": "template_backend_unavailable",
            "font_count": 0,
            "available": False,
        }

    h, w = gray_u8.shape[:2]
    if h < 24 or w < 24:
        return {
            "hits": 0,
            "backend": "no_template_match",
            "font_count": _latin_template_font_count(),
            "available": True,
        }

    targets = [gray_u8.astype(np.float32)]
    if max(h, w) >= 220:
        targets.append(cv2.resize(gray_u8, (160, 160), interpolation=cv2.INTER_AREA).astype(np.float32))

    hits = 0
    seen = set()
    letters = (
        # Prefer angular Latin letters; avoid O/C/S/I which false-match Arabic arcs.
        "A", "B", "E", "F", "H", "K", "L", "M", "N", "P", "R", "T", "W", "Y",
        "a", "e", "m", "n", "r", "t",
    )
    # Fixed small templates + one image-scaled template for large single letters.
    font_sizes = [28, max(36, int(0.28 * min(h, w)))]
    font_count = _latin_template_font_count()
    backend = "pillow_truetype" if font_count else "opencv_hershey"
    for font_size in font_sizes:
        templates = (
            _latin_pillow_templates(font_size)
            if font_count
            else _latin_hershey_templates(font_size)
        )
        for _template_backend, ch, tmpl in templates:
            ys, xs = np.where(tmpl < 200)
            if len(xs) < 10:
                continue
            tmpl = tmpl[max(ys.min() - 1, 0) : ys.max() + 2, max(xs.min() - 1, 0) : xs.max() + 2]
            base_h, base_w = tmpl.shape
            if base_h < 8 or base_w < 8:
                continue
            best = 0.0
            scales = (0.85, 1.0, 1.3, 1.7) if font_size <= 28 else (0.7, 0.85, 1.0, 1.15)
            for target in targets:
                th_img, tw_img = target.shape[:2]
                for scale in scales:
                    th = max(8, int(round(base_h * scale)))
                    tw = max(8, int(round(base_w * scale)))
                    if th >= th_img or tw >= tw_img:
                        continue
                    scaled = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)
                    try:
                        result = cv2.matchTemplate(
                            target, scaled.astype(np.float32), cv2.TM_CCOEFF_NORMED
                        )
                        best = max(best, float(result.max()) if result.size else 0.0)
                    except cv2.error:
                        continue
            if best >= 0.78 and ch not in seen:
                seen.add(ch)
                hits += 1
            if hits >= 5:
                return {
                    "hits": hits,
                    "backend": backend,
                    "font_count": font_count,
                    "available": True,
                }
    return {
        "hits": hits,
        "backend": backend if hits else "no_template_match",
        "font_count": font_count,
        "available": True,
    }


def _latin_template_match_hits(gray_u8: np.ndarray) -> int:
    """Backward-compatible integer API."""
    return int(_latin_template_match_diagnostics(gray_u8)["hits"])


def _dominant_blob_photo_scores(gray: np.ndarray) -> Dict:
    """Detect solid photographed objects (animal/person/thing) on light backgrounds.

    Calligraphy on paper has many thin stroke components. A cow/person photo on a
    white studio background is typically one dominant filled blob with smoother edges.
    """
    empty = {
        "is_dominant_blob_photo": False,
        "top_dark_fraction": 0.0,
        "significant_components": 0,
        "edge_ratio": 0.0,
        "paper_ratio": 0.0,
    }
    try:
        import cv2
        from PIL import ImageFilter
        from PIL import Image as PILImage

        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
        paper_ratio = float((gray_u8 > 200).mean())
        dark_mask = ((gray_u8 < 140) * 255).astype(np.uint8)
        dark_px = int(dark_mask.sum() // 255)
        if dark_px < 80:
            return empty

        edges = np.asarray(
            PILImage.fromarray(gray_u8, mode="L").filter(ImageFilter.FIND_EDGES),
            dtype=np.float32,
        )
        edge_ratio = float((edges > 40).mean())
        n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
        areas = sorted(
            [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)],
            reverse=True,
        )
        if not areas:
            return empty
        top = areas[0]
        top_frac = float(top) / float(dark_px)
        significant = sum(1 for a in areas if a >= max(25, dark_px * 0.01))
        # Studio/object photo: light backdrop + one dominant filled body.
        dark_ratio = float(dark_px) / float(gray_u8.size)
        is_blob = bool(
            paper_ratio >= 0.30
            and top_frac >= 0.72
            and edge_ratio <= 0.20
            and significant <= 40
            and dark_ratio >= 0.12  # animals/objects; exclude tiny single glyphs
        )
        # Softer rule for very dominant objects even with slightly higher edges.
        if (
            not is_blob
            and paper_ratio >= 0.45
            and top_frac >= 0.88
            and edge_ratio <= 0.24
            and significant <= 25
            and dark_ratio >= 0.15
        ):
            is_blob = True
        return {
            "is_dominant_blob_photo": is_blob,
            "top_dark_fraction": round(top_frac, 4),
            "significant_components": int(significant),
            "edge_ratio": round(edge_ratio, 4),
            "paper_ratio": round(paper_ratio, 4),
            "dark_ratio": round(dark_ratio, 4),
        }
    except Exception:
        return empty


def _creature_photo_scores(
    gray: np.ndarray,
    hue: np.ndarray,
    sat: np.ndarray,
    *,
    paper_ratio: float,
    mean_saturation: float,
    dark_ground_ratio: float,
    light_ink_ratio: float,
    looks_like_calligraphy: bool,
) -> Dict:
    """Detect animal/creature photos (e.g. chickens on a dark backdrop).

    Illuminated gold calligraphy on dark panels must not match — those already
    set looks_like_calligraphy via light-ink-on-dark stroke structure.
    """
    empty = {
        "is_creature_photo": False,
        "warm_fur_ratio": 0.0,
        "red_crest_ratio": 0.0,
        "chroma_mass": 0.0,
    }
    if gray is None or gray.size == 0 or looks_like_calligraphy:
        return empty
    try:
        # Brown/tan fur & plumage (chicken body, dog, cow hide).
        warm_fur_ratio = float(
            (
                (hue >= 8)
                & (hue < 48)
                & (sat > 0.38)
                & (gray > 45)
                & (gray < 200)
            ).mean()
        )
        # Red comb / wattle / beak accents common on birds.
        red_crest_ratio = float(
            (
                ((hue < 18) | (hue > 235))
                & (sat > 0.45)
                & (gray > 40)
                & (gray < 210)
            ).mean()
        )
        chroma_mass = float((sat > 0.42) & (gray > 40) & (gray < 215)).mean()
        # Bright body mass on a mostly dark ground (studio animal shot).
        body_ratio = float((gray > 85) & (gray < 220)).mean()
        is_creature = bool(
            paper_ratio < 0.18
            and mean_saturation >= 0.30
            and dark_ground_ratio >= 0.30
            and light_ink_ratio < 0.085
            and (
                (
                    mean_saturation >= 0.55
                    and (warm_fur_ratio + red_crest_ratio) >= 0.035
                    and chroma_mass >= 0.06
                )
                or (
                    red_crest_ratio >= 0.012
                    and warm_fur_ratio >= 0.04
                    and chroma_mass >= 0.08
                )
                or (
                    mean_saturation >= 0.42
                    and chroma_mass >= 0.12
                    and body_ratio >= 0.08
                    and body_ratio <= 0.55
                    and dark_ground_ratio >= 0.45
                )
            )
        )
        return {
            "is_creature_photo": is_creature,
            "warm_fur_ratio": round(warm_fur_ratio, 4),
            "red_crest_ratio": round(red_crest_ratio, 4),
            "chroma_mass": round(chroma_mass, 4),
            "body_ratio": round(body_ratio, 4),
        }
    except Exception:
        return empty


def _face_roi_has_skin(rgb_roi: np.ndarray) -> bool:
    """Validate a Haar face ROI using conservative skin-color evidence.

    Haar cascades often see circular calligraphy ornaments as faces. Monochrome
    ink has almost no chroma, while a real color face normally contains a useful
    amount of skin-like HSV/YCrCb pixels. Failure is conservative: the ROI is not
    accepted as a face.
    """
    if rgb_roi is None or rgb_roi.size == 0 or rgb_roi.ndim != 3:
        return False
    if rgb_roi.shape[0] < 12 or rgb_roi.shape[1] < 12:
        return False

    try:
        import cv2

        roi = np.clip(rgb_roi, 0, 255).astype(np.uint8)
        channels = roi.astype(np.float32)
        chroma = float(
            (
                np.abs(channels[:, :, 0] - channels[:, :, 1])
                + np.abs(channels[:, :, 1] - channels[:, :, 2])
                + np.abs(channels[:, :, 0] - channels[:, :, 2])
            ).mean()
            / 3.0
        )
        if chroma < 6.0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        hsv_skin = (
            ((hue < 25) | (hue > 160))
            & (sat > 30)
            & (sat < 230)
            & (val > 45)
            & (val < 250)
        )

        ycrcb = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
        ycrcb_skin = (
            (y > 40)
            & (cr >= 132)
            & (cr <= 185)
            & (cb >= 75)
            & (cb <= 140)
        )

        # Agreement between two color spaces is more robust than either alone.
        skin_fraction = float((hsv_skin & ycrcb_skin).mean())
        return skin_fraction >= 0.08
    except Exception:
        return False


def _detect_human_presence(rgb_image) -> Dict:
    """Detect people via skin-validated OpenCV Haar face cascades.

    Upper-body cascade is intentionally not used alone because calligraphy
    flourishes frequently trigger it. Raw Haar hits are retained for diagnostics,
    but only skin-validated face boxes can reject an image as a human photo.
    """
    empty = {
        "is_human_photo": False,
        "faces": 0,
        "raw_faces": 0,
        "upper_bodies": 0,
    }
    try:
        import cv2

        if hasattr(rgb_image, "size"):
            arr = np.asarray(rgb_image.convert("RGB"), dtype=np.uint8)
        else:
            arr = np.asarray(rgb_image, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[0] < 40 or arr.shape[1] < 40:
            return empty

        h, w = arr.shape[:2]
        scale = 480.0 / max(h, w)
        if scale < 1.0:
            arr = cv2.resize(
                arr,
                (max(40, int(w * scale)), max(40, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"

        raw_faces = 0
        skin_faces = 0
        for cascade_path, params in (
            (face_path, {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": (32, 32)}),
            (profile_path, {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": (32, 32)}),
        ):
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                continue
            found = cascade.detectMultiScale(gray, **params)
            raw_faces += len(found)
            for x, y, fw, fh in found:
                x0 = max(0, int(x))
                y0 = max(0, int(y))
                x1 = min(arr.shape[1], int(x + fw))
                y1 = min(arr.shape[0], int(y + fh))
                if x1 <= x0 or y1 <= y0:
                    continue
                if _face_roi_has_skin(arr[y0:y1, x0:x1]):
                    skin_faces += 1

        return {
            "is_human_photo": bool(skin_faces >= 1),
            "faces": int(skin_faces),
            "raw_faces": int(raw_faces),
            "upper_bodies": 0,
        }
    except Exception:
        return empty


def assess_calligraphy_content(image_path: str) -> Dict:
    """Fast visual gate: reject photos and Latin-alphabet writing that are not Arabic khat.

    Supports dark ink on light paper and light/gold ink on dark illuminated panels.
    Ornamental borders must not trigger plant/photo rejection when stroke structure exists.
    """
    from PIL import Image, ImageFilter, ImageOps

    empty = {
        "looks_like_calligraphy": True,
        "is_photographic_non_khat": False,
        "is_latin_script_non_khat": False,
        "is_non_khat_content": False,
        "available": False,
        "paper_ratio": None,
        "midtone_ratio": None,
        "edge_ratio": None,
        "mean_saturation": None,
        "scene_scores": None,
        "latin_gate": None,
        "rejection_reason": None,
    }
    if not image_path or not os.path.isfile(image_path):
        return empty

    try:
        with Image.open(image_path) as img:
            rgb = ImageOps.exif_transpose(img).convert("RGB")
            small = rgb.resize((192, 192), Image.LANCZOS)
            latin_src = rgb.resize((320, 320), Image.LANCZOS)
            human_src = rgb.resize((480, 480), Image.LANCZOS)
        gray = np.asarray(ImageOps.grayscale(small), dtype=np.float32)
        gray_latin = np.asarray(ImageOps.grayscale(latin_src), dtype=np.float32)
        hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1] / 255.0
        # White/cream paper + warm parchment (aged khat paper).
        bright_paper = (gray > 200) & (sat < 0.28)
        parchment = (gray > 155) & (gray <= 230) & (sat < 0.45) & (hue < 55)
        paper_ratio = float((bright_paper | parchment).mean())
        midtone_ratio = float(((gray >= 80) & (gray <= 180)).mean())
        edges = np.asarray(ImageOps.grayscale(small).filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        edge_ratio = float((edges > 40).mean())
        mean_saturation = float(sat.mean())
        dark_ink_ratio = float((gray < 140).mean())
        light_ink_ratio = float((gray > 170).mean())
        dark_ground_ratio = float((gray < 115).mean())
        ink_ratio = dark_ink_ratio
        # PIL HSV hue is 0–255 for a full circle.
        green_ratio = float(((hue > 45) & (hue < 110) & (sat > 0.22)).mean())
        blue_ratio = float(((hue > 125) & (hue < 200) & (sat > 0.18)).mean())
        # Regional outdoor cues (sky usually top; vegetation often bottom).
        top_h = max(1, int(gray.shape[0] * 0.34))
        bot_h = max(1, int(gray.shape[0] * 0.34))
        blue_mask = (hue > 125) & (hue < 200) & (sat > 0.15)
        green_mask = (hue > 45) & (hue < 110) & (sat > 0.18)
        top_blue_ratio = float(blue_mask[:top_h, :].mean())
        bottom_green_ratio = float(green_mask[-bot_h:, :].mean())
        # Skin-like / warm brown hues — also appear in gold ink and wooden panels.
        skin_ratio = float(
            (
                (hue > 5)
                & (hue < 45)
                & (sat > 0.12)
                & (sat < 0.70)
                & (gray > 55)
                & (gray < 230)
            ).mean()
        )
    except OSError:
        return empty

    blob_gate = _dominant_blob_photo_scores(gray)
    is_dominant_blob_photo = bool(blob_gate.get("is_dominant_blob_photo"))
    human_gate = _detect_human_presence(human_src)
    is_human_photo = bool(human_gate.get("is_human_photo"))
    # Saturated blue/green poster/photo fields are not illuminated calligraphy panels.
    is_color_poster_photo = bool(
        mean_saturation >= 0.35
        and paper_ratio < 0.25
        and (blue_ratio >= 0.35 or green_ratio >= 0.35)
    )
    # Outdoor photos: buildings, landscapes, sea, plants under open sky.
    # Mosque/building shots often fake "ink on paper" because walls are bright.
    is_outdoor_scene_photo = bool(
        (
            top_blue_ratio >= 0.35
            and blue_ratio >= 0.18
            and mean_saturation >= 0.12
        )
        or (
            top_blue_ratio >= 0.28
            and bottom_green_ratio >= 0.12
            and blue_ratio >= 0.15
        )
        or (
            blue_ratio >= 0.28
            and green_ratio >= 0.08
            and mean_saturation >= 0.15
            and paper_ratio < 0.55
        )
        or (
            # Strong sky field even without much green (sea / open yard).
            top_blue_ratio >= 0.45
            and blue_ratio >= 0.30
            and mean_saturation >= 0.12
        )
    )

    # Classic: dark strokes on light paper. Dense black-and-white circular
    # compositions can have higher edge density than ordinary pages. Allow the
    # wider 0.48 ceiling only for low-saturation paper-like images.
    ink_edge_upper = 0.48 if (paper_ratio >= 0.35 and mean_saturation <= 0.30) else 0.42
    looks_like_ink_on_paper = (
        (not is_dominant_blob_photo)
        and (not is_human_photo)
        and (not is_color_poster_photo)
        and (not is_outdoor_scene_photo)
        and paper_ratio >= 0.28
        and dark_ink_ratio >= 0.04
        and dark_ink_ratio <= 0.55
        and edge_ratio >= 0.035
        and edge_ratio <= ink_edge_upper
        and blue_ratio < 0.22
        and green_ratio < 0.18
    )
    # Illuminated panel: light/gold strokes on dark brown ground (+ ornate borders).
    # Exclude vivid blue/green photo posters that only look "dark" in grayscale.
    looks_like_light_ink_on_dark = (
        (not is_human_photo)
        and (not is_color_poster_photo)
        and (not is_outdoor_scene_photo)
        and dark_ground_ratio >= 0.30
        and light_ink_ratio >= 0.035
        and light_ink_ratio <= 0.50
        and edge_ratio >= 0.040
        and edge_ratio <= 0.42
        and paper_ratio < 0.45
        and blue_ratio < 0.28
        and green_ratio < 0.28
    )
    looks_like_calligraphy = bool(looks_like_ink_on_paper or looks_like_light_ink_on_dark)

    creature_gate = _creature_photo_scores(
        gray,
        hue,
        sat,
        paper_ratio=paper_ratio,
        mean_saturation=mean_saturation,
        dark_ground_ratio=dark_ground_ratio,
        light_ink_ratio=light_ink_ratio,
        looks_like_calligraphy=looks_like_calligraphy,
    )
    is_creature_photo = bool(creature_gate.get("is_creature_photo"))
    if is_creature_photo:
        looks_like_calligraphy = False

    is_ocean = (
        (not looks_like_calligraphy or is_outdoor_scene_photo)
        and blue_ratio > 0.35
        and paper_ratio < 0.40
        and mean_saturation > 0.12
        and (edge_ratio < 0.22 or blue_ratio > 0.50 or top_blue_ratio >= 0.40)
    )
    is_plant = (
        (not looks_like_calligraphy or is_outdoor_scene_photo)
        and green_ratio > 0.22
        and paper_ratio < 0.40
        and mean_saturation > 0.12
        and (edge_ratio < 0.22 or green_ratio > 0.35 or bottom_green_ratio >= 0.20)
    )
    # Real skin/body photos: smooth midtones, little paper, weak stroke structure.
    is_skin_or_body = (
        not looks_like_calligraphy
        and skin_ratio > 0.22
        and paper_ratio < 0.22
        and dark_ink_ratio < 0.20
        and light_ink_ratio < 0.20
        and (
            (edge_ratio < 0.10 and midtone_ratio > 0.35)
            or (skin_ratio > 0.35 and edge_ratio < 0.12 and paper_ratio < 0.15)
        )
    )
    is_color_field_photo = (
        not looks_like_calligraphy
        and paper_ratio < 0.12
        and mean_saturation > 0.35
        and (
            blue_ratio > 0.40
            or green_ratio > 0.35
            or skin_ratio > 0.22
            # Highly saturated subjects on dark grounds (birds, colorful animals).
            or mean_saturation >= 0.55
        )
    )
    is_photo_scene = (
        not looks_like_calligraphy
        and (
            (edge_ratio < 0.06 and midtone_ratio > 0.55)
            or (paper_ratio < 0.05 and mean_saturation > 0.32 and edge_ratio < 0.08)
            or (paper_ratio < 0.02 and midtone_ratio > 0.65 and edge_ratio < 0.09)
            or (paper_ratio < 0.08 and mean_saturation > 0.28 and midtone_ratio > 0.42 and edge_ratio < 0.11)
        )
    )
    is_photographic = bool(
        is_dominant_blob_photo
        or is_human_photo
        or is_color_poster_photo
        or is_outdoor_scene_photo
        or is_creature_photo
        or is_ocean
        or is_plant
        or is_skin_or_body
        or is_color_field_photo
        or is_photo_scene
    )
    if (
        is_dominant_blob_photo
        or is_human_photo
        or is_color_poster_photo
        or is_outdoor_scene_photo
        or is_creature_photo
    ):
        looks_like_calligraphy = False
    latin_gate = _detect_latin_alphabet_script(gray_latin)
    is_latin = bool(latin_gate.get("is_latin_script"))
    latin_hits = int(latin_gate.get("latin_template_hits") or 0)
    latin_score = float(latin_gate.get("latin_score") or 0)
    arabic_cursive = float(latin_gate.get("arabic_cursive_score") or 0)
    glyph_count = int(latin_gate.get("glyph_count") or 0)
    hc_raw = latin_gate.get("height_consistency")
    height_consistency = float(hc_raw) if hc_raw is not None else None
    arabic_baseline = float(latin_gate.get("arabic_baseline_score") or 0)
    dense_arabic_page = bool(latin_gate.get("dense_arabic_page"))
    latin_specimen_candidate = bool(latin_gate.get("latin_specimen_candidate"))

    # Suppress Latin false-positives only when Arabic evidence is strong and the
    # template evidence is weak. A verified specimen sheet must never be cleared
    # merely because one decorative Latin word appears cursive.
    if is_latin and not latin_specimen_candidate and (
        (arabic_cursive >= 0.35 and latin_hits < 3)
        or (
            looks_like_calligraphy
            and height_consistency is not None
            and height_consistency < 0.45
            and glyph_count >= 6
            and latin_hits < 3
        )
        or (
            looks_like_calligraphy
            and latin_hits < 2
            and latin_score < 0.62
            and arabic_cursive >= 0.20
        )
        or (
            looks_like_calligraphy
            and dense_arabic_page
            and arabic_baseline >= 0.70
            and latin_hits <= 2
        )
    ):
        is_latin = False
    # Prefer Latin rejection label when script evidence is stronger than blob photo.
    # Keep photographic label for real outdoor/people/plant scenes (not wordmarks).
    if is_latin and is_dominant_blob_photo and (latin_hits >= 1 or latin_score >= 0.55):
        if not (
            is_outdoor_scene_photo
            or is_ocean
            or is_plant
            or is_human_photo
            or is_color_poster_photo
            or is_skin_or_body
            or is_creature_photo
        ):
            is_photographic = False
        looks_like_calligraphy = False
    is_non_khat = bool(is_photographic or is_latin)

    scene_scores = {
        "green_ratio": round(green_ratio, 4),
        "blue_ratio": round(blue_ratio, 4),
        "skin_ratio": round(skin_ratio, 4),
        "ink_ratio": round(ink_ratio, 4),
        "dark_ink_ratio": round(dark_ink_ratio, 4),
        "light_ink_ratio": round(light_ink_ratio, 4),
        "dark_ground_ratio": round(dark_ground_ratio, 4),
        "looks_like_ink_on_paper": looks_like_ink_on_paper,
        "ink_edge_upper": round(ink_edge_upper, 4),
        "looks_like_light_ink_on_dark": looks_like_light_ink_on_dark,
        "looks_like_calligraphy": looks_like_calligraphy,
        "is_ocean": is_ocean,
        "is_plant": is_plant,
        "is_skin_or_body": is_skin_or_body,
        "is_color_field_photo": is_color_field_photo,
        "is_dominant_blob_photo": is_dominant_blob_photo,
        "is_human_photo": is_human_photo,
        "is_color_poster_photo": is_color_poster_photo,
        "is_outdoor_scene_photo": is_outdoor_scene_photo,
        "is_creature_photo": is_creature_photo,
        "top_blue_ratio": round(top_blue_ratio, 4),
        "bottom_green_ratio": round(bottom_green_ratio, 4),
        "blob_gate": blob_gate,
        "creature_gate": creature_gate,
        "human_gate": human_gate,
    }

    reason = None
    if is_ocean or is_plant or is_skin_or_body or is_photographic:
        reason = "Gambar bukan merupakan tulisan Arab."
    elif is_latin:
        reason = "Input bukan merupakan tulisan Arab."
    if is_latin:
        decision_code = "rejected_latin"
    elif is_photographic:
        decision_code = "rejected_photo"
    elif looks_like_calligraphy and dense_arabic_page:
        decision_code = "confirmed_khat_content"
    elif looks_like_calligraphy:
        decision_code = "likely_khat_content"
    else:
        decision_code = "uncertain_content"
    evidence_for = []
    evidence_against = []
    if looks_like_ink_on_paper:
        evidence_for.append("ink_on_paper")
    if looks_like_light_ink_on_dark:
        evidence_for.append("light_ink_on_dark")
    if dense_arabic_page:
        evidence_for.append("dense_arabic_page")
    if arabic_cursive >= 0.35:
        evidence_for.append("arabic_cursive_structure")
    if is_latin:
        evidence_against.append("latin_script_evidence")
    if is_photographic:
        evidence_against.append("photographic_scene_evidence")
    return {
        "looks_like_calligraphy": not is_non_khat,
        "is_photographic_non_khat": is_photographic,
        "is_latin_script_non_khat": is_latin,
        "is_non_khat_content": is_non_khat,
        "available": True,
        "paper_ratio": round(paper_ratio, 4),
        "midtone_ratio": round(midtone_ratio, 4),
        "edge_ratio": round(edge_ratio, 4),
        "mean_saturation": round(mean_saturation, 4),
        "scene_scores": scene_scores,
        "latin_gate": latin_gate,
        "rejection_reason": reason,
        "decision_code": decision_code,
        "decision_reason": reason or latin_gate.get("decision_reason") or decision_code,
        "evidence_for_khat": evidence_for,
        "evidence_against_khat": evidence_against,
        "triggered_rules": evidence_for + evidence_against,
        "feature_snapshot": {
            "paper_ratio": round(paper_ratio, 4),
            "midtone_ratio": round(midtone_ratio, 4),
            "edge_ratio": round(edge_ratio, 4),
            "mean_saturation": round(mean_saturation, 4),
            "arabic_cursive_score": round(arabic_cursive, 4),
            "arabic_baseline_score": round(arabic_baseline, 4),
            "latin_template_hits": latin_hits,
            "raw_face_hits": int((human_gate or {}).get("raw_faces") or 0),
            "skin_validated_faces": int((human_gate or {}).get("faces") or 0),
        },
    }


def _content_rejection_result(
    *,
    content: Dict,
    detector_available: bool,
    skipped: bool,
    khat_probability: float = 0.0,
    non_khat_probability: float = 1.0,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict:
    thresholds = thresholds or {}
    if content.get("is_latin_script_non_khat"):
        message = "Input bukan merupakan tulisan Arab."
        explanation = (
            "Validasi konten menolak citra karena terdeteksi sebagai alfabet non-Arab "
            "(Latin, angka, simbol, CJK, QR/barcode), bukan tulisan Arab."
        )
    else:
        message = "Gambar bukan merupakan tulisan Arab."
        explanation = (
            "Validasi konten menolak citra karena karakteristik visualnya menyerupai foto "
            "(manusia/hewan/tumbuhan/lautan/benda), bukan tulisan Arab."
        )
    return {
        "detector_available": detector_available,
        "is_khat": False,
        "khat_probability": round(float(khat_probability), 4),
        "non_khat_probability": round(float(non_khat_probability), 4),
        "input_status": "non_khat",
        "detection_status": "non_khat",
        "detection_decision": "rejected",
        "detection_status_label": DETECTION_STATUS_LABELS["non_khat"],
        "detection_decision_label": DETECTION_DECISION_LABELS["rejected"],
        "manual_review_required": False,
        "stage2_allowed": False,
        "stage2_permission": "Blocked",
        "title": "Rejected Non-Khat",
        "message": message,
        "moderate_confidence_note": None,
        "accept_threshold": thresholds.get("accept"),
        "reject_threshold": thresholds.get("reject"),
        "borderline_threshold": thresholds.get("borderline"),
        "skipped": skipped,
        "rejection_reason": content.get("rejection_reason") or message,
        "detection_explanation": explanation,
        "content_gate": content,
    }


def get_thresholds(config=None) -> Dict[str, float]:
    if config is None:
        config = current_app.config
    defaults = {
        "accept": float(config.get("KHAT_ACCEPT_THRESHOLD", 0.70)),
        "reject": float(config.get("KHAT_REJECT_THRESHOLD", 0.50)),
        "borderline": float(config.get("KHAT_BORDERLINE_THRESHOLD", 0.65)),
        "high_confidence": float(config.get("KHAT_HIGH_CONFIDENCE_THRESHOLD", 0.85)),
    }
    settings_path = config.get("KHAT_DETECTOR_SETTINGS_PATH")
    if settings_path and os.path.isfile(settings_path):
        saved = load_json(settings_path, {})
        for key in ("accept", "reject", "borderline", "high_confidence"):
            if key in saved:
                defaults[key] = float(saved[key])
    return defaults


def save_thresholds(accept: float, reject: float, borderline: Optional[float] = None, config=None) -> Dict:
    if config is None:
        config = current_app.config
    if accept <= reject:
        raise ValueError("Accept threshold must be greater than reject threshold.")
    if accept - reject < 0.10:
        raise ValueError("Recommended gap between accept and reject thresholds is at least 0.10.")
    borderline_val = float(borderline if borderline is not None else config.get("KHAT_BORDERLINE_THRESHOLD", 0.65))
    if borderline_val < reject or borderline_val >= accept:
        raise ValueError("Borderline threshold must be between reject and accept thresholds.")
    payload = {
        "accept": round(float(accept), 4),
        "reject": round(float(reject), 4),
        "borderline": round(borderline_val, 4),
        "high_confidence": float(config.get("KHAT_HIGH_CONFIDENCE_THRESHOLD", 0.85)),
        "updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }
    save_json(config["KHAT_DETECTOR_SETTINGS_PATH"], payload)
    return payload


def resolve_detection(
    khat_probability: float,
    config=None,
    force_classify: bool = False,
) -> Dict:
    """Resolve the Stage 1 Khat-detection decision.

    Inputs at or above the accept threshold continue normally.
    Inputs in the borderline band continue with mandatory manual review.
    Inputs below the borderline threshold remain rejected unless the
    operator explicitly force-continues classification.
    """
    if config is None:
        config = current_app.config
    thresholds = get_thresholds(config)
    accept = thresholds["accept"]
    reject = thresholds["reject"]
    borderline = thresholds["borderline"]
    high_conf = thresholds["high_confidence"]
    khat_probability = float(khat_probability)
    khat_pct = round(khat_probability * 100, 2)
    accept_pct = round(accept * 100, 2)

    if khat_probability >= accept:
        detection_status = "confirmed_khat" if khat_probability >= high_conf else "moderate_khat"
        return {
            "input_status": "khat",
            "detection_status": detection_status,
            "is_khat": True,
            "detection_decision": "continue",
            "manual_review_required": detection_status == "moderate_khat",
            "stage2_allowed": True,
            "stage2_permission": "Enabled",
            "title": (
                "Khat Image Confirmed"
                if detection_status == "confirmed_khat"
                else "Khat Image Detected — Moderate Confidence"
            ),
            "message": (
                "The uploaded image was confidently detected as Arabic Khat calligraphy."
                if detection_status == "confirmed_khat"
                else (
                    "The system detected this image as Arabic Khat calligraphy with moderate confidence. "
                    "Classification continues, but manual review is recommended."
                )
            ),
            "moderate_confidence_note": (
                "Input passed Khat detection with moderate confidence."
                if detection_status == "moderate_khat"
                else None
            ),
            "rejection_reason": None,
            "explanation_text": None,
        }

    # Borderline Khat input: continue Stage 2 with mandatory review.
    # The hard content gate has already rejected photos, Latin writing,
    # and other clearly non-Arab content before this decision.
    if khat_probability >= borderline:
        return {
            "input_status": "khat",
            "detection_status": "borderline_khat",
            "is_khat": True,
            "detection_decision": "continue_caution",
            "manual_review_required": True,
            "stage2_allowed": True,
            "stage2_permission": "Enabled with caution",
            "title": "Borderline Khat Input — Continued with Caution",
            "message": (
                f"Probabilitas khat {khat_pct}% berada pada rentang "
                f"borderline dan di bawah ambang terima {accept_pct}%. "
                "Klasifikasi jenis khat tetap dilanjutkan, tetapi hasil "
                "wajib divalidasi secara manual."
            ),
            "moderate_confidence_note": (
                "Stage 2 dilanjutkan karena citra berada pada rentang "
                "borderline khat. Manual review wajib dilakukan."
            ),
            "rejection_reason": None,
            "explanation_text": (
                f"Probabilitas khat {khat_pct}% berada di antara ambang "
                f"borderline dan ambang terima ({accept_pct}%). "
                "Citra diteruskan ke klasifikasi 4 kelas dengan kehati-hatian."
            ),
        }

    # Values below the borderline threshold remain rejected by default.
    is_mid_band = khat_probability >= reject

    if is_mid_band:
        detection_status = "uncertain_khat"
        title = "Rejected — Uncertain Non-Khat"
    else:
        detection_status = "non_khat"
        title = "Rejected Non-Khat"

    if force_classify:
        return {
            "input_status": "uncertain",
            "detection_status": (
                detection_status
                if detection_status != "non_khat"
                else "uncertain_khat"
            ),
            "is_khat": True,
            "detection_decision": "continue_caution",
            "manual_review_required": True,
            "stage2_allowed": True,
            "stage2_permission": "Enabled with caution",
            "title": title.replace("Rejected — ", "") + " (Forced Continue)",
            "message": (
                f"Khat probability is {khat_pct}% "
                f"(accept ≥{accept_pct}%). "
                "Classification was explicitly continued at user request. "
                "Manual review is required."
            ),
            "moderate_confidence_note": (
                "Classification continued despite failing Stage 1 "
                "Khat acceptance."
            ),
            "rejection_reason": None,
            "explanation_text": (
                f"Operator force-continued classification although khat "
                f"probability {khat_pct}% is below the accept threshold "
                f"({accept_pct}%)."
            ),
        }

    return {
        "input_status": "non_khat",
        "detection_status": detection_status,
        "is_khat": False,
        "detection_decision": "rejected",
        "manual_review_required": False,
        "stage2_allowed": False,
        "stage2_permission": "Blocked",
        "title": title,
        "message": "Gambar bukan merupakan tulisan Arab.",
        "moderate_confidence_note": None,
        "rejection_reason": (
            "Gambar bukan merupakan tulisan Arab."
        ),
        "explanation_text": (
            f"Probabilitas khat {khat_pct}% berada di bawah ambang "
            f"borderline ({round(borderline * 100, 2)}%). "
            "Klasifikasi 4 kelas diblokir agar gambar di luar domain "
            "tulisan Arab tidak diberi label."
        ),
    }


def resolve_input_status(khat_probability: float, config=None) -> Tuple[str, bool, str]:
    """Backward-compatible tuple API."""
    resolved = resolve_detection(khat_probability, config)
    return resolved["input_status"], resolved["is_khat"], resolved["message"]


def _collect_khat_images(config) -> List[str]:
    paths = []
    for root in (config["TRAIN_DIR"], config["VALIDATION_DIR"], config["TEST_DIR"], config["RAW_DATASET_DIR"]):
        if not os.path.isdir(root):
            continue
        for cls in config["CLASS_LABELS"]:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for name in os.listdir(cls_dir):
                if _is_image_file(name):
                    path = os.path.join(cls_dir, name)
                    if os.path.isfile(path):
                        paths.append(path)
    return list(set(paths))


def _collect_non_khat_images(config) -> List[str]:
    paths = []
    for root in (
        config["NON_KHAT_DIR"],
        os.path.join(config.get("KHAT_DETECTOR_TRAIN_DIR", ""), "non_khat"),
        os.path.join(config.get("KHAT_DETECTOR_TEST_DIR", ""), "non_khat"),
    ):
        if not root or not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if _is_image_file(name):
                path = os.path.join(root, name)
                if os.path.isfile(path):
                    paths.append(path)
    return list(set(paths))


def _collect_detector_split_images(config, split: str, label: str) -> List[str]:
    split_map = {
        "train": config.get("KHAT_DETECTOR_TRAIN_DIR"),
        "validation": config.get("KHAT_DETECTOR_VALIDATION_DIR"),
        "test": config.get("KHAT_DETECTOR_TEST_DIR"),
    }
    root = split_map.get(split)
    if not root:
        return []
    cls_dir = os.path.join(root, label)
    if not os.path.isdir(cls_dir):
        return []
    return [
        os.path.join(cls_dir, name)
        for name in os.listdir(cls_dir)
        if _is_image_file(name) and os.path.isfile(os.path.join(cls_dir, name))
    ]


def sync_khat_detector_dataset(config=None, val_ratio: float = 0.15, test_ratio: float = 0.10, seed: int = 42) -> Dict:
    """Build dataset/khat_detector/{train,validation,test}/{khat,non_khat}."""
    from sklearn.model_selection import train_test_split

    if config is None:
        config = current_app.config

    bootstrap_non_khat_dataset(config)
    khat_paths = _collect_khat_images(config)
    non_khat_paths = _collect_non_khat_images(config)

    for split in ("train", "validation", "test"):
        for label in DETECTOR_CLASSES:
            path = os.path.join(config["KHAT_DETECTOR_DATASET_DIR"], split, label)
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

    def _copy_items(items: List[Dict], split: str) -> int:
        count = 0
        for item in items:
            dest_dir = os.path.join(config["KHAT_DETECTOR_DATASET_DIR"], split, item["label"])
            ext = os.path.splitext(item["path"])[1].lower() or ".jpg"
            dest = os.path.join(dest_dir, f"{item['label']}_{count:05d}{ext}")
            try:
                shutil.copy2(item["path"], dest)
                count += 1
            except OSError:
                continue
        return count

    khat_items = [{"path": p, "label": "khat"} for p in khat_paths]
    non_items = [{"path": p, "label": "non_khat"} for p in non_khat_paths]
    all_items = khat_items + non_items
    if not all_items:
        return {"khat_total": 0, "non_khat_total": 0, "warning": "No images available for detector dataset."}

    labels = [x["label"] for x in all_items]
    train_items, holdout = train_test_split(all_items, test_size=val_ratio + test_ratio, random_state=seed, stratify=labels)
    relative_test = test_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) else 0.5
    holdout_labels = [x["label"] for x in holdout]
    val_items, test_items = train_test_split(
        holdout, test_size=relative_test, random_state=seed, stratify=holdout_labels if len(set(holdout_labels)) > 1 else None
    )

    counts = {
        "train": _copy_items(train_items, "train"),
        "validation": _copy_items(val_items, "validation"),
        "test": _copy_items(test_items, "test"),
    }
    khat_total = sum(1 for x in all_items if x["label"] == "khat")
    non_khat_total = sum(1 for x in all_items if x["label"] == "non_khat")
    min_required = int(config.get("KHAT_DETECTOR_MIN_SAMPLES", 500))
    warning = None
    if khat_total < min_required or non_khat_total < min_required:
        warning = (
            f"Non-Khat dataset is too small ({non_khat_total} images; Khat: {khat_total}). "
            f"Recommended minimum: {min_required} per class. Detection may be unreliable."
        )

    balance = build_balance_report(config)
    return {
        "synced_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "khat_total": khat_total,
        "non_khat_total": non_khat_total,
        "split_counts": counts,
        "warning": warning,
        "balance": balance,
    }


def build_balance_report(config=None) -> Dict:
    if config is None:
        config = current_app.config

    khat_count = len(_collect_detector_split_images(config, "train", "khat"))
    non_count = len(_collect_detector_split_images(config, "train", "non_khat"))
    if khat_count == 0 and non_count == 0:
        khat_count = len(_collect_khat_images(config))
        non_count = len(_collect_non_khat_images(config))

    total = khat_count + non_count
    ratio = round(max(khat_count, non_count) / max(min(khat_count, non_count), 1), 2) if total else 0
    recommendations = []
    min_required = int(config.get("KHAT_DETECTOR_MIN_SAMPLES", 500))
    if non_count < min_required:
        recommendations.append(f"Add more Non-Khat images (current: {non_count}, recommended: {min_required}+).")
    if khat_count < min_required:
        recommendations.append(f"Add more Khat images (current: {khat_count}, recommended: {min_required}+).")
    if ratio > 3:
        recommendations.append("Class imbalance is high. Use class weights and balanced sampling during training.")

    report = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "khat_images": khat_count,
        "non_khat_images": non_count,
        "total_images": total,
        "balance_ratio": ratio,
        "balanced": ratio <= 2.0 and khat_count >= 100 and non_count >= 100,
        "recommendations": recommendations,
    }
    save_json(config["KHAT_DETECTOR_BALANCE_REPORT_PATH"], report)
    return report


def bootstrap_non_khat_dataset(config=None, min_images: int = 80) -> Dict:
    """Generate synthetic non-Khat samples when the folder is sparse."""
    from PIL import Image, ImageDraw

    if config is None:
        config = current_app.config
    root = config["NON_KHAT_DIR"]
    os.makedirs(root, exist_ok=True)

    existing = _collect_non_khat_images(config)
    created = 0
    if len(existing) >= min_images:
        return {"created": 0, "total": len(existing), "path": root}

    target = min_images - len(existing)
    colors = [(240, 240, 240), (30, 30, 30), (200, 220, 255), (255, 230, 200), (180, 180, 180)]
    for i in range(target):
        kind = i % 6
        img = Image.new("RGB", (random.randint(320, 900), random.randint(240, 700)), random.choice(colors))
        draw = ImageDraw.Draw(img)
        if kind == 0:
            for y in range(0, img.height, 40):
                draw.line([(0, y), (img.width, y)], fill=(220, 220, 220), width=1)
            draw.rectangle([20, 20, img.width - 20, 60], fill=(59, 130, 246))
            draw.text((30, 30), "Application Dashboard", fill=(255, 255, 255))
        elif kind == 1:
            draw.ellipse([50, 50, img.width - 50, img.height - 50], outline=(100, 100, 100), width=4)
            draw.text((80, img.height // 2), "LOGO", fill=(50, 50, 50))
        elif kind == 2:
            for _ in range(30):
                x1, y1 = random.randint(0, img.width), random.randint(0, img.height)
                x2, y2 = x1 + random.randint(20, 120), y1 + random.randint(10, 40)
                draw.rectangle(
                    [x1, y1, x2, y2],
                    fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                )
        elif kind == 3:
            arr = np.random.randint(0, 255, (img.height, img.width, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
        elif kind == 4:
            draw.rectangle([0, 0, img.width, img.height], fill=(255, 255, 255))
            for row in range(8):
                y = 40 + row * 35
                draw.line([(40, y), (img.width - 40, y)], fill=(180, 180, 180))
        else:
            draw.text((40, 40), "Document / Table / Chart", fill=(0, 0, 0))
            draw.rectangle([40, 100, img.width - 40, img.height - 40], outline=(0, 0, 0))

        out = os.path.join(root, f"synthetic_non_khat_{len(existing) + created + 1:04d}.jpg")
        img.save(out, format="JPEG", quality=90)
        created += 1

    total = len(_collect_non_khat_images(config))
    return {"created": created, "total": total, "path": root}


def _build_detector_datasets(config, val_ratio: float = 0.15, seed: int = 42):
    from sklearn.model_selection import train_test_split

    sync_khat_detector_dataset(config, val_ratio=val_ratio, test_ratio=0.10, seed=seed)
    train_items = (
        [{"path": p, "label": "khat"} for p in _collect_detector_split_images(config, "train", "khat")]
        + [{"path": p, "label": "non_khat"} for p in _collect_detector_split_images(config, "train", "non_khat")]
    )
    val_items = (
        [{"path": p, "label": "khat"} for p in _collect_detector_split_images(config, "validation", "khat")]
        + [{"path": p, "label": "non_khat"} for p in _collect_detector_split_images(config, "validation", "non_khat")]
    )
    if len(train_items) < 20 or len(val_items) < 10:
        khat_paths = list(set(_collect_khat_images(config)))
        non_khat_paths = list(set(_collect_non_khat_images(config)))
        if len(khat_paths) < 20:
            raise ValueError("Not enough Khat images for detector training. Process dataset first.")
        if len(non_khat_paths) < 20:
            bootstrap_non_khat_dataset(config)
            non_khat_paths = list(set(_collect_non_khat_images(config)))
        items = [{"path": p, "label": "khat"} for p in khat_paths] + [{"path": p, "label": "non_khat"} for p in non_khat_paths]
        labels = [x["label"] for x in items]
        train_items, val_items = train_test_split(items, test_size=val_ratio, random_state=seed, stratify=labels)
    return train_items, val_items


def _compute_class_weights(items: List[Dict]) -> Dict[int, float]:
    counts = {0: 0, 1: 0}
    for item in items:
        counts[0 if item["label"] == "khat" else 1] += 1
    total = sum(counts.values()) or 1
    return {idx: total / (2 * max(count, 1)) for idx, count in counts.items()}


def train_khat_detector(architecture: str = "efficientnetb0", epochs: int = 8, batch_size: int = 16, config=None) -> Dict:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    if config is None:
        config = current_app.config

    get_keras()
    bootstrap_non_khat_dataset(config)
    train_items, val_items = _build_detector_datasets(config)
    balance = build_balance_report(config)
    class_weights = _compute_class_weights(train_items)
    architecture = (architecture or "efficientnetb0").lower()

    def _generator(items, batch_size, shuffle=True):
        indices = list(range(len(items)))
        while True:
            if shuffle:
                random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                xs, ys = [], []
                for idx in batch_idx:
                    item = items[idx]
                    if not os.path.isfile(item["path"]):
                        continue
                    try:
                        xs.append(preprocess_calligraphy_image(item["path"], architecture=architecture)[0])
                        ys.append(0 if item["label"] == "khat" else 1)
                    except Exception:
                        continue
                if xs:
                    yield np.array(xs), tf.keras.utils.to_categorical(ys, 2)

    from services.model_builder_service import build_classifier

    model, _ = build_classifier(architecture, 2, 1e-4, trainable_backbone=False)
    steps_train = max(1, len(train_items) // batch_size)
    steps_val = max(1, len(val_items) // batch_size)

    checkpoint = config["KHAT_DETECTOR_PATH"]
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
        ModelCheckpoint(
            checkpoint.replace(".keras", "_checkpoint.weights.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
        ),
    ]

    history = model.fit(
        _generator(train_items, batch_size, True),
        validation_data=_generator(val_items, batch_size, False),
        steps_per_epoch=steps_train,
        validation_steps=steps_val,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    weights_path = checkpoint.replace(".keras", "_checkpoint.weights.h5")
    if os.path.isfile(weights_path):
        model.load_weights(weights_path)
    model.save(checkpoint)
    model.save_weights(checkpoint.replace(".keras", ".weights.h5"))

    hist = history.history
    best_val = float(max(hist["val_accuracy"])) if hist.get("val_accuracy") else None
    thresholds = get_thresholds(config)
    metadata = {
        "architecture": architecture,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_samples": len(train_items),
        "val_samples": len(val_items),
        "khat_samples": sum(1 for x in train_items + val_items if x["label"] == "khat"),
        "non_khat_samples": sum(1 for x in train_items + val_items if x["label"] == "non_khat"),
        "class_weights": class_weights,
        "best_val_accuracy": best_val,
        "trained_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "classes": DETECTOR_CLASSES,
        "accept_threshold": thresholds["accept"],
        "reject_threshold": thresholds["reject"],
    }
    save_json(config["KHAT_DETECTOR_METADATA_PATH"], metadata)
    save_json(config["KHAT_DETECTOR_INDICES_PATH"], {"class_indices": {"khat": 0, "non_khat": 1}})

    calibrate_detector_thresholds(config)
    evaluate_non_khat_detection(config)
    return metadata


def load_detector_model(config=None):
    from services.model_cache_service import get_cached_classifier

    if config is None:
        config = current_app.config
    path = config["KHAT_DETECTOR_PATH"]
    arch = "efficientnetb0"
    meta = load_json(config.get("KHAT_DETECTOR_METADATA_PATH"), {})
    if meta.get("architecture"):
        arch = meta["architecture"]
    return get_cached_classifier(path, arch, 2)


def detect_khat(image_path: str, config=None, force_classify: bool = False) -> Dict:
    if config is None:
        config = current_app.config

    if not os.path.isfile(image_path):
        return {
            "detector_available": is_detector_available(config),
            "is_khat": False,
            "khat_probability": 0.0,
            "non_khat_probability": 1.0,
            "input_status": "non_khat",
            "detection_status": "non_khat",
            "detection_decision": "rejected",
            "message": "Image file not found.",
            "skipped": True,
            "rejection_reason": "Image file not found",
        }

    content = assess_calligraphy_content(image_path)
    thresholds = get_thresholds(config)

    # Hard reject photo / Latin-abjad / non-calligraphy OOD before class labeling.
    # Never allow force_classify to override this content gate.
    is_non_khat_content = bool(
        content.get("is_non_khat_content")
        or content.get("is_photographic_non_khat")
        or content.get("is_latin_script_non_khat")
    )
    if is_non_khat_content:
        return _content_rejection_result(
            content=content,
            detector_available=is_detector_available(config),
            skipped=not is_detector_available(config),
            khat_probability=0.0,
            non_khat_probability=1.0,
            thresholds=thresholds,
        )

    if not is_detector_available(config):
        # No trained detector: content gate already passed; Stage 2 must still be strict.
        return {
            "detector_available": False,
            "is_khat": False,
            "khat_probability": None,
            "non_khat_probability": None,
            "input_status": "uncertain",
            "detection_status": "detector_unavailable",
            "detection_decision": "continue_caution",
            "manual_review_required": True,
            "stage2_allowed": True,
            "stage2_permission": "Enabled with caution",
            "title": "Khat Detector Not Trained",
            "message": (
                "Detektor Stage 1 belum dilatih. Sistem tetap memproses, "
                "tetapi hanya menerima prediksi kelas yang jelas (4 kelas khat)."
            ),
            "skipped": True,
            "rejection_reason": None,
            "content_gate": content,
            "accept_threshold": thresholds["accept"],
            "reject_threshold": thresholds["reject"],
            "borderline_threshold": thresholds["borderline"],
        }

    try:
        model = load_detector_model(config)
    except (ValueError, OSError, MemoryError):
        return {
            "detector_available": False,
            "is_khat": False,
            "khat_probability": None,
            "non_khat_probability": None,
            "input_status": "uncertain",
            "detection_status": "detector_unavailable",
            "detection_decision": "continue_caution",
            "manual_review_required": True,
            "stage2_allowed": True,
            "stage2_permission": "Enabled with caution",
            "title": "Khat Detector Unavailable",
            "message": (
                "Detektor Stage 1 tidak dapat dimuat. Klasifikasi dilanjutkan dengan pintu ketat Stage 2 "
                "(hanya 4 kelas khat yang valid)."
            ),
            "skipped": True,
            "rejection_reason": None,
            "content_gate": content,
            "accept_threshold": thresholds["accept"],
            "reject_threshold": thresholds["reject"],
            "borderline_threshold": thresholds["borderline"],
        }

    meta = load_json(config.get("KHAT_DETECTOR_METADATA_PATH"), {})
    architecture = meta.get("architecture", "efficientnetb0")
    batch = preprocess_calligraphy_image(image_path, architecture=architecture)
    probs = model.predict(batch, verbose=0)[0]
    khat_prob = float(probs[0])
    non_khat_prob = float(probs[1])
    resolved = resolve_detection(khat_prob, config, force_classify=force_classify)

    return {
        "detector_available": True,
        "is_khat": resolved["is_khat"],
        "khat_probability": round(khat_prob, 4),
        "non_khat_probability": round(non_khat_prob, 4),
        "input_status": resolved["input_status"],
        "detection_status": resolved["detection_status"],
        "detection_decision": resolved["detection_decision"],
        "detection_status_label": DETECTION_STATUS_LABELS.get(resolved["detection_status"], resolved["detection_status"]),
        "detection_decision_label": DETECTION_DECISION_LABELS.get(resolved["detection_decision"], resolved["detection_decision"]),
        "manual_review_required": resolved["manual_review_required"],
        "stage2_allowed": resolved["stage2_allowed"],
        "title": resolved["title"],
        "message": resolved["message"],
        "moderate_confidence_note": resolved.get("moderate_confidence_note"),
        "accept_threshold": thresholds["accept"],
        "reject_threshold": thresholds["reject"],
        "borderline_threshold": thresholds["borderline"],
        "skipped": False,
        "rejection_reason": resolved.get("rejection_reason"),
        "stage2_permission": resolved.get("stage2_permission"),
        "detection_explanation": resolved.get("explanation_text"),
        "content_gate": content,
    }


def _predict_khat_probability(image_path: str, config) -> Optional[float]:
    if not os.path.isfile(image_path) or not is_detector_available(config):
        return None
    try:
        model = load_detector_model(config)
        meta = load_json(config.get("KHAT_DETECTOR_METADATA_PATH"), {})
        architecture = meta.get("architecture", "efficientnetb0")
        batch = preprocess_calligraphy_image(image_path, architecture=architecture)
        return float(model.predict(batch, verbose=0)[0][0])
    except Exception:
        return None


def calibrate_detector_thresholds(config=None) -> Dict:
    if config is None:
        config = current_app.config
    if not is_detector_available(config):
        return {"available": False}

    val_khat = _collect_detector_split_images(config, "validation", "khat")
    val_non = _collect_detector_split_images(config, "validation", "non_khat")
    if not val_khat:
        val_khat = _collect_khat_images(config)[:80]
    if not val_non:
        val_non = _collect_non_khat_images(config)[:80]

    samples = [(p, True) for p in val_khat if os.path.isfile(p)] + [(p, False) for p in val_non if os.path.isfile(p)]
    probs = []
    for path, expected_khat in samples:
        prob = _predict_khat_probability(path, config)
        if prob is not None:
            probs.append({"expected_khat": expected_khat, "khat_probability": prob})

    sweep = []
    best = None
    for threshold in [round(x, 2) for x in np.arange(0.40, 0.91, 0.05)]:
        false_rejection = 0
        false_acceptance = 0
        khat_total = sum(1 for s in probs if s["expected_khat"])
        non_total = len(probs) - khat_total
        for sample in probs:
            accepted = sample["khat_probability"] >= threshold
            if sample["expected_khat"] and not accepted:
                false_rejection += 1
            if not sample["expected_khat"] and accepted:
                false_acceptance += 1
        fr_rate = false_rejection / khat_total if khat_total else 0
        fa_rate = false_acceptance / non_total if non_total else 0
        row = {
            "threshold": threshold,
            "false_rejection_rate": round(fr_rate, 4),
            "false_acceptance_rate": round(fa_rate, 4),
        }
        sweep.append(row)
        score = fr_rate * 2 + fa_rate
        if best is None or score < best["score"]:
            best = {**row, "score": score}

    recommended_accept = 0.70
    if best:
        recommended_accept = min(0.75, max(0.65, best["threshold"]))

    report = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "validation_samples": len(probs),
        "threshold_sweep": sweep,
        "recommended_accept_threshold": recommended_accept,
        "recommended_reject_threshold": round(max(0.45, recommended_accept - 0.20), 2),
        "best_balanced_threshold": best,
    }
    save_json(config["KHAT_DETECTOR_THRESHOLD_REPORT_PATH"], report)
    return report


def evaluate_non_khat_detection(config=None) -> Dict:
    if config is None:
        config = current_app.config

    test_non = _collect_detector_split_images(config, "test", "non_khat") or _collect_non_khat_images(config)
    test_khat = _collect_detector_split_images(config, "test", "khat") or _collect_khat_images(config)[:100]
    test_khat = test_khat[: min(100, len(test_khat))]

    results = []
    tp = fp = tn = fn = 0

    for path in test_non + test_khat:
        if not os.path.isfile(path):
            continue
        expected_khat = path in test_khat
        det = detect_khat(path, config)
        predicted_accept = det.get("stage2_allowed") and det.get("input_status") == "khat"
        predicted_reject = det.get("input_status") == "non_khat"
        if expected_khat and predicted_accept:
            tp += 1
        elif expected_khat and not predicted_accept:
            fn += 1
        elif not expected_khat and predicted_reject or (not expected_khat and det.get("input_status") == "uncertain"):
            tn += 1
        elif not expected_khat and predicted_accept:
            fp += 1

        results.append({
            "path": path,
            "expected_khat": expected_khat,
            "predicted_khat": predicted_accept,
            "khat_probability": det.get("khat_probability"),
            "input_status": det.get("input_status"),
            "detection_status": det.get("detection_status"),
            "correct": predicted_accept == expected_khat,
        })

    total_non = sum(1 for r in results if not r["expected_khat"])
    total_khat = sum(1 for r in results if r["expected_khat"])
    non_khat_rejection = sum(1 for r in results if not r["expected_khat"] and r["input_status"] in ("non_khat", "uncertain"))
    khat_acceptance = sum(1 for r in results if r["expected_khat"] and r["predicted_khat"])
    false_acceptance = fp
    false_rejection = fn

    precision_khat = tp / (tp + fp) if (tp + fp) else 0
    recall_khat = tp / (tp + fn) if (tp + fn) else 0
    precision_non = tn / (tn + fn) if (tn + fn) else 0
    recall_non = tn / (tn + fp) if (tn + fp) else 0
    accuracy = (tp + tn) / len(results) if results else 0

    thresholds = get_thresholds(config)
    report = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "detector_available": is_detector_available(config),
        "accept_threshold": thresholds["accept"],
        "reject_threshold": thresholds["reject"],
        "non_khat_test_count": total_non,
        "khat_test_count": total_khat,
        "binary_accuracy": round(accuracy, 4),
        "khat_precision": round(precision_khat, 4),
        "khat_recall": round(recall_khat, 4),
        "non_khat_precision": round(precision_non, 4),
        "non_khat_recall": round(recall_non, 4),
        "non_khat_rejection_accuracy": round(non_khat_rejection / total_non, 4) if total_non else None,
        "khat_acceptance_accuracy": round(khat_acceptance / total_khat, 4) if total_khat else None,
        "false_acceptance_rate": round(false_acceptance / total_non, 4) if total_non else None,
        "false_rejection_rate": round(false_rejection / total_khat, 4) if total_khat else None,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "too_strict_warning": (
            "Khat detector is too strict and rejects valid calligraphy images."
            if total_khat and (false_rejection / total_khat) > 0.15
            else None
        ),
        "results": results[:200],
    }
    save_json(config["NON_KHAT_DETECTION_REPORT_PATH"], report)
    save_json(config["KHAT_DETECTOR_EVALUATION_PATH"], report)
    return report
