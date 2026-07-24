"""Bangun konteks halaman Detail Perhitungan Algoritma untuk pengguna."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from flask import current_app, url_for

from models import ProbabilitasPrediksi, RiwayatKlasifikasi
from services.db_compat import normalize_class_slug
from sqlalchemy import or_


CNN_LAYERS = [
    {
        "nama": "Convolution Layer",
        "ikon": "bi-grid-3x3-gap",
        "penjelasan": (
            "Digunakan untuk mendeteksi pola visual seperti garis, lengkungan, tepi huruf, "
            "ketebalan goresan, dan bentuk khas kaligrafi Arab."
        ),
    },
    {
        "nama": "ReLU (Rectified Linear Unit)",
        "ikon": "bi-lightning",
        "penjelasan": "Mengaktifkan fitur penting dan menghilangkan nilai negatif agar model fokus pada pola yang relevan.",
    },
    {
        "nama": "Pooling Layer",
        "ikon": "bi-aspect-ratio",
        "penjelasan": "Mengurangi ukuran feature map tetapi tetap mempertahankan informasi penting dari citra.",
    },
    {
        "nama": "Flatten Layer",
        "ikon": "bi-list-ul",
        "penjelasan": "Mengubah feature map menjadi vektor satu dimensi untuk tahap klasifikasi akhir.",
    },
    {
        "nama": "Fully Connected Layer",
        "ikon": "bi-diagram-3",
        "penjelasan": "Menentukan kelas akhir jenis khat berdasarkan fitur yang telah diekstraksi CNN.",
    },
]

DEFAULT_PREPROCESSING_STEPS = [
    "Gambar diunggah oleh user",
    "Validasi format JPG, JPEG, atau PNG",
    "Resize gambar menjadi 224 × 224 piksel",
    "Konversi gambar menjadi array piksel",
    "Normalisasi nilai piksel dengan membagi 255",
    "Membentuk tensor input model dengan shape (1, 224, 224, 3)",
    "Gambar diproses menggunakan model CNN",
]


def _normalize_softmax(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    score = float(value)
    if score > 1:
        return round(score / 100.0, 6)
    return round(score, 6)


def _display_names_map() -> Dict[str, str]:
    names = dict(current_app.config.get("CLASS_DISPLAY_NAMES", {}))
    names.setdefault("diwani_jali", "Diwani Jali")
    names.setdefault("tsuluts", "Tsuluts")
    return names


def _slug_from_label(label: str) -> str:
    display = _display_names_map()
    normalized = str(label).strip().lower()
    for slug, name in display.items():
        if name.strip().lower() == normalized:
            return slug
    slug = re.sub(r"^khat\s+", "", normalized, flags=re.I)
    slug = slug.replace("'", "").replace(" ", "_").replace("-", "_")
    return normalize_class_slug(slug) or slug


def _label_for_slug(slug: str) -> str:
    display = _display_names_map()
    key = normalize_class_slug(slug) or slug
    if key in display:
        return display[key]
    from models import KelasKhat

    kelas = KelasKhat.query.filter_by(slug=key).first()
    if kelas:
        return kelas.nama_kelas
    return key.replace("_", " ").title()


def _parse_json_mapping(raw: Optional[str]) -> Dict[str, float]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    scores: Dict[str, float] = {}
    for key, value in data.items():
        if value is None:
            continue
        slug = _slug_from_label(str(key)) if not str(key).replace("_", "").isalnum() else normalize_class_slug(str(key)) or str(key)
        score = _normalize_softmax(float(value))
        if score is not None:
            scores[slug] = score
    return scores


def _scores_from_row(row: RiwayatKlasifikasi) -> Dict[str, float]:
    scores: Dict[str, float] = {}

    if row.probabilitas:
        for prob in row.probabilitas.all():
            slug = prob.kelas.slug if prob.kelas else None
            if not slug:
                continue
            score = _normalize_softmax(prob.skor_softmax)
            if score is not None:
                scores[slug] = score

    calc = row.perhitungan
    if calc and calc.rumus_json:
        for slug, score in _parse_json_mapping(calc.rumus_json).items():
            scores.setdefault(slug, score)

    extra_scores = _parse_json_mapping(row.softmax_scores_json)
    for slug, score in extra_scores.items():
        scores.setdefault(slug, score)

    legacy_slugs = ["naskhi", "riqah", "diwani", "kufi", "diwani_jali", "tsuluts"]
    for slug in legacy_slugs:
        raw = row._get_extra(f"{slug}_score")
        score = _normalize_softmax(raw)
        if score is not None and score > 0:
            scores.setdefault(slug, score)

    if not scores and row.confidence is not None and row.kelas_prediksi:
        top_slug = row.kelas_prediksi.slug
        scores[top_slug] = _normalize_softmax(row.confidence) or 0.0
        if row.kelas_top2 and row.skor_top2 is not None:
            scores[row.kelas_top2.slug] = _normalize_softmax(row.skor_top2) or 0.0

    return scores


def _probability_rows_from_scores(scores: Dict[str, float]) -> List[Dict[str, Any]]:
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    rows: List[Dict[str, Any]] = []
    for rank, (slug, score) in enumerate(ranked, start=1):
        pct = round(score * 100, 2)
        rows.append({
            "no": rank,
            "slug": slug,
            "name": _label_for_slug(slug),
            "softmax": round(score, 4),
            "percent": pct,
            "rank": rank,
        })
    return rows


def _confidence_category(pct: Optional[float]) -> Dict[str, str]:
    if pct is None:
        return {"label": "Tidak Yakin", "rule": "Confidence tidak tersedia."}
    if pct >= 80:
        return {"label": "Sangat Yakin", "rule": "≥ 80% → prediksi sangat kuat."}
    if pct >= 60:
        return {"label": "Cukup Yakin", "rule": "60%–79.99% → prediksi cukup dapat dipercaya."}
    if pct >= 40:
        return {"label": "Kurang Yakin", "rule": "40%–59.99% → disarankan gambar lebih jelas."}
    return {"label": "Tidak Yakin", "rule": "< 40% → hasil tidak stabil."}


def _margin_category(margin_pct: Optional[float]) -> Dict[str, str]:
    if margin_pct is None:
        return {"label": "—", "explanation": "Margin tidak tersedia."}
    if margin_pct >= 20:
        return {"label": "Separasi Kuat", "explanation": "Selisih Top-1 dan Top-2 besar — prediksi dominan."}
    if margin_pct >= 10:
        return {"label": "Separasi Sedang", "explanation": "Prediksi utama lebih tinggi, tetapi masih ada alternatif."}
    return {"label": "Prediksi Ambigu", "explanation": "Selisih kecil — review manual disarankan."}


def build_calculation_walkthrough(
    row: RiwayatKlasifikasi,
    image: Dict[str, Any],
    preprocessing: Dict[str, Any],
    prob_rows: List[Dict[str, Any]],
    decision: Dict[str, Any],
) -> List[Dict[str, Any]]:
    top = prob_rows[0] if prob_rows else None
    second = prob_rows[1] if len(prob_rows) > 1 else None
    conf_pct = decision.get("confidence_pct")
    top2_pct = decision.get("second_pct")
    margin_pct = decision.get("margin_pct")
    conf_cat = _confidence_category(conf_pct)
    margin_cat = _margin_category(margin_pct)
    model_name = row.model.nama_model if row.model else (row.model_name or "CNN Transfer Learning")

    steps: List[Dict[str, Any]] = [
        {
            "step": 1,
            "title": "Masukan Citra",
            "narrative": (
                f"Sistem menerima file <strong>{image.get('filename') or '—'}</strong>. "
                f"Ukuran asli: <strong>{preprocessing.get('original_size', '—')}</strong>. "
                "Citra divalidasi formatnya sebelum masuk ke tahap preprocessing."
            ),
            "bullets": [
                f"Format file: {image.get('format', '—')}",
                f"Ukuran file: {image.get('file_size_kb') or '—'} KB" if image.get("file_size_kb") else "Ukuran file: —",
                "Format didukung: JPG, JPEG, PNG",
            ],
            "formula": None,
            "calculation": None,
            "result": "Citra diterima sebagai input klasifikasi.",
        },
        {
            "step": 2,
            "title": "Preprocessing & Pembentukan Tensor",
            "narrative": (
                f"Citra diubah ke RGB, diresize ke <strong>{preprocessing.get('resized_size', '224 × 224 px')}</strong>, "
                f"dinormalisasi ({preprocessing.get('normalization', 'pixel / 255')}), "
                f"lalu dibentuk menjadi tensor <code>{preprocessing.get('tensor_shape', '(1, 224, 224, 3)')}</code>."
            ),
            "bullets": [step["text"] for step in preprocessing.get("steps", [])[:7]],
            "formula": "Tensor Input = normalize(resize(RGB(citra)))",
            "calculation": (
                f"{preprocessing.get('original_size', '—')} → {preprocessing.get('resized_size', '224 × 224 px')} "
                f"→ {preprocessing.get('tensor_shape', '(1, 224, 224, 3)')}"
            ),
            "result": "Tensor input siap untuk inferensi model CNN.",
        },
        {
            "step": 3,
            "title": "Inferensi Model CNN",
            "narrative": (
                f"Model <strong>{model_name}</strong> memproses tensor input melalui lapisan convolution, "
                "ReLU, pooling, flatten, dan fully connected untuk menghasilkan skor mentah tiap kelas."
            ),
            "bullets": [layer["nama"] + ": " + layer["penjelasan"] for layer in CNN_LAYERS],
            "formula": "Logit_k = f_CNN(tensor_input)_k",
            "calculation": "Output mentah model diubah menjadi distribusi probabilitas dengan fungsi Softmax.",
            "result": f"Model menghasilkan {len(prob_rows)} skor probabilitas kelas.",
        },
    ]

    if prob_rows:
        softmax_lines = [
            f"{item['name']}: {item['softmax']:.4f} × 100 = {item['percent']:.2f}%"
            for item in prob_rows
        ]
        steps.append({
            "step": 4,
            "title": "Perhitungan Probabilitas Softmax",
            "narrative": (
                "Setiap skor softmax dikalikan 100 untuk mendapatkan persentase probabilitas. "
                "Jumlah probabilitas seluruh kelas ≈ 100%."
            ),
            "bullets": softmax_lines,
            "formula": "Probabilitas (%) = Skor Softmax × 100",
            "calculation": " ; ".join(softmax_lines[:4]),
            "result": f"Total {len(prob_rows)} kelas terhitung.",
        })
        rank_lines = [f"Peringkat {item['rank']}: {item['name']} — {item['percent']:.2f}%" for item in prob_rows]
        argmax_values = ", ".join(f"{item['percent']:.2f}%" for item in prob_rows)
        steps.append({
            "step": 5,
            "title": "Ranking & Prediksi Utama (argmax)",
            "narrative": "Probabilitas diurutkan dari tertinggi ke terendah. Kelas tertinggi menjadi prediksi utama (Top-1).",
            "bullets": rank_lines,
            "formula": "Predicted Class = argmax(probabilitas)",
            "calculation": f"argmax([{argmax_values}]) = {decision.get('top_class', '—')}",
            "result": f"Prediksi utama: <strong>{decision.get('top_class', '—')}</strong>.",
        })
        steps.append({
            "step": 6,
            "title": "Perhitungan Confidence",
            "narrative": "Confidence diambil dari probabilitas kelas Top-1 — mengukur seberapa yakin model terhadap prediksinya.",
            "bullets": [
                f"Top-1 Score = {conf_pct:.2f}%" if conf_pct is not None else "Top-1 Score = —",
                f"Kategori: {conf_cat['label']}",
                conf_cat["rule"],
            ],
            "formula": "Confidence = Probabilitas Top-1",
            "calculation": (
                f"Confidence = {conf_pct:.2f}% → {conf_cat['label']}"
                if conf_pct is not None
                else "Confidence tidak tersedia."
            ),
            "result": conf_cat["rule"],
        })
        steps.append({
            "step": 7,
            "title": "Perhitungan Top-2 Margin",
            "narrative": "Margin mengukur jarak antara prediksi utama dan alternatif terdekat.",
            "bullets": [
                f"Top-1: {decision.get('top_class', '—')} = {conf_pct:.2f}%" if conf_pct is not None else f"Top-1: {decision.get('top_class', '—')}",
                (
                    f"Top-2: {decision.get('second_class', '—')} = {top2_pct:.2f}%"
                    if top2_pct is not None
                    else f"Top-2: {decision.get('second_class', '—')}"
                ),
                margin_cat["explanation"],
            ],
            "formula": "Top-2 Margin = Top-1 Score − Top-2 Score",
            "calculation": (
                f"Top-2 Margin = {conf_pct:.2f}% − {top2_pct:.2f}% = {margin_pct:.2f}%"
                if conf_pct is not None and top2_pct is not None and margin_pct is not None
                else "Margin tidak dapat dihitung."
            ),
            "result": (
                f"{margin_cat['label']} (margin = {margin_pct:.2f}%)."
                if margin_pct is not None
                else margin_cat["explanation"]
            ),
        })

    steps.append({
        "step": len(steps) + 1,
        "title": "Keputusan Akhir Klasifikasi",
        "narrative": decision.get("final_text", "Keputusan berdasarkan probabilitas tertinggi model CNN."),
        "bullets": [
            f"Kelas tertinggi: {decision.get('top_class', '—')}",
            f"Confidence: {conf_pct:.2f}%" if conf_pct is not None else "Confidence: —",
            f"Level confidence: {conf_cat['label']}",
        ],
        "formula": "Keputusan Akhir = kelas dengan probabilitas tertinggi",
        "calculation": decision.get("final_text"),
        "result": f"Citra diklasifikasikan sebagai <strong>{decision.get('top_class', '—')}</strong>.",
    })
    return steps


def build_formula_summary(prob_rows: List[Dict[str, Any]], decision: Dict[str, Any]) -> List[Dict[str, str]]:
    formulas: List[Dict[str, str]] = [
        {
            "title": "Konversi Softmax ke Persentase",
            "code": "Probabilitas (%) = Skor Softmax × 100",
            "example": "; ".join(
                f"{row['name']}: {row['softmax']} × 100 = {row['percent']}%"
                for row in prob_rows[:4]
            ) or "—",
        },
        {
            "title": "Confidence",
            "code": "Confidence = Probabilitas Top-1",
            "example": (
                f"Confidence = {decision['confidence_pct']}%"
                if decision.get("confidence_pct") is not None
                else "—"
            ),
        },
        {
            "title": "Top-2 Margin",
            "code": "Top-2 Margin = Top-1 Score − Top-2 Score",
            "example": (
                f"{decision['confidence_pct']}% − {decision['second_pct']}% = {decision['margin_pct']}%"
                if decision.get("confidence_pct") is not None
                and decision.get("second_pct") is not None
                and decision.get("margin_pct") is not None
                else "—"
            ),
        },
        {
            "title": "Prediksi Utama",
            "code": "Predicted Class = argmax(probabilitas)",
            "example": (
                f"argmax → {decision.get('top_class', '—')}"
                if prob_rows
                else "—"
            ),
        },
    ]
    return formulas


def confidence_level_info(confidence: Optional[float]) -> Dict[str, str]:
    if confidence is None:
        return {"label": "Tidak Yakin", "tone": "danger", "pct": None}
    pct = confidence * 100 if confidence <= 1 else float(confidence)
    if pct >= 80:
        return {"label": "Sangat Yakin", "tone": "success", "pct": round(pct, 2)}
    if pct >= 60:
        return {"label": "Cukup Yakin", "tone": "info", "pct": round(pct, 2)}
    if pct >= 40:
        return {"label": "Kurang Yakin", "tone": "warn", "pct": round(pct, 2)}
    return {"label": "Tidak Yakin", "tone": "danger", "pct": round(pct, 2)}


def _parse_preprocessing_steps(raw: Optional[str]) -> List[Dict[str, Any]]:
    if not raw:
        return [{"step": i + 1, "text": text} for i, text in enumerate(DEFAULT_PREPROCESSING_STEPS)]
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return [{"step": i + 1, "text": text} for i, text in enumerate(DEFAULT_PREPROCESSING_STEPS)]

    if isinstance(data, dict):
        items = []
        for key in sorted(data.keys()):
            items.append({"step": len(items) + 1, "text": data[key]})
        return items or [{"step": i + 1, "text": t} for i, t in enumerate(DEFAULT_PREPROCESSING_STEPS)]

    if isinstance(data, list):
        steps = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                steps.append({
                    "step": item.get("urutan") or item.get("step") or i + 1,
                    "text": item.get("langkah") or item.get("text") or str(item),
                })
            else:
                steps.append({"step": i + 1, "text": str(item)})
        return steps

    return [{"step": i + 1, "text": text} for i, text in enumerate(DEFAULT_PREPROCESSING_STEPS)]


def _image_url(row: RiwayatKlasifikasi) -> Optional[str]:
    if not row.path_upload:
        return None
    rel = row.path_upload.replace("static/", "", 1) if row.path_upload.startswith("static/") else row.path_upload
    return url_for("static", filename=rel)


def _file_meta(row: RiwayatKlasifikasi) -> Dict[str, Any]:
    filename = row.nama_file_upload or ""
    ext = filename.rsplit(".", 1)[-1].upper() if "." in filename else "—"
    calc = row.perhitungan
    width = calc.lebar_asli if calc and calc.lebar_asli else row._get_extra("original_width")
    height = calc.tinggi_asli if calc and calc.tinggi_asli else row._get_extra("original_height")
    file_size = None
    if row.path_upload:
        abs_path = os.path.join(current_app.config["BASE_DIR"], row.path_upload)
        if os.path.isfile(abs_path):
            file_size = round(os.path.getsize(abs_path) / 1024, 2)
    return {
        "filename": filename,
        "format": ext,
        "width": width,
        "height": height,
        "file_size_kb": file_size,
        "preview_url": _image_url(row),
    }


def _probability_rows(row: RiwayatKlasifikasi) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    probs = (
        row.probabilitas.order_by(ProbabilitasPrediksi.urutan_ranking.asc()).all()
        if row.probabilitas
        else []
    )
    if probs:
        display_names = current_app.config.get("CLASS_DISPLAY_NAMES", {})
        for prob in probs:
            slug = prob.kelas.slug if prob.kelas else ""
            name = prob.kelas.nama_kelas if prob.kelas else display_names.get(slug, slug)
            score = float(prob.skor_softmax or 0)
            pct = float(prob.persentase_probabilitas or (score * 100))
            rows.append({
                "no": prob.urutan_ranking,
                "slug": slug,
                "name": name,
                "softmax": round(score, 4),
                "percent": round(pct, 2),
                "rank": prob.urutan_ranking,
            })
        return rows
    return _probability_rows_from_scores(_scores_from_row(row))


def _decision_block(row: RiwayatKlasifikasi, prob_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    top = prob_rows[0] if prob_rows else None
    second = prob_rows[1] if len(prob_rows) > 1 else None
    conf = row.confidence
    conf_pct = (conf * 100 if conf and conf <= 1 else conf) if conf is not None else (top["percent"] if top else None)
    margin = row.margin_top2
    if margin is None and top and second:
        margin = round(top["percent"] - second["percent"], 2)
    top_name = top["name"] if top else (row.keputusan_akhir or "—")
    second_name = second["name"] if second else (row.kelas_top2.nama_kelas if row.kelas_top2 else "—")
    second_pct = second["percent"] if second else (
        (row.skor_top2 * 100 if row.skor_top2 and row.skor_top2 <= 1 else row.skor_top2) if row.skor_top2 else None
    )
    final_text = (
        f"Citra diklasifikasikan sebagai {top_name} karena memiliki nilai probabilitas tertinggi "
        f"({conf_pct:.2f}%) dibandingkan kelas lainnya."
        if top_name and conf_pct is not None
        else "Keputusan klasifikasi berdasarkan probabilitas tertinggi model CNN."
    )
    if row.perhitungan and row.perhitungan.catatan_perhitungan:
        final_text = row.perhitungan.catatan_perhitungan
    return {
        "top_class": top_name,
        "confidence_pct": round(conf_pct, 2) if conf_pct is not None else None,
        "second_class": second_name,
        "second_pct": round(second_pct, 2) if second_pct is not None else None,
        "margin_pct": round(margin, 2) if margin is not None else None,
        "final_text": final_text,
    }


def _user_owns_row(row: RiwayatKlasifikasi, user_id: Optional[int]) -> bool:
    if user_id is None:
        return True
    if row.id_pengguna is None and current_app.config.get("APP_SIMPLE_MODE"):
        return True
    return row.id_pengguna == user_id


def user_riwayat_query(user_id: Optional[int] = None):
    query = RiwayatKlasifikasi.query
    if user_id is None:
        return query
    if current_app.config.get("APP_SIMPLE_MODE"):
        return query.filter(
            or_(RiwayatKlasifikasi.id_pengguna == user_id, RiwayatKlasifikasi.id_pengguna.is_(None))
        )
    return query.filter_by(id_pengguna=user_id)


def resolve_detail_riwayat_id(user_id: Optional[int], preferred_id: Optional[int] = None) -> Optional[int]:
    """Cari ID riwayat yang boleh dibuka user (session terakhir atau riwayat terbaru)."""
    if preferred_id:
        row = RiwayatKlasifikasi.query.get(preferred_id)
        if row and _user_owns_row(row, user_id):
            return row.id
    latest = user_riwayat_query(user_id).order_by(RiwayatKlasifikasi.dibuat_pada.desc()).first()
    return latest.id if latest else None


def get_riwayat_for_user(riwayat_id: int, user_id: Optional[int] = None) -> Optional[RiwayatKlasifikasi]:
    row = RiwayatKlasifikasi.query.get(riwayat_id)
    if not row:
        return None
    if not _user_owns_row(row, user_id):
        return None
    return row


def build_classification_detail_context(
    riwayat_id: int,
    user_id: Optional[int] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Kembalikan (context, error_message).
    error_message: 'not_found' | 'forbidden' | None
    """
    row = RiwayatKlasifikasi.query.get(riwayat_id)
    if not row:
        return None, "not_found"
    if not _user_owns_row(row, user_id):
        return None, "forbidden"

    calc = row.perhitungan
    prob_rows = _probability_rows(row)
    has_prediction = bool(
        prob_rows
        or calc
        or row.kelas_prediksi
        or row.keputusan_akhir
        or row.confidence is not None
    )

    if not has_prediction:
        return {
            "available": False,
            "riwayat_id": riwayat_id,
            "message": "Detail perhitungan belum tersedia untuk data ini.",
        }, None

    conf_info = confidence_level_info(row.confidence)
    decision = _decision_block(row, prob_rows)
    image = _file_meta(row)

    chart_labels = [p["name"] for p in prob_rows] or [decision["top_class"]]
    chart_values = [p["percent"] for p in prob_rows] or ([decision["confidence_pct"]] if decision.get("confidence_pct") is not None else [0])

    low_warning = None
    if conf_info["pct"] is not None and conf_info["pct"] < 60:
        low_warning = (
            "Hasil klasifikasi memiliki confidence rendah. "
            "Disarankan menggunakan gambar yang lebih jelas."
        )

    preprocessing = {
        "steps": _parse_preprocessing_steps(calc.langkah_preprocessing_json if calc else None),
        "original_size": (
            f"{calc.lebar_asli} × {calc.tinggi_asli} px"
            if calc and calc.lebar_asli and calc.tinggi_asli
            else (
                f"{image['width']} × {image['height']} px"
                if image.get("width") and image.get("height")
                else "—"
            )
        ),
        "resized_size": (
            f"{calc.lebar_proses} × {calc.tinggi_proses} px"
            if calc and calc.lebar_proses
            else "224 × 224 px"
        ),
        "tensor_shape": (calc.bentuk_tensor if calc else "(1, 224, 224, 3)") or "(1, 224, 224, 3)",
        "model_input": (calc.ukuran_input_model if calc else "224x224") or "224x224",
        "normalization": "pixel / 255.0 (normalisasi channel sesuai arsitektur CNN)",
    }

    walkthrough = build_calculation_walkthrough(row, image, preprocessing, prob_rows, decision)
    formulas = build_formula_summary(prob_rows, decision)

    return {
        "available": True,
        "riwayat_id": riwayat_id,
        "image": image,
        "summary": {
            "predicted": decision["top_class"],
            "confidence_pct": decision["confidence_pct"],
            "confidence_level": conf_info["label"],
            "confidence_tone": conf_info["tone"],
            "low_confidence_warning": low_warning,
            "created_at": row.dibuat_pada.strftime("%Y-%m-%d %H:%M:%S") if row.dibuat_pada else "—",
        },
        "preprocessing": preprocessing,
        "cnn_layers": CNN_LAYERS,
        "probabilities": prob_rows,
        "chart": {"labels": chart_labels, "values": chart_values},
        "decision": decision,
        "calculation_notes": calc.catatan_perhitungan if calc else decision.get("final_text"),
        "walkthrough": walkthrough,
        "formulas": formulas,
    }, None
