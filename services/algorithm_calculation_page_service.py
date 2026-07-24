"""Context builder for the standalone Perhitungan Algoritma documentation page."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from flask import current_app, url_for

from models import ClassificationResult
from services.algorithm_calculation_service import safe_build_algorithm_calculation
from services.cnn_calculation_page_service import (
    build_class_dataset_rows,
    build_evaluation_metrics,
    build_split_math_summary,
)
from services.dashboard_ui_service import build_dataset_insights
from services.dataset_path_service import get_dataset_inventory
from services.dataset_readiness_service import get_dataset_readiness
from services.evaluation_dataset_service import load_split_report, resolve_evaluation_counts
from services.evaluation_ui_service import build_confusion_interpretation, display_class
from services.external_assets_service import should_use_keras_model
from services.khat_detector_service import get_thresholds, is_detector_available
from services.metrics_service import ACCURACY_TARGET
from services.model_context_service import get_active_model_path, get_model_context, load_model_metadata
from services.prediction_audit_service import load_confusion_pair_report, load_misclassification_report
from services.teachable_machine_service import get_tm_display_info, is_teachable_machine_available


EXAMPLE_SOFTMAX = [
    {"key": "diwani", "label": "Diwani", "softmax_score": 0.0210, "percent": 2.10, "rank": 3},
    {"key": "diwani_jali", "label": "Diwani Jali", "softmax_score": 0.0152, "percent": 1.52, "rank": 4},
    {"key": "naskhi", "label": "Naskhi", "softmax_score": 0.0612, "percent": 6.12, "rank": 2},
    {"key": "tsuluts", "label": "Tsuluts", "softmax_score": 0.9026, "percent": 90.26, "rank": 1},
]

EXAMPLE_SAMPLE = {
    "is_example": True,
    "filename": "nkgambar.jpg",
    "original_size": "967 × 701 px",
    "model_input": "224 × 224 px",
    "predicted_class": "Tsuluts",
    "predicted_display": "Tsuluts",
    "top2_class": "Naskhi",
    "top2_display": "Naskhi",
    "confidence_pct": 90.26,
    "top2_pct": 6.12,
    "margin_pct": 84.14,
    "confidence_level": "Strong Prediction",
    "expected_class": "Unknown",
    "validation_status": "No Expected Class Provided",
    "final_decision": "Accepted with Caution",
    "interpretation_id": (
        "Model sangat yakin memprediksi Tsuluts karena probabilitas dominan. "
        "Namun karena tidak ada kelas pembanding, hasil perlu ditafsirkan dengan hati-hati "
        "dan dapat direview secara manual."
    ),
    "softmax_table": EXAMPLE_SOFTMAX,
}


def _load_eval_json(config: Dict) -> Dict:
    path = config.get("EVALUATION_RESULT_PATH")
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def get_model_info(config: Optional[Dict] = None) -> Dict[str, Any]:
    if config is None:
        config = current_app.config
    labels = config.get("CLASS_LABELS", [])
    display_labels = [display_class(label) for label in labels]
    tm = get_tm_display_info() if is_teachable_machine_available(config) and not should_use_keras_model(config) else None
    if tm:
        return {
            "source": "Teachable Machine",
            "runtime": "TensorFlow.js",
            "input_size": "224 × 224 px",
            "tensor_shape": "[1, 224, 224, 3]",
            "output_count": len(labels),
            "model_file": url_for("static", filename="model/model.json"),
            "metadata_file": url_for("static", filename="model/metadata.json"),
            "weights_file": url_for("static", filename="model/weights.bin"),
            "labels": display_labels,
            "label_keys": labels,
            "model_name": tm.get("model_name") or "Khat Classifier",
            "architecture": "teachable_machine",
            "architecture_label": "Teachable Machine",
            "is_teachable_machine": True,
            "preprocessing": "Teachable Machine (piksel / 127.5 − 1)",
        }

    ctx = get_model_context(config)
    meta = load_model_metadata(config)
    model_path = get_active_model_path(config)
    arch_key = ctx.get("model_architecture_key") or meta.get("model_architecture_key") or "efficientnetb0"
    arch_label = ctx.get("model_architecture") or meta.get("architecture_label") or "EfficientNetB0 Transfer Learning"
    preprocessing = meta.get("preprocessing") or arch_key
    if preprocessing in ("teachable_machine", "keras_h5", "external_h5"):
        preprocess_label = "Normalisasi (piksel / 127.5 − 1)"
    elif preprocessing == "vgg16":
        preprocess_label = "VGG16 ImageNet mean subtraction (BGR)"
    else:
        preprocess_label = "EfficientNet preprocess_input"

    return {
        "source": arch_label,
        "runtime": "TensorFlow / Keras",
        "input_size": "224 × 224 px",
        "tensor_shape": "[1, 224, 224, 3]",
        "output_count": len(labels),
        "model_file": os.path.basename(model_path) if model_path else config.get("MODEL_PATH", ""),
        "metadata_file": config.get("MODEL_METADATA_PATH", ""),
        "weights_file": "",
        "labels": display_labels,
        "label_keys": labels,
        "model_name": arch_label,
        "architecture": arch_key,
        "architecture_label": arch_label,
        "is_teachable_machine": False,
        "preprocessing": preprocess_label,
        "validation_accuracy_pct": (
            round(float(ctx.get("validation_accuracy") or meta.get("validation_accuracy") or 0) * 100, 2)
            if (ctx.get("validation_accuracy") or meta.get("validation_accuracy"))
            else None
        ),
    }


def get_thresholds_block(config: Optional[Dict] = None) -> Dict[str, Any]:
    if config is None:
        config = current_app.config
    det = get_thresholds(config)
    return {
        "strong_confidence": 85,
        "moderate_confidence": 70,
        "weak_confidence": 60,
        "khat_accept": round(det.get("accept", 0.70) * 100, 2),
        "khat_reject": round(det.get("reject", 0.50) * 100, 2),
        "khat_borderline": round(det.get("borderline", 0.65) * 100, 2),
        "strong_margin": 15,
        "minimum_margin": 10,
        "detector_available": is_detector_available(config),
    }


def get_dataset_summary(config: Optional[Dict] = None) -> Dict[str, Any]:
    if config is None:
        config = current_app.config
    labels = list(config.get("CLASS_LABELS", []))
    inventory = get_dataset_inventory(config, force_refresh=True)
    split_report = load_split_report(config)
    class_rows = build_class_dataset_rows(config)

    per_class_keys = {row["slug"]: row["raw_count"] for row in class_rows}
    per_class_display = {row["display"]: row["raw_count"] for row in class_rows}
    per_class_split = {
        row["display"]: {
            "train": row["train_count"],
            "validation": row["validation_count"],
            "test": row["test_count"],
        }
        for row in class_rows
    }

    dataset_totals = {
        "raw": inventory.get("raw_total") or sum(per_class_keys.values()),
        "train": inventory.get("train") or sum(r["train_count"] for r in class_rows),
        "validation": inventory.get("validation") or sum(r["validation_count"] for r in class_rows),
        "test": inventory.get("test") or sum(r["test_count"] for r in class_rows),
        "processed": inventory.get("processed_total") or 0,
    }
    if not dataset_totals["processed"]:
        dataset_totals["processed"] = (
            dataset_totals["train"] + dataset_totals["validation"] + dataset_totals["test"]
        )

    split_math = build_split_math_summary(dataset_totals)
    readiness = get_dataset_readiness(config)
    distribution = {row["slug"]: row["raw_count"] for row in class_rows}
    insights = build_dataset_insights(distribution, labels)

    return {
        "total_database": sum(r.get("db_count", 0) for r in class_rows),
        "total_raw": dataset_totals["raw"],
        "processed_total": dataset_totals["processed"],
        "per_class": per_class_display,
        "per_class_keys": per_class_keys,
        "per_class_split": per_class_split,
        "class_rows": class_rows,
        "imbalance_warning": insights.get("imbalance_warning") or insights.get("warning"),
        "balance_status": "balanced" if inventory.get("is_balanced") else insights.get("balance_status", "unknown"),
        "recommended_min": 300,
        "recommended_max": 500,
        "split_ready": readiness.get("is_ready", False),
        "train_count": dataset_totals["train"],
        "validation_count": dataset_totals["validation"],
        "test_count": dataset_totals["test"],
        "train_data_count": split_math.get("train_data_count"),
        "train_data_pct": split_math.get("train_data_pct"),
        "test_data_pct": split_math.get("test_data_pct"),
        "split_math": split_math,
        "split_report": split_report,
        "split_ratio": split_report.get("split_ratio") or "80/20 (train+validation / test holdout)",
        "effective_ratio": split_report.get("effective_ratio") or "68/12/20 train/validation/test",
    }


def get_evaluation_summary(config: Optional[Dict] = None) -> Dict[str, Any]:
    if config is None:
        config = current_app.config
    eval_json = _load_eval_json(config)
    if not eval_json:
        return {"available": False}
    acc = eval_json.get("accuracy")
    test_samples = eval_json.get("test_samples")
    misclassified = eval_json.get("misclassified_count") or eval_json.get("incorrect_predictions")
    correct = eval_json.get("correct_predictions")
    if correct is None and test_samples is not None and misclassified is not None:
        correct = test_samples - misclassified

    def _pct(key: str) -> Optional[float]:
        val = eval_json.get(key)
        return round(float(val) * 100, 2) if val is not None else None

    return {
        "available": True,
        "accuracy": acc,
        "accuracy_pct": _pct("accuracy"),
        "precision": eval_json.get("precision"),
        "recall": eval_json.get("recall"),
        "f1_score": eval_json.get("f1_score"),
        "precision_pct": _pct("precision"),
        "recall_pct": _pct("recall"),
        "f1_pct": _pct("f1_score"),
        "precision_macro_pct": _pct("precision_macro"),
        "recall_macro_pct": _pct("recall_macro"),
        "f1_macro_pct": _pct("f1_macro"),
        "cohen_kappa": eval_json.get("cohen_kappa"),
        "test_samples": test_samples,
        "correct_predictions": correct,
        "incorrect_predictions": misclassified,
        "evaluated_at": eval_json.get("evaluated_at"),
        "target_pct": int(ACCURACY_TARGET * 100),
        "target_achieved": bool(acc is not None and float(acc) >= ACCURACY_TARGET),
        "accuracy_formula_text": eval_json.get("accuracy_formula"),
    }


def get_confusion_matrix_data(config: Optional[Dict] = None) -> Dict[str, Any]:
    if config is None:
        config = current_app.config
    eval_json = _load_eval_json(config)
    matrix = eval_json.get("confusion_matrix") or []
    labels = config.get("CLASS_LABELS", [])
    display_labels = [display_class(label) for label in labels]
    interpretation = build_confusion_interpretation(matrix)
    return {
        "available": bool(matrix),
        "matrix": matrix,
        "labels": display_labels,
        "label_keys": labels,
        "interpretation": interpretation,
    }


def get_error_pair_analysis(config: Optional[Dict] = None) -> Dict[str, Any]:
    if config is None:
        config = current_app.config
    pairs_report = load_confusion_pair_report()
    pairs = pairs_report.get("pairs") if pairs_report else []
    if not pairs:
        eval_json = _load_eval_json(config)
        pairs = eval_json.get("confused_pairs") or eval_json.get("pairs") or []
    formatted: List[Dict[str, Any]] = []
    for pair in pairs[:8]:
        formatted.append(
            {
                "true_class": pair.get("true_class"),
                "predicted_class": pair.get("predicted_class"),
                "true_display": pair.get("true_display") or display_class(pair.get("true_class")),
                "predicted_display": pair.get("predicted_display") or display_class(pair.get("predicted_class")),
                "count": pair.get("count", 0),
                "percentage": pair.get("percentage"),
            }
        )
    return {
        "available": bool(formatted),
        "pairs": formatted,
        "top_pair": formatted[0] if formatted else None,
    }


def get_latest_prediction_sample(config: Optional[Dict] = None) -> Dict[str, Any]:
    if config is None:
        config = current_app.config
    row = ClassificationResult.query.order_by(ClassificationResult.dibuat_pada.desc()).first()
    if not row:
        return dict(EXAMPLE_SAMPLE)
    calc = safe_build_algorithm_calculation(row)
    softmax = calc.get("softmax_table") or []
    top = softmax[0] if softmax else {}
    second = softmax[1] if len(softmax) > 1 else {}
    margin_block = calc.get("margin_calculation") or {}
    conf_block = calc.get("confidence_calculation") or {}
    decision_block = calc.get("final_decision_block") or {}
    image_meta = calc.get("summary_grid") or []
    original_size = "—"
    for item in image_meta:
        if item.get("label") in ("Original Size", "Ukuran Asli"):
            original_size = item.get("value", "—")
            break
    return {
        "is_example": False,
        "filename": row.uploaded_filename or row.filename or "—",
        "original_size": original_size,
        "model_input": calc.get("model_input_display") or "224 × 224 px",
        "predicted_class": row.predicted_class,
        "predicted_display": display_class(row.predicted_class),
        "top2_class": second.get("key") or row.top_2_class,
        "top2_display": second.get("label") or display_class(row.top_2_class),
        "confidence_pct": top.get("percent") or round((row.confidence or 0) * 100, 2),
        "top2_pct": second.get("percent") or (round((row.top_2_score or 0) * 100, 2) if row.top_2_score and row.top_2_score <= 1 else row.top_2_score),
        "margin_pct": margin_block.get("margin") or row.top2_margin or (
            round((top.get("percent") or 0) - (second.get("percent") or 0), 2) if second else None
        ),
        "confidence_level": conf_block.get("category") or calc.get("confidence_level") or calc.get("confidence_category"),
        "expected_class": display_class(row.expected_class) if row.expected_class else "Unknown",
        "validation_status": row.validation_status or calc.get("validation_status") or "—",
        "final_decision": decision_block.get("final_decision") or row.final_decision,
        "interpretation_id": calc.get("interpretation_id") or calc.get("interpretation"),
        "softmax_table": softmax or EXAMPLE_SOFTMAX,
        "created_at": row.created_at.strftime("%Y-%m-%d %H:%M") if row.created_at else None,
        "khat_pct": round(float(row.khat_probability) * 100, 2) if row.khat_probability is not None else None,
        "pipeline_steps": calc.get("pipeline_steps") or [],
        "summary_grid": calc.get("summary_grid") or [],
        "calc": calc,
    }


def get_workflow_steps(model_info: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    info = model_info or {}
    is_tm = info.get("is_teachable_machine", False)
    runtime_label = info.get("runtime") or ("TensorFlow.js" if is_tm else "TensorFlow / Keras")
    model_label = info.get("architecture_label") or info.get("source") or "CNN / Keras"
    train_ref = "pelatihan model" if is_tm else "pelatihan EfficientNetB0 transfer learning"
    infer_title = (
        "Inferensi Model TensorFlow.js"
        if is_tm
        else "Inferensi Model Keras (EfficientNetB0)"
    )
    infer_desc = (
        "Tensor dimasukkan ke model Teachable Machine yang diekspor sebagai model.json + weights.bin."
        if is_tm
        else f"Tensor dimasukkan ke model {model_label} yang disimpan sebagai file .keras."
    )
    return [
        {
            "step": 1,
            "title": "Unggah / Pilih Gambar",
            "description": "Pengguna memilih citra kaligrafi Arab. Sistem menerima file dan memeriksa apakah file dapat dibaca.",
            "detail": "Tahap ini hanya memastikan gambar valid sebagai input — belum ada prediksi kelas.",
        },
        {
            "step": 2,
            "title": "Validasi Format Gambar",
            "description": "Sistem memeriksa ekstensi (JPG, PNG, WEBP), ukuran file, dan integritas citra.",
            "detail": "Jika file rusak atau format tidak didukung, proses dihentikan sebelum preprocessing.",
        },
        {
            "step": 3,
            "title": "Preprocessing Citra",
            "description": "Gambar dikonversi ke RGB, dirotasi sesuai EXIF bila ada, lalu disesuaikan ke format yang dibutuhkan model.",
            "detail": "Preprocessing membuat semua input seragam sehingga model tidak bingung karena perbedaan ukuran atau warna.",
        },
        {
            "step": 4,
            "title": "Resize ke 224 × 224 piksel",
            "description": f"Citra diubah ukurannya agar sama persis dengan input yang digunakan saat {train_ref}.",
            "detail": "Ukuran 224 × 224 piksel adalah syarat wajib model — bukan pilihan opsional.",
        },
        {
            "step": 5,
            "title": "Konversi ke Tensor",
            "description": "Nilai piksel diubah menjadi tensor numerik berbentuk [1, 224, 224, 3].",
            "detail": "Angka 1 = satu gambar; 224 × 224 = lebar × tinggi; 3 = channel warna RGB.",
        },
        {
            "step": 6,
            "title": infer_title,
            "description": infer_desc,
            "detail": "Model mengembalikan empat skor mentah — satu skor untuk setiap kelas khat.",
        },
        {
            "step": 7,
            "title": "Softmax → Probabilitas",
            "description": "Skor mentah diubah menjadi probabilitas (%) sehingga total keempat kelas = 100%.",
            "detail": "Rumus: Probabilitas (%) = Skor Softmax × 100.",
        },
        {
            "step": 8,
            "title": "Ranking Kelas",
            "description": "Keempat probabilitas diurutkan dari tertinggi ke terendah.",
            "detail": "Kelas peringkat 1 (Top-1) menjadi prediksi utama; peringkat 2 (Top-2) dipakai untuk menghitung margin.",
        },
        {
            "step": 9,
            "title": "Hitung Confidence",
            "description": "Confidence = probabilitas kelas Top-1. Semakin tinggi, semakin yakin model.",
            "detail": "≥85% = kuat · 70–84% = sedang · 60–69% = lemah · <60% = rendah.",
        },
        {
            "step": 10,
            "title": "Hitung Top-2 Margin",
            "description": "Margin = selisih probabilitas Top-1 dan Top-2.",
            "detail": "Margin besar berarti prediksi jelas; margin kecil berarti model ragu antara dua kelas.",
        },
        {
            "step": 11,
            "title": "Validasi Label Pembanding",
            "description": "Jika ada expected class, prediksi dibandingkan: cocok atau tidak cocok.",
            "detail": "Tanpa label pembanding, validasi otomatis tidak dapat menyatakan benar/salah.",
        },
        {
            "step": 12,
            "title": "Keputusan Akhir",
            "description": "Sistem menggabungkan confidence, margin, validasi, dan hasil deteksi Khat menjadi satu keputusan.",
            "detail": "Contoh: Diterima · Diterima dengan Catatan · Review Manual · Ditolak Non-Khat.",
        },
        {
            "step": 13,
            "title": "Simpan ke Riwayat",
            "description": "Semua hasil perhitungan disimpan ke database riwayat klasifikasi.",
            "detail": "Riwayat dapat dibuka kembali untuk audit penelitian atau laporan skripsi.",
        },
        {
            "step": 14,
            "title": "Tampilkan Hasil",
            "description": "Pengguna melihat kelas prediksi, probabilitas, confidence, margin, dan interpretasi.",
            "detail": "Halaman ini (Perhitungan Algoritma) menjelaskan seluruh logika di balik tampilan tersebut.",
        },
    ]


def _confidence_category_id(confidence_pct: float) -> Dict[str, str]:
    if confidence_pct >= 85:
        return {"label": "Prediksi Kuat", "rule": f"{confidence_pct:.2f}% ≥ 85%", "tone": "ok"}
    if confidence_pct >= 70:
        return {"label": "Prediksi Sedang", "rule": f"70% ≤ {confidence_pct:.2f}% < 85%", "tone": "info"}
    if confidence_pct >= 60:
        return {"label": "Prediksi Lemah", "rule": f"60% ≤ {confidence_pct:.2f}% < 70%", "tone": "warn"}
    return {"label": "Confidence Rendah", "rule": f"{confidence_pct:.2f}% < 60%", "tone": "danger"}


def _margin_category_id(margin_pct: float) -> Dict[str, str]:
    if margin_pct >= 15:
        return {"label": "Separasi Kuat", "explanation": "Prediksi utama jauh lebih tinggi dari kelas kedua."}
    if margin_pct >= 10:
        return {"label": "Separasi Sedang", "explanation": "Prediksi utama lebih tinggi, tetapi masih ada kemungkinan kebingungan."}
    return {"label": "Prediksi Ambigu", "explanation": "Selisih Top-1 dan Top-2 kecil — review manual disarankan."}


def build_detailed_walkthrough(
    latest: Dict[str, Any],
    thresholds: Dict[str, Any],
    model_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Langkah perhitungan lengkap dengan narasi Indonesia dan angka aktual."""
    calc = latest.get("calc") or {}
    softmax = latest.get("softmax_table") or EXAMPLE_SOFTMAX
    sorted_softmax = sorted(softmax, key=lambda r: r.get("rank", 99))
    top = sorted_softmax[0] if sorted_softmax else {}
    second = sorted_softmax[1] if len(sorted_softmax) > 1 else {}
    conf_pct = float(latest.get("confidence_pct") or top.get("percent") or 0)
    top2_pct = float(latest.get("top2_pct") or (second.get("percent") if second else 0) or 0)
    margin_pct = float(latest.get("margin_pct") or round(conf_pct - top2_pct, 2))
    conf_cat = _confidence_category_id(conf_pct)
    margin_cat = _margin_category_id(margin_pct)
    margin_block = calc.get("margin_calculation") or {}
    conf_block = calc.get("confidence_calculation") or {}
    decision_block = calc.get("final_decision_block") or {}
    known_status = decision_block.get("known_class_status") or "—"

    steps: List[Dict[str, Any]] = []

    steps.append(
        {
            "step": 1,
            "title": "Masukan Citra",
            "narrative": (
                f"Sistem menerima file <strong>{latest.get('filename', '—')}</strong>. "
                f"Ukuran asli citra: <strong>{latest.get('original_size', '—')}</strong>. "
                "Sebelum masuk ke model, citra harus lolos validasi format dan preprocessing."
            ),
            "bullets": [
                "Format yang didukung: JPG, JPEG, PNG, WEBP",
                "Citra harus dapat dibuka dan dibaca oleh library gambar",
                f"Setelah preprocessing, citra diseragamkan ke {latest.get('model_input', '224 × 224 px')}",
            ],
            "formula": None,
            "calculation": None,
            "result": "Citra diterima sebagai input klasifikasi.",
        }
    )

    stage1_narrative = (
        "Tahap ini memfilter citra yang bukan kaligrafi Khat Arab. "
        "Jika citra bukan Khat, klasifikasi empat kelas tidak dilanjutkan."
    )
    if thresholds.get("detector_available") and latest.get("khat_pct") is not None:
        khat_pct = latest["khat_pct"]
        non_khat_pct = round(100 - khat_pct, 2)
        if khat_pct >= thresholds["khat_accept"]:
            stage1_result = f"Khat Probability {khat_pct:.2f}% ≥ {thresholds['khat_accept']}% → citra diterima sebagai Khat."
            stage1_status = "Khat Terkonfirmasi"
        elif khat_pct >= thresholds["khat_reject"]:
            stage1_result = (
                f"Khat Probability {khat_pct:.2f}% berada di zona ragu "
                f"({thresholds['khat_reject']}% – {thresholds['khat_accept']}%) → lanjut dengan review manual."
            )
            stage1_status = "Input Ragu"
        else:
            stage1_result = f"Khat Probability {khat_pct:.2f}% < {thresholds['khat_reject']}% → citra ditolak sebagai Non-Khat."
            stage1_status = "Non-Khat"
        steps.append(
            {
                "step": 2,
                "title": "Validasi Khat / Non-Khat (Tahap 1)",
                "narrative": stage1_narrative,
                "bullets": [
                    f"Skor detektor Khat: {khat_pct:.2f}%",
                    f"Skor Non-Khat: {non_khat_pct:.2f}% (dihitung: 100% − {khat_pct:.2f}%)",
                    f"Ambang terima: ≥ {thresholds['khat_accept']}% · Ambang tolak: < {thresholds['khat_reject']}%",
                ],
                "formula": "Non-Khat (%) = 100 − Khat (%)",
                "calculation": f"Non-Khat = 100 − {khat_pct:.2f} = {non_khat_pct:.2f}%",
                "result": f"{stage1_status}. {stage1_result}",
            }
        )
    else:
        steps.append(
            {
                "step": 2,
                "title": "Validasi Khat / Non-Khat (Tahap 1)",
                "narrative": stage1_narrative + " Detektor Khat tidak aktif atau belum tercatat untuk prediksi ini.",
                "bullets": [
                    "Sistem langsung melanjutkan ke klasifikasi empat kelas",
                    f"Ambang baku sistem: terima ≥ {thresholds['khat_accept']}%, tolak < {thresholds['khat_reject']}%",
                ],
                "formula": "Non-Khat (%) = 100 − Khat (%)",
                "calculation": None,
                "result": "Tahap 1 dilewati — lanjut ke klasifikasi jenis Khat.",
            }
        )

    steps.append(
        {
            "step": 3,
            "title": "Preprocessing & Persiapan Tensor",
            "narrative": (
                f"Citra diubah ke RGB, diresize ke <strong>{latest.get('model_input', '224 × 224 px')}</strong>, "
                "dan dinormalisasi (nilai piksel 0–1). Kemudian dibentuk menjadi tensor "
                f"<strong>{model_info.get('tensor_shape', '[1, 224, 224, 3]')}</strong>."
            ),
            "bullets": [
                "Resize wajib agar sesuai input model Teachable Machine",
                "Normalisasi membuat skala piksel konsisten antar gambar",
                "Tensor siap diinferensikan oleh runtime TensorFlow.js",
            ],
            "formula": None,
            "calculation": None,
            "result": "Tensor input siap — model dapat dijalankan.",
        }
    )

    softmax_lines = []
    for row in sorted_softmax:
        score = row.get("softmax_score", 0)
        pct = row.get("percent", 0)
        softmax_lines.append(f"{row.get('label')}: {score:.4f} × 100 = {pct:.2f}%")

    steps.append(
        {
            "step": 4,
            "title": "Output Model → Probabilitas Softmax",
            "narrative": (
                f"Model {model_info.get('source')} menghasilkan empat skor probabilitas. "
                "Setiap skor menunjukkan seberapa kuat model mengaitkan citra dengan satu kelas. "
                "Total probabilitas keempat kelas = 100%."
            ),
            "bullets": softmax_lines,
            "formula": "Probabilitas (%) = Skor Softmax × 100",
            "calculation": None,
            "result": f"Kelas tertinggi: {top.get('label')} dengan {conf_pct:.2f}%.",
        }
    )

    rank_lines = [f"Peringkat {r.get('rank')}: {r.get('label')} — {r.get('percent', 0):.2f}%" for r in sorted_softmax]
    steps.append(
        {
            "step": 5,
            "title": "Ranking & Prediksi Utama (argmax)",
            "narrative": (
                "Probabilitas diurutkan dari tertinggi ke terendah. "
                "Kelas dengan nilai tertinggi ditetapkan sebagai prediksi utama (Top-1)."
            ),
            "bullets": rank_lines,
            "formula": "Predicted Class = argmax(probabilitas)",
            "calculation": (
                "argmax(["
                + ", ".join(f"{r.get('percent', 0):.2f}%" for r in sorted_softmax)
                + f"]) = {latest.get('predicted_display', top.get('label', '—'))}"
            ),
            "result": f"Prediksi utama: {latest.get('predicted_display', '—')}.",
        }
    )

    steps.append(
        {
            "step": 6,
            "title": "Perhitungan Confidence",
            "narrative": (
                "Confidence diambil langsung dari probabilitas kelas Top-1 — bukan rata-rata atau nilai lain. "
                "Confidence menjawab pertanyaan: seberapa yakin model terhadap prediksinya?"
            ),
            "bullets": [
                f"Top-1 Score = {conf_pct:.2f}%",
                f"Aturan kategori: {conf_cat['rule']}",
                f"Kategori: {conf_block.get('category') or conf_cat['label']}",
            ],
            "formula": "Confidence = Probabilitas Top-1",
            "calculation": f"Confidence = {conf_pct:.2f}% → {conf_cat['label']}",
            "result": conf_block.get("threshold_line") or conf_cat["rule"],
        }
    )

    steps.append(
        {
            "step": 7,
            "title": "Perhitungan Top-2 Margin",
            "narrative": (
                "Margin mengukur jarak antara prediksi utama dan alternatif terdekat. "
                "Margin kecil berarti model hampir equally yakin pada dua kelas — bukan hanya satu."
            ),
            "bullets": [
                f"Top-1: {latest.get('predicted_display')} = {conf_pct:.2f}%",
                f"Top-2: {latest.get('top2_display', '—')} = {top2_pct:.2f}%",
                margin_cat["explanation"],
            ],
            "formula": margin_block.get("formula") or "Top-2 Margin = Top-1 Score − Top-2 Score",
            "calculation": margin_block.get("calculation_line")
            or f"Top-2 Margin = {conf_pct:.2f}% − {top2_pct:.2f}% = {margin_pct:.2f}%",
            "result": f"{margin_cat['label']} (margin = {margin_pct:.2f}%).",
        }
    )

    expected = latest.get("expected_class") or "Tidak tersedia"
    validation = latest.get("validation_status") or "—"
    if expected == "Unknown" or expected == "Tidak tersedia":
        val_narrative = (
            "Tidak ada label pembanding (expected class). "
            "Sistem tidak dapat otomatis menyatakan prediksi benar atau salah — hanya menampilkan hasil model."
        )
    elif validation == "Prediction Match":
        val_narrative = (
            f"Prediksi ({latest.get('predicted_display')}) sama dengan label pembanding ({expected}). "
            "Validasi otomatis: cocok."
        )
    else:
        val_narrative = (
            f"Prediksi ({latest.get('predicted_display')}) berbeda dari label pembanding ({expected}). "
            "Validasi otomatis: tidak cocok — review manual wajib."
        )

    steps.append(
        {
            "step": 8,
            "title": "Validasi Label Pembanding",
            "narrative": val_narrative,
            "bullets": [
                f"Predicted Class: {latest.get('predicted_display', '—')}",
                f"Expected Class: {expected}",
                f"Validation Status: {validation}",
            ],
            "formula": "Validasi = (Predicted Class == Expected Class)",
            "calculation": None,
            "result": validation,
        }
    )

    steps.append(
        {
            "step": 9,
            "title": "Status Kelas & Keputusan Akhir",
            "narrative": (
                "Sistem menggabungkan confidence, margin, validasi label, dan (jika ada) hasil deteksi Khat "
                "menjadi satu keputusan akhir yang dapat dipahami pengguna."
            ),
            "bullets": [
                f"Known Class Status: {known_status}",
                f"Confidence Level: {latest.get('confidence_level') or conf_cat['label']}",
                f"Top-2 Margin: {margin_pct:.2f}%",
            ],
            "formula": None,
            "calculation": None,
            "result": (
                f"<strong>{latest.get('final_decision') or decision_block.get('final_decision') or '—'}</strong>"
                f" — {decision_block.get('explanation') or latest.get('interpretation_id') or ''}"
            ),
        }
    )

    return steps


def build_evaluation_calculation_narrative(
    ev: Dict[str, Any],
    dataset_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not ev.get("available"):
        return {
            "available": False,
            "intro": "Data evaluasi belum tersedia. Jalankan evaluasi model pada menu Evaluation terlebih dahulu.",
        }
    total = ev.get("test_samples") or 0
    correct = ev.get("correct_predictions") or 0
    wrong = ev.get("incorrect_predictions") or 0
    acc_pct = ev.get("accuracy_pct")
    ds = dataset_summary or {}
    processed = ds.get("processed_total") or ds.get("total_raw") or 0
    train_n = ds.get("train_count") or 0
    val_n = ds.get("validation_count") or 0
    train_val = ds.get("train_data_count") or (train_n + val_n)

    split_lines = []
    if processed:
        split_lines.append(
            f"Dataset diproses: <strong>{processed}</strong> citra "
            f"(train {train_n} + validation {val_n} + test {total})."
        )
        split_lines.append(
            f"Data latih (80%): <strong>{train_val}</strong> citra — tidak dipakai dalam evaluasi final."
        )

    return {
        "available": True,
        "intro": (
            f"Evaluasi dilakukan pada <strong>{total}</strong> citra holdout test set "
            f"(±20% dari {processed or 'total'} citra diproses). "
            f"Dari {total} citra uji, <strong>{correct}</strong> diprediksi benar dan "
            f"<strong>{wrong}</strong> salah."
            + (" " + " ".join(split_lines) if split_lines else "")
        ),
        "accuracy_formula": "Akurasi = (Prediksi Benar / Total Citra Uji) × 100",
        "accuracy_calculation": (
            f"Akurasi = ({correct} / {total}) × 100 = {acc_pct}%"
            if total and acc_pct is not None
            else ev.get("accuracy_formula_text")
        ),
        "precision_note": (
            f"Presisi (weighted) = {ev.get('precision_pct')}%. "
            "Presisi tinggi berarti ketika model memprediksi suatu kelas, prediksi tersebut cenderung benar."
        ),
        "recall_note": (
            f"Recall (weighted) = {ev.get('recall_pct')}%. "
            "Recall tinggi berarti model berhasil menemukan sebagian besar citra dari kelas sebenarnya."
        ),
        "f1_note": (
            f"F1-Score (weighted) = {ev.get('f1_pct')}%. "
            f"F1 macro = {ev.get('f1_macro_pct') or '—'}%. "
            "F1 menggabungkan presisi dan recall — metrik utama selain akurasi."
        ),
        "target_note": (
            f"Target riset: {ev.get('target_pct')}%. "
            + ("Target tercapai." if ev.get("target_achieved") else "Target belum tercapai — perbaikan dataset/training disarankan.")
        ),
        "split_formulas": (ds.get("split_math") or {}).get("formulas") or [],
    }


def build_algorithm_calculation_page_context(config: Optional[Dict] = None) -> Dict[str, Any]:
    if config is None:
        config = current_app.config
    latest = get_latest_prediction_sample(config)
    softmax_rows = latest.get("softmax_table") or EXAMPLE_SOFTMAX
    if latest.get("is_example"):
        latest["sample_label"] = "Contoh perhitungan"
    else:
        latest["sample_label"] = "Perhitungan prediksi terbaru"
    recent_count = ClassificationResult.query.count()
    model_info = get_model_info(config)
    thresholds = get_thresholds_block(config)
    dataset_summary = get_dataset_summary(config)
    evaluation_summary = get_evaluation_summary(config)
    class_rows = dataset_summary.get("class_rows") or build_class_dataset_rows(config)
    evaluation_metrics = build_evaluation_metrics(config, class_rows)
    eval_counts = resolve_evaluation_counts(config, _load_eval_json(config))
    return {
        "model_info": model_info,
        "thresholds": thresholds,
        "dataset_summary": dataset_summary,
        "evaluation_summary": evaluation_summary,
        "evaluation_metrics": evaluation_metrics,
        "eval_counts": eval_counts,
        "confusion_data": get_confusion_matrix_data(config),
        "error_pairs": get_error_pair_analysis(config),
        "latest_prediction": latest,
        "softmax_rows": softmax_rows,
        "workflow_steps": get_workflow_steps(model_info),
        "detailed_walkthrough": build_detailed_walkthrough(latest, thresholds, model_info),
        "eval_calculation": build_evaluation_calculation_narrative(evaluation_summary, dataset_summary),
        "recent_prediction_count": recent_count,
        "accuracy_target_pct": int(ACCURACY_TARGET * 100),
    }
