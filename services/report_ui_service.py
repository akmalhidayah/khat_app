"""UI helpers for the research summary report — metrics from real data only."""

import json
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from services.metrics_service import ACCURACY_TARGET, target_status

CLASS_LABELS = {
    "naskhi": "Naskhi",
    "diwani": "Diwani",
    "diwani_jali": "Diwani Jali",
    "tsuluts": "Tsuluts",
}

CLASS_LABELS_LIST = "Naskhi, Diwani, Diwani Jali, Tsuluts"

INVALID_PREDICTED = {None, "", "unknown", "none", "-", "rejected"}


def _fmt_dt(value) -> str:
    if not value:
        return "—"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


def is_valid_khat_prediction(record) -> bool:
    status = (record.input_status or "khat").lower()
    if status != "khat":
        return False
    pred = (record.predicted_class or "").lower().strip()
    return pred not in INVALID_PREDICTED and pred in CLASS_LABELS


def build_history_report_summary(all_records: List) -> Dict[str, Any]:
    total = len(all_records)
    valid_khat = [r for r in all_records if is_valid_khat_prediction(r)]
    rejected = [r for r in all_records if (r.input_status or "") == "non_khat"]
    uncertain = [r for r in all_records if (r.input_status or "") == "uncertain"]

    confidences = [r.confidence for r in valid_khat if r.confidence is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else None

    class_counts = Counter(r.predicted_class for r in valid_khat if r.predicted_class)
    most_predicted = None
    most_predicted_display = "Belum ada prediksi valid"
    if class_counts:
        most_predicted = class_counts.most_common(1)[0][0]
        most_predicted_display = CLASS_LABELS.get(most_predicted, most_predicted.replace("_", " ").title())

    latest_valid = valid_khat[0] if valid_khat else None
    latest_rejected = rejected[:5]

    mismatch_count = 0
    for r in valid_khat:
        expected = (r.expected_class or "").strip().lower()
        predicted = (r.predicted_class or "").strip().lower()
        if expected and predicted and expected != predicted:
            mismatch_count += 1

    return {
        "total": total,
        "valid_khat_count": len(valid_khat),
        "rejected_count": len(rejected),
        "uncertain_count": len(uncertain),
        "mismatch_count": mismatch_count,
        "avg_confidence_valid": avg_confidence,
        "most_predicted_class": most_predicted,
        "most_predicted_display": most_predicted_display,
        "latest_prediction_date": _fmt_dt(all_records[0].created_at) if all_records else "—",
        "valid_khat_records": valid_khat[:10],
        "latest_rejected": latest_rejected,
    }


def detect_class_imbalance(distribution: Dict[str, int], threshold_ratio: float = 2.0) -> bool:
    counts = [c for c in distribution.values() if c > 0]
    if len(counts) < 2:
        return False
    return max(counts) / min(counts) >= threshold_ratio


def build_target_card(latest_eval) -> Dict[str, Any]:
    acc = latest_eval.accuracy if latest_eval else None
    f1 = latest_eval.f1_score if latest_eval else None
    acc_status = target_status(acc, ACCURACY_TARGET)
    f1_status = target_status(f1, ACCURACY_TARGET)
    acc_pct = acc_status["percent"]
    f1_pct = f1_status["percent"]
    gap = round(ACCURACY_TARGET * 100 - acc_pct, 2) if acc_pct is not None else None

    if acc is None:
        tone = "muted"
        message = "Evaluasi model belum tersedia. Jalankan training dan evaluasi untuk menilai pencapaian target riset."
    elif acc >= ACCURACY_TARGET:
        tone = "success"
        message = "Model telah mencapai target akurasi riset 85%."
    elif acc >= 0.70:
        tone = "warning"
        message = (
            "Model belum mencapai target akurasi riset 85%. "
            "Diperlukan penyeimbangan dataset, audit label, augmentasi, dan retraining."
        )
    else:
        tone = "critical"
        message = (
            "Performa model masih jauh di bawah target riset. "
            "Perbaikan dataset dan retraining wajib dilakukan sebelum pelaporan final."
        )

    return {
        "target_pct": round(ACCURACY_TARGET * 100, 2),
        "accuracy_pct": acc_pct,
        "f1_pct": f1_pct,
        "achieved": acc_status["achieved"],
        "status_label": "Target Tercapai" if acc_status["achieved"] else "Belum Mencapai Target",
        "status_label_en": acc_status["label"],
        "gap_pct": gap,
        "tone": tone,
        "message": message,
    }


def build_eval_metric_rows(latest_eval) -> List[Dict[str, Any]]:
    if not latest_eval:
        return []

    rows = []
    specs = [
        ("accuracy", "Akurasi", latest_eval.accuracy, "Proporsi prediksi benar secara keseluruhan"),
        ("precision", "Precision", latest_eval.precision_score, "Keandalan kelas yang diprediksi"),
        ("recall", "Recall", latest_eval.recall_score, "Kemampuan mendeteksi kelas sebenarnya"),
        ("f1", "F1-Score", latest_eval.f1_score, "Keseimbangan precision dan recall"),
    ]
    for key, label, value, interpretation in specs:
        status = target_status(value, ACCURACY_TARGET)
        rows.append(
            {
                "key": key,
                "label": label,
                "value_pct": status["percent"],
                "interpretation": interpretation,
                "status_label": "Di bawah target" if not status["achieved"] else "Target tercapai",
                "tone": status["tone"],
            }
        )
    return rows


def build_executive_summary(
    arch_label: str,
    total_dataset: int,
    latest_eval,
    history_summary: Dict,
    target_card: Dict,
    is_tm: bool = False,
) -> str:
    acc_text = f"{target_card['accuracy_pct']}%" if target_card["accuracy_pct"] is not None else "belum tersedia"
    model_phrase = (
        "Teachable Machine Image Model"
        if is_tm
        else arch_label
    )
    target_note = (
        f"Evaluasi terakhir menunjukkan akurasi {acc_text}, sehingga model belum mencapai target riset sebesar {target_card['target_pct']}%."
        if not target_card["achieved"] and target_card["accuracy_pct"] is not None
        else (
            f"Evaluasi terakhir menunjukkan akurasi {acc_text}, memenuhi target riset sebesar {target_card['target_pct']}%."
            if target_card["achieved"]
            else "Evaluasi model belum tersedia untuk menilai pencapaian target riset."
        )
    )
    return (
        f"Laporan ini merangkum pengembangan dan evaluasi sistem klasifikasi Khat Arab berbasis {model_phrase}. "
        f"Sistem mengklasifikasikan empat jenis Khat, yaitu Naskhi, Diwani, Diwani Jali, dan Tsuluts. "
        f"Dataset berisi {total_dataset} gambar yang telah diproses. "
        f"{target_note} "
        f"Tersimpan {history_summary['total']} record prediksi "
        f"({history_summary['valid_khat_count']} prediksi valid, "
        f"{history_summary['rejected_count']} input non-Khat ditolak, "
        f"{history_summary['uncertain_count']} input tidak pasti)."
    )


def build_interpretation_sections(
    distribution: Dict[str, int],
    is_imbalanced: bool,
    latest_eval,
    history_summary: Dict,
    target_card: Dict,
) -> Dict[str, str]:
    total_dist = sum(distribution.values()) or 0
    dist_text = ", ".join(
        f"{CLASS_LABELS.get(k, k)} ({v})" for k, v in distribution.items() if v > 0
    ) or "belum tersedia"

    dataset_text = (
        f"Dataset terdiri dari {total_dist} gambar dengan distribusi: {dist_text}. "
    )
    if is_imbalanced:
        dataset_text += (
            "Terdeteksi ketidakseimbangan kelas; class weighting dan augmentasi seimbang direkomendasikan saat training."
        )
    else:
        dataset_text += "Distribusi kelas relatif seimbang untuk keempat jenis Khat."

    if latest_eval and target_card["accuracy_pct"] is not None:
        model_text = (
            f"Akurasi {target_card['accuracy_pct']}%, precision {round(latest_eval.precision_score * 100, 2)}%, "
            f"recall {round(latest_eval.recall_score * 100, 2)}%, dan F1-score {target_card['f1_pct']}%. "
            f"Status target: {target_card['status_label']}. "
        )
        if target_card["achieved"]:
            model_text += "Performa model memenuhi target riset 85%."
        else:
            model_text += (
                f"Performa model termasuk moderat namun belum memenuhi target riset "
                f"(gap {target_card['gap_pct']}%)."
            )
    else:
        model_text = "Metrik evaluasi model belum tersedia."

    avg_conf = history_summary["avg_confidence_valid"]
    avg_text = f"{round(avg_conf * 100, 2)}%" if avg_conf is not None else "—"
    history_text = (
        f"Riwayat klasifikasi: {history_summary['total']} record total, "
        f"{history_summary['valid_khat_count']} prediksi Khat valid, "
        f"{history_summary['rejected_count']} input Non-Khat ditolak, "
        f"{history_summary['uncertain_count']} input tidak pasti. "
        f"Rata-rata confidence (Khat valid): {avg_text}. "
        f"Kelas paling sering diprediksi: {history_summary['most_predicted_display']}."
    )

    if target_card["achieved"]:
        final_text = "Model siap untuk pelaporan riset final berdasarkan pencapaian target akurasi."
    elif target_card["accuracy_pct"] is not None and target_card["accuracy_pct"] >= 70:
        final_text = (
            f"Performa model dikategorikan moderat (akurasi {target_card['accuracy_pct']}%), "
            f"namun belum mencapai target riset 85%. Perbaikan dataset dan retraining diperlukan."
        )
    else:
        final_text = (
            "Model belum siap untuk pelaporan riset final. "
            "Hasil saat ini berguna untuk pengujian sistem dan analisis awal."
        )

    return {
        "dataset": dataset_text,
        "model": model_text,
        "history": history_text,
        "final": final_text,
    }


def build_recommendations(
    latest_eval,
    is_imbalanced: bool,
    training_mode_key: Optional[str] = None,
    arch_key: Optional[str] = None,
) -> List[str]:
    """Formal ordered recommendations for the research report."""
    recs = [
        "Menambah jumlah data pada kelas Naskhi dan Diwani.",
        "Melakukan audit label untuk mengurangi data salah kelas.",
        "Menghapus duplikasi dan gambar berkualitas rendah.",
        "Melakukan augmentasi data secara aman.",
        "Mengaktifkan class weights saat training.",
        "Melakukan retraining model dengan konfigurasi riset.",
        "Mengevaluasi model menggunakan confusion matrix, precision, recall, dan F1-score.",
        "Melakukan validasi manual pada hasil prediksi mismatch atau confidence rendah.",
    ]
    if training_mode_key in ("ultra_fast", "fast"):
        recs.insert(5, "Beralih dari mode demo ke Mode Riset Akurasi untuk hasil evaluasi final.")
    if latest_eval and latest_eval.accuracy is not None and latest_eval.accuracy < ACCURACY_TARGET:
        recs.append(
            f"Akurasi saat ini {round(latest_eval.accuracy * 100, 2)}% — di bawah target 85%. "
            "Model belum siap untuk pelaporan riset final."
        )
    return list(dict.fromkeys(recs))[:10]


def build_research_insights(
    target_card: Dict,
    is_imbalanced: bool,
    distribution: Dict[str, int],
    latest_eval,
    history_summary: Dict,
) -> List[str]:
    insights = []
    if not target_card["achieved"]:
        insights.append("Model belum mencapai target akurasi riset 85%.")
    if is_imbalanced:
        sorted_dist = sorted(distribution.items(), key=lambda x: x[1])
        minority = [CLASS_LABELS.get(k, k) for k, _ in sorted_dist[:2] if _ > 0]
        dominant = [CLASS_LABELS.get(k, k) for k, _ in sorted_dist[-2:] if _ > 0]
        if minority and dominant:
            insights.append(
                f"Dataset masih belum seimbang. Kelas {', '.join(dominant)} lebih dominan dibanding {', '.join(minority)}."
            )
        else:
            insights.append("Dataset masih belum seimbang antar kelas Khat.")
    else:
        insights.append("Distribusi dataset relatif seimbang untuk keempat kelas Khat.")
    insights.append("Kelas minoritas memerlukan penambahan data untuk mengurangi bias model.")
    insights.append("Class weights dan augmentasi aman disarankan saat training eksperimental.")
    insights.append("Evaluasi lanjutan menggunakan confusion matrix diperlukan untuk analisis per kelas.")
    insights.append(
        "Manual review tetap diperlukan untuk prediksi confidence rendah atau mismatch."
    )
    if history_summary.get("mismatch_count", 0) > 0:
        insights.append(
            f"Terdapat {history_summary['mismatch_count']} prediksi mismatch yang perlu ditinjau ulang."
        )
    if latest_eval is None:
        insights.append("Metrik evaluasi model belum lengkap — jalankan evaluasi pada test set.")
    return insights


def build_distribution_rows(distribution: Dict[str, int]) -> List[Dict[str, Any]]:
    total = sum(distribution.values()) or 0
    rows = []
    for key in CLASS_LABELS:
        count = distribution.get(key, 0)
        pct = round((count / total) * 100, 2) if total else 0
        rows.append({
            "key": key,
            "label": CLASS_LABELS[key],
            "count": count,
            "percent": pct,
        })
    return rows


def build_dataset_balance_note(distribution: Dict[str, int], is_imbalanced: bool) -> str:
    if not is_imbalanced:
        return "Distribusi kelas relatif seimbang untuk keempat jenis Khat."
    sorted_dist = sorted(distribution.items(), key=lambda x: x[1])
    minority = [CLASS_LABELS.get(k, k) for k, v in sorted_dist[:2] if v > 0]
    dominant = [CLASS_LABELS.get(k, k) for k, v in sorted_dist[-2:] if v > 0]
    if minority and dominant:
        return (
            f"Dataset belum seimbang. Kelas {', '.join(dominant)} memiliki jumlah data lebih dominan "
            f"dibanding {', '.join(minority)}."
        )
    return "Dataset belum seimbang. Class weighting dan augmentasi seimbang direkomendasikan."


def build_technical_model_info(tm_info: Optional[Dict], model_ctx: Dict) -> Dict[str, str]:
    if tm_info and tm_info.get("available"):
        return {
            "model_source": tm_info.get("model_source", "Teachable Machine"),
            "model_name": tm_info.get("model_name", "tm-my-image-model"),
            "model_runtime": tm_info.get("model_runtime", "TensorFlow.js"),
            "model_file": tm_info.get("model_json_url", "/static/model/model.json"),
            "metadata_file": tm_info.get("model_metadata_url", "/static/model/metadata.json"),
            "weights_file": tm_info.get("model_weights_url", "/static/model/weights.bin"),
            "input_size": tm_info.get("model_input_size", "224x224").replace("x", " × "),
            "labels": tm_info.get("labels_display", "diwani, diwani jali, naskhi, tsuluts"),
            "architecture": tm_info.get("model_architecture", "Teachable Machine Image Model"),
        }
    arch = model_ctx.get("model_architecture", "Research Model")
    return {
        "model_source": "Keras Transfer Learning",
        "model_name": os.path.basename(model_ctx.get("active_model_path", "model.keras")),
        "model_runtime": "TensorFlow / Keras",
        "model_file": model_ctx.get("active_model_path", "—"),
        "metadata_file": "—",
        "weights_file": "—",
        "input_size": "224 × 224",
        "labels": "naskhi, diwani, diwani_jali, tsuluts",
        "architecture": arch,
    }


def build_confusion_matrix_table(
    confusion_matrix: Optional[List],
    class_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    labels = class_labels or list(CLASS_LABELS.keys())
    if not confusion_matrix:
        return {"available": False, "labels": labels, "rows": []}
    rows = []
    for i, label in enumerate(labels):
        if i >= len(confusion_matrix):
            break
        row_vals = confusion_matrix[i]
        rows.append({
            "true_label": CLASS_LABELS.get(label, label),
            "cells": row_vals,
        })
    return {"available": True, "labels": [CLASS_LABELS.get(l, l) for l in labels], "rows": rows}


def parse_eval_json(latest_eval) -> Dict[str, Any]:
    if not latest_eval:
        return {}
    try:
        return json.loads(latest_eval.classification_report or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


def build_training_metric_rows(latest_eval) -> List[Dict[str, Any]]:
    if not latest_eval:
        return []
    specs = [
        ("training_accuracy", "Training Accuracy", latest_eval.training_accuracy),
        ("validation_accuracy", "Validation Accuracy", latest_eval.validation_accuracy),
        ("training_loss", "Training Loss", latest_eval.training_loss),
        ("validation_loss", "Validation Loss", latest_eval.validation_loss),
    ]
    rows = []
    for key, label, value in specs:
        if value is None:
            display = "Belum tersedia"
        elif "loss" in key:
            display = f"{float(value):.4f}"
        else:
            display = f"{round(float(value) * 100, 2)}%"
        rows.append({"key": key, "label": label, "value": display})
    return rows


def build_conclusion(target_card: Dict) -> str:
    if target_card["achieved"]:
        return (
            "Model telah mencapai target akurasi riset dan dapat dipertimbangkan siap untuk pelaporan final."
        )
    return (
        "Model belum mencapai target akurasi riset. "
        "Hasil saat ini berguna untuk pengujian sistem dan analisis awal, "
        "namun perbaikan lebih lanjut diperlukan sebelum pelaporan riset final."
    )


def serialize_report_row(record, image_url: str) -> Dict[str, Any]:
    conf_pct = round(record.confidence * 100, 2) if record.confidence is not None else None
    return {
        "id": record.id,
        "filename": record.filename,
        "image_url": image_url,
        "predicted_class": record.predicted_class,
        "predicted_display": CLASS_LABELS.get(record.predicted_class or "", record.predicted_class or "—"),
        "confidence_pct": conf_pct,
        "reliability": record.reliability_level or "—",
        "khat_probability_pct": round(record.khat_probability * 100, 1) if record.khat_probability is not None else None,
        "created_at": _fmt_dt(record.created_at),
    }
