from services.algorithm_calculation_service import safe_build_algorithm_calculation
from services.khat_characteristics_service import build_characteristics_context
from services.khat_recognition_messages import STATUS_LABEL, STATUS_LABEL_SHORT

from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional


def _similarity_tone(score: Optional[float]) -> str:
    if score is None:
        return "neutral"
    if score >= 95:
        return "danger"
    if score >= 85:
        return "warn"
    if score >= 70:
        return "info"
    return "ok"

CLASS_LABELS = {
    "naskhi": "Khat Naskhi",
    "riqah": "Khat Riq'ah",
    "diwani": "Khat Diwani",
    "kufi": "Khat Kufi",
}

INPUT_STATUS_LABELS = {
    "khat": "Khat Valid",
    "non_khat": "Non-Khat",
    "uncertain": "Tidak Pasti",
    "unrecognized": STATUS_LABEL,
}

CONFIDENCE_LABELS = {
    "high": "High Confidence",
    "medium": "Medium Confidence",
    "low": "Low Confidence",
    "rejected": "Rejected",
    "uncertain": "Uncertain",
}

INVALID_PREDICTED = {"", "unknown", "none", "-", "rejected"}


def _fmt_dt(value) -> str:
    if not value:
        return "—"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


def is_valid_khat_prediction(row) -> bool:
    status = (row.input_status or "khat").lower()
    if status != "khat":
        return False
    cls = (row.predicted_class or "").strip().lower()
    return bool(cls) and cls not in INVALID_PREDICTED


def confidence_band(confidence: Optional[float]) -> Optional[str]:
    if confidence is None:
        return None
    pct = confidence * 100 if confidence <= 1 else confidence
    if pct >= 85:
        return "high"
    if pct >= 50:
        return "medium"
    return "low"


def confidence_label(band: Optional[str], input_status: str) -> str:
    if input_status == "non_khat":
        return CONFIDENCE_LABELS["rejected"]
    if input_status == "unrecognized":
        return STATUS_LABEL
    if input_status == "uncertain":
        return CONFIDENCE_LABELS["uncertain"]
    if band == "high":
        return CONFIDENCE_LABELS["high"]
    if band == "medium":
        return CONFIDENCE_LABELS["medium"]
    if band == "low":
        return CONFIDENCE_LABELS["low"]
    return "—"


def row_tone(input_status: str) -> str:
    if input_status == "non_khat":
        return "rejected"
    if input_status == "unrecognized":
        return "rejected"
    if input_status == "uncertain":
        return "uncertain"
    return "valid"


def input_type_label(row) -> str:
    det = row.detection_status or row.input_status or "khat"
    if det == "confirmed_khat":
        return "Khat Terkonfirmasi"
    if det == "moderate_khat":
        return "Khat Moderat"
    if row.input_status == "non_khat":
        return "Non-Khat"
    if row.input_status == "unrecognized":
        return STATUS_LABEL
    if row.input_status == "uncertain":
        return "Tidak Pasti"
    return "Khat Terkonfirmasi"


def status_badge(row) -> Dict:
    status = row.input_status or "khat"
    if status == "khat":
        tone = "ok"
        label = "Diterima"
    elif status == "uncertain":
        tone = "warn"
        label = "Perlu Review"
    elif status == "unrecognized":
        tone = "danger"
        label = STATUS_LABEL
    else:
        tone = "danger"
        label = "Ditolak"
    return {"tone": tone, "label": label}


def export_status_label(input_status: str) -> str:
    mapping = {
        "khat": "Diterima",
        "non_khat": "Ditolak",
        "uncertain": "Perlu Review",
        "unrecognized": STATUS_LABEL,
    }
    return mapping.get(input_status or "khat", input_status or "—")


def build_history_record(
    row,
    image_url: str,
    delete_url: str,
    model_architecture: str,
    model_training_mode: str = "",
) -> Dict:
    conf = row.confidence
    conf_pct = round(conf * 100, 2) if conf is not None else None
    band = confidence_band(conf) if is_valid_khat_prediction(row) else None
    status = row.input_status or "khat"
    inp_badge = status_badge(row)
    is_mismatch = (
        row.validation_status == "Prediction Mismatch"
        or (
            row.expected_class
            and row.predicted_class
            and row.expected_class != row.predicted_class
        )
    )

    return {
        "id": row.id,
        "filename": row.filename or "Unknown",
        "image_url": image_url or "",
        "image_path": row.image_path or "",
        "input_status": status,
        "input_type_label": input_type_label(row),
        "input_type_tone": row_tone(status),
        "status_label": inp_badge["label"],
        "status_tone": inp_badge["tone"],
        "predicted_class": row.predicted_class,
        "predicted_display": (
            STATUS_LABEL
            if status == "unrecognized"
            else CLASS_LABELS.get(row.predicted_class or "", row.predicted_class or "—")
        ),
        "confidence": conf,
        "confidence_pct": conf_pct,
        "confidence_band": band,
        "confidence_label": confidence_label(band, status),
        "khat_probability_pct": round(row.khat_probability * 100, 1) if row.khat_probability is not None else None,
        "non_khat_probability_pct": round(row.non_khat_probability * 100, 1) if row.non_khat_probability is not None else None,
        "manual_review_required": bool(row.manual_review_required),
        "reliability_level": row.reliability_level or row.review_status or "—",
        "expected_class": row.expected_class,
        "expected_display": CLASS_LABELS.get(row.expected_class or "", row.expected_class or "—"),
        "validation_status": row.validation_status,
        "review_status": row.review_status,
        "rejection_reason": row.rejection_reason,
        "detection_status": row.detection_status,
        "detection_decision": row.detection_decision,
        "naskhi_score_pct": round(row.naskhi_score * 100, 2) if row.naskhi_score is not None else None,
        "diwani_score_pct": round(row.diwani_score * 100, 2) if row.diwani_score is not None else None,
        "diwani_jali_score_pct": round(row.diwani_jali_score * 100, 2) if row.diwani_jali_score is not None else None,
        "tsuluts_score_pct": round(row.tsuluts_score * 100, 2) if row.tsuluts_score is not None else None,
        "created_at": _fmt_dt(row.created_at),
        "created_at_iso": row.created_at.isoformat() if row.created_at else "",
        "model_architecture": model_architecture,
        "model_training_mode": model_training_mode or "—",
        "delete_url": delete_url,
        "row_tone": row_tone(status),
        "is_valid_khat": is_valid_khat_prediction(row),
        "characteristics": build_characteristics_context(
            row.predicted_class,
            expected_class=row.expected_class,
            validation_status=row.validation_status,
            filename_mismatch=is_mismatch,
        ),
        "similarity_status": row.similarity_status,
        "similarity_score": row.similarity_score,
        "similarity_score_pct": round(row.similarity_score, 1) if row.similarity_score is not None else None,
        "nearest_dataset_image": row.nearest_dataset_image,
        "nearest_dataset_class": row.nearest_dataset_class,
        "similarity_risk_level": row.similarity_risk_level,
        "similarity_message": row.similarity_message,
        "similarity_recommendation": row.similarity_recommendation,
        "similarity_tone": _similarity_tone(row.similarity_score),
        "similarity_available": row.similarity_score is not None,
        "manual_expected_class": row.manual_expected_class,
        "manual_expected_display": CLASS_LABELS.get(row.manual_expected_class or "", row.manual_expected_class or "—"),
        "correction_label": row.correction_label,
        "correction_notes": row.correction_notes,
        "source_type": row.source_type,
        "preprocessing_mode": row.preprocessing_mode or "standard",
        "known_class_status": row.known_class_status,
        "final_decision": row.final_decision,
        "confidence_label": row.confidence_label or row.confidence_level,
        "top_2_class": row.top_2_class,
        "top_2_score_pct": round(row.top_2_score * 100, 2) if row.top_2_score is not None and row.top_2_score <= 1 else (
            round(row.top_2_score, 2) if row.top_2_score is not None else None
        ),
        "top2_margin_pct": round(row.top2_margin, 2) if row.top2_margin is not None and row.top2_margin > 1 else (
            round(row.top2_margin * 100, 2) if row.top2_margin is not None else None
        ),
        "algorithm_calculation": safe_build_algorithm_calculation(row, input_source=getattr(row, "input_source", None) or "upload"),
    }


def build_history_summary(all_records: List) -> Dict:
    class_counts = Counter(
        r.predicted_class for r in all_records if is_valid_khat_prediction(r)
    )
    status_counts = Counter((r.input_status or "khat") for r in all_records)
    total = len(all_records)
    valid_khat = sum(1 for r in all_records if is_valid_khat_prediction(r))
    rejected = status_counts.get("non_khat", 0) + status_counts.get("unrecognized", 0)
    uncertain = status_counts.get("uncertain", 0)
    unrecognized = status_counts.get("unrecognized", 0)
    confidences = [r.confidence for r in all_records if is_valid_khat_prediction(r) and r.confidence is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else None
    most_predicted = class_counts.most_common(1)[0][0] if class_counts else None
    latest = max((r.created_at for r in all_records if r.created_at), default=None)

    return {
        "total": total,
        "valid_khat": valid_khat,
        "rejected": rejected,
        "unrecognized": unrecognized,
        "uncertain": uncertain,
        "avg_confidence": avg_confidence,
        "class_counts": dict(class_counts),
        "status_counts": dict(status_counts),
        "most_predicted": most_predicted,
        "most_predicted_display": CLASS_LABELS.get(most_predicted or "", most_predicted or "—"),
        "latest_prediction_date": _fmt_dt(latest),
    }


def build_prediction_insights(records: List) -> Dict:
    """Insight card metrics from a record set (filtered or overall)."""
    valid_khat = [r for r in records if is_valid_khat_prediction(r)]
    class_counts = Counter(r.predicted_class for r in valid_khat)
    most = class_counts.most_common(1)[0] if class_counts else None

    conf_pairs = [(r, r.confidence) for r in valid_khat if r.confidence is not None]
    avg_confidence = sum(c for _, c in conf_pairs) / len(conf_pairs) if conf_pairs else None

    highest = max(conf_pairs, key=lambda x: x[1]) if conf_pairs else None
    lowest = min(conf_pairs, key=lambda x: x[1]) if conf_pairs else None

    rejected = sum(1 for r in records if (r.input_status or "") in ("non_khat", "unrecognized"))
    uncertain = sum(1 for r in records if (r.input_status or "") == "uncertain")
    latest = max((r.created_at for r in records if r.created_at), default=None)

    def _conf_display(val):
        if val is None:
            return "—"
        pct = val * 100 if val <= 1 else val
        return f"{pct:.1f}%"

    return {
        "has_valid_khat": bool(valid_khat),
        "most_predicted_display": (
            CLASS_LABELS.get(most[0], most[0].replace("_", " ").title())
            if most
            else "Belum ada prediksi Khat valid."
        ),
        "avg_confidence": avg_confidence,
        "avg_confidence_display": _conf_display(avg_confidence),
        "highest_confidence_display": _conf_display(highest[1]) if highest else "—",
        "highest_confidence_file": highest[0].filename if highest else "—",
        "lowest_confidence_display": _conf_display(lowest[1]) if lowest else "—",
        "lowest_confidence_file": lowest[0].filename if lowest else "—",
        "rejected_count": rejected,
        "uncertain_count": uncertain,
        "latest_prediction_date": _fmt_dt(latest),
        "record_count": len(records),
    }


def apply_confidence_filter(records: List, level: str) -> List:
    """Filter valid Khat predictions by confidence band."""
    if not level:
        return records
    out = []
    for row in records:
        if not is_valid_khat_prediction(row) or row.confidence is None:
            continue
        band = confidence_band(row.confidence)
        if level == band:
            out.append(row)
    return out


def serialize_records_for_export(records: List, model_architecture: str, model_training_mode: str = "") -> List[Dict]:
    rows = []
    for row in records:
        band = confidence_band(row.confidence) if is_valid_khat_prediction(row) else None
        status = row.input_status or "khat"
        rows.append(
            {
                "id": row.id,
                "uploaded_filename": row.uploaded_filename or row.filename,
                "filename": row.filename,
                "expected_class": row.expected_class,
                "predicted_class": row.predicted_class,
                "confidence_score": row.confidence_score if row.confidence_score is not None else row.confidence,
                "probability_diwani": row.probability_diwani if row.probability_diwani is not None else row.diwani_score,
                "probability_diwani_jali": row.probability_diwani_jali if row.probability_diwani_jali is not None else row.diwani_jali_score,
                "probability_naskhi": row.probability_naskhi if row.probability_naskhi is not None else row.naskhi_score,
                "probability_tsuluts": row.probability_tsuluts if row.probability_tsuluts is not None else row.tsuluts_score,
                "confidence_label": row.confidence_label,
                "review_status": row.review_status,
                "final_decision": row.final_decision,
                "explanation_text": row.explanation_text,
                "top_2_class": row.top_2_class,
                "top_2_score": row.top_2_score,
                "top_2_margin": row.top2_margin,
                "input_status": status,
                "status": export_status_label(status),
                "khat_probability": row.khat_probability,
                "non_khat_probability": row.non_khat_probability,
                "confidence": row.confidence,
                "reliability_level": row.reliability_level or row.review_status,
                "model_architecture": model_architecture,
                "model_training_mode": model_training_mode,
                "created_at": _fmt_dt(row.created_at),
                "confidence_band": band,
                "detection_status": row.detection_status,
                "rejection_reason": row.rejection_reason,
                "manual_review_required": bool(row.manual_review_required),
                "similarity_status": row.similarity_status,
                "similarity_score": row.similarity_score,
                "nearest_dataset_image": row.nearest_dataset_image,
                "nearest_dataset_class": row.nearest_dataset_class,
                "similarity_risk_level": row.similarity_risk_level,
                "manual_expected_class": row.manual_expected_class,
                "correction_label": row.correction_label,
                "correction_notes": row.correction_notes,
                "source_type": row.source_type,
                "preprocessing_mode": row.preprocessing_mode,
                "known_class_status": row.known_class_status,
            }
        )
    return rows
