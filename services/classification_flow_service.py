"""Alur klasifikasi pengguna: upload → CNN → simpan ke riwayat_klasifikasi + probabilitas + perhitungan."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, Optional, Tuple

from flask import current_app
from PIL import Image
from werkzeug.utils import secure_filename

from models import RiwayatKlasifikasi, VersiModel, db
from services.algorithm_detail_service import confidence_level_info
from services.db_compat import (
    finalize_classification_save,
    get_active_model_id,
    get_class_id_by_slug,
    get_class_name_by_id,
)
from services.prediction_service import predict_image
from services.dataset_path_service import safe_relpath
from services.khat_recognition_messages import (
    STATUS_LABEL,
    flash_message,
    should_treat_as_unrecognized,
    unrecognized_from_system_error,
)
from services.class_mapping_service import format_trained_class_list
from services.result_interpretation_service import build_image_metadata


def allowed_classification_file(filename: str) -> bool:
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    allowed = current_app.config.get("CLASSIFICATION_ALLOWED_EXTENSIONS") or {"jpg", "jpeg", "png"}
    return ext in allowed


def _confidence_level(confidence: Optional[float]) -> str:
    return confidence_level_info(confidence)["label"]


def _build_preprocessing_steps_json() -> str:
    steps = {
        "step_1": "Gambar diunggah oleh user",
        "step_2": "Validasi format JPG, JPEG, atau PNG",
        "step_3": "Resize gambar menjadi 224 x 224 piksel",
        "step_4": "Konversi gambar menjadi array piksel",
        "step_5": "Normalisasi nilai piksel dengan membagi 255",
        "step_6": "Membentuk tensor input model dengan shape (1, 224, 224, 3)",
        "step_7": "Gambar diproses menggunakan model CNN",
    }
    return json.dumps(steps, ensure_ascii=False)


def _build_probabilities_json(scores: Dict[str, float]) -> str:
    display = current_app.config.get("CLASS_DISPLAY_NAMES", {})
    payload = {}
    for slug, score in scores.items():
        label = display.get(slug, slug.replace("_", " ").title())
        pct = score * 100 if score <= 1 else score
        payload[label] = round(pct, 2)
    return json.dumps(payload, ensure_ascii=False)


def _build_preprocessing_steps(preprocessing_mode: str = "standard") -> str:
    return _build_preprocessing_steps_json()


def ensure_active_cnn_model_version() -> Optional[int]:
    """Pastikan ada versi model aktif di tabel versi_model (baca dari file CNN)."""
    active = VersiModel.query.filter_by(aktif=True).first()
    if active:
        return active.id

    from services.dataset_readiness_service import is_valid_trained_model
    from services.model_context_service import get_active_model_path, get_model_context

    model_path = get_active_model_path()
    if not is_valid_trained_model(model_path):
        return None

    ctx = get_model_context()
    labels = current_app.config.get("CLASS_LABELS", [])
    row = VersiModel(
        versi="v1.0",
        nama_model=os.path.basename(model_path),
        sumber_model="CNN Transfer Learning",
        runtime="TensorFlow Keras",
        file_model=model_path,
        lebar_input=224,
        tinggi_input=224,
        label_json=json.dumps(labels, ensure_ascii=False),
        status="active",
        jumlah_kelas=len(labels),
        aktif=True,
        catatan=ctx.get("model_architecture") or "EfficientNetB0",
    )
    db.session.add(row)
    db.session.commit()
    return row.id


def save_uploaded_image(file_storage) -> Tuple[str, str, str]:
    """Simpan file upload; kembalikan (abs_path, rel_path, safe_name)."""
    ext = file_storage.filename.rsplit(".", 1)[1].lower()
    stored_name = f"{uuid.uuid4().hex}.{ext}"
    abs_path = os.path.join(current_app.config["UPLOAD_FOLDER"], stored_name)
    file_storage.save(abs_path)

    with Image.open(abs_path) as img:
        img.verify()
    with Image.open(abs_path) as img:
        img.convert("RGB")

    rel_path = safe_relpath(abs_path, current_app.config["BASE_DIR"])
    return abs_path, rel_path, secure_filename(file_storage.filename)


def persist_classification_result(
    *,
    user_id: int,
    abs_path: str,
    rel_path: str,
    original_filename: str,
    prediction: Dict[str, Any],
    preprocessing_mode: str = "standard",
) -> RiwayatKlasifikasi:
    input_status = prediction.get("input_status", "khat")
    scores = prediction.get("scores") or {}
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_slug = prediction.get("predicted_class")
    second = sorted_scores[1] if len(sorted_scores) > 1 else None
    confidence = prediction.get("confidence")
    conf_pct = (confidence * 100 if confidence <= 1 else confidence) if confidence is not None else None
    margin = None
    if second and conf_pct is not None:
        second_pct = second[1] * 100 if second[1] <= 1 else second[1]
        margin = round(conf_pct - second_pct, 4)

    model_id = ensure_active_cnn_model_version() or get_active_model_id()
    image_meta = build_image_metadata(abs_path)

    if input_status in ("unrecognized", "non_khat", "uncertain"):
        is_non_khat = input_status == "non_khat" or (
            input_status == "uncertain" and not prediction.get("stage2_ran")
        )
        decision_label = (
            STATUS_LABEL
            if input_status == "unrecognized"
            else "Ditolak Non-Khat"
        )
        row = RiwayatKlasifikasi(
            id_pengguna=user_id,
            id_model=model_id,
            nama_file_upload=secure_filename(original_filename),
            path_upload=rel_path,
            id_kelas_prediksi=None,
            confidence=None,
            id_kelas_top2=None,
            skor_top2=None,
            margin_top2=None,
            status_validasi=decision_label,
            keputusan_akhir=decision_label,
            sumber_input="upload",
            input_status="non_khat" if is_non_khat and input_status != "unrecognized" else input_status,
            is_khat=False,
            rejection_reason=prediction.get("rejection_reason") or prediction.get("message"),
            reliability_level="Rejected",
            review_status="Rejected",
            explanation_text=prediction.get("message") or prediction.get("rejection_reason"),
            detection_status=prediction.get("detection_status"),
            detection_decision=prediction.get("detection_decision") or "rejected",
            khat_probability=prediction.get("khat_probability"),
            non_khat_probability=prediction.get("non_khat_probability"),
            manual_review_required=False,
        )
        db.session.add(row)
        db.session.flush()
        algo_fields = {
            "original_width": image_meta.get("width"),
            "original_height": image_meta.get("height"),
            "processed_width": 224,
            "processed_height": 224,
            "model_input_size": "224x224",
            "tensor_shape": "(1, 224, 224, 3)",
            "preprocessing_steps_json": _build_preprocessing_steps(preprocessing_mode),
            "khat_probability": prediction.get("khat_probability"),
            "stage1_decision": "Ditolak",
            "confidence_level": "Rejected",
            "known_class_status": decision_label,
            "calculation_notes": (
                prediction.get("stage2_message")
                or prediction.get("message")
                or prediction.get("detail_message")
                or prediction.get("rejection_reason")
                or "Citra ditolak karena di luar domain khat / 4 kelas yang didukung."
            ),
        }
        finalize_classification_save(row, scores=None, algorithm_fields=algo_fields)
        db.session.commit()
        return row

    row = RiwayatKlasifikasi(
        id_pengguna=user_id,
        id_model=model_id,
        nama_file_upload=secure_filename(original_filename),
        path_upload=rel_path,
        id_kelas_prediksi=get_class_id_by_slug(top_slug),
        confidence=confidence,
        id_kelas_top2=get_class_id_by_slug(second[0]) if second else None,
        skor_top2=second[1] if second else None,
        margin_top2=margin,
        status_validasi="selesai",
        keputusan_akhir=get_class_name_by_id(get_class_id_by_slug(top_slug)) or top_slug,
        sumber_input="upload",
        input_status="khat",
        is_khat=True,
        khat_probability=prediction.get("khat_probability"),
        non_khat_probability=prediction.get("non_khat_probability"),
        detection_status=prediction.get("detection_status"),
        detection_decision=prediction.get("detection_decision"),
    )
    row.softmax_scores_json = json.dumps(scores, ensure_ascii=False)
    db.session.add(row)
    db.session.flush()

    top_display = get_class_name_by_id(get_class_id_by_slug(top_slug)) or top_slug
    decision_note = (
        f"Citra diklasifikasikan sebagai {top_display} karena memiliki nilai probabilitas tertinggi "
        f"({conf_pct:.2f}%) dibandingkan kelas lainnya."
        if conf_pct is not None
        else f"Prediksi CNN: {top_display}."
    )

    algo_fields = {
        "original_width": image_meta.get("width"),
        "original_height": image_meta.get("height"),
        "processed_width": 224,
        "processed_height": 224,
        "model_input_size": "224x224",
        "tensor_shape": "(1, 224, 224, 3)",
        "preprocessing_steps_json": _build_preprocessing_steps(preprocessing_mode),
        "khat_probability": prediction.get("khat_probability") if prediction.get("khat_probability") is not None else confidence,
        "stage1_decision": top_display,
        "confidence_level": _confidence_level(confidence),
        "known_class_status": "Dikenali",
        "formula_summary_json": _build_probabilities_json(scores),
        "calculation_notes": decision_note,
    }
    finalize_classification_save(row, scores=scores, algorithm_fields=algo_fields)
    db.session.commit()
    return row


def run_user_classification(
    user_id: int,
    file_storage,
    preprocessing_mode: str = "standard",
) -> Tuple[RiwayatKlasifikasi, Dict[str, Any]]:
    if not allowed_classification_file(file_storage.filename):
        raise ValueError("Format file tidak didukung. Gunakan JPG, JPEG, atau PNG.")

    abs_path, rel_path, _ = save_uploaded_image(file_storage)
    display_list = format_trained_class_list()
    try:
        # Full pipeline: Stage 1 Khat detector + Stage 2 4-class gate.
        prediction = predict_image(abs_path, preprocessing_mode=preprocessing_mode)
        if prediction.get("input_status") == "error":
            raise RuntimeError(prediction.get("message") or "Model CNN belum siap atau prediksi gagal.")
        row = persist_classification_result(
            user_id=user_id,
            abs_path=abs_path,
            rel_path=rel_path,
            original_filename=file_storage.filename,
            prediction=prediction,
            preprocessing_mode=preprocessing_mode,
        )
        return row, prediction
    except Exception as exc:
        # Path/mount quirks must not blank out a valid calligraphy prediction.
        # Only convert to unrecognized when prediction never produced a class label.
        if should_treat_as_unrecognized(exc):
            try:
                from services.prediction_service import (
                    _apply_class_recognition_gate,
                    _classify_khat_type,
                )

                type_result = _classify_khat_type(abs_path, preprocessing_mode=preprocessing_mode)
                type_result = _apply_class_recognition_gate(
                    type_result,
                    image_path=abs_path,
                    stage1_confirmed=True,
                )
                if type_result.get("predicted_class"):
                    fallback = {
                        **type_result,
                        "input_status": "khat",
                        "is_khat": True,
                        "stage2_ran": True,
                    }
                    row = persist_classification_result(
                        user_id=user_id,
                        abs_path=abs_path,
                        rel_path=rel_path,
                        original_filename=file_storage.filename,
                        prediction=fallback,
                        preprocessing_mode=preprocessing_mode,
                    )
                    return row, fallback
            except Exception:
                pass
            prediction = unrecognized_from_system_error(display_list, exc)
            row = persist_classification_result(
                user_id=user_id,
                abs_path=abs_path,
                rel_path=rel_path,
                original_filename=file_storage.filename,
                prediction=prediction,
                preprocessing_mode=preprocessing_mode,
            )
            return row, prediction
        if os.path.isfile(abs_path):
            os.remove(abs_path)
        raise
