"""Write operations for the 9-table Indonesian schema."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from flask import session

from models import (
    AlgorithmCalculation,
    ClassificationHistory,
    ClassificationResult,
    CorrectionLog,
    Dataset,
    DatasetImage,
    EvaluasiModel,
    GambarDataset,
    KhatClass,
    ModelEvaluation,
    PerhitunganAlgoritma,
    PredictionProbability,
    ProbabilitasPrediksi,
    RiwayatKlasifikasi,
    User,
    VersiModel,
    db,
)
from models.evaluasi_model import TIPE_DETAIL, TIPE_RINGKASAN
from services.db_compat import get_class_id_by_slug, get_active_model_id


def get_active_model_version_id() -> Optional[int]:
    return get_active_model_id()


def _resolve_reviewer_user_id() -> Optional[int]:
    uid = session.get("user_id")
    if uid:
        return uid
    username = session.get("username")
    if username:
        user = User.query.filter_by(username=username).first()
        return user.id if user else None
    return None


def resolve_dataset_image_id_from_path(filename: Optional[str] = None, image_path: Optional[str] = None) -> Optional[int]:
    candidates = []
    for value in (filename, image_path):
        if not value:
            continue
        norm = str(value).replace("\\", "/").strip()
        candidates.append(norm)
        candidates.append(os.path.basename(norm))
    for token in candidates:
        row = GambarDataset.query.filter(
            db.or_(
                GambarDataset.nama_file == token,
                GambarDataset.path_asli.endswith(token),
                GambarDataset.path_asli.contains(token),
            )
        ).first()
        if row:
            return row.id
    return None


def sync_dataset_image_from_legacy(legacy: Dataset, class_slug: str) -> Optional[GambarDataset]:
    existing = GambarDataset.query.filter_by(id_legacy=legacy.id).first()
    if existing:
        return existing
    from services.db_compat import create_gambar_dataset

    row = create_gambar_dataset(
        class_name=class_slug,
        filename=legacy.filename,
        original_filename=legacy.original_filename,
        image_path=legacy.image_path,
        data_type=legacy.data_type or "raw",
        image_format=legacy.image_format or "",
        image_size=legacy.image_size or "",
        legacy_id=legacy.id,
    )
    db.session.flush()
    return row


def sync_classification_history_from_legacy(
    legacy: ClassificationResult,
    user_id: Optional[int] = None,
    algorithm_calc: Optional[Dict[str, Any]] = None,
) -> Optional[RiwayatKlasifikasi]:
    existing = RiwayatKlasifikasi.query.filter_by(id_legacy=legacy.id).first()
    if existing:
        return existing
    from services.db_compat import finalize_classification_save

    hist = RiwayatKlasifikasi(
        id_pengguna=user_id or session.get("user_id"),
        id_model=get_active_model_id(),
        nama_file_upload=legacy.uploaded_filename or legacy.filename,
        path_upload=legacy.image_path,
        id_kelas_diharapkan=get_class_id_by_slug(legacy.expected_class or legacy.manual_expected_class),
        id_kelas_prediksi=get_class_id_by_slug(legacy.predicted_class),
        id_kelas_top2=get_class_id_by_slug(getattr(legacy, "top_2_class", None)),
        confidence=legacy.confidence or legacy.confidence_score,
        skor_top2=legacy.top_2_score,
        margin_top2=legacy.top2_margin,
        status_validasi=legacy.validation_status,
        keputusan_akhir=legacy.final_decision,
        sumber_input=legacy.input_source or "upload",
        id_legacy=legacy.id,
    )
    db.session.add(hist)
    db.session.flush()
    scores = {
        "naskhi": float(legacy.naskhi_score or 0),
        "diwani": float(legacy.diwani_score or 0),
        "diwani_jali": float(legacy.diwani_jali_score or 0),
        "tsuluts": float(legacy.tsuluts_score or 0),
    }
    finalize_classification_save(hist, scores=scores, algorithm_fields={"khat_probability": legacy.khat_probability})
    return hist


def sync_evaluation_session_from_legacy(
    legacy: ModelEvaluation,
    eval_json: Optional[Dict[str, Any]] = None,
) -> EvaluasiModel:
    existing = EvaluasiModel.query.filter_by(tipe_record=TIPE_RINGKASAN, id_legacy=legacy.id).first()
    if existing:
        return existing
    payload = eval_json or {}
    row = EvaluasiModel(
        id_model=get_active_model_id(),
        total_gambar=payload.get("test_samples"),
        prediksi_benar=payload.get("correct_predictions"),
        prediksi_salah=payload.get("incorrect_predictions") or payload.get("misclassified_count"),
        akurasi=legacy.accuracy,
        presisi=legacy.precision_score,
        recall=legacy.recall_score,
        f1_score=legacy.f1_score,
        tipe_record=TIPE_RINGKASAN,
        id_legacy=legacy.id,
        dibuat_pada=legacy.dibuat_pada,
    )
    db.session.add(row)
    db.session.flush()
    return row


def sync_evaluation_results_from_misclassified(
    session_row: EvaluasiModel,
    misclassified_rows: list,
    image_url_builder=None,
) -> int:
    from services.db_compat import save_evaluation_details

    return save_evaluation_details(session_row, misclassified_rows, commit=False)


def sync_correction_for_classification(
    result_row: ClassificationResult,
    correct_class: str,
    note: str = "",
    dataset_image_id: Optional[int] = None,
) -> Optional[CorrectionLog]:
    hist = RiwayatKlasifikasi.query.filter_by(id_legacy=result_row.id).first()
    if not hist:
        hist = sync_classification_history_from_legacy(result_row)
    corrected_id = get_class_id_by_slug(correct_class)
    if not corrected_id:
        return None
    old_id = get_class_id_by_slug(result_row.predicted_class) or (hist.id_kelas_prediksi if hist else None)
    log = CorrectionLog(
        id_riwayat=hist.id if hist else None,
        id_gambar_dataset=dataset_image_id,
        id_kelas_lama=old_id,
        id_kelas_koreksi=corrected_id,
        catatan_koreksi=(note or "").strip() or None,
        status_koreksi="corrected",
        direview_oleh=_resolve_reviewer_user_id(),
        direview_pada=datetime.utcnow(),
    )
    db.session.add(log)
    db.session.flush()
    return log


def sync_correction_for_evaluation(
    image_rel: str,
    correct_class: str,
    note: str = "",
    dataset_image_id: Optional[int] = None,
) -> Optional[CorrectionLog]:
    corrected_id = get_class_id_by_slug(correct_class)
    if not corrected_id:
        return None
    eval_result = (
        EvaluasiModel.query.filter_by(tipe_record=TIPE_DETAIL)
        .filter(
            db.or_(
                EvaluasiModel.nama_file_simpan == image_rel,
                EvaluasiModel.nama_file_simpan.endswith(image_rel),
                EvaluasiModel.url_gambar.contains(image_rel),
            )
        )
        .order_by(EvaluasiModel.id.desc())
        .first()
    )
    log = CorrectionLog(
        id_evaluasi=eval_result.id if eval_result else None,
        id_gambar_dataset=dataset_image_id or (eval_result.id_gambar_dataset if eval_result else None),
        id_kelas_lama=eval_result.id_kelas_asli if eval_result else None,
        id_kelas_koreksi=corrected_id,
        catatan_koreksi=(note or "").strip() or None,
        status_koreksi="corrected",
        direview_oleh=_resolve_reviewer_user_id(),
        direview_pada=datetime.utcnow(),
    )
    db.session.add(log)
    db.session.flush()
    return log


def list_khat_classes() -> list:
    return KhatClass.query.order_by(KhatClass.id.asc()).all()
