"""Relational read helpers using SQLAlchemy joins."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import joinedload

from models import EvaluasiModel, KelasKhat, PerhitunganAlgoritma, ProbabilitasPrediksi, RiwayatKlasifikasi, db
from models.evaluasi_model import TIPE_DETAIL, TIPE_RINGKASAN


def get_class_label_map() -> Dict[int, str]:
    return {row.id: row.nama_kelas for row in KelasKhat.query.order_by(KelasKhat.id.asc()).all()}


def serialize_classification_history(row: RiwayatKlasifikasi) -> Dict[str, Any]:
    label_map = get_class_label_map()
    probs = (
        ProbabilitasPrediksi.query.filter_by(id_riwayat=row.id)
        .order_by(ProbabilitasPrediksi.urutan_ranking.asc())
        .all()
    )
    calc = PerhitunganAlgoritma.query.filter_by(id_riwayat=row.id).first()
    return {
        "id": row.id,
        "uploaded_filename": row.nama_file_upload,
        "predicted_class": label_map.get(row.id_kelas_prediksi),
        "expected_class": label_map.get(row.id_kelas_diharapkan),
        "top2_class": label_map.get(row.id_kelas_top2),
        "confidence_score": row.confidence,
        "model_version_id": row.id_model,
        "validation_status": row.status_validasi,
        "final_decision": row.keputusan_akhir,
        "created_at": row.dibuat_pada.isoformat() if row.dibuat_pada else None,
        "probabilities": [
            {
                "class": label_map.get(p.id_kelas),
                "softmax_score": p.skor_softmax,
                "probability_percent": p.persentase_probabilitas,
                "rank_order": p.urutan_ranking,
            }
            for p in probs
        ],
        "algorithm_calculation_id": calc.id if calc else None,
    }


def list_classification_history(limit: int = 50) -> List[Dict[str, Any]]:
    rows = (
        RiwayatKlasifikasi.query.options(
            joinedload(RiwayatKlasifikasi.kelas_prediksi),
            joinedload(RiwayatKlasifikasi.kelas_diharapkan),
        )
        .order_by(RiwayatKlasifikasi.dibuat_pada.desc())
        .limit(limit)
        .all()
    )
    return [serialize_classification_history(row) for row in rows]


def get_latest_classification_with_joins() -> Optional[Dict[str, Any]]:
    row = RiwayatKlasifikasi.query.order_by(RiwayatKlasifikasi.dibuat_pada.desc()).first()
    return serialize_classification_history(row) if row else None


def list_evaluation_results_for_session(session_id: int) -> List[Dict[str, Any]]:
    label_map = get_class_label_map()
    rows = (
        EvaluasiModel.query.filter_by(tipe_record=TIPE_DETAIL, id_induk=session_id)
        .order_by(EvaluasiModel.id.asc())
        .all()
    )
    return [
        {
            "id": r.id,
            "dataset_image_id": r.id_gambar_dataset,
            "true_class": label_map.get(r.id_kelas_asli),
            "predicted_class": label_map.get(r.id_kelas_prediksi),
            "confidence_score": r.confidence,
            "is_correct": r.benar,
            "error_type": r.jenis_error,
            "image_url": r.url_gambar,
        }
        for r in rows
    ]


def get_latest_evaluation_session_summary() -> Optional[Dict[str, Any]]:
    session_row = EvaluasiModel.query.filter_by(tipe_record=TIPE_RINGKASAN).order_by(EvaluasiModel.id.desc()).first()
    if not session_row:
        return None
    return {
        "id": session_row.id,
        "model_version_id": session_row.id_model,
        "total_images": session_row.total_gambar,
        "accuracy": session_row.akurasi,
        "results_count": EvaluasiModel.query.filter_by(tipe_record=TIPE_DETAIL, id_induk=session_row.id).count(),
    }
