"""Database compatibility helpers for Indonesian 3NF schema."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from flask import session

from models.database import db


# Alias nama folder / label model → slug kelas di database.
CLASS_SLUG_DB_ALIASES = {
    "diwani jali": "diwani_jali",
    "diwanijali": "diwani_jali",
    "thuluth": "tsuluts",
    "thuluts": "tsuluts",
    "suluth": "tsuluts",
    # Legacy slug names (jika masih dipakai di data lama)
    "riqah": "riqah",
    "kufi": "kufi",
}


def normalize_class_slug(slug: Optional[str]) -> Optional[str]:
    if not slug:
        return None
    key = str(slug).strip().lower().replace(" ", "_").replace("-", "_")
    return CLASS_SLUG_DB_ALIASES.get(key, key)


def get_class_id_by_slug(slug: Optional[str]) -> Optional[int]:
    key = normalize_class_slug(slug)
    if not key:
        return None
    from models import KelasKhat

    row = KelasKhat.query.filter_by(slug=key).first()
    return row.id if row else None


def filter_dataset_by_class(query, class_slug: str):
    from models import GambarDataset, KelasKhat

    kelas = KelasKhat.query.filter_by(slug=class_slug).first()
    if not kelas:
        return query.filter(db.text("1=0"))
    return query.filter(GambarDataset.id_kelas == kelas.id)


def count_dataset_by_class(class_slug: str) -> int:
    from models import GambarDataset, KelasKhat

    kelas = KelasKhat.query.filter_by(slug=class_slug).first()
    if not kelas:
        return 0
    return GambarDataset.query.filter_by(id_kelas=kelas.id).count()


def get_class_name_by_id(class_id: Optional[int]) -> Optional[str]:
    if not class_id:
        return None
    from models import KelasKhat

    row = KelasKhat.query.get(class_id)
    return row.nama_kelas if row else None


def get_class_slug_by_id(class_id: Optional[int]) -> Optional[str]:
    if not class_id:
        return None
    from models import KelasKhat

    row = KelasKhat.query.get(class_id)
    return row.slug if row else None


def get_active_model():
    from models import VersiModel

    return VersiModel.query.filter_by(aktif=True).first()


def get_active_model_id() -> Optional[int]:
    mv = get_active_model()
    return mv.id if mv else None


def create_gambar_dataset(
    class_name: str,
    filename: str,
    original_filename: str,
    image_path: str,
    data_type: str = "raw",
    image_format: str = "",
    image_size: str = "",
    legacy_id: Optional[int] = None,
):
    from models import GambarDataset

    class_id = get_class_id_by_slug(class_name)
    if not class_id:
        raise ValueError(f"Kelas tidak dikenal: {class_name}")
    lebar, tinggi = None, None
    if image_size and "x" in image_size.lower():
        parts = image_size.lower().split("x")
        try:
            lebar, tinggi = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            pass
    row = GambarDataset(
        id_kelas=class_id,
        nama_file=filename,
        nama_file_asli=original_filename,
        path_asli=image_path,
        format_file=image_format,
        lebar=lebar,
        tinggi=tinggi,
        sumber_data=data_type,
        id_legacy=legacy_id,
    )
    db.session.add(row)
    return row


def save_prediction_probabilities(riwayat_id: int, scores: Dict[str, float]) -> None:
    from models import ProbabilitasPrediksi

    ProbabilitasPrediksi.query.filter_by(id_riwayat=riwayat_id).delete()
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (slug, score) in enumerate(ranked, start=1):
        cid = get_class_id_by_slug(slug)
        if not cid:
            continue
        pct = score * 100 if score <= 1 else score
        db.session.add(
            ProbabilitasPrediksi(
                id_riwayat=riwayat_id,
                id_kelas=cid,
                skor_softmax=score if score <= 1 else score / 100,
                persentase_probabilitas=round(pct, 4),
                urutan_ranking=rank,
            )
        )


def save_perhitungan_algoritma(riwayat_id: int, fields: Dict[str, Any]) -> None:
    from models import PerhitunganAlgoritma

    existing = PerhitunganAlgoritma.query.filter_by(id_riwayat=riwayat_id).first()
    if existing:
        row = existing
    else:
        row = PerhitunganAlgoritma(id_riwayat=riwayat_id)
        db.session.add(row)

    mapping = {
        "original_width": "lebar_asli",
        "original_height": "tinggi_asli",
        "processed_width": "lebar_proses",
        "processed_height": "tinggi_proses",
        "model_input_size": "ukuran_input_model",
        "tensor_shape": "bentuk_tensor",
        "preprocessing_steps_json": "langkah_preprocessing_json",
        "khat_probability": "probabilitas_khat",
        "non_khat_probability": "probabilitas_non_khat",
        "stage1_decision": "keputusan_tahap1",
        "detection_decision": "keputusan_tahap1",
        "confidence_level": "level_confidence",
        "known_class_status": "status_kelas_dikenal",
        "formula_summary_json": "rumus_json",
        "calculation_notes": "catatan_perhitungan",
    }
    for src, dest in mapping.items():
        if src in fields and fields[src] is not None:
            setattr(row, dest, fields[src])


def finalize_classification_save(row, scores: Optional[Dict[str, float]] = None, algorithm_fields: Optional[Dict] = None) -> None:
    """Persist probabilities and algorithm calculation after riwayat row flush."""
    db.session.flush()
    if scores:
        save_prediction_probabilities(row.id, scores)
    if algorithm_fields:
        save_perhitungan_algoritma(row.id, algorithm_fields)


def get_dataset_summary() -> Dict[str, Any]:
    from models import GambarDataset, KelasKhat

    total = GambarDataset.query.count()
    per_class = {}
    for kelas in KelasKhat.query.order_by(KelasKhat.id.asc()).all():
        per_class[kelas.slug] = GambarDataset.query.filter_by(id_kelas=kelas.id).count()
    return {"total": total, "per_class": per_class}


def get_evaluation_summary() -> Optional[Dict[str, Any]]:
    from models import EvaluasiModel
    from models.evaluasi_model import TIPE_DETAIL, TIPE_RINGKASAN

    ringkasan = EvaluasiModel.query.filter_by(tipe_record=TIPE_RINGKASAN).order_by(EvaluasiModel.id.desc()).first()
    if not ringkasan:
        return None
    detail_count = EvaluasiModel.query.filter_by(tipe_record=TIPE_DETAIL, id_induk=ringkasan.id).count()
    return {
        "id": ringkasan.id,
        "accuracy": ringkasan.akurasi,
        "total_images": ringkasan.total_gambar,
        "wrong_predictions": ringkasan.prediksi_salah,
        "results_count": detail_count,
    }


def save_evaluation_details(summary_row, misclassified_rows: list, commit: bool = True) -> int:
    from models import EvaluasiModel
    from models.evaluasi_model import TIPE_DETAIL

    if not misclassified_rows:
        return 0
    EvaluasiModel.query.filter_by(tipe_record=TIPE_DETAIL, id_induk=summary_row.id).delete()
    count = 0
    for row in misclassified_rows:
        true_id = get_class_id_by_slug(row.get("true_label"))
        pred_id = get_class_id_by_slug(row.get("predicted_label"))
        top2_id = get_class_id_by_slug(row.get("top2_class"))
        from services.relational_sync_service import resolve_dataset_image_id_from_path

        dataset_image_id = resolve_dataset_image_id_from_path(
            filename=row.get("filename") or row.get("preview_filename"),
            image_path=row.get("image_path") or row.get("image_rel"),
        )
        db.session.add(
            EvaluasiModel(
                id_model=summary_row.id_model,
                id_induk=summary_row.id,
                id_gambar_dataset=dataset_image_id,
                id_kelas_asli=true_id,
                id_kelas_prediksi=pred_id,
                confidence=row.get("confidence")
                or (row.get("confidence_pct", 0) / 100 if row.get("confidence_pct") else None),
                id_kelas_top2=top2_id,
                skor_top2=row.get("top2_probability"),
                margin_top2=row.get("margin_pct"),
                benar=False,
                jenis_error=row.get("error_type"),
                rekomendasi=row.get("recommendation") or row.get("error_hint"),
                url_gambar=row.get("image_url"),
                nama_file_simpan=row.get("filename") or row.get("preview_filename"),
                tipe_record=TIPE_DETAIL,
                dibuat_pada=summary_row.dibuat_pada,
            )
        )
        count += 1
    if commit:
        db.session.commit()
    return count


def save_evaluation_run(result: Dict[str, Any], model_name: str = "model", history: Optional[Dict] = None) -> EvaluasiModel:
    from models import EvaluasiModel
    from models.evaluasi_model import TIPE_RINGKASAN

    history = history or {}
    summary = EvaluasiModel(
        id_model=get_active_model_id(),
        total_gambar=result.get("test_samples"),
        prediksi_benar=result.get("correct_predictions"),
        prediksi_salah=result.get("incorrect_predictions") or result.get("misclassified_count"),
        akurasi=result.get("accuracy"),
        presisi=result.get("precision"),
        recall=result.get("recall"),
        f1_score=result.get("f1_score"),
        tipe_record=TIPE_RINGKASAN,
    )
    summary.model_name = model_name
    summary.confusion_matrix = json.dumps(result.get("confusion_matrix", []))
    summary.classification_report = json.dumps(result.get("classification_report", {}))
    if history.get("accuracy"):
        summary.training_accuracy = history["accuracy"][-1]
    if history.get("val_accuracy"):
        summary.validation_accuracy = history["val_accuracy"][-1]
    if history.get("loss"):
        summary.training_loss = history["loss"][-1]
    if history.get("val_loss"):
        summary.validation_loss = history["val_loss"][-1]
    db.session.add(summary)
    db.session.flush()
    save_evaluation_details(summary, result.get("misclassified_images") or [], commit=False)
    return summary


def get_latest_evaluation_summary() -> Optional[EvaluasiModel]:
    from models import EvaluasiModel
    from models.evaluasi_model import TIPE_RINGKASAN

    return EvaluasiModel.query.filter_by(tipe_record=TIPE_RINGKASAN).order_by(EvaluasiModel.dibuat_pada.desc()).first()


def list_kelas_khat() -> List:
    from models import KelasKhat

    return KelasKhat.query.order_by(KelasKhat.id.asc()).all()
