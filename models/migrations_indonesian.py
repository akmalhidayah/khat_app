"""Migration from English schema to Indonesian 3NF tables."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import unquote_plus

from sqlalchemy import inspect, text

from models.database import db
from services.db_compat import get_class_id_by_slug


ENGLISH_TABLES = [
    "users",
    "datasets",
    "classification_results",
    "model_evaluations",
    "model_versions",
    "khat_classes",
    "dataset_images",
    "classification_history",
    "prediction_probabilities",
    "algorithm_calculations",
    "evaluation_sessions",
    "evaluation_results",
    "correction_logs",
]

INDONESIAN_TABLES = [
    "pengguna",
    "kelas_khat",
    "gambar_dataset",
    "versi_model",
    "riwayat_klasifikasi",
    "probabilitas_prediksi",
    "perhitungan_algoritma",
    "evaluasi_model",
    "log_koreksi",
]

OLD_SPLIT_TABLES = ["sesi_evaluasi", "hasil_evaluasi"]

BACKUP_SUFFIX = "_backup_en"


def _tables() -> List[str]:
    return inspect(db.engine).get_table_names()


def _table_exists(name: str) -> bool:
    return name in _tables()


def backup_database(config: Dict) -> Optional[str]:
    backup_dir = os.path.join(config.get("INSTANCE_DIR") or config["BASE_DIR"], "db_backups")
    os.makedirs(backup_dir, exist_ok=True)
    db_name = config["SQLALCHEMY_DATABASE_URI"].rsplit("/", 1)[-1]
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(backup_dir, f"{db_name}_id_{ts}.sql")
    try:
        from config import Config

        password = unquote_plus(Config._DB_PASSWORD)
        cmd = [
            "mysqldump",
            f"-h{Config._DB_HOST}",
            f"-P{Config._DB_PORT}",
            f"-u{Config._DB_USER}",
            f"-p{password}",
            "--single-transaction",
            db_name,
        ]
        with open(outfile, "w", encoding="utf-8") as handle:
            subprocess.run(cmd, stdout=handle, stderr=subprocess.PIPE, check=True, text=True)
        return outfile
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        marker = os.path.join(backup_dir, f"{db_name}_id_{ts}_backup_skipped.txt")
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write("Backup skipped — mysqldump unavailable.\n")
        return None


def _rename_old_tables() -> List[str]:
    renamed = []
    for table in ENGLISH_TABLES:
        if not _table_exists(table):
            continue
        backup_name = f"{table}{BACKUP_SUFFIX}"
        if _table_exists(backup_name):
            continue
        db.session.execute(text(f"RENAME TABLE `{table}` TO `{backup_name}`"))
        renamed.append(table)
    if renamed:
        db.session.commit()
    return renamed


def seed_kelas_khat(config: Dict) -> Dict[str, int]:
    from models import KelasKhat
    from services.khat_characteristics_service import KHAT_CHARACTERISTICS

    defaults = [
        ("naskhi", "Khat Naskhi"),
        ("riqah", "Khat Riq'ah"),
        ("diwani", "Khat Diwani"),
        ("kufi", "Khat Kufi"),
    ]
    slug_map: Dict[str, int] = {}
    for slug, display in defaults:
        row = KelasKhat.query.filter_by(slug=slug).first()
        if not row:
            meta = KHAT_CHARACTERISTICS.get(slug, {})
            row = KelasKhat(
                nama_kelas=display,
                slug=slug,
                deskripsi=meta.get("description"),
                ciri_utama=json.dumps(meta.get("characteristics", []), ensure_ascii=False),
            )
            db.session.add(row)
            db.session.flush()
        slug_map[slug] = row.id
    db.session.commit()
    return slug_map


def _copy_from_english_kelas(slug_map: Dict[str, int]) -> None:
    backup = f"khat_classes{BACKUP_SUFFIX}"
    source = backup if _table_exists(backup) else ("khat_classes" if _table_exists("khat_classes") else None)
    if not source:
        return
    from models import KelasKhat

    rows = db.session.execute(text(f"SELECT * FROM `{source}`")).mappings().all()
    for row in rows:
        slug = row.get("slug")
        if not slug or KelasKhat.query.filter_by(slug=slug).first():
            continue
        k = KelasKhat(
            nama_kelas=row.get("name") or row.get("nama_kelas"),
            slug=slug,
            deskripsi=row.get("description") or row.get("deskripsi"),
            ciri_utama=row.get("main_characteristics") or row.get("ciri_utama"),
        )
        db.session.add(k)
    db.session.commit()
    for k in KelasKhat.query.all():
        slug_map[k.slug] = k.id


def _migrate_pengguna() -> int:
    from models import Pengguna

    if Pengguna.query.count():
        return 0
    source = f"users{BACKUP_SUFFIX}" if _table_exists(f"users{BACKUP_SUFFIX}") else ("users" if _table_exists("users") else None)
    if not source:
        return 0
    rows = db.session.execute(text(f"SELECT * FROM `{source}`")).mappings().all()
    count = 0
    for row in rows:
        db.session.add(
            Pengguna(
                id=row["id"],
                nama_lengkap=row.get("name") or row.get("nama_lengkap") or "User",
                username=row["username"],
                password_hash=row.get("password") or row.get("password_hash"),
                peran=row.get("role") or row.get("peran") or "admin",
                status=row.get("status") or "aktif",
                dibuat_pada=row.get("created_at") or row.get("dibuat_pada") or datetime.utcnow(),
            )
        )
        count += 1
    if count:
        db.session.commit()
    return count


def _migrate_versi_model(config: Dict) -> int:
    from models import VersiModel

    if VersiModel.query.count():
        return 0
    source = (
        f"model_versions{BACKUP_SUFFIX}"
        if _table_exists(f"model_versions{BACKUP_SUFFIX}")
        else ("model_versions" if _table_exists("model_versions") else None)
    )
    if not source:
        return 0
    rows = db.session.execute(text(f"SELECT * FROM `{source}`")).mappings().all()
    count = 0
    for row in rows:
        db.session.add(
            VersiModel(
                id=row["id"],
                versi=row.get("version") or row.get("model_version") or "1.0",
                nama_model=row.get("model_name") or row.get("nama_model") or "model",
                sumber_model=row.get("model_source") or row.get("source") or "teachable_machine",
                runtime=row.get("model_runtime") or row.get("runtime"),
                file_model=row.get("model_file") or config.get("TEACHABLE_MODEL_JSON"),
                file_metadata=row.get("metadata_file") or config.get("TEACHABLE_MODEL_METADATA"),
                file_weights=row.get("weights_file") or config.get("TEACHABLE_MODEL_WEIGHTS"),
                lebar_input=row.get("input_width") or 224,
                tinggi_input=row.get("input_height") or 224,
                label_json=row.get("labels_json") or json.dumps(config.get("CLASS_LABELS", [])),
                status=row.get("status") or ("active" if row.get("is_active") else "archived"),
                akurasi=row.get("evaluation_accuracy") or row.get("akurasi"),
                presisi=row.get("precision_score") or row.get("presisi"),
                recall=row.get("recall_score") or row.get("recall"),
                f1_score=row.get("f1_score"),
                jumlah_kelas=row.get("num_classes") or 4,
                jumlah_gambar_dataset=row.get("dataset_images"),
                catatan=row.get("notes") or row.get("catatan"),
                aktif=bool(row.get("is_active")),
                dibuat_pada=row.get("created_at") or datetime.utcnow(),
                diaktifkan_pada=row.get("activated_at") or row.get("training_date"),
            )
        )
        count += 1
    if count:
        db.session.commit()
    return count


def _migrate_gambar_dataset(slug_map: Dict[str, int]) -> int:
    from models import GambarDataset

    if GambarDataset.query.count():
        return 0
    sources = []
    if _table_exists(f"dataset_images{BACKUP_SUFFIX}"):
        sources.append(f"dataset_images{BACKUP_SUFFIX}")
    elif _table_exists("dataset_images"):
        sources.append("dataset_images")
    if _table_exists(f"datasets{BACKUP_SUFFIX}"):
        sources.append(f"datasets{BACKUP_SUFFIX}")
    elif _table_exists("datasets"):
        sources.append("datasets")

    count = 0
    seen_legacy = set()
    for source in sources:
        rows = db.session.execute(text(f"SELECT * FROM `{source}`")).mappings().all()
        for row in rows:
            legacy_id = row.get("legacy_dataset_id") or row.get("id")
            if legacy_id in seen_legacy:
                continue
            class_slug = row.get("class_name")
            class_id = row.get("class_id") or row.get("id_kelas") or get_class_id_by_slug(class_slug)
            if not class_id:
                continue
            seen_legacy.add(legacy_id)
            w, h = row.get("width"), row.get("height")
            if not w and row.get("image_size") and "x" in str(row["image_size"]).lower():
                parts = str(row["image_size"]).lower().split("x")
                try:
                    w, h = int(parts[0]), int(parts[1])
                except (ValueError, IndexError):
                    pass
            db.session.add(
                GambarDataset(
                    id_kelas=class_id,
                    nama_file=row.get("filename") or row.get("nama_file"),
                    nama_file_asli=row.get("original_filename") or row.get("nama_file_asli") or row.get("filename"),
                    path_asli=row.get("stored_path") or row.get("image_path") or row.get("path_asli"),
                    path_proses=row.get("processed_path") or row.get("path_proses"),
                    path_model_ready=row.get("model_ready_path") or row.get("path_model_ready"),
                    format_file=row.get("file_format") or row.get("image_format") or row.get("format_file"),
                    lebar=w,
                    tinggi=h,
                    sumber_data=row.get("source_type") or row.get("data_type") or row.get("sumber_data") or "raw",
                    hash_file=row.get("hash_value") or row.get("hash_file"),
                    status=row.get("status") or "active",
                    id_legacy=legacy_id,
                    dibuat_pada=row.get("created_at") or row.get("dibuat_pada") or datetime.utcnow(),
                )
            )
            count += 1
    if count:
        db.session.commit()
    return count


def _extra_from_legacy_classification(row: dict) -> str:
    keys = [
        "filename", "input_status", "is_khat", "rejection_reason", "reliability_level",
        "review_status", "confidence_label", "explanation_text", "model_name", "model_source",
        "model_runtime", "similarity_status", "similarity_score", "nearest_dataset_image",
        "nearest_dataset_class", "similarity_risk_level", "similarity_message",
        "similarity_recommendation", "manual_expected_class", "correction_label",
        "correction_notes", "source_type", "preprocessing_mode", "detection_status",
        "detection_decision", "softmax_scores_json", "preprocessing_steps_json",
        "calculation_notes", "predicted_class", "expected_class", "top_2_class",
        "naskhi_score", "diwani_score", "diwani_jali_score", "tsuluts_score",
        "khat_probability", "non_khat_probability", "known_class_status",
        "original_width", "original_height", "processed_width", "processed_height",
        "manual_review_required",
    ]
    extra = {k: row[k] for k in keys if k in row and row[k] is not None}
    return json.dumps(extra, ensure_ascii=False) if extra else None


def _migrate_riwayat(slug_map: Dict[str, int]) -> int:
    from models import RiwayatKlasifikasi, VersiModel
    from services.db_compat import save_perhitungan_algoritma, save_prediction_probabilities

    if RiwayatKlasifikasi.query.count():
        return 0
    mv = VersiModel.query.filter_by(aktif=True).first()
    mv_id = mv.id if mv else None

    sources = []
    if _table_exists(f"classification_history{BACKUP_SUFFIX}"):
        sources.append(("relational", f"classification_history{BACKUP_SUFFIX}"))
    elif _table_exists("classification_history"):
        sources.append(("relational", "classification_history"))
    if _table_exists(f"classification_results{BACKUP_SUFFIX}"):
        sources.append(("legacy", f"classification_results{BACKUP_SUFFIX}"))
    elif _table_exists("classification_results"):
        sources.append(("legacy", "classification_results"))

    count = 0
    seen_legacy = set()
    for kind, source in sources:
        rows = db.session.execute(text(f"SELECT * FROM `{source}`")).mappings().all()
        for row in rows:
            legacy_id = row.get("legacy_result_id") or row.get("id")
            if legacy_id in seen_legacy:
                continue
            seen_legacy.add(legacy_id)
            if kind == "relational":
                pred_id = row.get("predicted_class_id") or row.get("id_kelas_prediksi")
                exp_id = row.get("expected_class_id") or row.get("id_kelas_diharapkan")
                top2_id = row.get("top2_class_id") or row.get("id_kelas_top2")
                rk = RiwayatKlasifikasi(
                    id_pengguna=row.get("user_id") or row.get("id_pengguna"),
                    id_model=row.get("model_version_id") or row.get("id_model") or mv_id,
                    id_gambar_dataset=row.get("dataset_image_id") or row.get("id_gambar_dataset"),
                    nama_file_upload=row.get("uploaded_filename") or row.get("nama_file_upload"),
                    path_upload=row.get("uploaded_path") or row.get("path_upload"),
                    id_kelas_diharapkan=exp_id,
                    id_kelas_prediksi=pred_id,
                    sumber_kelas_diharapkan=row.get("expected_class_source") or row.get("sumber_kelas_diharapkan"),
                    confidence=row.get("confidence_score") or row.get("confidence"),
                    id_kelas_top2=top2_id,
                    skor_top2=row.get("top2_score") or row.get("top_2_score") or row.get("skor_top2"),
                    margin_top2=row.get("top2_margin") or row.get("margin_top2"),
                    status_validasi=row.get("validation_status") or row.get("status_validasi"),
                    keputusan_akhir=row.get("final_decision") or row.get("keputusan_akhir"),
                    sumber_input=row.get("input_source") or row.get("sumber_input") or "upload",
                    id_legacy=legacy_id,
                    dibuat_pada=row.get("created_at") or row.get("dibuat_pada") or datetime.utcnow(),
                )
            else:
                rk = RiwayatKlasifikasi(
                    id_model=mv_id,
                    nama_file_upload=row.get("uploaded_filename") or row.get("filename"),
                    path_upload=row.get("image_path") or row.get("path_upload"),
                    id_kelas_prediksi=get_class_id_by_slug(row.get("predicted_class")),
                    id_kelas_diharapkan=get_class_id_by_slug(row.get("expected_class") or row.get("manual_expected_class")),
                    id_kelas_top2=get_class_id_by_slug(row.get("top_2_class")),
                    confidence=row.get("confidence") or row.get("confidence_score"),
                    skor_top2=row.get("top_2_score"),
                    margin_top2=row.get("top2_margin"),
                    status_validasi=row.get("validation_status"),
                    keputusan_akhir=row.get("final_decision"),
                    sumber_input=row.get("input_source") or "upload",
                    sumber_kelas_diharapkan=row.get("expected_class_source") or row.get("source_type"),
                    id_legacy=legacy_id,
                    data_ekstra_json=_extra_from_legacy_classification(dict(row)),
                    dibuat_pada=row.get("created_at") or datetime.utcnow(),
                )
            db.session.add(rk)
            db.session.flush()

            scores = {}
            if row.get("softmax_scores_json"):
                try:
                    scores = json.loads(row["softmax_scores_json"])
                except (json.JSONDecodeError, TypeError):
                    pass
            if not scores:
                scores = {
                    "naskhi": float(row.get("naskhi_score") or row.get("probability_naskhi") or 0),
                    "diwani": float(row.get("diwani_score") or row.get("probability_diwani") or 0),
                    "diwani_jali": float(row.get("diwani_jali_score") or row.get("probability_diwani_jali") or 0),
                    "tsuluts": float(row.get("tsuluts_score") or row.get("probability_tsuluts") or 0),
                }
            save_prediction_probabilities(rk.id, scores)
            save_perhitungan_algoritma(
                rk.id,
                {
                    "original_width": row.get("original_width"),
                    "original_height": row.get("original_height"),
                    "processed_width": row.get("processed_width") or 224,
                    "processed_height": row.get("processed_height") or 224,
                    "model_input_size": row.get("model_input_size") or "224x224",
                    "tensor_shape": "[1, 224, 224, 3]",
                    "preprocessing_steps_json": row.get("preprocessing_steps_json"),
                    "khat_probability": row.get("khat_probability"),
                    "non_khat_probability": row.get("non_khat_probability"),
                    "stage1_decision": row.get("stage1_decision") or row.get("detection_decision"),
                    "confidence_level": row.get("confidence_level") or row.get("confidence_label"),
                    "known_class_status": row.get("known_class_status"),
                    "calculation_notes": row.get("calculation_notes"),
                },
            )
            count += 1
    if count:
        db.session.commit()
    return count


def _ensure_gambar_dataset_columns() -> None:
    if not _table_exists("gambar_dataset"):
        return
    cols = {c["name"] for c in inspect(db.engine).get_columns("gambar_dataset")}
    if "split_data" not in cols:
        db.session.execute(text("ALTER TABLE gambar_dataset ADD COLUMN split_data VARCHAR(20) NULL"))
        db.session.commit()


def _ensure_log_koreksi_columns() -> None:
    """Rename id_hasil_evaluasi → id_evaluasi and point FK to evaluasi_model."""
    if not _table_exists("log_koreksi"):
        return
    cols = {c["name"] for c in inspect(db.engine).get_columns("log_koreksi")}
    if "id_evaluasi" in cols:
        return
    if "id_hasil_evaluasi" not in cols:
        return
    for fk in inspect(db.engine).get_foreign_keys("log_koreksi"):
        if "id_hasil_evaluasi" in fk.get("constrained_columns", []):
            db.session.execute(text(f"ALTER TABLE log_koreksi DROP FOREIGN KEY `{fk['name']}`"))
    db.session.execute(
        text("ALTER TABLE log_koreksi CHANGE COLUMN id_hasil_evaluasi id_evaluasi INT NULL")
    )
    if _table_exists("evaluasi_model"):
        db.session.execute(
            text(
                "ALTER TABLE log_koreksi ADD CONSTRAINT log_koreksi_id_evaluasi_fk "
                "FOREIGN KEY (id_evaluasi) REFERENCES evaluasi_model(id)"
            )
        )
    db.session.commit()


def _rename_leftover_tables() -> List[str]:
    """Rename stray English tables that were not caught by the main rename pass."""
    renamed = []
    pairs = [
        ("model_versions", f"model_versions{BACKUP_SUFFIX}"),
    ]
    for table, backup in pairs:
        if not _table_exists(table) or _table_exists(backup):
            continue
        db.session.execute(text(f"RENAME TABLE `{table}` TO `{backup}`"))
        renamed.append(table)
    if renamed:
        db.session.commit()
    return renamed


def drop_unused_backup_tables() -> List[str]:
    """Remove legacy English and backup tables after data has been migrated to the 9 core tables."""
    core = set(INDONESIAN_TABLES)
    dropped: List[str] = []
    candidates = []
    for table in sorted(_tables()):
        if table in core:
            continue
        is_backup = table.endswith(BACKUP_SUFFIX) or table.endswith("_backup")
        is_legacy = table in ENGLISH_TABLES or table in OLD_SPLIT_TABLES
        if is_backup or is_legacy:
            candidates.append(table)
    if not candidates:
        return dropped
    db.session.execute(text("SET FOREIGN_KEY_CHECKS=0"))
    try:
        for table in candidates:
            db.session.execute(text(f"DROP TABLE IF EXISTS `{table}`"))
            dropped.append(table)
        db.session.commit()
    finally:
        db.session.execute(text("SET FOREIGN_KEY_CHECKS=1"))
        db.session.commit()
    return dropped


def _rename_old_split_tables() -> List[str]:
    renamed = []
    for table in OLD_SPLIT_TABLES:
        if not _table_exists(table):
            continue
        backup = f"{table}_backup"
        if _table_exists(backup):
            continue
        db.session.execute(text(f"RENAME TABLE `{table}` TO `{backup}`"))
        renamed.append(table)
    if renamed:
        db.session.commit()
    return renamed


def _migrate_evaluasi_model(config: Dict, slug_map: Dict[str, int]) -> int:
    from models import EvaluasiModel, VersiModel
    from models.evaluasi_model import TIPE_DETAIL, TIPE_RINGKASAN

    if EvaluasiModel.query.count() and not any(_table_exists(t) for t in OLD_SPLIT_TABLES):
        return 0

    mv = VersiModel.query.filter_by(aktif=True).first()
    mv_id = mv.id if mv else None
    count = 0
    now = datetime.utcnow()

    sesi_source = None
    for name in ("sesi_evaluasi", "sesi_evaluasi_backup", f"sesi_evaluasi{BACKUP_SUFFIX}"):
        if _table_exists(name):
            sesi_source = name
            break

    if sesi_source:
        sesi_rows = db.session.execute(text(f"SELECT * FROM `{sesi_source}`")).mappings().all()
        for row in sesi_rows:
            summary = EvaluasiModel(
                id_model=row.get("id_model") or mv_id,
                total_gambar=row.get("total_gambar") or row.get("total_images"),
                prediksi_benar=row.get("prediksi_benar") or row.get("correct_predictions"),
                prediksi_salah=row.get("prediksi_salah") or row.get("wrong_predictions"),
                akurasi=row.get("akurasi") or row.get("accuracy"),
                presisi=row.get("presisi") or row.get("precision_score"),
                recall=row.get("recall") or row.get("recall_score"),
                f1_score=row.get("f1_score"),
                tipe_record=TIPE_RINGKASAN,
                id_legacy=row.get("id_legacy") or row.get("legacy_evaluation_id") or row.get("id"),
                dibuat_pada=row.get("selesai_pada") or row.get("dibuat_pada") or row.get("completed_at") or now,
            )
            extra = {}
            for key in ("matriks_kebingungan_json", "laporan_klasifikasi_json", "data_ekstra_json"):
                if row.get(key):
                    extra[key] = row[key]
            if extra:
                summary.data_ekstra_json = json.dumps(extra, ensure_ascii=False)
            db.session.add(summary)
            db.session.flush()
            count += 1

            hasil_source = None
            for name in ("hasil_evaluasi", "hasil_evaluasi_backup", f"hasil_evaluasi{BACKUP_SUFFIX}"):
                if _table_exists(name):
                    hasil_source = name
                    break
            if hasil_source:
                detail_rows = db.session.execute(
                    text(f"SELECT * FROM `{hasil_source}` WHERE id_sesi_evaluasi = :sid"),
                    {"sid": row.get("id")},
                ).mappings().all()
                for drow in detail_rows:
                    db.session.add(
                        EvaluasiModel(
                            id_model=summary.id_model,
                            id_induk=summary.id,
                            id_gambar_dataset=drow.get("id_gambar_dataset") or drow.get("dataset_image_id"),
                            id_kelas_asli=drow.get("id_kelas_asli") or drow.get("true_class_id"),
                            id_kelas_prediksi=drow.get("id_kelas_prediksi") or drow.get("predicted_class_id"),
                            confidence=drow.get("confidence") or drow.get("confidence_score"),
                            id_kelas_top2=drow.get("id_kelas_top2") or drow.get("top2_class_id"),
                            skor_top2=drow.get("skor_top2") or drow.get("top2_score"),
                            margin_top2=drow.get("margin_top2") or drow.get("top2_margin"),
                            benar=drow.get("benar") if drow.get("benar") is not None else drow.get("is_correct"),
                            jenis_error=drow.get("jenis_error") or drow.get("error_type"),
                            rekomendasi=drow.get("rekomendasi") or drow.get("recommendation"),
                            url_gambar=drow.get("url_gambar") or drow.get("image_url"),
                            nama_file_simpan=drow.get("nama_file_simpan") or drow.get("stored_filename"),
                            tipe_record=TIPE_DETAIL,
                            dibuat_pada=drow.get("dibuat_pada") or drow.get("created_at") or summary.dibuat_pada,
                        )
                    )
                    count += 1
        if count:
            db.session.commit()
        return count

    eval_path = config.get("EVALUATION_RESULT_PATH")
    if eval_path and os.path.isfile(eval_path):
        try:
            with open(eval_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            summary = EvaluasiModel(
                id_model=mv_id,
                total_gambar=payload.get("test_samples"),
                prediksi_benar=payload.get("correct_predictions"),
                prediksi_salah=payload.get("incorrect_predictions") or payload.get("misclassified_count"),
                akurasi=payload.get("accuracy"),
                presisi=payload.get("precision"),
                recall=payload.get("recall"),
                f1_score=payload.get("f1_score"),
                tipe_record=TIPE_RINGKASAN,
                dibuat_pada=now,
            )
            summary.data_ekstra_json = json.dumps(
                {
                    "confusion_matrix": payload.get("confusion_matrix"),
                    "classification_report": payload.get("classification_report"),
                    "model_name": payload.get("model_name"),
                },
                ensure_ascii=False,
            )
            db.session.add(summary)
            db.session.flush()
            count += 1
            from services.db_compat import save_evaluation_details

            count += save_evaluation_details(summary, payload.get("misclassified_images") or [], commit=False)
            db.session.commit()
        except (json.JSONDecodeError, OSError):
            pass
    return count


def _migrate_sesi_evaluasi(config: Dict) -> int:
    return 0


def _migrate_hasil_evaluasi(slug_map: Dict[str, int]) -> int:
    return 0


def build_schema_report() -> Dict[str, Any]:
    from models import (
        EvaluasiModel,
        GambarDataset,
        KelasKhat,
        LogKoreksi,
        Pengguna,
        PerhitunganAlgoritma,
        ProbabilitasPrediksi,
        RiwayatKlasifikasi,
        VersiModel,
    )
    from models.evaluasi_model import TIPE_DETAIL, TIPE_RINGKASAN

    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    fk_map = {}
    for table in INDONESIAN_TABLES:
        if table not in tables:
            continue
        fk_map[table] = [
            {
                "column": fk["constrained_columns"][0] if fk.get("constrained_columns") else "",
                "references": f"{fk.get('referred_table')}.{fk.get('referred_columns', [''])[0]}",
            }
            for fk in inspector.get_foreign_keys(table)
        ]
    core_count = len([t for t in INDONESIAN_TABLES if t in tables])
    return {
        "schema": "indonesian_3nf_9_tables",
        "table_count": core_count,
        "max_tables": 9,
        "tables_present": [t for t in INDONESIAN_TABLES if t in tables],
        "backup_tables": [t for t in tables if t.endswith(BACKUP_SUFFIX) or t.endswith("_backup")],
        "foreign_keys": fk_map,
        "row_counts": {
            "pengguna": Pengguna.query.count(),
            "kelas_khat": KelasKhat.query.count(),
            "gambar_dataset": GambarDataset.query.count(),
            "versi_model": VersiModel.query.count(),
            "riwayat_klasifikasi": RiwayatKlasifikasi.query.count(),
            "probabilitas_prediksi": ProbabilitasPrediksi.query.count(),
            "perhitungan_algoritma": PerhitunganAlgoritma.query.count(),
            "evaluasi_model": EvaluasiModel.query.count(),
            "evaluasi_ringkasan": EvaluasiModel.query.filter_by(tipe_record=TIPE_RINGKASAN).count(),
            "evaluasi_detail": EvaluasiModel.query.filter_by(tipe_record=TIPE_DETAIL).count(),
            "log_koreksi": LogKoreksi.query.count(),
        },
        "is_3nf_relational": core_count == 9 and bool(fk_map.get("riwayat_klasifikasi")),
    }


def run_indonesian_migration(config: Optional[Dict] = None) -> Dict[str, Any]:
    from flask import current_app

    cfg = config or current_app.config
    backup_path = backup_database(cfg)
    db.create_all()
    _ensure_gambar_dataset_columns()
    _ensure_log_koreksi_columns()

    slug_map = seed_kelas_khat(cfg)
    _copy_from_english_kelas(slug_map)

    migrated = {
        "pengguna": _migrate_pengguna(),
        "gambar_dataset": _migrate_gambar_dataset(slug_map),
        "versi_model": _migrate_versi_model(cfg),
        "riwayat_klasifikasi": _migrate_riwayat(slug_map),
        "evaluasi_model": _migrate_evaluasi_model(cfg, slug_map),
    }

    renamed_split = _rename_old_split_tables()
    renamed_en = []
    if any(_table_exists(t) for t in ENGLISH_TABLES):
        renamed_en = _rename_old_tables()
    renamed_leftover = _rename_leftover_tables()
    dropped_backup = drop_unused_backup_tables()

    report = build_schema_report()
    return {
        "backup_path": backup_path,
        "seeded_classes": len(slug_map),
        "migrated_rows": migrated,
        "renamed_split_tables": renamed_split,
        "renamed_english_tables": renamed_en,
        "renamed_leftover_tables": renamed_leftover,
        "dropped_backup_tables": dropped_backup,
        "report": report,
    }
