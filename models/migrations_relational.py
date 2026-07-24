"""Relational schema migration, seeding, backfill, and integrity checks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import unquote_plus

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import aliased

from models.database import db
from models import (
    AlgorithmCalculation,
    ClassificationHistory,
    ClassificationResult,
    CorrectionLog,
    Dataset,
    DatasetImage,
    EvaluationResult,
    EvaluationSession,
    KhatClass,
    ModelEvaluation,
    ModelVersion,
    PredictionProbability,
)
from services.khat_characteristics_service import KHAT_CHARACTERISTICS, CLASS_DISPLAY


DEFAULT_CLASSES = [
    ("naskhi", "Naskhi"),
    ("diwani", "Diwani"),
    ("diwani_jali", "Diwani Jali"),
    ("tsuluts", "Tsuluts"),
]


def _table_exists(name: str) -> bool:
    return name in inspect(db.engine).get_table_names()


def _column_exists(table: str, column: str) -> bool:
    if not _table_exists(table):
        return False
    cols = {c["name"] for c in inspect(db.engine).get_columns(table)}
    return column in cols


def ensure_model_version_relational_columns() -> None:
    """Extend legacy model_versions with relational metadata columns."""
    if not _table_exists("model_versions"):
        return
    alters = []
    mapping = {
        "version": "VARCHAR(50) NULL",
        "source": "VARCHAR(100) NULL",
        "runtime": "VARCHAR(100) NULL",
        "model_file": "VARCHAR(512) NULL",
        "metadata_file": "VARCHAR(512) NULL",
        "weights_file": "VARCHAR(512) NULL",
        "input_width": "INT NULL",
        "input_height": "INT NULL",
        "labels_json": "TEXT NULL",
        "status": "VARCHAR(30) NULL DEFAULT 'active'",
        "precision_score": "FLOAT NULL",
        "recall_score": "FLOAT NULL",
        "f1_score": "FLOAT NULL",
        "activated_at": "DATETIME NULL",
    }
    for col, ddl in mapping.items():
        if not _column_exists("model_versions", col):
            alters.append(f"ADD COLUMN {col} {ddl}")
    for clause in alters:
        db.session.execute(text(f"ALTER TABLE model_versions {clause}"))
    if alters:
        db.session.commit()


def seed_khat_classes(config: Optional[Dict] = None) -> Dict[str, int]:
    """Ensure khat_classes rows exist; return slug -> id map."""
    slug_to_id: Dict[str, int] = {}
    for slug, display in DEFAULT_CLASSES:
        meta = KHAT_CHARACTERISTICS.get(slug, {})
        row = KhatClass.query.filter_by(slug=slug).first()
        if not row:
            row = KhatClass(
                name=display,
                slug=slug,
                description=meta.get("description"),
                main_characteristics=json.dumps(meta.get("characteristics", []), ensure_ascii=False),
            )
            db.session.add(row)
            db.session.flush()
        slug_to_id[slug] = row.id
    db.session.commit()
    return slug_to_id


def resolve_class_id(slug: Optional[str], slug_map: Optional[Dict[str, int]] = None) -> Optional[int]:
    if not slug:
        return None
    key = str(slug).strip().lower().replace(" ", "_")
    if slug_map and key in slug_map:
        return slug_map[key]
    row = KhatClass.query.filter_by(slug=key).first()
    return row.id if row else None


def backfill_dataset_images(slug_map: Dict[str, int]) -> int:
    if not _table_exists("dataset_images"):
        return 0
    count = 0
    for legacy in Dataset.query.all():
        if DatasetImage.query.filter_by(id_legacy=legacy.id).first():
            continue
        class_id = resolve_class_id(legacy.class_name, slug_map)
        if not class_id:
            continue
        w, h = None, None
        if legacy.image_size and "x" in legacy.image_size.lower():
            parts = legacy.image_size.lower().split("x")
            try:
                w, h = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                pass
        row = DatasetImage(
            class_id=class_id,
            filename=legacy.filename,
            original_filename=legacy.original_filename,
            stored_path=legacy.image_path,
            file_format=legacy.image_format,
            width=w,
            height=h,
            source_type=legacy.data_type or "raw",
            status="active",
            legacy_dataset_id=legacy.id,
            created_at=legacy.created_at or datetime.utcnow(),
            updated_at=legacy.created_at or datetime.utcnow(),
        )
        db.session.add(row)
        count += 1
    if count:
        db.session.commit()
    return count


def _scores_from_legacy(row: ClassificationResult) -> Dict[str, float]:
    scores = {}
    if row.softmax_scores_json:
        try:
            parsed = json.loads(row.softmax_scores_json)
            if isinstance(parsed, dict):
                scores = {k: float(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    if not scores:
        mapping = {
            "naskhi": row.naskhi_score or row.probability_naskhi,
            "diwani": row.diwani_score or row.probability_diwani,
            "diwani_jali": row.diwani_jali_score or row.probability_diwani_jali,
            "tsuluts": row.tsuluts_score or row.probability_tsuluts,
        }
        scores = {k: float(v or 0) for k, v in mapping.items()}
    return scores


def backfill_classification_history(slug_map: Dict[str, int]) -> int:
    if not _table_exists("classification_history"):
        return 0
    active_mv = ModelVersion.query.filter_by(is_active=True).first()
    mv_id = active_mv.id if active_mv else None
    count = 0
    for legacy in ClassificationResult.query.order_by(ClassificationResult.id.asc()).all():
        if ClassificationHistory.query.filter_by(id_legacy=legacy.id).first():
            continue
        predicted_id = resolve_class_id(legacy.predicted_class or legacy.top1_class, slug_map)
        expected_id = resolve_class_id(legacy.expected_class or legacy.manual_expected_class, slug_map)
        top2_id = resolve_class_id(getattr(legacy, "top_2_class", None), slug_map)
        hist = ClassificationHistory(
            user_id=None,
            model_version_id=mv_id,
            dataset_image_id=None,
            uploaded_filename=legacy.uploaded_filename or legacy.filename,
            uploaded_path=legacy.image_path,
            expected_class_id=expected_id,
            predicted_class_id=predicted_id,
            expected_class_source=legacy.expected_class_source or legacy.source_type,
            confidence_score=legacy.confidence or legacy.confidence_score,
            top2_class_id=top2_id,
            top2_score=legacy.top_2_score,
            top2_margin=legacy.top2_margin,
            validation_status=legacy.validation_status,
            final_decision=legacy.final_decision,
            input_source=legacy.input_source or "upload",
            legacy_result_id=legacy.id,
            created_at=legacy.created_at or datetime.utcnow(),
        )
        db.session.add(hist)
        db.session.flush()

        scores = _scores_from_legacy(legacy)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (slug, score) in enumerate(ranked, start=1):
            pct = score * 100 if score <= 1 else score
            cid = resolve_class_id(slug, slug_map)
            if not cid:
                continue
            db.session.add(
                PredictionProbability(
                    classification_id=hist.id,
                    class_id=cid,
                    softmax_score=score if score <= 1 else score / 100,
                    probability_percent=round(pct, 4),
                    rank_order=rank,
                )
            )

        db.session.add(
            AlgorithmCalculation(
                classification_id=hist.id,
                original_width=legacy.original_width,
                original_height=legacy.original_height,
                processed_width=legacy.processed_width,
                processed_height=legacy.processed_height,
                model_input_size=legacy.model_input_size,
                tensor_shape="[1, 224, 224, 3]",
                preprocessing_steps_json=legacy.preprocessing_steps_json,
                khat_probability=legacy.khat_probability,
                non_khat_probability=legacy.non_khat_probability,
                stage1_decision=legacy.stage1_decision or legacy.detection_decision,
                confidence_level=legacy.confidence_level or legacy.confidence_label,
                known_class_status=legacy.known_class_status,
                calculation_notes=legacy.calculation_notes,
            )
        )
        count += 1
    if count:
        db.session.commit()
    return count


def backfill_evaluation_sessions(slug_map: Dict[str, int]) -> int:
    if not _table_exists("evaluation_sessions"):
        return 0
    active_mv = ModelVersion.query.filter_by(is_active=True).first()
    mv_id = active_mv.id if active_mv else None
    count = 0
    for legacy in ModelEvaluation.query.order_by(ModelEvaluation.id.asc()).all():
        if EvaluationSession.query.filter_by(id_legacy=legacy.id).first():
            continue
        session_row = EvaluationSession(
            model_version_id=mv_id,
            total_images=None,
            correct_predictions=None,
            wrong_predictions=None,
            accuracy=legacy.accuracy,
            precision_score=legacy.precision_score,
            recall_score=legacy.recall_score,
            f1_score=legacy.f1_score,
            legacy_evaluation_id=legacy.id,
            completed_at=legacy.created_at,
        )
        db.session.add(session_row)
        count += 1
    if count:
        db.session.commit()
    return count


def backup_database_before_migration(config: Dict) -> Optional[str]:
    """Create a SQL dump before schema migration (MySQL). Returns backup path or None."""
    backup_dir = os.path.join(config.get("INSTANCE_DIR") or config["BASE_DIR"], "db_backups")
    os.makedirs(backup_dir, exist_ok=True)
    db_name = config["SQLALCHEMY_DATABASE_URI"].rsplit("/", 1)[-1]
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(backup_dir, f"{db_name}_{ts}.sql")
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
            "--routines",
            db_name,
        ]
        with open(outfile, "w", encoding="utf-8") as handle:
            subprocess.run(cmd, stdout=handle, stderr=subprocess.PIPE, check=True, text=True)
        return outfile
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        marker = os.path.join(backup_dir, f"{db_name}_{ts}_backup_skipped.txt")
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write("mysqldump unavailable or failed; relational migration proceeded without SQL dump.\n")
        return None


def backfill_evaluation_results(config: Dict, slug_map: Dict[str, int]) -> int:
    """Populate evaluation_results from JSON misclassification reports."""
    if not _table_exists("evaluation_results"):
        return 0
    from services.prediction_audit_service import load_misclassification_report
    from services.relational_sync_service import sync_evaluation_results_from_misclassified

    payload = load_misclassification_report(config)
    rows = payload.get("misclassified_images") or []
    if not rows:
        eval_path = config.get("EVALUATION_RESULT_PATH")
        if eval_path and os.path.isfile(eval_path):
            try:
                with open(eval_path, "r", encoding="utf-8") as handle:
                    eval_json = json.load(handle)
                rows = eval_json.get("misclassified_images") or []
            except (json.JSONDecodeError, OSError):
                rows = []

    if not rows:
        return 0

    session_row = EvaluationSession.query.order_by(EvaluationSession.id.desc()).first()
    if not session_row:
        return 0

    existing = EvaluationResult.query.filter_by(evaluation_session_id=session_row.id).count()
    if existing >= len(rows):
        return 0

    count = sync_evaluation_results_from_misclassified(session_row, rows)
    if count:
        session_row.wrong_predictions = count
        if session_row.total_images and session_row.correct_predictions is None:
            session_row.correct_predictions = max(session_row.total_images - count, 0)
        db.session.commit()
    return count


def backfill_correction_logs_from_json(config: Dict, slug_map: Dict[str, int]) -> int:
    """Migrate evaluation_corrections.json into correction_logs."""
    if not _table_exists("correction_logs"):
        return 0
    path = os.path.join(config["BASE_DIR"], "model", "evaluation_corrections.json")
    if not os.path.isfile(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return 0

    records = data.get("records") or {}
    count = 0
    for image_rel, rec in records.items():
        label = rec.get("correction_label")
        corrected_id = resolve_class_id(label, slug_map)
        if not corrected_id:
            continue
        eval_result = EvaluationResult.query.filter(
            EvaluationResult.stored_filename.contains(os.path.basename(image_rel))
        ).first()
        if eval_result and CorrectionLog.query.filter_by(evaluation_result_id=eval_result.id).first():
            continue
        db.session.add(
            CorrectionLog(
                evaluation_result_id=eval_result.id if eval_result else None,
                old_class_id=eval_result.true_class_id if eval_result else None,
                corrected_class_id=corrected_id,
                correction_note=rec.get("correction_note"),
                correction_status=rec.get("reviewed_status") or "corrected",
                reviewed_at=datetime.utcnow(),
            )
        )
        count += 1
    if count:
        db.session.commit()
    return count


def sync_model_version_relational_fields(config: Dict) -> None:
    """Copy TM file paths and metrics into extended model_versions columns."""
    for mv in ModelVersion.query.all():
        updated = False
        if not mv.version and mv.model_version:
            mv.version = mv.model_version
            updated = True
        if not mv.source and mv.model_source:
            mv.source = mv.model_source
            updated = True
        if not mv.runtime and mv.model_runtime:
            mv.runtime = mv.model_runtime
            updated = True
        if not mv.model_file:
            mv.model_file = config.get("TEACHABLE_MODEL_JSON")
            updated = True
        if not mv.metadata_file:
            mv.metadata_file = config.get("TEACHABLE_MODEL_METADATA")
            updated = True
        if not mv.weights_file:
            mv.weights_file = config.get("TEACHABLE_MODEL_WEIGHTS")
            updated = True
        if mv.input_width is None:
            mv.input_width = 224
            updated = True
        if mv.input_height is None:
            mv.input_height = 224
            updated = True
        if not mv.labels_json:
            mv.labels_json = json.dumps(config.get("CLASS_LABELS", []))
            updated = True
        if not mv.status:
            mv.status = "active" if mv.is_active else "archived"
            updated = True
        if mv.precision_score is None and mv.evaluation_accuracy is not None:
            pass
        if updated:
            db.session.add(mv)
    db.session.commit()


def run_relational_migration(config: Optional[Dict] = None) -> Dict[str, Any]:
    """Create relational tables, seed classes, backfill legacy data."""
    from flask import current_app

    cfg = config or current_app.config
    backup_path = backup_database_before_migration(cfg)
    db.create_all()
    ensure_model_version_relational_columns()

    slug_map = seed_khat_classes(cfg)
    sync_model_version_relational_fields(cfg)

    return {
        "backup_path": backup_path,
        "khat_classes": len(slug_map),
        "dataset_images_backfilled": backfill_dataset_images(slug_map),
        "classification_history_backfilled": backfill_classification_history(slug_map),
        "evaluation_sessions_backfilled": backfill_evaluation_sessions(slug_map),
        "evaluation_results_backfilled": backfill_evaluation_results(cfg, slug_map),
        "correction_logs_backfilled": backfill_correction_logs_from_json(cfg, slug_map),
    }


def build_relationship_report() -> Dict[str, Any]:
    """Analyze relational vs legacy schema state."""
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()

    relational_tables = [
        "khat_classes",
        "dataset_images",
        "classification_history",
        "prediction_probabilities",
        "algorithm_calculations",
        "evaluation_sessions",
        "evaluation_results",
        "correction_logs",
    ]
    legacy_tables = ["datasets", "classification_results", "model_evaluations", "model_versions", "users"]

    fk_map: Dict[str, List[Dict[str, str]]] = {}
    for table in relational_tables + ["model_versions"]:
        if table not in tables:
            continue
        fks = inspector.get_foreign_keys(table)
        fk_map[table] = [
            {
                "column": fk["constrained_columns"][0] if fk.get("constrained_columns") else "",
                "references": f"{fk.get('referred_table')}.{fk.get('referred_columns', [''])[0]}",
            }
            for fk in fks
        ]

    counts = {
        "khat_classes": KhatClass.query.count() if "khat_classes" in tables else 0,
        "dataset_images": DatasetImage.query.count() if "dataset_images" in tables else 0,
        "datasets_legacy": Dataset.query.count(),
        "classification_history": ClassificationHistory.query.count() if "classification_history" in tables else 0,
        "classification_results_legacy": ClassificationResult.query.count(),
        "prediction_probabilities": PredictionProbability.query.count() if "prediction_probabilities" in tables else 0,
        "algorithm_calculations": AlgorithmCalculation.query.count() if "algorithm_calculations" in tables else 0,
        "evaluation_sessions": EvaluationSession.query.count() if "evaluation_sessions" in tables else 0,
        "evaluation_results": EvaluationResult.query.count() if "evaluation_results" in tables else 0,
        "correction_logs": CorrectionLog.query.count() if "correction_logs" in tables else 0,
        "model_versions": ModelVersion.query.count(),
        "model_evaluations_legacy": ModelEvaluation.query.count(),
    }

    issues: List[str] = []
    if counts["datasets_legacy"] and counts["dataset_images"] < counts["datasets_legacy"]:
        issues.append(
            f"{counts['datasets_legacy'] - counts['dataset_images']} legacy dataset rows not yet linked in dataset_images."
        )
    if counts["classification_results_legacy"] and counts["classification_history"] < counts["classification_results_legacy"]:
        issues.append(
            f"{counts['classification_results_legacy'] - counts['classification_history']} legacy classification rows not yet linked in classification_history."
        )

    orphan_probs = 0
    if "prediction_probabilities" in tables:
        orphan_probs = (
            db.session.query(PredictionProbability)
            .outerjoin(ClassificationHistory, PredictionProbability.id_riwayat == ClassificationHistory.id)
            .filter(ClassificationHistory.id.is_(None))
            .count()
        )
        if orphan_probs:
            issues.append(f"{orphan_probs} prediction_probabilities rows without classification_history parent.")

    if "classification_history" in tables and ClassificationHistory.query.count():
        missing_mv = ClassificationHistory.query.filter(
            ClassificationHistory.id_model.is_(None)
        ).count()
        if missing_mv:
            issues.append(f"{missing_mv} classification_history rows without model_version_id.")

    if "evaluation_results" in tables and EvaluationResult.query.count():
        missing_img = EvaluationResult.query.filter(EvaluationResult.dataset_image_id.is_(None)).count()
        if missing_img:
            issues.append(
                f"{missing_img} evaluation_results rows without dataset_image_id (test holdout may be external)."
            )

    if "correction_logs" in tables and CorrectionLog.query.count():
        unlinked = CorrectionLog.query.filter(
            CorrectionLog.id_riwayat.is_(None),
            CorrectionLog.evaluation_result_id.is_(None),
        ).count()
        if unlinked:
            issues.append(f"{unlinked} correction_logs not linked to classification or evaluation result.")

    if "algorithm_calculations" in tables:
        orphan_calc = (
            db.session.query(AlgorithmCalculation)
            .outerjoin(ClassificationHistory, AlgorithmCalculation.id_riwayat == ClassificationHistory.id)
            .filter(ClassificationHistory.id.is_(None))
            .count()
        )
        if orphan_calc:
            issues.append(f"{orphan_calc} algorithm_calculations rows without classification_history parent.")

    return {
        "is_relational": bool(fk_map.get("classification_history")),
        "relational_tables_present": [t for t in relational_tables if t in tables],
        "legacy_tables_present": [t for t in legacy_tables if t in tables],
        "foreign_keys": fk_map,
        "row_counts": counts,
        "issues": issues,
        "summary": (
            "Database uses formal foreign keys and normalized relational tables."
            if fk_map.get("classification_history")
            else "Database is still primarily legacy flat tables without foreign keys."
        ),
    }


def run_integrity_checks() -> Dict[str, Any]:
    """Validate referential integrity for relational tables."""
    report = build_relationship_report()
    checks = []
    if ClassificationHistory.query.count():
        missing_predicted = ClassificationHistory.query.filter(
            ClassificationHistory.id_kelas_prediksi.is_(None)
        ).count()
        checks.append(
            {
                "name": "riwayat_klasifikasi.id_kelas_prediksi",
                "ok": missing_predicted == 0,
                "detail": f"{missing_predicted} rows missing id_kelas_prediksi",
            }
        )
        missing_mv = ClassificationHistory.query.filter(
            ClassificationHistory.id_model.is_(None)
        ).count()
        checks.append(
            {
                "name": "riwayat_klasifikasi.id_model",
                "ok": missing_mv == 0,
                "detail": f"{missing_mv} rows missing id_model",
            }
        )
    from models.evaluasi_model import EvaluasiModel, TIPE_DETAIL

    if EvaluasiModel.query.filter_by(tipe_record=TIPE_DETAIL).count():
        detail = aliased(EvaluasiModel)
        summary = aliased(EvaluasiModel)
        missing_session = (
            db.session.query(detail)
            .outerjoin(summary, detail.id_induk == summary.id)
            .filter(detail.tipe_record == TIPE_DETAIL, summary.id.is_(None))
            .count()
        )
        checks.append(
            {
                "name": "evaluasi_model.id_induk",
                "ok": missing_session == 0,
                "detail": f"{missing_session} orphan evaluasi detail rows",
            }
        )
    if AlgorithmCalculation.query.count():
        orphan_calc = (
            db.session.query(AlgorithmCalculation)
            .outerjoin(ClassificationHistory, AlgorithmCalculation.id_riwayat == ClassificationHistory.id)
            .filter(ClassificationHistory.id.is_(None))
            .count()
        )
        checks.append(
            {
                "name": "algorithm_calculations.classification_id",
                "ok": orphan_calc == 0,
                "detail": f"{orphan_calc} orphan algorithm_calculations",
            }
        )
    if PredictionProbability.query.count():
        orphan_probs = (
            db.session.query(PredictionProbability)
            .outerjoin(ClassificationHistory, PredictionProbability.id_riwayat == ClassificationHistory.id)
            .filter(ClassificationHistory.id.is_(None))
            .count()
        )
        checks.append(
            {
                "name": "prediction_probabilities.classification_id",
                "ok": orphan_probs == 0,
                "detail": f"{orphan_probs} orphan prediction_probabilities",
            }
        )
    report["integrity_checks"] = checks
    report["integrity_ok"] = all(c["ok"] for c in checks) if checks else True
    return report
