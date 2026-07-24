"""Tests for simplified 9-table schema."""

from app import create_app
from models import CORE_TABLES, EvaluasiModel, KelasKhat
from models.evaluasi_model import TIPE_DETAIL, TIPE_RINGKASAN
from models.migrations_indonesian import build_schema_report, run_indonesian_migration


def test_nine_core_tables():
    app = create_app()
    with app.app_context():
        report = build_schema_report()
        assert len(CORE_TABLES) == 9
        assert report["table_count"] == 9
        assert report["max_tables"] == 9
        assert "evaluasi_model" in report["tables_present"]
        assert "sesi_evaluasi" not in report["tables_present"]
        assert "hasil_evaluasi" not in report["tables_present"]


def test_evaluasi_model_tipe_record():
    app = create_app()
    with app.app_context():
        run_indonesian_migration(app.config)
        summary = EvaluasiModel.query.filter_by(tipe_record=TIPE_RINGKASAN).count()
        detail = EvaluasiModel.query.filter_by(tipe_record=TIPE_DETAIL).count()
        assert summary >= 0
        assert detail >= 0


def test_no_backup_tables_after_migration():
    app = create_app()
    with app.app_context():
        result = run_indonesian_migration(app.config)
        report = result["report"]
        assert report["table_count"] == 9
        assert report["backup_tables"] == []
        assert len(report["tables_present"]) == 9
