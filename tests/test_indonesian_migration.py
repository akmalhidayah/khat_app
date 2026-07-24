"""Tests for Indonesian 3NF schema migration."""

from app import create_app
from models import GambarDataset, KelasKhat, Pengguna, RiwayatKlasifikasi
from models.migrations_indonesian import build_schema_report, run_indonesian_migration
from services.db_compat import get_class_id_by_slug, get_dataset_summary


def test_indonesian_migration_runs():
    app = create_app()
    with app.app_context():
        result = run_indonesian_migration(app.config)
        assert result["seeded_classes"] == 4
        assert "report" in result


def test_kelas_khat_seeded():
    app = create_app()
    with app.app_context():
        run_indonesian_migration(app.config)
        assert KelasKhat.query.count() >= 4
        assert get_class_id_by_slug("naskhi") is not None


def test_schema_report_indonesian():
    app = create_app()
    with app.app_context():
        run_indonesian_migration(app.config)
        report = build_schema_report()
        assert report["schema"] == "indonesian_3nf_9_tables"
        assert "kelas_khat" in report["tables_present"]
        assert report["is_3nf_relational"]


def test_dataset_summary():
    app = create_app()
    with app.app_context():
        run_indonesian_migration(app.config)
        summary = get_dataset_summary()
        assert "total" in summary
        assert "per_class" in summary


def test_pengguna_table():
    app = create_app()
    with app.app_context():
        run_indonesian_migration(app.config)
        assert Pengguna.query.filter_by(username="admin").first() is not None


def test_riwayat_table_exists():
    app = create_app()
    with app.app_context():
        run_indonesian_migration(app.config)
        assert RiwayatKlasifikasi.query.count() >= 0
        assert GambarDataset.query.count() >= 0
