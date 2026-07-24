"""Tests for relational / Indonesian schema migration and reporting."""

from app import create_app
from models import ClassificationHistory, KhatClass
from models.migrations_indonesian import build_schema_report, run_indonesian_migration, seed_kelas_khat
from models.migrations_relational import run_integrity_checks
from services.relational_query_service import get_class_label_map


def test_seed_khat_classes():
    app = create_app()
    with app.app_context():
        slug_map = seed_kelas_khat(app.config)
        assert len(slug_map) == 4
        assert KhatClass.query.count() >= 4
        slugs = {row.slug for row in KhatClass.query.all()}
        assert slugs >= {"naskhi", "diwani", "diwani_jali", "tsuluts"}


def test_relational_migration_runs():
    app = create_app()
    with app.app_context():
        result = run_indonesian_migration(app.config)
        assert result["seeded_classes"] == 4
        assert "report" in result


def test_relationship_report_has_foreign_keys():
    app = create_app()
    with app.app_context():
        run_indonesian_migration(app.config)
        report = build_schema_report()
        assert "kelas_khat" in report["tables_present"]
        assert report["row_counts"]["kelas_khat"] >= 4
        assert report["is_3nf_relational"]


def test_integrity_checks_structure():
    app = create_app()
    with app.app_context():
        run_indonesian_migration(app.config)
        integrity = run_integrity_checks()
        assert "integrity_ok" in integrity
        assert "integrity_checks" in integrity


def test_class_label_map_uses_khat_classes():
    app = create_app()
    with app.app_context():
        run_indonesian_migration(app.config)
        label_map = get_class_label_map()
        assert len(label_map) >= 4
        assert "Naskhi" in label_map.values()


def test_classification_history_linked_to_classes():
    app = create_app()
    with app.app_context():
        run_indonesian_migration(app.config)
        row = ClassificationHistory.query.first()
        if row:
            assert row.id_kelas_prediksi is not None or row.id_legacy is not None
