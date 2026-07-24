"""Tests for evaluation image path resolution."""

from pathlib import Path

from config import Config
from services.evaluation_ui_service import enrich_misclassified_row, resolve_test_image_rel


def test_resolve_from_absolute_image_path():
    row = {
        "image_path": "/Applications/XAMPP/xamppfiles/htdocs/khat-classification-web/dataset/test/diwani_jali/abc.jpg",
        "filename": "abc.jpg",
        "true_label": "diwani_jali",
    }
    rel = resolve_test_image_rel(
        row,
        test_dir="/Applications/XAMPP/xamppfiles/htdocs/khat-classification-web/dataset/test",
    )
    assert rel == "diwani_jali/abc.jpg"


def test_resolve_from_true_label_and_basename():
    row = {
        "filename": "abc.jpg",
        "true_label": "tsuluts",
    }
    assert resolve_test_image_rel(row) == "tsuluts/abc.jpg"


def test_enrich_sets_image_rel(tmp_path):
    from app import create_app

    test_dir = tmp_path / "dataset" / "test" / "naskhi"
    test_dir.mkdir(parents=True)
    source = test_dir / "sample.jpg"
    source.write_bytes(b"fake-image")
    preview_dir = tmp_path / "static" / "evaluation_previews"
    preview_dir.mkdir(parents=True)
    preview_name = "naskhi__sample.jpg"
    (preview_dir / preview_name).write_bytes(b"fake-image")

    app = create_app()
    app.config.update(
        {
            "BASE_DIR": str(tmp_path),
            "TEST_DIR": str(tmp_path / "dataset" / "test"),
            "EVALUATION_PREVIEW_DIR": str(preview_dir),
        }
    )

    row = {
        "image_path": str(source),
        "filename": "sample.jpg",
        "true_label": "naskhi",
        "confidence": 0.9,
        "margin_pct": 12,
        "preview_filename": preview_name,
        "image_exists": True,
        "preview_status": "available",
    }

    with app.test_request_context():
        enriched = enrich_misclassified_row(
            row,
            row_id=0,
            test_dir=str(tmp_path / "dataset" / "test"),
            static_url_builder=lambda rel: f"/static/{rel}",
        )
    assert enriched["image_rel"] == "naskhi/sample.jpg"
    assert enriched["image_available"] is True
    assert enriched["image_url"] == f"/static/evaluation_previews/{preview_name}"
    assert enriched["recommendation"]
    assert enriched["error_analysis"]
