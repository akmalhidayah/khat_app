"""Tests for evaluation image preview resolution."""

import shutil
from pathlib import Path

from services.evaluation_image_service import (
    attach_preview_to_record,
    copy_evaluation_preview,
    locate_evaluation_image_file,
    preview_filename_for,
    resolve_evaluation_image,
)


def test_preview_filename_for_class():
    assert preview_filename_for("diwani_jali", "abc.jpg") == "diwani_jali__abc.jpg"


def test_locate_and_copy_preview(tmp_path):
    test_dir = tmp_path / "dataset" / "test" / "naskhi"
    test_dir.mkdir(parents=True)
    source = test_dir / "sample.jpg"
    source.write_bytes(b"fake-image")

    config = {
        "BASE_DIR": str(tmp_path),
        "TEST_DIR": str(tmp_path / "dataset" / "test"),
        "OPTIMIZED_MODEL_DIR": str(tmp_path / "dataset" / "optimized" / "model"),
        "RAW_DATASET_DIR": str(tmp_path / "dataset" / "raw"),
        "PROCESSED_DATASET_DIR": str(tmp_path / "dataset" / "processed"),
        "MODEL_READY_DIR": str(tmp_path / "dataset" / "model_ready"),
        "TRAIN_DIR": str(tmp_path / "dataset" / "train"),
        "VALIDATION_DIR": str(tmp_path / "dataset" / "validation"),
        "UPLOAD_FOLDER": str(tmp_path / "static" / "uploads"),
        "EVALUATION_FOLDER": str(tmp_path / "static" / "evaluation"),
        "EVALUATION_PREVIEW_DIR": str(tmp_path / "static" / "evaluation_previews"),
        "CLASS_LABELS": ["naskhi", "diwani", "diwani_jali", "tsuluts"],
    }

    row = {
        "filename": "sample.jpg",
        "true_label": "naskhi",
        "image_path": str(source),
    }
    located = locate_evaluation_image_file(row, config=config)
    assert located is not None
    assert located["path"].is_file()

    preview_name = copy_evaluation_preview(source, "naskhi", "sample.jpg", config=config)
    assert preview_name == "naskhi__sample.jpg"
    assert (Path(config["EVALUATION_PREVIEW_DIR"]) / preview_name).is_file()

    resolved = resolve_evaluation_image(
        {**row, "preview_filename": preview_name},
        config=config,
        static_url_builder=lambda rel: f"/static/{rel}",
    )
    assert resolved["image_available"] is True
    assert resolved["image_url"] == f"/static/evaluation_previews/{preview_name}"


def test_attach_preview_to_record(tmp_path):
    test_dir = tmp_path / "dataset" / "test" / "tsuluts"
    test_dir.mkdir(parents=True)
    source = test_dir / "err.jpg"
    source.write_bytes(b"fake-image")
    config = {
        "BASE_DIR": str(tmp_path),
        "TEST_DIR": str(tmp_path / "dataset" / "test"),
        "OPTIMIZED_MODEL_DIR": str(tmp_path / "dataset" / "optimized" / "model"),
        "RAW_DATASET_DIR": str(tmp_path / "dataset" / "raw"),
        "PROCESSED_DATASET_DIR": str(tmp_path / "dataset" / "processed"),
        "MODEL_READY_DIR": str(tmp_path / "dataset" / "model_ready"),
        "TRAIN_DIR": str(tmp_path / "dataset" / "train"),
        "VALIDATION_DIR": str(tmp_path / "dataset" / "validation"),
        "UPLOAD_FOLDER": str(tmp_path / "static" / "uploads"),
        "EVALUATION_FOLDER": str(tmp_path / "static" / "evaluation"),
        "EVALUATION_PREVIEW_DIR": str(tmp_path / "static" / "evaluation_previews"),
        "CLASS_LABELS": ["naskhi", "diwani", "diwani_jali", "tsuluts"],
    }
    row = attach_preview_to_record(
        {
            "image_path": str(source),
            "filename": "tsuluts/err.jpg",
            "true_label": "tsuluts",
        },
        config=config,
    )
    assert row["image_exists"] is True
    assert row["preview_filename"] == "tsuluts__err.jpg"
