"""Tests for evaluation dataset and metrics helpers."""

from services.evaluation_dataset_service import (
    build_dataset_summary,
    iter_test_images,
    resolve_evaluation_counts,
    validate_test_holdout_before_eval,
)
from services.evaluation_metrics_service import compute_classification_metrics


def _expect_raises(match: str, fn):
    try:
        fn()
    except ValueError as exc:
        if match not in str(exc):
            raise AssertionError(f"Expected {match!r} in {exc!r}")
        return
    raise AssertionError(f"Expected ValueError containing {match!r}")


def test_compute_classification_metrics_per_class_accuracy():
    labels = ["naskhi", "diwani", "diwani_jali", "tsuluts"]
    y_true = [0, 0, 1, 1, 2, 3, 3, 3]
    y_pred = [0, 1, 1, 1, 2, 3, 2, 3]
    metrics = compute_classification_metrics(y_true, y_pred, labels)

    assert metrics["accuracy"] == 0.75
    assert metrics["per_class_accuracy"]["naskhi"] == 0.5
    assert metrics["per_class_accuracy"]["diwani"] == 1.0
    assert len(metrics["confusion_matrix"]) == 4
    assert len(metrics["confusion_matrix_normalized"]) == 4


def test_build_dataset_summary_counts(tmp_path):
    class_labels = ["naskhi", "diwani"]
    test_dir = tmp_path / "test"
    for label, count in (("naskhi", 2), ("diwani", 3)):
        class_dir = test_dir / label
        class_dir.mkdir(parents=True)
        for idx in range(count):
            (class_dir / f"{label}_{idx}.jpg").write_bytes(b"fake")

    items = iter_test_images(str(test_dir), class_labels)
    summary = build_dataset_summary(str(test_dir), items, class_labels, config={"CLASS_LABELS": class_labels, "DATASET_SPLIT_REPORT_PATH": ""})

    assert summary["total_images"] == 5
    assert summary["per_class"][0]["actual_count"] == 2
    assert summary["per_class"][1]["actual_count"] == 3


def test_validate_test_holdout_rejects_full_dataset_as_test(tmp_path):
    class_labels = ["naskhi", "diwani"]
    raw_dir = tmp_path / "raw"
    test_dir = tmp_path / "test"
    for label in class_labels:
        for idx in range(3):
            (raw_dir / label).mkdir(parents=True, exist_ok=True)
            (test_dir / label).mkdir(parents=True, exist_ok=True)
            (raw_dir / label / f"{label}_{idx}.jpg").write_bytes(b"fake")
            (test_dir / label / f"{label}_{idx}.jpg").write_bytes(b"fake")

    config = {
        "CLASS_LABELS": class_labels,
        "RAW_DATASET_DIR": str(raw_dir),
        "TRAIN_DIR": str(tmp_path / "train"),
        "VALIDATION_DIR": str(tmp_path / "validation"),
        "TEST_DIR": str(test_dir),
        "OPTIMIZED_MODEL_DIR": str(tmp_path / "optimized" / "model"),
    }
    items = iter_test_images(str(test_dir), class_labels)
    _expect_raises("holdout", lambda: validate_test_holdout_before_eval(items, config))


def test_resolve_evaluation_counts_distinguishes_test_from_total(tmp_path):
    class_labels = ["naskhi", "diwani"]
    raw_dir = tmp_path / "raw"
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "validation"
    test_dir = tmp_path / "test"
    for label, raw_n, train_n, val_n, test_n in (
        ("naskhi", 10, 6, 1, 2),
        ("diwani", 10, 6, 1, 2),
    ):
        for split_dir, count in ((raw_dir, raw_n), (train_dir, train_n), (val_dir, val_n), (test_dir, test_n)):
            (split_dir / label).mkdir(parents=True, exist_ok=True)
            for idx in range(count):
                (split_dir / label / f"{label}_{idx}.jpg").write_bytes(b"fake")

    config = {
        "CLASS_LABELS": class_labels,
        "RAW_DATASET_DIR": str(raw_dir),
        "TRAIN_DIR": str(train_dir),
        "VALIDATION_DIR": str(val_dir),
        "TEST_DIR": str(test_dir),
        "OPTIMIZED_MODEL_DIR": str(tmp_path / "optimized" / "model"),
        "DATASET_SPLIT_REPORT_PATH": "",
    }
    counts = resolve_evaluation_counts(config, {"test_samples": 999})
    assert counts["test_samples"] == 4
    assert counts["total_dataset_processed"] == 18
    assert counts["total_dataset_raw"] == 20
    assert counts["split_valid"] is True
    assert counts["samples_match_saved"] is False


def test_resolve_evaluation_counts_reads_saved_dataset_summary_train_fields():
    config = {"CLASS_LABELS": ["naskhi", "diwani"]}
    counts = resolve_evaluation_counts(
        config,
        {
            "test_samples": 560,
            "dataset_summary": {
                "processed_images": 2800,
                "total_dataset_raw": 2800,
                "train_images": 1904,
                "validation_images": 336,
            },
        },
    )
    assert counts["train_images"] == 1904
    assert counts["validation_images"] == 336
    assert counts["total_dataset_processed"] == 2800
    assert counts["holdout_percent"] == 20.0
