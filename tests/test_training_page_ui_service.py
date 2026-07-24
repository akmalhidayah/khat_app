"""Tests for training page UI service."""

from services.training_page_ui_service import (
    build_eval_display,
    build_model_management_insights,
    build_training_page_insights,
    build_training_safety,
)


def test_target_gap_text_when_below():
    insights = build_training_page_insights(
        readiness={"train_count": 100, "validation_count": 20, "test_count": 20, "raw_count": 200},
        quality_report={"per_class_counts": {"naskhi": 50, "diwani": 60, "diwani_jali": 200, "tsuluts": 180}, "class_imbalance_percent": 75},
        target_validation={"percent": 77.53, "achieved": False},
        target_test={"percent": 84.44, "achieved": False},
        target_achieved=False,
        accuracy_target=0.85,
        below_target=True,
        eval_result={"accuracy": 0.8444},
        history={"accuracy": [0.8], "val_accuracy": [0.66]},
        is_tm_active=True,
        active_model_version={"model_name": "tm-my-image-model"},
    )
    assert insights["test_gap_text"] == "Kurang 0.56% dari target akurasi uji 85%."
    assert insights["val_below_target"] is True
    assert len(insights["checklist"]) == 7


def test_training_safety_blocks_when_not_ready():
    safety = build_training_safety(
        readiness={"is_ready": False, "train_count": 0, "validation_count": 0, "test_count": 0, "raw_count": 0},
        validation_report={"errors": ["missing dataset folder"]},
        classes_count=4,
        model_dir="/tmp",
        quality_report={},
    )
    assert safety["can_retrain"] is False
    assert safety["block_reason"]


def test_diagnosis_imbalance_message():
    insights = build_training_page_insights(
        readiness={"train_count": 100},
        quality_report={
            "per_class_counts": {"naskhi": 50, "diwani": 60, "diwani_jali": 200, "tsuluts": 180},
            "class_imbalance_percent": 75,
        },
        target_validation={"percent": None},
        target_test={"percent": None},
        target_achieved=False,
        accuracy_target=0.85,
        below_target=True,
        eval_result={},
        history=None,
        is_tm_active=True,
        active_model_version=None,
    )
    assert insights["is_imbalanced"] is True
    assert any("belum seimbang" in item["text"] for item in insights["diagnosis_items"])


def test_model_management_notice_and_workflow():
    mgmt = build_model_management_insights(
        readiness={"is_ready": True, "train_count": 100},
        quality_report={"per_class_counts": {"naskhi": 50, "diwani": 60, "diwani_jali": 200, "tsuluts": 180}, "class_imbalance_percent": 75},
        eval_result={"accuracy": 0.84},
        is_tm_active=True,
        below_target=True,
    )
    assert "Teachable Machine" in mgmt["notice"]
    assert len(mgmt["retraining_workflow"]) == 7
    assert len(mgmt["retraining_checklist"]) == 8
    assert any(item["status"] == "needs_attention" for item in mgmt["retraining_checklist"])


def test_eval_display_from_result():
    display = build_eval_display(
        {
            "accuracy": 0.8444,
            "precision": 0.86,
            "recall": 0.84,
            "f1_score": 0.84,
            "confusion_matrix": [[7, 1, 0, 1]],
            "per_class": {"naskhi": {"precision": 0.63, "recall": 0.77, "f1_score": 0.7, "support": 9}},
            "test_samples": 90,
            "correct_predictions": 76,
            "incorrect_predictions": 14,
        },
        ["naskhi", "diwani", "diwani_jali", "tsuluts"],
    )
    assert display["has_eval"] is True
    assert display["accuracy"] == 84.44
    assert display["test_samples"] == 90
