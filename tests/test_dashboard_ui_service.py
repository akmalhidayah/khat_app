"""Tests for dashboard UI insight helpers."""

from services.dashboard_ui_service import (
    build_dashboard_insights,
    build_dataset_insights,
    workflow_steps,
)


def test_dataset_imbalance_detected():
    distribution = {"naskhi": 60, "diwani": 64, "diwani_jali": 201, "tsuluts": 269}
    insights = build_dataset_insights(distribution)
    assert insights["total"] == 594
    assert insights["is_imbalanced"] is True
    assert insights["warning_message"] is not None
    assert "Naskhi" in insights["recommendation"]


def test_dashboard_insights_below_target():
    dataset = build_dataset_insights({"naskhi": 60, "diwani": 64, "diwani_jali": 201, "tsuluts": 269})
    insights = build_dashboard_insights(
        test_accuracy=0.844,
        accuracy_target=0.85,
        target_achieved=False,
        dataset_insights=dataset,
        detector_available=True,
        khat_thresholds={"accept": 0.7, "reject": 0.5},
    )
    assert insights["accuracy_gap_pct"] == 0.6
    assert insights["accuracy_status"] == "Mendekati target"
    assert insights["dataset_status"] == "Belum seimbang"


def test_workflow_steps_current_prediction():
    steps = workflow_steps(
        total_dataset=10,
        has_training=True,
        has_evaluation=True,
        total_predictions=0,
    )
    assert steps[4]["label"] == "Prediksi"
    assert steps[4]["current"] is True
