"""Tests for report UI service."""

from services.report_ui_service import (
    build_dataset_balance_note,
    build_executive_summary,
    build_research_insights,
    build_technical_model_info,
    detect_class_imbalance,
)


def test_executive_summary_uses_teachable_machine():
    text = build_executive_summary(
        "Teachable Machine Image Model",
        594,
        None,
        {"total": 86, "valid_khat_count": 70, "rejected_count": 11, "uncertain_count": 5},
        {"accuracy_pct": 76.67, "target_pct": 85.0, "achieved": False},
        is_tm=True,
    )
    assert "Teachable Machine Image Model" in text
    assert "CNN transfer learning" not in text


def test_imbalance_note():
    dist = {"naskhi": 50, "diwani": 60, "diwani_jali": 200, "tsuluts": 180}
    note = build_dataset_balance_note(dist, detect_class_imbalance(dist))
    assert "belum seimbang" in note.lower()


def test_technical_info_tm():
    info = build_technical_model_info(
        {
            "available": True,
            "model_source": "Teachable Machine",
            "model_name": "tm-my-image-model",
            "model_runtime": "TensorFlow.js",
            "model_json_url": "/static/model/model.json",
            "model_metadata_url": "/static/model/metadata.json",
            "model_weights_url": "/static/model/weights.bin",
            "model_input_size": "224x224",
            "labels_display": "diwani, diwani jali, naskhi, tsuluts",
            "model_architecture": "Teachable Machine Image Model",
        },
        {},
    )
    assert info["model_file"] == "/static/model/model.json"


def test_research_insights_below_target():
    insights = build_research_insights(
        {"achieved": False},
        True,
        {"naskhi": 50, "diwani": 60, "diwani_jali": 200, "tsuluts": 180},
        None,
        {"mismatch_count": 2},
    )
    assert any("85%" in i for i in insights)
