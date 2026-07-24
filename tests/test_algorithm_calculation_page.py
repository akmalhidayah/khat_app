"""Tests for Perhitungan Algoritma page context."""

from app import create_app
from services.algorithm_calculation_page_service import (
    build_algorithm_calculation_page_context,
    get_model_info,
    get_thresholds_block,
    get_workflow_steps,
)


def test_workflow_steps_count():
    assert len(get_workflow_steps()) == 14


def test_page_context_builds():
    app = create_app()
    with app.test_request_context("/algorithm-calculation/"):
        ctx = build_algorithm_calculation_page_context(app.config)
    assert ctx["model_info"]["runtime"] in ("TensorFlow.js", "TensorFlow / Keras")
    assert len(ctx["workflow_steps"]) == 14
    assert len(ctx["detailed_walkthrough"]) >= 7
    assert "thresholds" in ctx
    assert "eval_calculation" in ctx
    assert "evaluation_summary" in ctx
    assert "latest_prediction" in ctx
    assert ctx["softmax_rows"]


def test_model_info_labels():
    app = create_app()
    with app.test_request_context("/"):
        info = get_model_info(app.config)
    assert info["output_count"] == 4
    assert len(info["labels"]) == 4


def test_thresholds_block():
    app = create_app()
    th = get_thresholds_block(app.config)
    assert th["strong_confidence"] == 85
    assert th["khat_accept"] >= 50
