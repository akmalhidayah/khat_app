"""Tests for evaluation error classification."""

from services.evaluation_ui_service import classify_error_type


def test_high_confidence_error():
    result = classify_error_type(90, 80, "diwani_jali", "tsuluts")
    assert result["key"] == "high_confidence"
    assert "Review label" in result["recommendation"]


def test_ambiguous_margin():
    result = classify_error_type(75, 8, "naskhi", "diwani")
    assert result["key"] == "ambiguous"


def test_class_confusion_pair():
    result = classify_error_type(75, 20, "diwani_jali", "tsuluts")
    assert result["key"] == "class_confusion"
