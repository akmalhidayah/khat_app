"""Tests for detailed algorithm calculation service."""

from services.algorithm_calculation_service import (
    algorithm_confidence_category,
    algorithm_final_decision,
    algorithm_known_class_status,
    algorithm_margin_interpretation,
    build_academic_interpretation,
)


def test_confidence_categories():
    assert algorithm_confidence_category(90)["label"] == "Strong Prediction"
    assert algorithm_confidence_category(75)["label"] == "Moderate Prediction"
    assert algorithm_confidence_category(65)["label"] == "Weak Prediction"
    assert algorithm_confidence_category(45)["label"] == "Low Confidence"


def test_known_class_status():
    assert algorithm_known_class_status(90, 84) == "Known Class"
    assert algorithm_known_class_status(75, 12) == "Likely Known Class"
    assert algorithm_known_class_status(65, 20) == "Uncertain Class"
    assert algorithm_known_class_status(50, 5) == "Possible Unknown Class"


def test_margin_interpretation():
    assert algorithm_margin_interpretation(20)["level"] == "strong"
    assert algorithm_margin_interpretation(12)["level"] == "moderate"
    assert algorithm_margin_interpretation(5)["level"] == "ambiguous"


def test_final_decision_no_expected_high_confidence():
    result = algorithm_final_decision("tsuluts", None, 90, 84, input_status="khat")
    assert result["decision"] == "Accepted with Caution"


def test_final_decision_mismatch():
    result = algorithm_final_decision("tsuluts", "naskhi", 90, 84, input_status="khat")
    assert result["decision"] == "Manual Review Required"


def test_academic_interpretation():
    text = build_academic_interpretation(
        "Tsuluts",
        90.26,
        84.14,
        "Naskhi",
        "Strong Prediction",
        "Accepted with Caution",
        False,
        False,
    )
    assert "Tsuluts" in text
    assert "90.26" in text
