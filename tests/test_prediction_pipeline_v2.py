import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from services.class_mapping_service import CLASS_NAMES, validate_production_mapping
from services.preprocessing_service import prepare_local_classifier_input
from services.prediction_service import _class_recognition_gate
from services.khat_detector_service import detect_khat


def test_class_mapping_matches_model_output():
    result = validate_production_mapping(
        class_indices={name: index for index, name in enumerate(CLASS_NAMES)},
        num_model_outputs=4,
        metadata_class_mapping=CLASS_NAMES,
    )
    assert result["valid"]


@pytest.mark.parametrize(
    "indices,outputs",
    [
        ({"tsuluts": 0, "diwani_jali": 1, "diwani": 2, "naskhi": 3}, 4),
        ({"naskhi": 0, "diwani": 1, "diwani_jali": 3, "tsuluts": 4}, 4),
        ({"naskhi": 0, "diwani": 1, "diwani_jali": 2, "tsuluts": 3}, 5),
    ],
)
def test_invalid_mapping_or_output_is_rejected(indices, outputs):
    with pytest.raises(ValueError, match="Invalid production class mapping"):
        validate_production_mapping(class_indices=indices, num_model_outputs=outputs)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_preprocessing_shape_and_finite_values(tmp_path, mode):
    size = (80, 160) if mode != "L" else (160, 80)
    color = 128 if mode == "L" else ((128, 80, 20, 255) if mode == "RGBA" else (128, 80, 20))
    path = tmp_path / f"input_{mode}.png"
    Image.new(mode, size, color).save(path)
    batch, diagnostic = prepare_local_classifier_input(
        str(path), "efficientnetb0", return_diagnostics=True
    )
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert np.isfinite(batch).all()
    assert diagnostic["aspect_ratio_preserved"] is True
    # EfficientNetB0 preprocessing must not divide an already scaled image again.
    assert diagnostic["max"] > 1.0


def test_corrupt_preprocessing_input_rejected(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not an image")
    with pytest.raises(Exception):
        prepare_local_classifier_input(str(path))


def test_detector_unavailable_returns_uncertain(tmp_path, monkeypatch):
    path = tmp_path / "khat.png"
    Image.new("RGB", (64, 64), "white").save(path)
    monkeypatch.setattr(
        "services.khat_detector_service.assess_calligraphy_content",
        lambda unused: {"is_non_khat_content": False, "decision_code": "likely_khat_content"},
    )
    config = {
        "KHAT_DETECTOR_PATH": str(tmp_path / "missing.keras"),
        "KHAT_ACCEPT_THRESHOLD": 0.70,
        "KHAT_REJECT_THRESHOLD": 0.50,
        "KHAT_BORDERLINE_THRESHOLD": 0.65,
        "KHAT_HIGH_CONFIDENCE_THRESHOLD": 0.85,
        "KHAT_DETECTOR_SETTINGS_PATH": str(tmp_path / "settings.json"),
    }
    result = detect_khat(str(path), config=config)
    assert result["input_status"] == "uncertain"
    assert result["detection_status"] == "detector_unavailable"
    assert result["khat_probability"] is None
    assert result["manual_review_required"] is True


def test_stage1_hard_rejection_cannot_be_forced(tmp_path, monkeypatch):
    path = tmp_path / "latin.png"
    Image.new("RGB", (64, 64), "white").save(path)
    monkeypatch.setattr(
        "services.khat_detector_service.assess_calligraphy_content",
        lambda unused: {
            "is_non_khat_content": True,
            "is_latin_script_non_khat": True,
            "is_photographic_non_khat": False,
            "rejection_reason": "Input bukan merupakan tulisan Arab.",
        },
    )
    config = {
        "KHAT_DETECTOR_PATH": str(tmp_path / "missing.keras"),
        "KHAT_ACCEPT_THRESHOLD": 0.70,
        "KHAT_REJECT_THRESHOLD": 0.50,
        "KHAT_BORDERLINE_THRESHOLD": 0.65,
        "KHAT_HIGH_CONFIDENCE_THRESHOLD": 0.85,
        "KHAT_DETECTOR_SETTINGS_PATH": str(tmp_path / "settings.json"),
    }
    result = detect_khat(str(path), config=config, force_classify=True)
    assert result["stage2_allowed"] is False
    assert result["detection_decision"] == "rejected"


def test_low_margin_prediction_abstains():
    config = {
        "STAGE2_MIN_CONFIDENCE": 0.70,
        "STAGE2_MIN_MARGIN": 0.10,
        "STAGE2_MIN_CONFIDENCE_AFTER_STAGE1": 0.55,
        "STAGE2_MIN_MARGIN_AFTER_STAGE1": 0.10,
        "CLASS_LABELS": CLASS_NAMES,
        "CANONICAL_CLASS_ORDER": {name: i for i, name in enumerate(CLASS_NAMES)},
        "CLASS_DISPLAY_NAMES": {},
    }
    result = _class_recognition_gate(
        {
            "predicted_class": "naskhi",
            "confidence": 0.90,
            "raw_model_scores": {
                "naskhi": 0.46,
                "diwani": 0.44,
                "diwani_jali": 0.06,
                "tsuluts": 0.04,
            },
        },
        config=config,
        stage1_confirmed=True,
    )
    assert result["abstained"] is True
    assert result["final_class"] is None


def test_raw_and_refined_scores_are_not_max_merged():
    config = {
        "STAGE2_MIN_CONFIDENCE": 0.70,
        "STAGE2_MIN_MARGIN": 0.10,
        "STAGE2_MIN_CONFIDENCE_AFTER_STAGE1": 0.55,
        "STAGE2_MIN_MARGIN_AFTER_STAGE1": 0.10,
        "CLASS_LABELS": CLASS_NAMES,
        "CANONICAL_CLASS_ORDER": {name: i for i, name in enumerate(CLASS_NAMES)},
        "CLASS_DISPLAY_NAMES": {},
    }
    result = _class_recognition_gate(
        {
            "predicted_class": "naskhi",
            "confidence": 0.99,
            "gate_confidence": 0.99,
            "best_single_confidence": 0.99,
            "raw_model_scores": dict(zip(CLASS_NAMES, [0.30, 0.29, 0.22, 0.19])),
        },
        config=config,
        stage1_confirmed=True,
    )
    assert result["abstained"] is True


def test_real_regression_manifest_is_explicitly_skipped_when_missing():
    manifest = Path(__file__).parent / "fixtures" / "prediction_manifest.json"
    if not manifest.exists():
        pytest.skip("Real-image prediction manifest is not available in this checkout")
    assert isinstance(json.loads(manifest.read_text(encoding="utf-8")), list)
