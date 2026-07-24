"""Tests for Teachable Machine model upload validation."""

import json
import os
import tempfile

import pytest

from services.tm_model_validation_service import (
    TmModelValidationError,
    validate_tm_files,
    validate_tm_metadata_dict,
)


def _valid_metadata():
    return {
        "labels": ["Diwani", "Diwani Jali", "Naskhi", "Tsuluts"],
        "imageSize": 224,
        "modelName": "tm-my-image-model",
    }


def test_validate_metadata_accepts_canonical_labels():
    result = validate_tm_metadata_dict(_valid_metadata())
    assert result["image_size"] == 224
    assert "naskhi" in result["labels"]


def test_validate_metadata_rejects_missing_labels():
    meta = _valid_metadata()
    meta["labels"] = ["Diwani", "Naskhi"]
    with pytest.raises(TmModelValidationError):
        validate_tm_metadata_dict(meta)


def test_validate_metadata_rejects_wrong_input_size():
    meta = _valid_metadata()
    meta["imageSize"] = 299
    with pytest.raises(TmModelValidationError):
        validate_tm_metadata_dict(meta)


def test_validate_tm_files_requires_all_parts():
    with tempfile.TemporaryDirectory() as tmp:
        meta_path = os.path.join(tmp, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(_valid_metadata(), handle)
        with pytest.raises(TmModelValidationError):
            validate_tm_files({"metadata.json": meta_path})
