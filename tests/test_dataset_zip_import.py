"""Tests for ZIP dataset import helpers."""

import os
import tempfile
import zipfile

import pytest

from services.dataset_zip_import_service import ZipImportSummary, zip_is_ignored, zip_normalize_class, zip_safe_extract


def test_zip_ignored_system_files():
    assert zip_is_ignored(".DS_Store") is True
    assert zip_is_ignored("__MACOSX/foo") is True
    assert zip_is_ignored("folder/__MACOSX/bar") is True
    assert zip_is_ignored("naskhi/image.jpg") is False


def test_zip_normalize_class_aliases():
    assert zip_normalize_class("Diwani Jali") == "diwani_jali"
    assert zip_normalize_class("thuluth") == "tsuluts"
    assert zip_normalize_class("unknown") is None


def test_zip_safe_extract_blocks_traversal():

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "unsafe.zip")
        extract_dir = os.path.join(temp_dir, "extract")

        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("../escape.txt", "bad")

        summary = ZipImportSummary()
        with zipfile.ZipFile(zip_path, "r") as archive:
            zip_safe_extract(archive, extract_dir, summary)

        assert summary.skipped_unsafe == 1
        assert not os.path.exists(os.path.join(os.path.dirname(extract_dir), "escape.txt"))
