"""Tests for dataset page UI insight helpers."""

from services.dataset_page_ui_service import build_dataset_page_insights, build_prep_status_insights


def test_dataset_page_insights_imbalanced():
    balance = [
        {"label": "naskhi", "count": 61, "percent": 10.2},
        {"label": "diwani", "count": 64, "percent": 10.7},
        {"label": "diwani_jali", "count": 202, "percent": 33.8},
        {"label": "tsuluts", "count": 270, "percent": 45.2},
    ]
    qc = {
        "balance_status": "imbalanced",
        "balance_label": "Imbalanced",
        "missing_file_count": 0,
        "duplicate_filename_count": 0,
        "corrupted_count": 0,
    }
    insights = build_dataset_page_insights(
        balance=balance,
        total_images=597,
        qc_report=qc,
        dataset_status={"state_id": "Siap untuk Training", "badge": "ready"},
        prep_status={"progress": 99.5},
    )
    assert insights["is_imbalanced"] is True
    assert "Naskhi" in insights["minority_classes"][0]
    assert "Tambahkan data" in insights["primary_recommendation"]


def test_prep_status_almost_ready_with_unprocessed():
    prep = {
        "raw": 677,
        "processed": 594,
        "train": 403,
        "validation": 72,
        "test": 119,
        "progress": 87.7,
    }
    insights = build_prep_status_insights(prep_status=prep)
    assert insights["unprocessed"] == 83
    assert insights["state_id"] == "Hampir Siap"
    assert insights["badge"] == "almost_ready"
    assert "83 gambar" in insights["warning_emphasis"]
    assert "Dataset hampir siap" in insights["warning_message"]
    assert insights["show_recommendation"] is True
    assert insights["split_ratio"]["train"] == 67.8
    assert insights["split_ratio"]["validation"] == 12.1
    assert insights["split_ratio"]["test"] == 20.0


def test_prep_status_complete_when_all_processed():
    prep = {"raw": 594, "processed": 594, "train": 403, "validation": 72, "test": 119, "progress": 100.0}
    insights = build_prep_status_insights(prep_status=prep)
    assert insights["unprocessed"] == 0
    assert insights["warning_message"] == "Semua gambar raw telah berhasil diproses."
    assert insights["show_recommendation"] is False


def test_prep_status_needs_processing_below_85():
    prep = {"raw": 500, "processed": 300, "train": 200, "validation": 50, "test": 50, "progress": 60.0}
    insights = build_prep_status_insights(prep_status=prep)
    assert insights["state_id"] == "Perlu Diproses"
    assert insights["badge"] == "pending"
    assert insights["unprocessed"] == 200
    assert "Banyak gambar raw belum diproses" in insights["status_message"]
