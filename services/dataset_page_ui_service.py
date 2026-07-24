"""Dataset Management page UI helpers — insights and display labels."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

CLASS_LABELS_ID = {
    "naskhi": "Naskhi",
    "diwani": "Diwani",
    "diwani_jali": "Diwani Jali",
    "tsuluts": "Tsuluts",
    "riqah": "Riq'ah",
    "kufi": "Kufi",
}


def _display(key: str) -> str:
    return CLASS_LABELS_ID.get(key, key.replace("_", " ").title())


def build_dataset_page_insights(
    *,
    balance: List[Dict[str, Any]],
    total_images: int,
    qc_report: Optional[Dict[str, Any]],
    dataset_status: Dict[str, Any],
    prep_status: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    prep = prep_status or {}
    qc = qc_report or {}

    is_imbalanced = qc.get("balance_status") in ("imbalanced", "needs_more_data")
    if not is_imbalanced and balance and total_images:
        pcts = [row.get("percent", 0) for row in balance]
        is_imbalanced = (max(pcts) - min(pcts)) >= 25 if pcts else False

    sorted_balance = sorted(balance, key=lambda r: r.get("count", 0)) if balance else []
    minority = [_display(r["label"]) for r in sorted_balance[:2]] if sorted_balance else []
    dominant = [_display(r["label"]) for r in sorted_balance[-2:]] if len(sorted_balance) >= 2 else []

    missing = qc.get("missing_file_count", 0)
    duplicates = qc.get("duplicate_filename_count", 0)
    corrupted = qc.get("corrupted_count", 0)

    readiness = dataset_status.get("state_id") or dataset_status.get("state") or "—"
    quality_label = qc.get("balance_label") or ("Imbalanced" if is_imbalanced else "Balanced")

    recommendations: List[str] = []
    if is_imbalanced and minority:
        names = " dan ".join(minority[:2])
        recommendations.append(
            f"Tambahkan data {names} agar distribusi dataset lebih seimbang dan mengurangi bias model saat training."
        )
    if prep.get("progress", 0) >= 99 and total_images:
        recommendations.append(f"Dataset telah diproses {prep.get('progress', 0)}% dan siap untuk training.")
    if is_imbalanced:
        recommendations.append("Class weights disarankan saat training untuk mengurangi bias akibat imbalance.")
    if missing == 0 and duplicates == 0 and corrupted == 0 and total_images:
        recommendations.append("Tidak ditemukan missing files, corrupted images, atau duplicate filenames.")

    primary = recommendations[0] if recommendations else "Dataset siap ditinjau untuk kebutuhan penelitian."

    return {
        "is_imbalanced": is_imbalanced,
        "quality_label": quality_label,
        "quality_tone": "warn" if is_imbalanced else "success",
        "readiness": readiness,
        "readiness_tone": "success" if dataset_status.get("badge") == "ready" else "warn",
        "minority_classes": minority,
        "dominant_classes": dominant,
        "missing_files": missing,
        "duplicate_filenames": duplicates,
        "corrupted_removed": corrupted,
        "priority_addition": ", ".join(minority[:2]) if minority else "—",
        "recommendations": recommendations,
        "primary_recommendation": primary,
        "processed_pct": prep.get("progress", 0),
    }


def build_prep_status_insights(
    *,
    prep_status: Dict[str, Any],
    quality_report: Optional[Dict[str, Any]] = None,
    optimization_summary: Optional[Dict[str, Any]] = None,
    duplicate_report: Optional[Dict[str, Any]] = None,
    split_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute preparation status levels, messages, and process report for the prep card."""
    raw = int(prep_status.get("raw", 0) or 0)
    processed = int(prep_status.get("processed", 0) or 0)
    train = int(prep_status.get("train", 0) or 0)
    validation = int(prep_status.get("validation", 0) or 0)
    test = int(prep_status.get("test", 0) or 0)
    progress = float(prep_status.get("progress", 0) or 0)
    unprocessed = max(raw - processed, 0)

    split_ratio: Dict[str, float] = {}
    if processed > 0:
        split_ratio = {
            "train": round((train / processed) * 100, 1),
            "validation": round((validation / processed) * 100, 1),
            "test": round((test / processed) * 100, 1),
        }

    qc = quality_report or {}
    opt = optimization_summary or {}
    dup = duplicate_report or {}
    split = split_report or {}

    duplicates = int(
        dup.get("duplicate_count")
        or dup.get("duplicates_found")
        or qc.get("duplicates_removed", 0)
        or 0
    ) + int(qc.get("near_duplicates_removed", 0) or 0)
    corrupted = int(opt.get("corrupted", 0) or qc.get("corrupted_removed", 0) or 0)
    skipped = int(opt.get("skipped", 0) or qc.get("non_image_removed", 0) or 0)
    excluded_from_cleaning = max(
        int(qc.get("total_before_cleaning", 0) or 0) - int(qc.get("total_after_cleaning", 0) or 0),
        0,
    )
    excluded = excluded_from_cleaning if excluded_from_cleaning > 0 else max(unprocessed - skipped - corrupted - duplicates, 0)

    processing_date = (
        qc.get("generated_at")
        or opt.get("last_run")
        or split.get("created_at")
        or split.get("generated_at")
    )
    notes: List[str] = []
    for source in (qc.get("warnings"), qc.get("recommendations"), split.get("evaluation_policy")):
        if isinstance(source, list):
            notes.extend(str(item) for item in source if item)
        elif source:
            notes.append(str(source))
    if not notes and unprocessed > 0:
        notes.append(
            "Gambar raw yang belum masuk processed dapat disebabkan oleh gagal dibaca, format tidak sesuai, "
            "duplikat, resolusi bermasalah, atau dikecualikan saat cleaning."
        )

    process_report = {
        "raw_total": raw,
        "processed_total": processed,
        "unprocessed_total": unprocessed,
        "skipped": skipped,
        "corrupted": corrupted,
        "duplicate": duplicates,
        "excluded": excluded,
        "processing_date": processing_date or "—",
        "notes": notes,
    }

    if raw == 0:
        return {
            "unprocessed": 0,
            "split_ratio": split_ratio,
            "prep_level": "empty",
            "state": "Waiting for Upload",
            "state_id": "Menunggu Unggahan",
            "badge": "empty",
            "status_message": "",
            "warning_message": "",
            "warning_emphasis": "",
            "warning_lead": "",
            "warning_trail": "",
            "warning_tone": "info",
            "show_recommendation": False,
            "recommendation": "",
            "process_report": process_report,
            "progress_bar_tone": "warning",
        }

    if progress >= 98:
        prep_level = "complete"
        state_id = "Lengkap"
        badge = "ready"
        status_message = "Dataset telah diproses hampir seluruhnya dan siap digunakan."
        progress_bar_tone = "success"
    elif progress >= 85:
        prep_level = "almost_ready"
        state_id = "Hampir Siap"
        badge = "almost_ready"
        status_message = (
            "Sebagian besar dataset telah diproses, tetapi masih ada beberapa gambar "
            "yang belum masuk ke processed dataset."
        )
        progress_bar_tone = "almost"
    else:
        prep_level = "needs_processing"
        state_id = "Perlu Diproses"
        badge = "pending"
        status_message = "Banyak gambar raw belum diproses. Jalankan preprocessing sebelum training."
        progress_bar_tone = "warning"

    if unprocessed == 0:
        warning_message = "Semua gambar raw telah berhasil diproses."
        warning_emphasis = ""
        warning_lead = ""
        warning_trail = ""
        warning_tone = "success"
        show_recommendation = False
        recommendation = ""
        if progress >= 98:
            status_message = "Dataset telah diproses hampir seluruhnya dan siap digunakan."
    else:
        warning_emphasis = f"{unprocessed} gambar"
        show_recommendation = True
        if progress >= 85 and progress < 98:
            warning_tone = "almost"
            warning_lead = "Dataset hampir siap."
            warning_trail = (
                "raw belum diproses atau dikecualikan saat pembersihan. "
                "Periksa laporan proses untuk mengetahui penyebabnya."
            )
            warning_message = (
                f"Dataset hampir siap. {unprocessed} gambar raw belum diproses atau dikecualikan "
                f"saat pembersihan. Periksa laporan proses untuk mengetahui penyebabnya."
            )
            recommendation = (
                "Jalankan ulang preprocessing atau periksa laporan cleaning sebelum training ulang model."
            )
        elif progress >= 98:
            warning_tone = "info"
            warning_lead = ""
            warning_trail = (
                "raw belum diproses atau dikecualikan saat pembersihan. "
                "Periksa laporan proses untuk mengetahui penyebabnya."
            )
            warning_message = (
                f"{unprocessed} gambar raw belum diproses atau dikecualikan saat pembersihan. "
                f"Periksa laporan proses untuk mengetahui penyebabnya."
            )
            recommendation = (
                "Periksa gambar yang belum diproses dan pastikan tidak ada gambar rusak, duplikat, "
                "atau salah format sebelum training ulang."
            )
        else:
            warning_tone = "warn"
            warning_lead = ""
            warning_trail = (
                "raw belum masuk ke dataset processed. Gambar tersebut mungkin belum diproses, "
                "terdeteksi tidak valid, duplikat, rusak, atau dikecualikan saat pembersihan."
            )
            warning_message = (
                f"{unprocessed} gambar raw belum masuk ke dataset processed. Gambar tersebut mungkin belum "
                f"diproses, terdeteksi tidak valid, duplikat, rusak, atau dikecualikan saat pembersihan."
            )
            recommendation = (
                "Periksa gambar yang belum diproses, jalankan ulang preprocessing, dan pastikan tidak ada "
                "gambar rusak, duplikat, atau salah format sebelum melakukan training ulang."
            )

    return {
        "unprocessed": unprocessed,
        "split_ratio": split_ratio,
        "prep_level": prep_level,
        "state": state_id,
        "state_id": state_id,
        "badge": badge,
        "status_message": status_message,
        "warning_message": warning_message,
        "warning_emphasis": warning_emphasis,
        "warning_lead": warning_lead,
        "warning_trail": warning_trail,
        "warning_tone": warning_tone,
        "show_recommendation": show_recommendation,
        "recommendation": recommendation,
        "process_report": process_report,
        "progress_bar_tone": progress_bar_tone,
    }


def qc_row_short_recommendation(row: Dict[str, Any], min_target: int = 300) -> str:
    """Concise recommendation for QC class table."""
    count = row.get("count", 0)
    status = row.get("status", "")
    if status == "ok":
        return "Memenuhi target minimum"
    if count < min_target:
        return f"Di bawah minimum {min_target} gambar"
    if status == "low":
        return "Masih di bawah rekomendasi 300 gambar"
    return "Perlu penambahan data untuk keseimbangan"
