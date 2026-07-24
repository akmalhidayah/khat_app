"""Resolve evaluation test split and build dataset manifests for accurate metrics."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from flask import current_app

from services.dataset_readiness_service import count_images_per_class
from services.teachable_machine_service import get_tm_display_info, is_teachable_machine_available, load_tm_metadata, normalize_tm_label

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def resolve_evaluation_test_dir(config=None) -> Tuple[str, str]:
    """Return (absolute_test_dir, source_key) where source_key is 'test' or 'optimized_test'."""
    if config is None:
        config = current_app.config

    class_labels = config["CLASS_LABELS"]
    primary_dir = config["TEST_DIR"]
    optimized_dir = os.path.join(config["OPTIMIZED_MODEL_DIR"], "test")

    if os.path.isdir(optimized_dir) and iter_test_images(optimized_dir, class_labels):
        return optimized_dir, "optimized_test"
    return primary_dir, "test"


def iter_test_images(test_dir: str, class_labels: List[str]) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for label in class_labels:
        class_dir = os.path.join(test_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for name in sorted(os.listdir(class_dir)):
            if not name.lower().endswith(IMAGE_EXTENSIONS):
                continue
            items.append((os.path.join(class_dir, name), label))
    return items


def _count_split_totals(config, class_labels: List[str]) -> Dict[str, int]:
    if config.get("USE_EXTERNAL_DATASET"):
        from services.dataset_path_service import get_dataset_inventory

        inv = get_dataset_inventory(config)
        return {
            "raw": int(inv.get("raw_total") or 0),
            "train": int(inv.get("train") or 0),
            "validation": int(inv.get("validation") or 0),
            "test": int(inv.get("test") or 0),
            "processed": int(inv.get("processed_total") or 0),
        }

    def _safe_count(dir_key: str) -> int:
        folder = config.get(dir_key) or ""
        if not folder or not os.path.isdir(folder):
            return 0
        return sum(count_images_per_class(folder, class_labels).values())

    raw_total = _safe_count("RAW_DATASET_DIR")
    train_total = _safe_count("TRAIN_DIR")
    val_total = _safe_count("VALIDATION_DIR")
    test_total = _safe_count("TEST_DIR")
    return {
        "raw": raw_total,
        "train": train_total,
        "validation": val_total,
        "test": test_total,
        "processed": train_total + val_total + test_total,
    }


def validate_test_holdout_before_eval(
    test_items: List[Tuple[str, str]],
    config=None,
    *,
    test_count: Optional[int] = None,
) -> None:
    """Reject evaluation when the test folder holds the entire dataset instead of a holdout split."""
    if config is None:
        config = current_app.config

    class_labels = config["CLASS_LABELS"]
    test_count = test_count if test_count is not None else len(test_items)
    totals = _count_split_totals(config, class_labels)

    if test_count == 0:
        raise ValueError("Test dataset is empty. Split dataset before evaluation.")

    if totals["raw"] > 0 and test_count >= totals["raw"]:
        raise ValueError(
            f"Folder test berisi {test_count} gambar — sama dengan seluruh dataset mentah ({totals['raw']}). "
            "Jalankan Proses & Bagi Dataset agar evaluasi hanya pada holdout 20%, bukan semua gambar."
        )

    if totals["processed"] > 0 and totals["train"] == 0 and totals["validation"] == 0:
        raise ValueError(
            f"Folder test berisi {test_count} gambar tetapi train/validation masih kosong. "
            "Dataset belum dibagi. Jalankan Proses & Bagi Dataset terlebih dahulu."
        )

    if totals["processed"] > 0 and test_count >= totals["processed"]:
        raise ValueError(
            f"Jumlah gambar test ({test_count}) sama dengan total dataset diproses ({totals['processed']}). "
            "Evaluasi harus menggunakan holdout 20% saja, bukan seluruh dataset."
        )


def resolve_evaluation_counts(config=None, eval_json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Authoritative test vs total dataset counts for evaluation UI and validation."""
    if config is None:
        config = current_app.config

    eval_json = eval_json or {}
    class_labels = config["CLASS_LABELS"]
    saved_samples = eval_json.get("test_samples")
    dataset_summary = eval_json.get("dataset_summary") or {}

    if saved_samples is not None and (
        dataset_summary.get("processed_total") or dataset_summary.get("processed_images")
    ):
        processed = int(
            dataset_summary.get("processed_total") or dataset_summary.get("processed_images") or 0
        )
        test_count = int(saved_samples)
        holdout_pct = round(test_count / processed * 100, 1) if processed > 0 else None
        train_images = dataset_summary.get("train_images") or dataset_summary.get("train_total")
        validation_images = dataset_summary.get("validation_images") or dataset_summary.get("validation_total")
        return {
            "test_samples": test_count,
            "total_dataset_raw": dataset_summary.get("raw_total") or dataset_summary.get("total_dataset_raw"),
            "total_dataset_processed": processed,
            "train_images": train_images,
            "validation_images": validation_images,
            "holdout_percent": holdout_pct,
            "expected_test_total": dataset_summary.get("expected_test_total") or dataset_summary.get("expected_total"),
            "test_source": dataset_summary.get("test_source_key") or dataset_summary.get("test_source", "test"),
            "split_valid": bool(processed > 0 and test_count > 0 and test_count < processed),
            "samples_match_saved": True,
            "warnings": list(dataset_summary.get("warnings") or []),
        }

    test_dir, test_source = resolve_evaluation_test_dir(config)
    test_items = iter_test_images(test_dir, class_labels)
    test_count = len(test_items)
    totals = _count_split_totals(config, class_labels)
    split_report = load_split_report(config)
    expected_test = int((split_report.get("splits") or {}).get("test", {}).get("total") or 0)
    holdout_pct = (
        round(test_count / totals["processed"] * 100, 1) if totals["processed"] > 0 else None
    )
    saved_samples = eval_json.get("test_samples")

    warnings: List[str] = []
    if saved_samples is not None and int(saved_samples) != test_count:
        warnings.append(
            f"Laporan evaluasi tersimpan ({saved_samples} gambar) berbeda dari folder test saat ini ({test_count}). "
            "Jalankan ulang evaluasi setelah split dataset."
        )
    if totals["raw"] > 0 and test_count >= totals["raw"]:
        warnings.append(
            f"Folder test ({test_count}) sama dengan dataset mentah ({totals['raw']}). "
            "Evaluasi mungkin mencakup seluruh dataset, bukan holdout 20%."
        )
    elif totals["processed"] > 0 and test_count >= totals["processed"]:
        warnings.append(
            f"Jumlah test ({test_count}) sama dengan total dataset diproses ({totals['processed']}). "
            "Pastikan dataset sudah dibagi train/validation/test."
        )
    if expected_test and expected_test != test_count:
        warnings.append(
            f"Folder test saat ini ({test_count}) berbeda dari laporan split ({expected_test})."
        )

    split_valid = bool(
        totals["processed"] > 0
        and test_count > 0
        and test_count < totals["processed"]
        and totals["train"] > 0
        and totals["validation"] > 0
    )

    return {
        "test_samples": test_count,
        "total_dataset_raw": totals["raw"] or None,
        "total_dataset_processed": totals["processed"] or None,
        "train_images": totals["train"] or None,
        "validation_images": totals["validation"] or None,
        "holdout_percent": holdout_pct,
        "expected_test_total": expected_test or None,
        "test_source": test_source,
        "split_valid": split_valid,
        "samples_match_saved": saved_samples is None or int(saved_samples) == test_count,
        "warnings": warnings,
    }


def load_split_report(config=None) -> Dict[str, Any]:
    if config is None:
        config = current_app.config
    path = config.get("DATASET_SPLIT_REPORT_PATH")
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def validate_tm_class_alignment(class_labels: List[str], config=None) -> Dict[str, Any]:
    if not is_teachable_machine_available(config):
        return {"valid": True, "warnings": [], "tm_labels": [], "tm_labels_normalized": []}

    metadata = load_tm_metadata(config)
    raw_labels = metadata.get("labels") or []
    normalized = [normalize_tm_label(label) for label in raw_labels]
    warnings: List[str] = []

    missing_in_model = [label for label in class_labels if label not in normalized]
    extra_in_model = [label for label in normalized if label not in class_labels]
    if missing_in_model:
        warnings.append(
            "Model Teachable Machine tidak memiliki kelas: "
            + ", ".join(missing_in_model)
        )
    if extra_in_model:
        warnings.append(
            "Model memiliki kelas tambahan di luar dataset: "
            + ", ".join(extra_in_model)
        )
    if len(normalized) != len(set(normalized)):
        warnings.append("Label model TM tumpang tindih setelah normalisasi.")

    return {
        "valid": not missing_in_model,
        "warnings": warnings,
        "tm_labels": raw_labels,
        "tm_labels_normalized": normalized,
        "model_name": get_tm_display_info(config).get("model_name"),
        "image_size": int(metadata.get("imageSize") or 224),
    }


def build_dataset_summary(
    test_dir: str,
    test_items: List[Tuple[str, str]],
    class_labels: List[str],
    config=None,
    *,
    actual_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    if config is None:
        config = current_app.config

    if actual_counts is None:
        actual_counts = {label: 0 for label in class_labels}
        for _, label in test_items:
            if label in actual_counts:
                actual_counts[label] += 1
    else:
        actual_counts = dict(actual_counts)

    disk_counts = actual_counts if config.get("EVAL_FAST_MODE", True) else count_images_per_class(test_dir, class_labels)
    split_report = load_split_report(config)
    expected_counts = (
        (split_report.get("splits") or {}).get("test", {}).get("per_class") or {}
    )
    expected_total = int((split_report.get("splits") or {}).get("test", {}).get("total") or 0)

    per_class_rows = []
    warnings: List[str] = []
    for label in class_labels:
        actual = int(actual_counts.get(label, 0))
        expected = int(expected_counts.get(label, 0)) if expected_counts else None
        disk = int(disk_counts.get(label, 0))
        row = {
            "class_key": label,
            "actual_count": actual,
            "disk_count": disk,
            "expected_count": expected,
            "share_pct": round(actual / len(test_items) * 100, 2) if test_items else 0.0,
        }
        if expected is not None and expected != actual:
            warnings.append(
                f"Kelas {label}: {actual} gambar dievaluasi, laporan split mengharapkan {expected}."
            )
        if disk != actual:
            warnings.append(
                f"Kelas {label}: {actual} gambar terbaca untuk evaluasi, {disk} file di folder test."
            )
        per_class_rows.append(row)

    total_actual = len(test_items)
    if expected_total and expected_total != total_actual:
        warnings.append(
            f"Total evaluasi ({total_actual}) berbeda dari laporan split test ({expected_total}). "
            "Jalankan ulang proses split dataset jika dataset baru ditambahkan."
        )

    totals = _count_split_totals(config, class_labels)
    train_total = int((split_report.get("splits") or {}).get("train", {}).get("total") or 0) or totals["train"]
    val_total = int((split_report.get("splits") or {}).get("validation", {}).get("total") or 0) or totals["validation"]
    raw_total = totals["raw"]
    processed_total = train_total + val_total + (expected_total or total_actual)
    if not processed_total:
        processed_total = totals["processed"]

    return {
        "test_dir": test_dir,
        "test_source": os.path.basename(test_dir),
        "split_policy": split_report.get("evaluation_policy")
        or "Evaluate only on test holdout — never on training images.",
        "split_ratio": split_report.get("split_ratio") or "80/20 (train+validation / test holdout)",
        "effective_ratio": split_report.get("effective_ratio") or "68/12/20 train/validation/test",
        "train_holdout_percent": split_report.get("train_holdout_percent", 80),
        "stratified": bool(split_report.get("stratified", True)),
        "total_images": total_actual,
        "total_dataset_raw": raw_total or None,
        "total_dataset_processed": processed_total or None,
        "expected_total": expected_total or None,
        "train_images": train_total or None,
        "validation_images": val_total or None,
        "processed_images": processed_total or None,
        "holdout_percent": round(total_actual / processed_total * 100, 1) if processed_total else None,
        "per_class": per_class_rows,
        "warnings": warnings,
        "class_labels": class_labels,
    }
