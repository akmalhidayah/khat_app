"""Safe ZIP extraction helpers for dataset import."""

from __future__ import annotations

import os
import shutil
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

ZIP_IGNORED_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}
ZIP_IGNORED_PREFIXES = (".", "~$")
ZIP_CLASS_ALIASES = {
    "naskhi": "naskhi",
    "diwani": "diwani",
    "diwani jali": "diwani_jali",
    "diwani_jali": "diwani_jali",
    "tsuluts": "tsuluts",
    "thuluth": "tsuluts",
}


@dataclass
class ZipImportSummary:
    imported: int = 0
    total_found: int = 0
    optimized: int = 0
    model_ready: int = 0
    skipped_unsupported: int = 0
    skipped_ignored: int = 0
    skipped_duplicates: int = 0
    skipped_corrupted: int = 0
    skipped_unsafe: int = 0
    original_total_bytes: int = 0
    optimized_total_bytes: int = 0
    saved_bytes: int = 0
    compression_ratio: float = 0.0
    processing_date: str = ""
    per_class: Dict[str, int] = field(default_factory=dict)
    skipped_files: List[str] = field(default_factory=list)
    processing_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "imported": self.imported,
            "total_found": self.total_found,
            "optimized": self.optimized,
            "model_ready": self.model_ready,
            "skipped_unsupported": self.skipped_unsupported,
            "skipped_ignored": self.skipped_ignored,
            "skipped_duplicates": self.skipped_duplicates,
            "skipped_corrupted": self.skipped_corrupted,
            "skipped_unsafe": self.skipped_unsafe,
            "original_total_bytes": self.original_total_bytes,
            "optimized_total_bytes": self.optimized_total_bytes,
            "saved_bytes": self.saved_bytes,
            "compression_ratio": self.compression_ratio,
            "processing_date": self.processing_date,
            "per_class": self.per_class,
            "skipped_files": self.skipped_files,
            "processing_notes": self.processing_notes,
        }


def zip_is_ignored(name: str) -> bool:
    normalized = name.replace("\\", "/").lower()
    base = os.path.basename(name).strip()
    lower = base.lower()
    if lower in ZIP_IGNORED_NAMES:
        return True
    if lower.startswith("__macosx") or "/__macosx/" in normalized or normalized.startswith("__macosx/"):
        return True
    if any(lower.startswith(prefix) for prefix in ZIP_IGNORED_PREFIXES):
        return True
    return False


def zip_normalize_class(folder_name: str) -> Optional[str]:
    key = folder_name.strip().lower().replace("_", " ")
    key = " ".join(key.split())
    return ZIP_CLASS_ALIASES.get(key) or ZIP_CLASS_ALIASES.get(folder_name.strip().lower())


def zip_safe_extract(archive: zipfile.ZipFile, dest_dir: str, summary: ZipImportSummary) -> None:
    dest_real = os.path.realpath(dest_dir)
    os.makedirs(dest_real, exist_ok=True)

    for member in archive.infolist():
        member_name = member.filename.replace("\\", "/")
        if not member_name or member_name.endswith("/"):
            continue
        if ".." in member_name.split("/"):
            summary.skipped_unsafe += 1
            summary.skipped_files.append(member_name)
            continue
        if member_name.startswith("/") or (len(member_name) > 1 and member_name[1] == ":"):
            summary.skipped_unsafe += 1
            summary.skipped_files.append(member_name)
            continue
        if zip_is_ignored(member_name):
            summary.skipped_ignored += 1
            continue

        target_path = os.path.realpath(os.path.join(dest_real, member_name))
        if not (target_path == dest_real or target_path.startswith(dest_real + os.sep)):
            summary.skipped_unsafe += 1
            summary.skipped_files.append(member_name)
            continue

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with archive.open(member) as source, open(target_path, "wb") as target:
            shutil.copyfileobj(source, target)
