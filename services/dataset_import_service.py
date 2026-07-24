"""
Import Arabic Khat calligraphy images from a local folder into dataset/raw/.

Usage (from project root):
    python services/dataset_import_service.py
    python services/dataset_import_service.py --source "/path/to/source"
    python services/dataset_import_service.py --no-db
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import Config
from services.dataset_path_service import store_path

DEFAULT_SOURCE = ""
RAW_DATASET_DIR = Config.RAW_DATASET_DIR
CLASS_LABELS = Config.CLASS_LABELS
ALLOWED_EXTENSIONS = Config.ALLOWED_EXTENSIONS

CLASS_ALIASES = {
    "naskhi": "naskhi",
    "diwani": "diwani",
    "diwani jali": "diwani_jali",
    "diwani_jali": "diwani_jali",
    "tsuluts": "tsuluts",
    "thuluth": "tsuluts",
    "riqah": "riqah",
    "riq'ah": "riqah",
    "kufi": "kufi",
}

IGNORED_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}
IGNORED_PREFIXES = (".", "~$")


@dataclass
class ImportSummary:
    source: str
    destination: str
    copied: int = 0
    skipped_duplicates: int = 0
    skipped_unsupported: int = 0
    skipped_ignored: int = 0
    db_registered: int = 0
    db_skipped: int = 0
    per_class: Dict[str, int] = field(default_factory=dict)
    formats_found: Set[str] = field(default_factory=set)
    unsupported_files: List[str] = field(default_factory=list)


def normalize_class_name(folder_name: str) -> Optional[str]:
    key = folder_name.strip().lower().replace("_", " ")
    key = " ".join(key.split())
    normalized = CLASS_ALIASES.get(key) or CLASS_ALIASES.get(folder_name.strip().lower())
    if normalized in CLASS_LABELS:
        return normalized
    return None


def is_ignored_file(filename: str) -> bool:
    name = filename.strip()
    lower = name.lower()
    if lower in IGNORED_NAMES:
        return True
    if any(lower.startswith(prefix) for prefix in IGNORED_PREFIXES):
        return True
    return False


def get_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def is_supported_image(filename: str) -> bool:
    return get_extension(filename) in ALLOWED_EXTENSIONS


def unique_destination_path(dest_dir: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dest_dir, filename)
    if not os.path.exists(candidate):
        return candidate
    counter = 1
    while True:
        renamed = f"{base}_{counter}{ext}"
        candidate = os.path.join(dest_dir, renamed)
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def ensure_raw_dirs() -> None:
    os.makedirs(RAW_DATASET_DIR, exist_ok=True)
    for label in CLASS_LABELS:
        os.makedirs(os.path.join(RAW_DATASET_DIR, label), exist_ok=True)


def copy_dataset_from_source(source_dir: str, destination_dir: str = RAW_DATASET_DIR) -> ImportSummary:
    summary = ImportSummary(source=source_dir, destination=destination_dir)
    ensure_raw_dirs()

    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    for root, _, files in os.walk(source_dir):
        folder_name = os.path.basename(root)
        class_name = normalize_class_name(folder_name)
        if not class_name:
            continue

        dest_class_dir = os.path.join(destination_dir, class_name)
        os.makedirs(dest_class_dir, exist_ok=True)

        for filename in files:
            if is_ignored_file(filename):
                summary.skipped_ignored += 1
                continue

            if not is_supported_image(filename):
                if "." in filename:
                    summary.skipped_unsupported += 1
                    summary.unsupported_files.append(os.path.join(root, filename))
                continue

            ext = get_extension(filename)
            summary.formats_found.add(f".{ext}")

            source_path = os.path.join(root, filename)
            safe_name = filename.replace(" ", "_")
            target_path = unique_destination_path(dest_class_dir, safe_name)
            shutil.copy2(source_path, target_path)
            summary.copied += 1
            summary.per_class[class_name] = summary.per_class.get(class_name, 0) + 1

    return summary


def scan_raw_dataset(raw_dir: str = RAW_DATASET_DIR) -> Tuple[int, Dict[str, int], Set[str], List[str]]:
    per_class: Dict[str, int] = {label: 0 for label in CLASS_LABELS}
    formats_found: Set[str] = set()
    unsupported_files: List[str] = []
    total = 0

    if not os.path.isdir(raw_dir):
        return total, per_class, formats_found, unsupported_files

    for class_name in CLASS_LABELS:
        class_dir = os.path.join(raw_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if is_ignored_file(filename):
                continue
            path = os.path.join(class_dir, filename)
            if not os.path.isfile(path):
                continue
            if is_supported_image(filename):
                ext = get_extension(filename)
                formats_found.add(f".{ext}")
                per_class[class_name] += 1
                total += 1
            else:
                unsupported_files.append(path)

    return total, per_class, formats_found, unsupported_files


def sync_raw_dataset_to_db(raw_dir: str = RAW_DATASET_DIR) -> Tuple[int, int]:
    from app import create_app
    from models import Dataset, db

    registered = 0
    skipped = 0

    app = create_app()
    with app.app_context():
        existing_paths = {
            os.path.normpath(os.path.join(Config.BASE_DIR, item.image_path))
            for item in Dataset.query.all()
        }

        for class_name in CLASS_LABELS:
            class_dir = os.path.join(raw_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in os.listdir(class_dir):
                if is_ignored_file(filename) or not is_supported_image(filename):
                    continue

                abs_path = os.path.normpath(os.path.join(class_dir, filename))
                if abs_path in existing_paths:
                    skipped += 1
                    continue

                try:
                    image = Image.open(abs_path)
                    image_size = f"{image.width}x{image.height}"
                    ext = get_extension(filename)
                    rel_path = store_path(abs_path, Config.BASE_DIR)

                    dataset = Dataset(
                        filename=filename,
                        original_filename=filename,
                        class_name=class_name,
                        image_path=rel_path,
                        data_type="raw",
                        image_format=image.format or ext.upper(),
                        image_size=image_size,
                    )
                    db.session.add(dataset)
                    existing_paths.add(abs_path)
                    registered += 1
                except Exception:
                    skipped += 1

        db.session.commit()

    return registered, skipped


def print_summary(
    total: int,
    per_class: Dict[str, int],
    formats_found: Set[str],
    unsupported_files: List[str],
    import_summary: Optional[ImportSummary] = None,
) -> None:
    print("\n" + "=" * 56)
    print("  Arabic Khat Dataset Import Summary")
    print("=" * 56)

    if import_summary:
        print(f"  Source:      {import_summary.source}")
        print(f"  Destination: {import_summary.destination}")
        print(f"  Copied:      {import_summary.copied} image(s)")
        if import_summary.skipped_duplicates:
            print(f"  Renamed:     handled via unique filenames")
        if import_summary.skipped_ignored:
            print(f"  Ignored:     {import_summary.skipped_ignored} system/hidden file(s)")
        if import_summary.skipped_unsupported:
            print(f"  Unsupported: {import_summary.skipped_unsupported} file(s) in source")
        if import_summary.db_registered or import_summary.db_skipped:
            print(f"  DB added:    {import_summary.db_registered}")
            print(f"  DB skipped:  {import_summary.db_skipped}")

    print("\n  Dataset scan (dataset/raw/)")
    print(f"  Total images: {total}")
    print("\n  Images per class:")
    for label in CLASS_LABELS:
        display = label.replace("_", " ").title()
        print(f"    - {display:<14} {per_class.get(label, 0)}")

    print("\n  Supported formats found:")
    if formats_found:
        for fmt in sorted(formats_found):
            print(f"    - {fmt}")
    else:
        print("    - (none)")

    if unsupported_files:
        print(f"\n  Unsupported files ({len(unsupported_files)}):")
        for path in unsupported_files[:20]:
            print(f"    - {path}")
        if len(unsupported_files) > 20:
            print(f"    ... and {len(unsupported_files) - 20} more")
    else:
        print("\n  Unsupported files: none")

    print("=" * 56 + "\n")


def run_import(source: str = DEFAULT_SOURCE, sync_db: bool = True) -> ImportSummary:
    if not os.path.isdir(source):
        raise FileNotFoundError(f"Source folder not found: {source}")

    ensure_raw_dirs()
    summary = copy_dataset_from_source(source, RAW_DATASET_DIR)

    if sync_db:
        registered, skipped = sync_raw_dataset_to_db(RAW_DATASET_DIR)
        summary.db_registered = registered
        summary.db_skipped = skipped

    total, per_class, formats_found, unsupported_files = scan_raw_dataset(RAW_DATASET_DIR)
    print_summary(total, per_class, formats_found, unsupported_files, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Arabic Khat dataset into dataset/raw/")
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Source folder containing class subfolders",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Copy files only; do not register images in the database",
    )
    args = parser.parse_args()
    if not args.source:
        print("Error: --source is required. Example:")
        print('  python services/dataset_import_service.py --source "/path/to/dataset"')
        sys.exit(1)

    try:
        run_import(source=args.source, sync_db=not args.no_db)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"Import failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
