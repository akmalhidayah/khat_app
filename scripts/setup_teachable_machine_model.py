#!/usr/bin/env python3
"""Extract Teachable Machine export into static/model/ for local inference."""

import os
import shutil
import sys
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = os.path.join(BASE_DIR, "static", "model")
REQUIRED_FILES = ("model.json", "metadata.json", "weights.bin")

ZIP_CANDIDATES = [
    os.path.join(BASE_DIR, "tm-my-image-model.zip"),
    os.path.expanduser("~/Downloads/tm-my-image-model.zip"),
]


def _find_model_files(root: str):
    found = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name in REQUIRED_FILES:
                found[name] = os.path.join(dirpath, name)
    return found


def _already_installed() -> bool:
    return all(os.path.isfile(os.path.join(TARGET_DIR, name)) for name in REQUIRED_FILES)


def main() -> int:
    os.makedirs(TARGET_DIR, exist_ok=True)

    if _already_installed():
        print(f"Teachable Machine model already present in {TARGET_DIR}")
        return 0

    zip_path = None
    for candidate in ZIP_CANDIDATES:
        if os.path.isfile(candidate):
            zip_path = candidate
            break

    if not zip_path:
        print("Teachable Machine ZIP not found.")
        print("Copy tm-my-image-model.zip to one of:")
        for candidate in ZIP_CANDIDATES:
            print(f"  - {candidate}")
        return 1

    extract_root = os.path.join(BASE_DIR, ".tm_extract")
    if os.path.isdir(extract_root):
        shutil.rmtree(extract_root)
    os.makedirs(extract_root, exist_ok=True)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_root)

    found = _find_model_files(extract_root)
    missing = [name for name in REQUIRED_FILES if name not in found]
    if missing:
        print(f"Missing files in archive: {', '.join(missing)}")
        shutil.rmtree(extract_root, ignore_errors=True)
        return 1

    for name in REQUIRED_FILES:
        dest = os.path.join(TARGET_DIR, name)
        shutil.copy2(found[name], dest)
        print(f"Installed {name} -> {dest}")

    shutil.rmtree(extract_root, ignore_errors=True)
    print("Teachable Machine model setup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
