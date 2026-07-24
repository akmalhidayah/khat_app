#!/usr/bin/env python3
"""Validate Arabic Khat AI installation and print a readable report."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from installer_lib import (
    PROJECT_ROOT,
    check_mysql_connection,
    check_packages,
    check_python_version,
    check_virtualenv,
    check_write_permissions,
    dataset_has_images,
    ensure_directories,
    model_files_ready,
    print_banner,
    project_path,
    validate_model_files,
)


def _print_line(status: str, label: str, detail: str = "") -> None:
    tag = {"OK": "[OK]", "WARNING": "[WARNING]", "ERROR": "[ERROR]"}.get(status, f"[{status}]")
    suffix = f" — {detail}" if detail else ""
    print(f"{tag} {label}{suffix}")


def main() -> int:
    print_banner()
    errors = 0
    warnings = 0

    status, version, detail = check_python_version()
    _print_line(status, f"Python version ({version})", detail)
    if status == "ERROR":
        errors += 1
    elif status == "WARNING":
        warnings += 1

    venv_status, venv_detail = check_virtualenv()
    _print_line(venv_status, "Virtual environment", venv_detail)
    if venv_status == "ERROR":
        errors += 1

    perm_status, perm_detail = check_write_permissions()
    _print_line(perm_status, "Write permissions", perm_detail)
    if perm_status == "ERROR":
        errors += 1

    created = ensure_directories()
    _print_line("OK" if not created else "OK", "Required folders", "all present" if not created else f"created {len(created)} folder(s)")

    for rel in [
        "static",
        "static/uploads",
        "templates",
        "dataset",
        "logs",
        "temp",
    ]:
        path = project_path(rel)
        if path.exists():
            _print_line("OK", f"Folder {rel}")
        else:
            _print_line("ERROR", f"Folder {rel}", "missing")
            errors += 1

    for rel, status, detail in validate_model_files():
        _print_line(status, f"Model file {rel}", detail)
        if status == "ERROR":
            errors += 1
        elif status == "WARNING":
            warnings += 1

    if not model_files_ready():
        print()
        print("Recommendation:")
        print("  Model Teachable Machine belum tersedia. Salin file model.json, metadata.json,")
        print("  dan weights.bin ke folder static/model/.")

    db_status, db_detail = check_mysql_connection()
    _print_line(db_status, "MySQL database", db_detail)
    if db_status == "ERROR":
        errors += 1

    ds_status, ds_detail = dataset_has_images()
    _print_line(ds_status, "Dataset content", ds_detail)
    if ds_status == "WARNING":
        warnings += 1

    print()
    print("Python packages:")
    for pkg, pkg_status in check_packages():
        _print_line(pkg_status, pkg)
        if pkg_status == "ERROR":
            errors += 1
        elif pkg_status == "WARNING":
            warnings += 1

    print()
    print("-" * 60)
    if errors:
        print(f"Result: {errors} error(s), {warnings} warning(s).")
        print("Jalankan installer/repair_windows.bat atau baca installer/README_INSTALL_WINDOWS.md.")
        return 1
    if warnings:
        print(f"Result: OK with {warnings} warning(s). Application can run with limited features.")
        return 0
    print("Result: All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
