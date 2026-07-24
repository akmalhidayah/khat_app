#!/usr/bin/env python3
"""First-run setup: folders, .env, database, default admin."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from installer_lib import (
    copy_env_from_example,
    ensure_directories,
    initialize_application,
    log_message,
    model_files_ready,
    print_banner,
    project_path,
    validate_model_files,
)


def main() -> int:
    print_banner("Arabic Khat AI Setup")
    log_file = project_path("logs", "install.log")
    log_message("Starting setup_app.py", log_file)

    created = ensure_directories()
    if created:
        log_message(f"Created folders: {', '.join(created[:8])}{'...' if len(created) > 8 else ''}", log_file)
        print(f"[OK] Created {len(created)} folder(s).")
    else:
        print("[OK] Required folders already exist.")

    copied, env_msg = copy_env_from_example()
    print(f"[{'OK' if copied else 'INFO'}] {env_msg}")
    log_message(env_msg, log_file)

    print()
    print("Model validation:")
    model_ok = True
    for rel, status, detail in validate_model_files():
        tag = {"OK": "[OK]", "WARNING": "[WARNING]", "ERROR": "[ERROR]"}.get(status, status)
        print(f"  {tag} {rel} — {detail}")
        if status == "ERROR":
            model_ok = False

    if not model_ok:
        print()
        print("[WARNING] Model Teachable Machine belum tersedia.")
        print("  Salin model.json, metadata.json, dan weights.bin ke static/model/.")
        print("  Aplikasi tetap dapat dijalankan, tetapi prediksi TM tidak aktif.")
        log_message("Teachable Machine model files missing.", log_file)

    print()
    print("Initializing database and default admin...")
    ok, note = initialize_application(log_file)
    if ok:
        print(f"[OK] {note}")
        print("[WARNING] Segera ganti password admin setelah login.")
        print("         Default: Username admin / Password admin123")
        log_message("Setup completed successfully.", log_file)
        return 0

    print(f"[ERROR] {note}")
    print("Pastikan MySQL (XAMPP) sudah berjalan dan pengaturan .env benar.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
