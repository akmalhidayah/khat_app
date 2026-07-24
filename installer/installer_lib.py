"""Shared helpers for Arabic Khat AI Windows/macOS/Linux installer scripts."""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project root = parent of installer/
INSTALLER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INSTALLER_DIR.parent

CLASS_LABELS = ["naskhi", "diwani", "diwani_jali", "tsuluts"]

REQUIRED_FOLDERS = [
    "static",
    "static/model",
    "static/uploads",
    "static/dataset_preview",
    "static/evaluation",
    "templates",
    "dataset",
    "dataset/raw",
    "dataset/processed",
    "dataset/model_ready",
    "dataset/train",
    "dataset/validation",
    "dataset/test",
    "dataset/tmp",
    "model",
    "logs",
    "temp",
    "instance",
]

MODEL_FILES = [
    "static/model/model.json",
    "static/model/metadata.json",
    "static/model/weights.bin",
]

REQUIRED_PACKAGES = [
    "flask",
    "flask_sqlalchemy",
    "pymysql",
    "dotenv",
    "werkzeug",
    "PIL",
    "numpy",
    "pandas",
    "sklearn",
    "cv2",
]

PACKAGE_IMPORT_MAP = {
    "flask": "flask",
    "flask_sqlalchemy": "flask_sqlalchemy",
    "pymysql": "pymysql",
    "dotenv": "dotenv",
    "werkzeug": "werkzeug",
    "PIL": "PIL",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "sklearn",
    "cv2": "cv2",
}


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def log_message(message: str, log_file: Optional[Path] = None) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line)
    target = log_file or project_path("logs", "install.log")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(line + os.linesep)
    except OSError:
        pass


def ensure_project_on_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def read_env_file() -> Dict[str, str]:
    env_path = project_path(".env")
    values: Dict[str, str] = {}
    if not env_path.is_file():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        values[key.strip()] = val.strip()
    return values


def copy_env_from_example(force: bool = False) -> Tuple[bool, str]:
    example = project_path(".env.example")
    env_file = project_path(".env")
    if env_file.is_file() and not force:
        return False, ".env already exists (not overwritten)."
    if not example.is_file():
        return False, ".env.example not found."
    shutil.copy2(example, env_file)
    return True, ".env created from .env.example."


def ensure_directories() -> List[str]:
    created: List[str] = []
    for rel in REQUIRED_FOLDERS:
        path = project_path(rel)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(rel)
    for split in ("raw", "processed", "train", "validation", "test", "model_ready"):
        for cls in CLASS_LABELS:
            path = project_path("dataset", split, cls)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                created.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    raw_non_khat = project_path("dataset", "raw", "non_khat")
    if not raw_non_khat.exists():
        raw_non_khat.mkdir(parents=True, exist_ok=True)
        created.append("dataset/raw/non_khat")
    return created


def validate_model_files() -> List[Tuple[str, str, str]]:
    """Return list of (relative_path, status, detail). status: OK|ERROR|WARNING."""
    results: List[Tuple[str, str, str]] = []
    for rel in MODEL_FILES:
        path = project_path(rel)
        if path.is_file() and path.stat().st_size > 0:
            size_mb = path.stat().st_size / (1024 * 1024)
            results.append((rel, "OK", f"{size_mb:.2f} MB"))
        elif path.is_file():
            results.append((rel, "WARNING", "file exists but is empty"))
        else:
            results.append((rel, "ERROR", "not found"))
    return results


def model_files_ready() -> bool:
    return all(status == "OK" for _, status, _ in validate_model_files())


def check_python_version() -> Tuple[str, str, str]:
    version = sys.version_info
    text = f"{version.major}.{version.minor}.{version.micro}"
    if (version.major, version.minor) in ((3, 10), (3, 11)):
        return "OK", text, "Recommended Python version."
    if version.major == 3 and version.minor >= 9:
        return "WARNING", text, "Python 3.10 or 3.11 is recommended."
    return "ERROR", text, "Python 3.10 or 3.11 is required."


def check_virtualenv() -> Tuple[str, str]:
    venv_dir = project_path(".venv")
    if platform.system() == "Windows":
        python_exe = venv_dir / "Scripts" / "python.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
    if venv_dir.is_dir() and python_exe.is_file():
        return "OK", str(venv_dir)
    return "ERROR", "Virtual environment .venv not found. Run install_windows.bat first."


def check_packages() -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    for label, module_name in PACKAGE_IMPORT_MAP.items():
        try:
            importlib.import_module(module_name)
            results.append((label, "OK"))
        except ImportError:
            results.append((label, "ERROR"))
    # TensorFlow is optional for TM-only inference but needed for training
    try:
        importlib.import_module("tensorflow")
        results.append(("tensorflow", "OK"))
    except ImportError:
        results.append(("tensorflow", "WARNING"))
    return results


def check_write_permissions() -> Tuple[str, str]:
    test_dir = project_path("temp", "_perm_test")
    try:
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file = test_dir / "write_test.tmp"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        test_dir.rmdir()
        return "OK", "Project folder is writable."
    except OSError as exc:
        return "ERROR", f"Cannot write to project folder: {exc}"


def check_mysql_connection() -> Tuple[str, str]:
    ensure_project_on_path()
    copy_env_from_example()
    env = read_env_file()
    host = env.get("DB_HOST", "localhost")
    port = int(env.get("DB_PORT", "3306"))
    user = env.get("DB_USER", "root")
    password = env.get("DB_PASSWORD", "")
    db_name = env.get("DB_NAME", "db_khat_classification")
    try:
        import pymysql

        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            charset="utf8mb4",
            connect_timeout=5,
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
            conn.commit()
        finally:
            conn.close()
        return "OK", f"MySQL reachable ({host}:{port}, database `{db_name}`)."
    except Exception as exc:
        return "ERROR", (
            f"MySQL connection failed: {exc}. "
            "Start MySQL (XAMPP) and verify DB_* settings in .env."
        )


def initialize_application(log: Optional[Path] = None) -> Tuple[bool, str]:
    """Run Flask app setup: folders, DB tables, default admin."""
    ensure_project_on_path()
    ensure_directories()
    copy_env_from_example()
    try:
        from app import create_app

        app = create_app()
        with app.app_context():
            from models import User

            admin = User.query.filter_by(username="admin").first()
            admin_note = "Default admin already exists." if admin else "Default admin created (admin / admin123)."
        log_message(f"Application initialized. {admin_note}", log)
        return True, admin_note
    except Exception as exc:
        msg = f"Application initialization failed: {exc}"
        log_message(msg, log)
        return False, msg


def clean_temp_files() -> List[str]:
    cleaned: List[str] = []
    temp_root = project_path("temp")
    if not temp_root.is_dir():
        return cleaned
    for item in temp_root.iterdir():
        if item.name.startswith("_perm_test"):
            continue
        try:
            if item.is_file():
                item.unlink()
                cleaned.append(str(item.relative_to(PROJECT_ROOT)))
            elif item.is_dir():
                shutil.rmtree(item)
                cleaned.append(str(item.relative_to(PROJECT_ROOT)))
        except OSError:
            pass
    dataset_tmp = project_path("dataset", "tmp")
    if dataset_tmp.is_dir():
        for item in dataset_tmp.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                    cleaned.append(str(item.relative_to(PROJECT_ROOT)))
            except OSError:
                pass
    return cleaned


def dataset_has_images() -> Tuple[str, str]:
    raw = project_path("dataset", "raw")
    count = 0
    if raw.is_dir():
        for cls in CLASS_LABELS:
            cls_dir = raw / cls
            if cls_dir.is_dir():
                count += sum(1 for f in cls_dir.iterdir() if f.is_file())
    if count > 0:
        return "OK", f"{count} raw dataset image(s) found."
    return "WARNING", "Dataset/raw class folders are empty."


def print_banner(title: str = "Arabic Khat AI System Check") -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"Project : {PROJECT_ROOT}")
    print(f"OS      : {platform.system()} {platform.release()}")
    print(f"Python  : {platform.python_version()}")
    print("-" * 60)
