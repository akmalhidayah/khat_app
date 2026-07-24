"""Pre-start checks used by run.bat before launching the Flask app."""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

DEFAULT_ENV = """SECRET_KEY=your-strong-secret-key
DB_HOST=localhost
DB_PORT=3306
DB_NAME=db_khat_classification
DB_USER=root
DB_PASSWORD=
"""


def ensure_env_file() -> bool:
    if os.path.exists(ENV_PATH):
        return True

    with open(ENV_PATH, "w", encoding="utf-8") as handle:
        handle.write(DEFAULT_ENV)
    print("[INFO] File .env dibuat otomatis (MySQL user: root, password kosong).")
    return True


def check_imports() -> bool:
    modules = (
        ("flask", "Flask"),
        ("flask_sqlalchemy", "Flask-SQLAlchemy"),
        ("pymysql", "PyMySQL"),
        ("dotenv", "python-dotenv"),
    )
    missing = []
    for module_name, package_name in modules:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        print(f"[ERROR] Paket belum terinstall: {', '.join(missing)}")
        print("Jalankan setup.bat terlebih dahulu.")
        return False
    return True


def check_mysql() -> bool:
    from dotenv import load_dotenv
    import pymysql

    load_dotenv(ENV_PATH)

    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "3306"))
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "")

    try:
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            charset="utf8mb4",
            connect_timeout=5,
        )
        conn.close()
        print(f"[OK] MySQL terhubung ({user}@{host}:{port}).")
        return True
    except Exception as exc:
        print(f"[ERROR] Tidak dapat terhubung ke MySQL: {exc}")
        print("Pastikan MySQL di XAMPP Control Panel sudah Start.")
        print("Periksa DB_USER dan DB_PASSWORD di file .env")
        return False


def main() -> int:
    if not ensure_env_file():
        return 1
    if not check_imports():
        return 1
    if not check_mysql():
        return 1
    print("[OK] Semua pengecekan awal berhasil.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
