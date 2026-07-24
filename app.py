import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from flask import Flask, flash, redirect, request, url_for
from werkzeug.exceptions import RequestEntityTooLarge
import pymysql

from config import Config
from models import User, db
from models.migrations import ensure_classification_result_columns
from services.dataset_upload_settings_service import apply_upload_settings_to_config
from routes import (
    auth_bp,
    algorithm_calculation_bp,
    classification_bp,
    cnn_calculation_bp,
    dashboard_bp,
    dataset_bp,
    evaluation_bp,
    report_bp,
    legacy_training_bp,
    model_management_bp,
    db_admin_bp,
)


def ensure_database_exists(app: Flask) -> None:
    db_name = app.config["SQLALCHEMY_DATABASE_URI"].rsplit("/", 1)[-1]
    host = Config._DB_HOST
    port = int(Config._DB_PORT)
    user = Config._DB_USER
    password = Config._DB_PASSWORD

    # Config password is URL-encoded for URI use; decode before direct pymysql connect.
    from urllib.parse import unquote_plus
    password = unquote_plus(password)

    conn = pymysql.connect(host=host, port=port, user=user, password=password, charset="utf8mb4")
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        conn.commit()
    finally:
        conn.close()


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    apply_upload_settings_to_config(app.config)

    db.init_app(app)
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(model_management_bp)
    app.register_blueprint(legacy_training_bp)
    app.register_blueprint(evaluation_bp)
    app.register_blueprint(classification_bp)
    app.register_blueprint(report_bp)
    app.register_blueprint(algorithm_calculation_bp)
    app.register_blueprint(cnn_calculation_bp)
    app.register_blueprint(db_admin_bp)

    @app.after_request
    def allow_camera_access(response):
        response.headers["Permissions-Policy"] = "camera=(self)"
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def request_entity_too_large(error):
        max_mb = app.config.get("UPLOAD_MAX_SIZE_MB", 1024)
        message = (
            f"File ZIP terlalu besar. Maksimal ukuran upload adalah {max_mb} MB. "
            "Silakan kompres gambar atau bagi dataset menjadi beberapa ZIP kecil."
        )
        if request.path.startswith("/dataset/upload-zip") or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            from flask import jsonify
            return jsonify({"success": False, "message": message}), 413
        flash(message, "danger")
        return redirect(url_for("dataset.upload_dataset"))

    @app.context_processor
    def inject_active_model():
        try:
            from services.model_context_service import (
                get_brand_chip_label,
                get_model_context,
                get_model_short_name,
                get_model_subtitle,
                is_teachable_machine_active,
            )
            from services.teachable_machine_service import get_tm_display_info

            ctx = get_model_context()
            short = get_model_short_name(ctx)
            tm_active = is_teachable_machine_active(ctx)
            tm_info = get_tm_display_info() if tm_active else None
            return {
                "global_model_short_name": short,
                "global_model_subtitle": get_model_subtitle(ctx),
                "global_model_architecture": ctx.get("model_architecture", f"{short} Transfer Learning"),
                "global_brand_chip": get_brand_chip_label(ctx),
                "global_is_teachable_machine": tm_active,
                "global_tm_model_info": tm_info,
            }
        except Exception:
            return {
                "global_model_short_name": "EfficientNetB0",
                "global_model_subtitle": "Arabic Calligraphy Khat Classification with EfficientNetB0 Transfer Learning",
                "global_model_architecture": "EfficientNetB0 Transfer Learning",
                "global_brand_chip": "EfficientNetB0 AI",
                "global_is_teachable_machine": False,
                "global_tm_model_info": None,
            }

    @app.route("/")
    def index():
        from flask import session
        if app.config.get("APP_SIMPLE_MODE"):
            if session.get("user_id"):
                if session.get("role") == "admin":
                    return redirect(url_for("dashboard.dashboard"))
                return redirect(url_for("classification.classify_image"))
            return redirect(url_for("auth.login"))
        if session.get("user_id"):
            return redirect(url_for("dashboard.dashboard"))
        return redirect(url_for("auth.login"))

    @app.route("/health")
    def health():
        return {"status": "ok"}

    @app.route("/home")
    def home():
        from flask import session
        if app.config.get("APP_SIMPLE_MODE"):
            if session.get("role") == "admin":
                return redirect(url_for("dashboard.dashboard"))
            return redirect(url_for("classification.classify_image"))
        return redirect(url_for("dashboard.dashboard"))

    with app.app_context():
        ensure_database_exists(app)
        db.create_all()
        for folder in [
            app.config["UPLOAD_FOLDER"],
            app.config["DATASET_PREVIEW_FOLDER"],
            app.config["EVALUATION_FOLDER"],
            app.config["MODEL_DIR"],
            app.config["TEACHABLE_MODEL_DIR"],
            app.config["NON_KHAT_DIR"],
            app.config.get("DATASET_TMP_DIR"),
            app.config.get("PROCESSED_DATASET_DIR"),
            app.config.get("MODEL_READY_DIR"),
            app.config.get("LOGS_DIR"),
            app.config.get("TEMP_DIR"),
            app.config.get("INSTANCE_DIR"),
        ]:
            if folder:
                os.makedirs(folder, exist_ok=True)
        ensure_classification_result_columns()
        from models.migrations import ensure_model_versions_table
        ensure_model_versions_table()
        try:
            from models.migrations_indonesian import run_indonesian_migration
            run_indonesian_migration(app.config)
        except Exception as exc:
            import logging
            db.session.rollback()
            logging.getLogger(__name__).warning("Indonesian migration skipped or partial: %s", exc)
        try:
            from services.model_version_service import register_active_tm_model
            from services.teachable_machine_service import is_teachable_machine_available
            if is_teachable_machine_available(app.config):
                register_active_tm_model(notes="Teachable Machine model auto-registered on startup.")
        except Exception:
            pass
        try:
            from services.classification_flow_service import ensure_active_cnn_model_version
            ensure_active_cnn_model_version()
        except Exception:
            pass
        try:
            from services.external_assets_service import ensure_external_model_ready
            ensure_external_model_ready(app.config)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("External model sync skipped: %s", exc)
        if not User.query.filter_by(username="user").first():
            demo = User(name="Pengguna Demo", username="user", role="pengguna")
            demo.set_password("user123")
            db.session.add(demo)
            db.session.commit()
        if not User.query.filter_by(username="admin").first():
            admin = User(name="Administrator", username="admin", role="admin")
            admin.set_password("admin123")
            db.session.add(admin)
            db.session.commit()

    return app


if __name__ == "__main__":
    application = create_app()
    host = os.environ.get("APP_HOST", application.config.get("APP_HOST", "127.0.0.1"))
    port = int(os.environ.get("APP_PORT", os.environ.get("FLASK_PORT", application.config.get("APP_PORT", 5002))))
    debug = os.environ.get("FLASK_ENV", "development") != "production"
    open_browser = os.environ.get("OPEN_BROWSER", "0") == "1"

    print()
    print("=" * 56)
    print(f"  Arabic Khat AI running at http://{host}:{port}")
    print("  Press Ctrl+C to stop.")
    print("=" * 56)
    print()

    if open_browser:
        import threading
        import time
        import urllib.error
        import urllib.request
        import webbrowser

        def _open_browser_when_ready():
            health_url = f"http://{host}:{port}/health"
            app_url = f"http://{host}:{port}"
            for _ in range(120):
                try:
                    with urllib.request.urlopen(health_url, timeout=1) as response:
                        if response.status == 200:
                            webbrowser.open(app_url)
                            return
                except (urllib.error.URLError, TimeoutError, OSError):
                    time.sleep(0.5)

        threading.Thread(target=_open_browser_when_ready, daemon=True).start()

    application.run(debug=debug, host=host, port=port, use_reloader=not open_browser)
