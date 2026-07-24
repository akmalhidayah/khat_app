import json
import os
import time
from datetime import datetime

from flask import Blueprint, current_app, flash, jsonify, redirect, render_template, request, url_for

from models import Dataset
from routes.utils import admin_required, login_required
from services.dataset_readiness_service import (
    build_pipeline_status,
    build_readiness_cards,
    get_dataset_readiness,
    get_model_status,
    load_training_history,
    load_training_summary,
)
from services.dataset_service import split_and_copy_dataset
from services.training_service import train_model
from services.training_status_service import (
    format_duration,
    load_training_status,
    mark_training_failed,
    mark_training_started,
    reconcile_training_status,
    resolve_button_mode,
)
from services.training_utils import log_training_error, read_last_training_error
from services.training_validation_service import validate_training_dataset
from services.dataset_quality_service import load_quality_report
from services.metrics_service import ACCURACY_TARGET, target_status
from services.khat_detector_service import get_thresholds

from services.model_builder_service import ARCHITECTURES
from services.model_context_service import get_model_context, is_teachable_machine_active
from services.model_version_service import (
    activate_model_version,
    archive_model_version,
    delete_model_version,
    get_active_model_version,
    list_model_versions,
    model_version_to_dict,
    sync_active_model_version_metrics,
)
from services.teachable_machine_service import get_tm_display_info, is_teachable_machine_available
from services.training_page_ui_service import (
    build_eval_display,
    build_model_management_insights,
    build_training_page_insights,
    build_training_safety,
)
from services.tm_model_upload_service import TmModelValidationError, process_tm_upload

model_management_bp = Blueprint("model_management", __name__, url_prefix="/model-management")
legacy_training_bp = Blueprint("training", __name__, url_prefix="/training")
TRAINING_META_PATH = "training_meta.json"
_TRAINING_CTX_CACHE = {"key": None, "value": None, "expires": 0.0}

ARCHITECTURE_SHORT = {
    "vgg16": "VGG16",
    "efficientnetb0": "EfficientNetB0",
    "mobilenetv2": "MobileNetV2",
}

MODE_LABELS = {
    "ultra_fast": {"en": "Ultra Fast Demo", "id": "Demo Ultra Cepat", "demo": True},
    "fast": {"en": "Fast Demo", "id": "Demo Cepat", "demo": True},
    "research": {"en": "Research Accuracy", "id": "Mode Riset Akurasi", "demo": False},
    "normal": {"en": "Research Accuracy", "id": "Mode Riset Akurasi", "demo": False},
}


def _architecture_label(key: str) -> str:
    return ARCHITECTURE_SHORT.get((key or "").lower(), "EfficientNetB0")


def _resolve_active_architecture(summary_config: dict, training_meta: dict) -> dict:
    key = (
        summary_config.get("model_architecture_key")
        or training_meta.get("model_architecture")
        or "efficientnetb0"
    )
    key = (key or "efficientnetb0").lower()
    if key not in ARCHITECTURE_SHORT:
        key = "efficientnetb0"
    label = _architecture_label(key)
    return {
        "key": key,
        "label": label,
        "badge": f"{label} Architecture",
        "full_label": ARCHITECTURES.get(key, f"{label} Transfer Learning"),
    }


def _resolve_training_mode(summary_config: dict, training_meta: dict) -> dict:
    mode = summary_config.get("training_mode_key") or training_meta.get("training_mode") or "research"
    if mode not in MODE_LABELS:
        mode = "research" if mode in ("normal",) else "fast"
    info = MODE_LABELS.get(mode, MODE_LABELS["research"])
    return {
        "key": mode,
        "label_en": info["en"],
        "label_id": info["id"],
        "is_demo": info["demo"],
        "is_research": not info["demo"],
    }


def _load_training_meta() -> dict:
    meta_path = os.path.join(current_app.config["MODEL_DIR"], TRAINING_META_PATH)
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_training_meta(result: dict) -> None:
    meta_path = os.path.join(current_app.config["MODEL_DIR"], TRAINING_META_PATH)
    os.makedirs(current_app.config["MODEL_DIR"], exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "epochs": result.get("epochs"),
                "batch_size": result.get("batch_size"),
                "learning_rate": result.get("learning_rate"),
                "training_mode": result.get("training_mode", "fast"),
                "duration_seconds": result.get("duration_seconds"),
                "trained_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            },
            f,
            indent=2,
        )


def _parse_int(value: str, default: int, field_name: str) -> int:
    cleaned = (value or "").strip().replace(",", ".")
    try:
        return int(float(cleaned))
    except (ValueError, TypeError):
        raise ValueError(f"Invalid {field_name}. Please enter a whole number.")


def _parse_learning_rate(value: str) -> float:
    cleaned = (value or "0.0001").strip().replace(",", ".")
    try:
        rate = float(cleaned)
    except (ValueError, TypeError):
        raise ValueError("Invalid learning rate. Use dot decimal format, for example 0.0001.")
    if rate < 0.000001 or rate > 1:
        raise ValueError("Learning rate must be between 0.000001 and 1.")
    return rate


def _parse_bool(value: str) -> bool:
    return (value or "").strip().lower() in {"1", "true", "on", "yes"}


def _training_context(force_refresh: bool = False):
    config = current_app.config
    ttl = float(config.get("MODEL_PAGE_CACHE_TTL", 30))
    cache_key = (
        os.path.getmtime(config["EVALUATION_RESULT_PATH"]) if os.path.isfile(config["EVALUATION_RESULT_PATH"]) else 0,
        os.path.getmtime(config["MODEL_PATH"]) if os.path.isfile(config["MODEL_PATH"]) else 0,
        os.path.getmtime(config["TRAINING_HISTORY_PATH"]) if os.path.isfile(config["TRAINING_HISTORY_PATH"]) else 0,
        config.get("USE_EXTERNAL_DATASET"),
        config.get("USE_EXTERNAL_MODEL"),
    )
    now = time.monotonic()
    if (
        not force_refresh
        and _TRAINING_CTX_CACHE["key"] == cache_key
        and now < _TRAINING_CTX_CACHE["expires"]
    ):
        return _TRAINING_CTX_CACHE["value"]

    readiness = get_dataset_readiness()
    model_status = get_model_status(
        current_app.config["MODEL_DIR"],
        current_app.config["MODEL_PATH"],
        current_app.config["TRAINING_HISTORY_PATH"],
        current_app.config["EVALUATION_RESULT_PATH"],
    )
    training_summary = load_training_summary(current_app.config["TRAINING_HISTORY_PATH"])
    history = load_training_history(current_app.config["TRAINING_HISTORY_PATH"])
    training_meta = _load_training_meta()
    training_status = load_training_status()
    has_history = bool(history)

    training_status = reconcile_training_status(
        training_status,
        model_status["exists"],
        has_history,
    )

    summary_config = training_summary.get("config", {})
    last_training_time = summary_config.get("training_date") or training_meta.get("trained_at")
    if not last_training_time and model_status["modified_at"]:
        last_training_time = datetime.utcfromtimestamp(model_status["modified_at"]).strftime("%Y-%m-%d %H:%M UTC")

    summary_state = "empty"
    if history:
        summary_state = "history"
    elif model_status["exists"]:
        summary_state = "model_only"

    button_mode = resolve_button_mode(readiness, model_status, has_history, training_status)
    low_accuracy_warning = False
    if history and history.get("val_accuracy"):
        low_accuracy_warning = float(history["val_accuracy"][-1]) < 0.70

    duration_seconds = summary_config.get("duration_seconds") or training_meta.get("duration_seconds")
    duration_display = format_duration(duration_seconds)

    eval_result = {}
    eval_path = current_app.config["EVALUATION_RESULT_PATH"]
    if os.path.isfile(eval_path):
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                eval_result = json.load(f)
        except (json.JSONDecodeError, OSError):
            eval_result = {}

    best_val = summary_config.get("best_validation_accuracy") or (
        float(max(history["val_accuracy"])) if history and history.get("val_accuracy") else None
    )
    target_validation = target_status(best_val, ACCURACY_TARGET)
    target_test = target_status(eval_result.get("accuracy"), ACCURACY_TARGET)
    target_achieved = summary_config.get("target_achieved", False)
    active_architecture = _resolve_active_architecture(summary_config, training_meta)
    training_mode_info = _resolve_training_mode(summary_config, training_meta)
    val_accuracy_percent = target_validation.get("percent")
    test_accuracy_percent = target_test.get("percent")
    below_target = not target_achieved and (
        (val_accuracy_percent is not None and val_accuracy_percent < ACCURACY_TARGET * 100)
        or (test_accuracy_percent is not None and test_accuracy_percent < ACCURACY_TARGET * 100)
    )

    ctx = {
        "readiness": readiness,
        "model_status": model_status,
        "history": history,
        "training_summary": training_summary,
        "summary_config": summary_config,
        "training_meta": training_meta,
        "training_status": training_status,
        "button_mode": button_mode,
        "summary_state": summary_state,
        "last_training_time": last_training_time,
        "low_accuracy_warning": low_accuracy_warning,
        "duration_display": duration_display,
        "is_training_running": training_status.get("status") == "running" and not (model_status["exists"] and has_history),
        "accuracy_target": ACCURACY_TARGET,
        "target_validation": target_validation,
        "target_test": target_test,
        "quality_report": load_quality_report(),
        "target_achieved": target_achieved,
        "active_architecture": active_architecture,
        "training_mode_info": training_mode_info,
        "val_accuracy_percent": val_accuracy_percent,
        "test_accuracy_percent": test_accuracy_percent,
        "below_target": below_target,
        "eval_result": eval_result,
    }
    _TRAINING_CTX_CACHE["key"] = cache_key
    _TRAINING_CTX_CACHE["value"] = ctx
    _TRAINING_CTX_CACHE["expires"] = now + ttl
    return ctx


@model_management_bp.route("/")
@login_required
def model_management_page():
    ctx = _training_context()
    readiness = ctx["readiness"]
    model_status = ctx["model_status"]

    validation_report = validate_training_dataset(fast=True)
    model_ctx = get_model_context()
    is_tm_active = is_teachable_machine_active(model_ctx)
    tm_info = get_tm_display_info() if is_teachable_machine_available() else None
    if is_tm_active:
        sync_active_model_version_metrics()
    active_version = get_active_model_version()
    active_model_version = model_version_to_dict(active_version) if active_version else None

    detector_eval = {}
    detector_eval_path = current_app.config.get("KHAT_DETECTOR_EVALUATION_PATH")
    if detector_eval_path and os.path.isfile(detector_eval_path):
        try:
            with open(detector_eval_path, "r", encoding="utf-8") as f:
                detector_eval = json.load(f)
        except (json.JSONDecodeError, OSError):
            detector_eval = {}

    page_insights = build_training_page_insights(
        readiness=readiness,
        quality_report=ctx["quality_report"],
        target_validation=ctx["target_validation"],
        target_test=ctx["target_test"],
        target_achieved=ctx["target_achieved"],
        accuracy_target=ctx["accuracy_target"],
        below_target=ctx["below_target"],
        eval_result=ctx["eval_result"],
        history=ctx["history"],
        is_tm_active=is_tm_active,
        active_model_version=active_model_version,
    )
    model_management = build_model_management_insights(
        readiness=readiness,
        quality_report=ctx["quality_report"],
        eval_result=ctx["eval_result"],
        is_tm_active=is_tm_active,
        below_target=ctx["below_target"],
        target_achieved=ctx["target_achieved"],
    )
    eval_display = build_eval_display(ctx["eval_result"], current_app.config["CLASS_LABELS"])
    eval_result = ctx["eval_result"] or {}
    if eval_display.get("has_eval"):
        dataset_summary = eval_result.get("dataset_summary") or {}
        if eval_result.get("test_samples") is not None:
            eval_display["test_samples"] = eval_result["test_samples"]
        if dataset_summary.get("processed_total") is not None:
            eval_display["total_dataset_processed"] = dataset_summary["processed_total"]
        elif readiness.get("processed_count"):
            eval_display["total_dataset_processed"] = readiness["processed_count"]
        processed = eval_display.get("total_dataset_processed") or 0
        test_n = eval_display.get("test_samples") or 0
        if processed > 0 and test_n > 0:
            eval_display["holdout_percent"] = round(test_n / processed * 100, 1)
    all_versions = [model_version_to_dict(row) for row in list_model_versions(limit=20)]
    model_versions = all_versions
    model_versions_preview = all_versions[:5]
    model_versions_has_more = len(all_versions) > 5
    training_safety = build_training_safety(
        readiness=readiness,
        validation_report=validation_report,
        classes_count=len(current_app.config["CLASS_LABELS"]),
        model_dir=current_app.config["MODEL_DIR"],
        quality_report=ctx["quality_report"],
    )

    debug_info = None
    if current_app.debug:
        debug_info = {
            "train_dir": current_app.config["TRAIN_DIR"],
            "validation_dir": current_app.config["VALIDATION_DIR"],
            "train_count": readiness["train_count"],
            "validation_count": readiness["validation_count"],
            "class_labels": current_app.config["CLASS_LABELS"],
            "model_path": current_app.config["MODEL_PATH"],
            "detected_classes": validation_report.get("detected_classes", {}),
            "last_error": read_last_training_error(current_app.config["BASE_DIR"]),
            "training_status": ctx["training_status"],
        }

    return render_template(
        "model_management.html",
        history=ctx["history"],
        training_summary=ctx["training_summary"],
        summary_config=ctx["summary_config"],
        training_meta=ctx["training_meta"],
        training_status=ctx["training_status"],
        total_dataset=max(Dataset.query.count(), readiness["raw_count"]),
        readiness=readiness,
        raw_count=readiness["raw_count"],
        processed_count=readiness["processed_count"],
        train_count=readiness["train_count"],
        validation_count=readiness["validation_count"],
        test_count=readiness["test_count"],
        classes_count=len(current_app.config["CLASS_LABELS"]),
        dataset_ready=readiness["is_ready"],
        model_status=model_status,
        model_exists=model_status["exists"],
        model_path=model_status["path"] or current_app.config["MODEL_PATH"],
        last_training_time=ctx["last_training_time"],
        duration_display=ctx["duration_display"],
        low_accuracy_warning=ctx["low_accuracy_warning"],
        pipeline_steps=build_pipeline_status(readiness, model_status),
        readiness_cards=build_readiness_cards(readiness, model_status),
        split_preview=readiness["split_preview"],
        button_mode=ctx["button_mode"],
        summary_state=ctx["summary_state"],
        is_training_running=ctx["is_training_running"],
        accuracy_target=ctx["accuracy_target"],
        target_validation=ctx["target_validation"],
        target_test=ctx["target_test"],
        quality_report=ctx["quality_report"],
        target_achieved=ctx["target_achieved"],
        active_architecture=ctx["active_architecture"],
        training_mode_info=ctx["training_mode_info"],
        val_accuracy_percent=ctx["val_accuracy_percent"],
        test_accuracy_percent=ctx["test_accuracy_percent"],
        below_target=ctx["below_target"],
        eval_result=ctx["eval_result"],
        debug_info=debug_info,
        khat_thresholds=get_thresholds(),
        is_tm_active=is_tm_active,
        tm_info=tm_info,
        active_model_version=active_model_version,
        page_insights=page_insights,
        model_management=model_management,
        eval_display=eval_display,
        model_versions=model_versions,
        model_versions_preview=model_versions_preview,
        model_versions_has_more=model_versions_has_more,
        model_versions_total=len(all_versions),
        training_safety=training_safety,
        validation_report=validation_report,
        detector_eval=detector_eval,
    )


@model_management_bp.route("/status")
@login_required
def training_status_api():
    ctx = _training_context()
    status = ctx["training_status"]
    return jsonify(
        {
            "status": status.get("status", "idle"),
            "message": status.get("message", ""),
            "current_stage": status.get("current_stage"),
            "current_epoch": status.get("current_epoch"),
            "total_epochs": status.get("total_epochs"),
            "button_mode": ctx["button_mode"],
            "model_exists": ctx["model_status"]["exists"],
            "history_exists": bool(ctx["history"]),
            "has_history": bool(ctx["history"]),
            "progress_percent": status.get("progress_percent"),
            "phase": status.get("phase") or status.get("training_phase"),
            "training_phase": status.get("training_phase"),
            "training_accuracy": status.get("training_accuracy") or status.get("current_accuracy"),
            "validation_accuracy": status.get("validation_accuracy") or status.get("current_val_accuracy"),
            "training_loss": status.get("training_loss") or status.get("current_loss"),
            "validation_loss": status.get("validation_loss") or status.get("current_val_loss"),
            "current_accuracy": status.get("current_accuracy"),
            "current_val_accuracy": status.get("current_val_accuracy"),
        }
    )


@model_management_bp.route("/process-dataset", methods=["POST"])
@admin_required
def process_dataset():
    readiness = get_dataset_readiness()
    if readiness["raw_count"] == 0:
        flash("No raw dataset images found. Please import or upload dataset images first.", "warning")
        return redirect(url_for("model_management.model_management_page"))
    try:
        result = split_and_copy_dataset()
        flash(
            "Dataset processed successfully. Training is now enabled. "
            f"Train: {result['train']}, Validation: {result['validation']}, Test: {result['test']}.",
            "success",
        )
    except ValueError as exc:
        flash(str(exc), "warning")
    except Exception:
        flash("Dataset processing failed. Please check folder permissions and dataset structure.", "danger")
    return redirect(url_for("model_management.model_management_page"))


@model_management_bp.route("/start", methods=["POST"])
@admin_required
def start_training():
    readiness = get_dataset_readiness()
    if not readiness["is_ready"]:
        flash("Dataset is not ready. Please process and split the dataset before training.", "warning")
        return redirect(url_for("model_management.model_management_page"))

    try:
        epochs = _parse_int(request.form.get("epochs", "12"), 12, "epochs")
        batch_size = _parse_int(request.form.get("batch_size", "16"), 16, "batch size")
        learning_rate = _parse_learning_rate(request.form.get("learning_rate", "0.0001"))
        training_mode = request.form.get("training_mode", "fast").strip()
        model_architecture = request.form.get("model_architecture", "efficientnetb0").strip()
        use_optimized_images = _parse_bool(request.form.get("use_optimized_images", "on"))
        cache_to_memory = _parse_bool(request.form.get("cache_to_memory", ""))
        auto_batch_fallback = _parse_bool(request.form.get("auto_batch_fallback", "on"))
        enable_class_weights = _parse_bool(request.form.get("enable_class_weights", "on"))
        enable_augmentation = _parse_bool(request.form.get("enable_augmentation", "on"))
        if training_mode not in ("ultra_fast", "fast", "research", "normal"):
            training_mode = "fast"
        if model_architecture not in ("vgg16", "efficientnetb0", "mobilenetv2"):
            model_architecture = "vgg16"
    except ValueError as exc:
        flash(str(exc), "warning")
        return redirect(url_for("model_management.model_management_page"))

    if epochs < 1 or epochs > 100:
        flash("Epochs must be between 1 and 100.", "warning")
        return redirect(url_for("model_management.model_management_page"))
    if batch_size < 1 or batch_size > 128:
        flash("Batch size must be between 1 and 128.", "warning")
        return redirect(url_for("model_management.model_management_page"))

    try:
        result = train_model(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            training_mode=training_mode,
            model_architecture=model_architecture,
            use_optimized_images=use_optimized_images,
            cache_to_memory=cache_to_memory,
            auto_batch_fallback=auto_batch_fallback,
            enable_class_weights=enable_class_weights,
            enable_augmentation=enable_augmentation,
        )
        _save_training_meta(result)
        mode_label = {
            "ultra_fast": "Ultra Fast Demo",
            "fast": "Fast Demo",
            "research": "Research Accuracy",
            "normal": "Research Accuracy",
        }.get(training_mode, training_mode)
        arch_label = model_architecture.upper()
        flash(
            "Training completed successfully. Model saved and training history generated. "
            f"Validation accuracy: {result['final_val_accuracy']:.4f}. "
            f"Best validation: {result.get('best_val_accuracy', result['final_val_accuracy']):.4f}. "
            f"Duration: {format_duration(result['duration_seconds'])}. Mode: {mode_label}. Architecture: {arch_label}."
            + (" Batch size was automatically reduced due to memory limitations." if result.get("batch_size_reduced") else ""),
            "success",
        )
        if result.get("target_achieved"):
            flash("Validation accuracy target (85%) achieved.", "success")
        else:
            flash(
                "Validation accuracy is below the 85% target. Run evaluation on the test set and consider "
                "Research Accuracy Mode with more epochs.",
                "warning",
            )
        for rec in result.get("recommendations", [])[:3]:
            flash(rec, "info")
        if result["final_val_accuracy"] < 0.70:
            flash(
                "Model accuracy is still low. Consider increasing epochs, improving dataset balance, "
                "or using data augmentation/class weights.",
                "warning",
            )
    except MemoryError as exc:
        log_path = log_training_error(current_app.config["BASE_DIR"], str(exc))
        mark_training_failed(str(exc))
        flash(f"{exc} See {log_path} for details.", "danger")
        current_app.logger.exception("Training failed: %s", exc)
    except ImportError as exc:
        log_path = log_training_error(current_app.config["BASE_DIR"], str(exc))
        mark_training_failed(str(exc))
        flash(f"{exc} See {log_path} for full traceback.", "danger")
        current_app.logger.exception("Training failed: %s", exc)
    except ValueError as exc:
        log_path = log_training_error(current_app.config["BASE_DIR"], str(exc))
        mark_training_failed(str(exc))
        flash(f"{exc}", "danger")
        current_app.logger.exception("Training failed: %s", exc)
    except Exception as exc:
        log_path = log_training_error(current_app.config["BASE_DIR"], str(exc))
        mark_training_failed(str(exc))
        flash(
            f"Training failed: {exc}. See logs/training_error.log for full traceback.",
            "danger",
        )
        current_app.logger.exception("Training failed: %s", exc)
    return redirect(url_for("model_management.model_management_page"))


@model_management_bp.route("/train-detector", methods=["POST"])
@admin_required
def train_khat_detector_route():
    try:
        from services.khat_detector_service import (
            bootstrap_non_khat_dataset,
            evaluate_non_khat_detection,
            sync_khat_detector_dataset,
            train_khat_detector,
        )

        bootstrap_non_khat_dataset()
        sync_khat_detector_dataset()
        epochs = int(request.form.get("detector_epochs", 8))
        result = train_khat_detector(epochs=epochs)
        eval_report = evaluate_non_khat_detection()
        flash(
            f"Khat detector trained. Val accuracy: {(result.get('best_val_accuracy') or 0) * 100:.1f}%. "
            f"Khat acceptance: {(eval_report.get('khat_acceptance_accuracy') or 0) * 100:.1f}%. "
            f"Non-Khat rejection: {(eval_report.get('non_khat_rejection_accuracy') or 0) * 100:.1f}%.",
            "success",
        )
    except Exception as exc:
        flash(f"Khat detector training failed: {exc}", "danger")
    return redirect(url_for("model_management.model_management_page"))


@model_management_bp.route("/detector-settings", methods=["POST"])
@admin_required
def update_detector_settings():
    try:
        from services.khat_detector_service import save_thresholds

        accept = float(request.form.get("accept_threshold", 0.70))
        reject = float(request.form.get("reject_threshold", 0.50))
        borderline = request.form.get("borderline_threshold")
        borderline_val = float(borderline) if borderline not in (None, "") else None
        save_thresholds(accept, reject, borderline=borderline_val)
        bl_label = f"{borderline_val:.2f}" if borderline_val is not None else "default"
        flash(
            f"Detector thresholds updated: accept={accept:.2f}, borderline={bl_label}, reject={reject:.2f}",
            "success",
        )
    except Exception as exc:
        flash(f"Could not update thresholds: {exc}", "danger")
    return redirect(url_for("model_management.model_management_page"))


@model_management_bp.route("/sync-detector-dataset", methods=["POST"])
@admin_required
def sync_detector_dataset_route():
    try:
        from services.khat_detector_service import sync_khat_detector_dataset

        result = sync_khat_detector_dataset()
        msg = f"Detector dataset synced: {result.get('khat_total', 0)} Khat, {result.get('non_khat_total', 0)} Non-Khat."
        if result.get("warning"):
            flash(f"{msg} Warning: {result['warning']}", "warning")
        else:
            flash(msg, "success")
    except Exception as exc:
        flash(f"Detector dataset sync failed: {exc}", "danger")
    return redirect(url_for("model_management.model_management_page"))


@model_management_bp.route("/upload-tm-model", methods=["POST"])
@admin_required
def upload_tm_model():
    try:
        result = process_tm_upload(
            model_json=request.files.get("model_json"),
            metadata_json=request.files.get("metadata_json"),
            weights_bin=request.files.get("weights_bin"),
            zip_file=request.files.get("tm_model_zip"),
        )
        version = result["version"]
        flash(
            f"Model Teachable Machine berhasil diunggah dan diaktifkan. "
            f"Nama: {result['model_name']} · Versi: {version.model_version}.",
            "success",
        )
    except TmModelValidationError as exc:
        flash(str(exc), "warning")
    except Exception as exc:
        current_app.logger.exception("TM model upload failed: %s", exc)
        flash(f"Upload model gagal: {exc}", "danger")
    return redirect(url_for("model_management.model_management_page") + "#upload-model")


@model_management_bp.route("/model-versions/<int:version_id>/activate", methods=["POST"])
@admin_required
def activate_model_version_route(version_id: int):
    try:
        row = activate_model_version(version_id)
        flash(f"Model {row.model_name} ({row.model_version}) berhasil diaktifkan.", "success")
    except ValueError as exc:
        flash(str(exc), "warning")
    except Exception as exc:
        flash(f"Aktivasi model gagal: {exc}", "danger")
    return redirect(url_for("model_management.model_management_page") + "#model-versions")


@model_management_bp.route("/model-versions/<int:version_id>/archive", methods=["POST"])
@admin_required
def archive_model_version_route(version_id: int):
    try:
        row = archive_model_version(version_id)
        flash(f"Versi {row.model_version} diarsipkan.", "success")
    except ValueError as exc:
        flash(str(exc), "warning")
    except Exception as exc:
        flash(f"Arsip model gagal: {exc}", "danger")
    return redirect(url_for("model_management.model_management_page") + "#model-versions")


@model_management_bp.route("/model-versions/<int:version_id>/delete", methods=["POST"])
@admin_required
def delete_model_version_route(version_id: int):
    try:
        delete_model_version(version_id)
        flash("Versi model berhasil dihapus.", "success")
    except ValueError as exc:
        flash(str(exc), "warning")
    except Exception as exc:
        flash(f"Hapus model gagal: {exc}", "danger")
    return redirect(url_for("model_management.model_management_page") + "#model-versions")


@legacy_training_bp.route("/", defaults={"path": ""})
@legacy_training_bp.route("/<path:path>")
def legacy_training_redirect(path: str):
    """Backward-compatible redirect from /training/ to /model-management/."""
    suffix = path.strip("/")
    target = "/model-management/" + suffix if suffix else "/model-management/"
    if request.query_string:
        target = f"{target}?{request.query_string.decode()}"
    return redirect(target, code=301)
