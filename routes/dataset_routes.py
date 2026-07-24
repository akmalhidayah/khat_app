import os
import shutil
import uuid
from typing import Optional

from flask import Blueprint, abort, current_app, flash, jsonify, redirect, render_template, request, send_file, session, url_for
from sqlalchemy import text

from models import Dataset, db
from services.db_compat import count_dataset_by_class, filter_dataset_by_class
from routes.utils import admin_required, login_required
from services.dataset_path_service import (
    count_dataset_images_on_disk,
    count_source_class_images,
    get_dataset_inventory,
    enrich_dataset_item,
    infer_image_source,
    resolve_dataset_image_path,
    resolve_dataset_serving_path,
)
from services.image_optimization_service import (
    is_optimization_running,
    load_optimization_report,
    load_optimization_status,
    start_optimization_background,
)
from services.dataset_zip_import_service import ZipImportSummary
from services.dataset_service import (
    DuplicateImageError,
    allowed_file,
    import_dataset_zip,
    save_dataset_file,
    save_dataset_files,
    split_and_copy_dataset,
)
from services.dataset_upload_settings_service import apply_upload_settings_to_config, load_upload_settings, save_upload_settings
from services.dataset_cleaning_service import load_duplicate_report, run_duplicate_audit
from services.dataset_quality_service import load_quality_report
from services.label_audit_service import load_label_audit_report, run_label_audit
from services.preprocessing_service import load_split_report
from services.prediction_audit_service import (
    load_confusion_pair_report,
    load_misclassification_report,
    load_prediction_audit_report,
    audit_predictions_on_dataset,
)
from services.dataset_qc_dashboard_service import build_dataset_qc_report
from services.dataset_page_ui_service import build_dataset_page_insights, build_prep_status_insights
from services.dataset_zip_chunk_service import get_chunk_upload_store
from services.label_mismatch_service import generate_label_mismatch_report, load_label_mismatch_report

dataset_bp = Blueprint("dataset", __name__, url_prefix="/dataset")


def _upload_max_bytes() -> int:
    return int(current_app.config.get("MAX_CONTENT_LENGTH", 1024 * 1024 * 1024))


def _wants_json_response() -> bool:
    return request.headers.get("X-Requested-With") == "XMLHttpRequest"


def _upload_too_large_response():
    max_mb = current_app.config.get("UPLOAD_MAX_SIZE_MB", 1024)
    message = (
        f"Ukuran file terlalu besar. Maksimal upload {max_mb} MB. "
        "Silakan kompres gambar atau unggah dataset dalam beberapa bagian."
    )
    if _wants_json_response():
        return jsonify({"success": False, "message": message}), 413
    flash(message, "danger")
    return redirect(url_for("dataset.upload_dataset"))


def _flash_zip_import_summary(summary: ZipImportSummary) -> None:
    class_parts = [
        f"{label.replace('_', ' ').title()}: {count}"
        for label, count in sorted(summary.per_class.items())
        if count > 0
    ]
    class_text = " · ".join(class_parts) if class_parts else "—"
    saved_mb = round(summary.saved_bytes / (1024 * 1024), 2)

    if summary.imported > 0:
        flash(
            f"Import ZIP selesai. {summary.imported} gambar diimpor, "
            f"{summary.optimized} dioptimasi, {summary.model_ready} model-ready.",
            "success",
        )
    else:
        flash("Import ZIP selesai. Tidak ada gambar baru yang ditambahkan.", "warning")

    detail = (
        f"Ringkasan: ditemukan {summary.total_found}, diimpor {summary.imported}, "
        f"dioptimasi {summary.optimized}, dilewati "
        f"{summary.skipped_unsupported + summary.skipped_ignored + summary.skipped_duplicates + summary.skipped_corrupted}, "
        f"rusak {summary.skipped_corrupted}, duplikat {summary.skipped_duplicates}. "
        f"Ukuran awal {round(summary.original_total_bytes / (1024 * 1024), 2)} MB → "
        f"optimized {round(summary.optimized_total_bytes / (1024 * 1024), 2)} MB "
        f"(hemat {saved_mb} MB, {summary.compression_ratio}%). Distribusi: {class_text}."
    )
    flash(detail, "info")
    for note in summary.processing_notes[:4]:
        flash(note, "info")


def _store_zip_import_flash(summary: ZipImportSummary) -> None:
    session["pending_zip_import"] = summary.to_dict()


def _flush_pending_zip_import_flash() -> None:
    data = session.pop("pending_zip_import", None)
    if not data:
        return
    summary = ZipImportSummary()
    for key, value in data.items():
        if hasattr(summary, key):
            setattr(summary, key, value)
    _flash_zip_import_summary(summary)


def _zip_import_json_response(summary: ZipImportSummary) -> dict:
    payload = summary.to_dict()
    payload.update({
        "success": summary.imported > 0 or summary.total_found == 0,
        "redirect_url": url_for("dataset.list_dataset"),
        "status": "completed",
    })
    return payload


def _count_images_in_dir(root_dir: str) -> int:
    if not os.path.exists(root_dir):
        return 0
    total = 0
    for _, _, files in os.walk(root_dir):
        total += len(
            [
                f
                for f in files
                if "." in f and f.rsplit(".", 1)[1].lower() in current_app.config["ALLOWED_EXTENSIONS"]
            ]
        )
    return total


def _count_images_in_class_dir(root_dir: str, class_name: str) -> int:
    from services.dataset_import_service import normalize_class_name

    allowed = current_app.config["ALLOWED_EXTENSIONS"]
    total = 0
    if not os.path.isdir(root_dir):
        return 0
    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        label = normalize_class_name(entry) or entry
        if label != class_name:
            continue
        total += len(
            [
                f
                for f in os.listdir(entry_path)
                if os.path.isfile(os.path.join(entry_path, f))
                and "." in f
                and f.rsplit(".", 1)[1].lower() in allowed
            ]
        )
    return total


def _load_upload_report() -> dict:
    path = current_app.config.get("DATASET_UPLOAD_REPORT_PATH")
    if not path or not os.path.isfile(path):
        return {}
    import json
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


def _build_prep_status(inventory: Optional[dict] = None) -> dict:
    inventory = inventory or get_dataset_inventory(current_app.config)
    processed_dir = current_app.config.get("PROCESSED_DATASET_DIR")
    model_ready_dir = current_app.config.get("MODEL_READY_DIR")
    raw_total = inventory["raw_total"]
    prep_status = {
        "raw": raw_total,
        "processed_images": _count_images_in_dir(processed_dir) if processed_dir else 0,
        "optimized_images": _count_images_in_dir(processed_dir) if processed_dir else 0,
        "model_ready_images": _count_images_in_dir(model_ready_dir) if model_ready_dir else 0,
        "train": inventory["train"],
        "validation": inventory["validation"],
        "test": inventory["test"],
    }
    prep_status["processed"] = inventory["processed_total"]
    prep_status["progress"] = (
        round((prep_status["processed"] / prep_status["raw"]) * 100, 1) if prep_status["raw"] > 0 else 0
    )
    raw = prep_status["raw"]
    est_test = int(raw * 0.20)
    est_train_val = raw - est_test
    est_val = int(est_train_val * 0.15)
    est_train = max(est_train_val - est_val, 0)
    prep_status["estimated_split"] = {"train": est_train, "validation": est_val, "test": est_test}
    prep_status["unprocessed"] = max(prep_status["raw"] - prep_status["processed"], 0)
    prep_status["raw_unprocessed"] = prep_status["unprocessed"]
    if prep_status["processed"] > 0:
        prep_status["split_ratio"] = {
            "train": round((prep_status["train"] / prep_status["processed"]) * 100, 1),
            "validation": round((prep_status["validation"] / prep_status["processed"]) * 100, 1),
            "test": round((prep_status["test"] / prep_status["processed"]) * 100, 1),
        }
    else:
        prep_status["split_ratio"] = {}
    return prep_status


def _build_dataset_status(prep_status: dict, orphaned_count: int, optimization_summary: dict) -> dict:
    if orphaned_count > 0:
        return {
            "state": "Missing Files Detected",
            "state_id": "File Hilang Terdeteksi",
            "badge": "error",
        }
    if prep_status.get("processed", 0) > 0:
        return {
            "state": "Ready for Training",
            "state_id": "Siap untuk Training",
            "badge": "ready",
        }
    if not optimization_summary.get("is_complete") and prep_status.get("raw", 0) > 0:
        return {
            "state": "Needs Optimization",
            "state_id": "Perlu Optimasi",
            "badge": "warning",
        }
    if prep_status.get("raw", 0) > 0:
        return {
            "state": "Not Processed",
            "state_id": "Belum Diproses",
            "badge": "pending",
        }
    return {
        "state": "Waiting for Upload",
        "state_id": "Menunggu Unggahan",
        "badge": "empty",
    }


def _build_optimization_summary(
    total_images: int,
    optimization_report: dict,
    optimization_status: dict,
    optimized_dir_count: int,
) -> dict:
    report = optimization_report or {}
    status = optimization_status or {}
    total_scanned = report.get("total_scanned") or total_images or 0
    optimized = report.get("optimized")
    if optimized is None:
        optimized = optimized_dir_count
    skipped = report.get("skipped", 0)
    corrupted = report.get("corrupted", 0)
    missing = max(total_scanned - optimized - skipped - corrupted, 0) if total_scanned else 0

    is_running = status.get("status") == "running"
    is_complete = (
        total_scanned > 0
        and optimized >= total_scanned
        and not is_running
    ) or (total_scanned > 0 and optimized >= total_scanned * 0.95 and not is_running)

    return {
        "total_scanned": total_scanned,
        "optimized": optimized,
        "skipped": skipped,
        "missing": missing,
        "corrupted": corrupted,
        "original_size_mb": report.get("original_size_mb"),
        "optimized_size_mb": report.get("optimized_size_mb"),
        "saved_mb": report.get("saved_mb"),
        "compression_percent": report.get("compression_percent"),
        "last_run": report.get("processing_date") or status.get("finished_at"),
        "is_complete": is_complete,
        "is_running": is_running,
        "status_label": "Optimized" if is_complete else "Not fully optimized",
        "status_label_id": "Sudah Dioptimasi" if is_complete else "Belum Lengkap",
        "needs_full_run": optimized <= 1 and total_images > 1,
        "progress": status.get("progress", 100 if is_complete else 0),
    }


def _build_workflow(total_images: int, prep_status: dict) -> dict:
    imported = total_images > 0
    validated = total_images > 0
    split_done = prep_status["processed"] > 0
    return {
        "imported": imported,
        "validated": validated,
        "split": split_done,
        "training_ready": split_done,
        "imported_label": "Completed" if imported else "Pending",
        "validated_label": "Completed" if validated else "Pending",
        "split_label": "Completed" if split_done else "Pending",
        "training_ready_label": "Yes" if split_done else "No",
    }


def _build_balance(counts: dict, total_images: int) -> list:
    balance = []
    for label in current_app.config["CLASS_LABELS"]:
        count = counts.get(label, 0)
        pct = round((count / total_images) * 100, 1) if total_images else 0
        balance.append({"label": label, "count": count, "percent": pct})
    return balance


def _flash_upload_result(result, redirect_to_list: bool = True):
    if result.added > 0 and result.duplicates == 0 and result.invalid == 0:
        if result.added == 1:
            flash("Dataset image uploaded successfully.", "success")
        else:
            flash(f"Dataset upload completed: {result.added} new images added.", "success")
    elif result.added > 0:
        flash(
            f"Dataset upload completed: {result.added} new images added, "
            f"{result.duplicates} duplicate images skipped, {result.invalid} invalid files ignored.",
            "success" if result.added > result.duplicates else "warning",
        )
    elif result.duplicates > 0 and result.invalid == 0:
        if result.duplicates == 1:
            flash(
                "Duplicate image detected. The selected image already exists in the dataset and was not added again.",
                "warning",
            )
        else:
            flash(
                f"Upload finished: {result.duplicates} duplicate images skipped. No new images were added.",
                "warning",
            )
    elif result.invalid > 0 and result.duplicates == 0:
        flash("No images were uploaded. Please check file format (JPG, JPEG, PNG, WEBP).", "warning")
    else:
        flash(
            f"Upload finished: {result.duplicates} duplicates skipped, {result.invalid} invalid files ignored.",
            "warning",
        )


@dataset_bp.route("/")
@login_required
def list_dataset():
    _flush_pending_zip_import_flash()
    selected_class = request.args.get("class_name", "").strip()
    search_query = request.args.get("search", "").strip()
    format_filter = request.args.get("format", "").strip().upper()
    source_filter = request.args.get("source", "").strip().lower()
    page = max(request.args.get("page", 1, type=int), 1)
    per_page = request.args.get("per_page", 25, type=int)
    if per_page not in (10, 25, 50, 100):
        per_page = 25

    query = Dataset.query.order_by(Dataset.id.asc())
    if selected_class:
        query = filter_dataset_by_class(query, selected_class)
    if search_query:
        like = f"%{search_query}%"
        query = query.filter(
            db.or_(
                Dataset.nama_file_asli.ilike(like),
                Dataset.nama_file.ilike(like),
            )
        )
    if format_filter:
        query = query.filter(Dataset.format_file.ilike(format_filter))
    if source_filter == "uploaded":
        query = query.filter(text("nama_file REGEXP '^[a-f0-9]{32}\\.'"))
    elif source_filter == "imported":
        query = query.filter(text("nama_file NOT REGEXP '^[a-f0-9]{32}\\.'"))

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    placeholder_url = url_for("static", filename="images/image-placeholder.svg")
    enriched_items = []
    for row in pagination.items:
        entry = enrich_dataset_item(row)
        entry["image_url"] = (
            url_for("dataset.dataset_image", dataset_id=entry["id"])
            if entry["file_exists"]
            else placeholder_url
        )
        enriched_items.append(entry)

    orphaned_count = sum(1 for row in Dataset.query.all() if enrich_dataset_item(row)["is_orphan"])

    inventory = get_dataset_inventory(current_app.config)
    counts = inventory["per_class"]
    if not any(counts.values()):
        counts = {
            name: max(
                count_dataset_by_class(name),
                _count_images_in_class_dir(current_app.config["RAW_DATASET_DIR"], name),
            )
            for name in current_app.config["CLASS_LABELS"]
        }
        inventory = {
            **inventory,
            "per_class": counts,
            "raw_total": sum(counts.values()),
        }
    class_total = inventory["raw_total"]
    total_images = class_total
    total_classes = inventory["active_classes"]
    processed_images = inventory["processed_total"]

    prep_status = _build_prep_status(inventory)
    optimization_status = load_optimization_status(current_app.config)
    optimization_report = load_optimization_report(current_app.config)
    upload_report = _load_upload_report()
    prep_status["storage_saved_mb"] = upload_report.get("saved_mb", 0)
    prep_status["compression_ratio"] = upload_report.get("compression_ratio", 0)
    prep_status["last_optimization_date"] = (
        upload_report.get("processing_date")
        or optimization_report.get("processing_date")
        or optimization_status.get("finished_at")
        or "—"
    )
    prep_status["compression_status"] = (
        "Optimized" if prep_status.get("optimized_images", 0) > 0 else "Belum Dioptimasi"
    )
    workflow = _build_workflow(total_images, prep_status)
    balance = _build_balance(counts, class_total or total_images)
    class_percentages = {row["label"]: row["percent"] for row in balance}

    dominant_class = max(balance, key=lambda x: x["count"])["label"] if balance and total_images else ""

    optimization_running = is_optimization_running(current_app.config)
    optimized_image_count = _count_images_in_dir(current_app.config["OPTIMIZED_DISPLAY_DIR"])
    optimization_summary = _build_optimization_summary(
        total_images,
        optimization_report,
        optimization_status,
        optimized_image_count,
    )
    dataset_status = _build_dataset_status(prep_status, orphaned_count, optimization_summary)

    label_audit = load_label_audit_report(current_app.config)
    duplicate_report = load_duplicate_report(current_app.config)
    quality_report = load_quality_report(current_app.config)
    split_report = load_split_report(current_app.config.get("DATASET_SPLIT_REPORT_PATH", ""))
    prediction_audit = load_prediction_audit_report(current_app.config)
    misclassification_report = load_misclassification_report(current_app.config)
    confusion_pairs = load_confusion_pair_report(current_app.config)
    label_mismatch = load_label_mismatch_report(current_app.config)

    missing_rows = [e for e in enriched_items if not e.get("file_exists")]
    qc_report = build_dataset_qc_report(db_rows=missing_rows)
    if label_mismatch:
        for item in label_mismatch.get("mismatches", [])[:10]:
            qc_report["mislabeled_warnings"].append(
                item.get("message")
                or f"Possible mislabel: {item.get('filename')}"
            )

    retrain_workflow = {
        "steps": [
            {"key": "audit", "label_id": "Audit Dataset", "icon": "clipboard2-check", "done": bool(label_audit or qc_report)},
            {"key": "clean", "label_id": "Bersihkan Dataset", "icon": "brush", "done": bool(quality_report)},
            {"key": "reprocess", "label_id": "Reprocess Gambar", "icon": "arrow-repeat", "done": optimization_summary["optimized"] > 0},
            {"key": "split", "label_id": "Split Dataset", "icon": "diagram-3", "done": prep_status["processed"] > 0},
            {"key": "train", "label_id": "Retrain Model", "icon": "cpu", "done": False},
            {"key": "evaluate", "label_id": "Evaluasi", "icon": "graph-up", "done": bool(prediction_audit or misclassification_report)},
            {"key": "compare", "label_id": "Bandingkan Hasil", "icon": "bar-chart-line", "done": bool(misclassification_report)},
        ]
    }
    prep_insights = build_prep_status_insights(
        prep_status=prep_status,
        quality_report=quality_report,
        optimization_summary=optimization_summary,
        duplicate_report=duplicate_report,
        split_report=split_report,
    )
    prep_status.update(prep_insights)
    if orphaned_count > 0:
        prep_status["state"] = "Missing Files Detected"
        prep_status["state_id"] = "File Hilang Terdeteksi"
        prep_status["badge"] = "error"
        prep_status["status_message"] = "Beberapa entri dataset tidak memiliki file gambar di disk."
        prep_status["warning_tone"] = "warn"

    page_insights = build_dataset_page_insights(
        balance=balance,
        total_images=total_images,
        qc_report=qc_report,
        dataset_status=dataset_status,
        prep_status=prep_status,
    )

    return render_template(
        "dataset.html",
        items=enriched_items,
        pagination=pagination,
        counts=counts,
        class_percentages=class_percentages,
        total_images=total_images,
        dataset_inventory=inventory,
        prep_status=prep_status,
        workflow=workflow,
        balance=balance,
        dominant_class=dominant_class,
        selected_class=selected_class,
        search_query=search_query,
        format_filter=format_filter,
        source_filter=source_filter,
        per_page=per_page,
        class_labels=current_app.config["CLASS_LABELS"],
        placeholder_url=placeholder_url,
        debug_mode=current_app.debug,
        orphaned_count=orphaned_count,
        optimization_status=optimization_status,
        optimization_report=optimization_report,
        optimization_running=optimization_running,
        optimized_image_count=optimized_image_count,
        optimization_summary=optimization_summary,
        dataset_status=dataset_status,
        total_classes=total_classes,
        processed_images=processed_images,
        label_audit=label_audit,
        duplicate_report=duplicate_report,
        quality_report=quality_report,
        split_report=split_report,
        prediction_audit=prediction_audit,
        misclassification_report=misclassification_report,
        confusion_pairs=confusion_pairs,
        retrain_workflow=retrain_workflow,
        qc_report=qc_report,
        label_mismatch=label_mismatch,
        upload_report=upload_report,
        page_insights=page_insights,
    )


def _send_dataset_file(abs_path: str, as_attachment: bool = False):
    import mimetypes

    mime, _ = mimetypes.guess_type(abs_path)
    return send_file(abs_path, mimetype=mime or "application/octet-stream", as_attachment=as_attachment)


@dataset_bp.route("/image/<int:dataset_id>")
@login_required
def dataset_image(dataset_id: int):
    item = Dataset.query.get_or_404(dataset_id)
    abs_path, _source = resolve_dataset_serving_path(item)
    if not abs_path or not os.path.isfile(abs_path):
        placeholder = os.path.join(current_app.static_folder, "images", "image-placeholder.svg")
        if os.path.isfile(placeholder):
            return send_file(placeholder, mimetype="image/svg+xml")
        abort(404)
    return _send_dataset_file(abs_path)


@dataset_bp.route("/image")
@login_required
def serve_dataset_image():
    """Legacy query-path route; prefer /dataset/image/<id>."""
    dataset_id = request.args.get("id", type=int)
    if dataset_id:
        return dataset_image(dataset_id)

    rel_path = request.args.get("path", "")
    if not rel_path or ".." in rel_path.replace("\\", "/"):
        abort(404)

    base_dir = os.path.normpath(current_app.config["BASE_DIR"])
    abs_path = rel_path if os.path.isabs(rel_path) else os.path.normpath(os.path.join(base_dir, rel_path))
    if not abs_path.startswith(base_dir) or not os.path.isfile(abs_path):
        abort(404)
    return _send_dataset_file(abs_path)


@dataset_bp.route("/cleanup-orphans", methods=["POST"])
@admin_required
def cleanup_orphan_records():
    removed = 0
    for item in Dataset.query.all():
        if enrich_dataset_item(item)["is_orphan"]:
            db.session.delete(item)
            removed += 1
    db.session.commit()
    if removed:
        flash(f"Removed {removed} orphaned dataset record(s) with missing image files.", "success")
    else:
        flash("No orphaned dataset records found.", "info")
    return redirect(url_for("dataset.list_dataset"))


@dataset_bp.route("/upload", methods=["GET", "POST"])
@admin_required
def upload_dataset():
    if request.method == "POST":
        class_name = request.form.get("class_name", "").strip().lower().replace(" ", "_")
        files = request.files.getlist("images") or request.files.getlist("image")
        files = [f for f in files if f and f.filename]

        if class_name not in current_app.config["CLASS_LABELS"]:
            flash("Invalid class label.", "danger")
            return redirect(url_for("dataset.upload_dataset"))
        if not files:
            flash("Please select at least one image.", "danger")
            return redirect(url_for("dataset.upload_dataset"))

        try:
            if len(files) == 1:
                try:
                    save_dataset_file(files[0], class_name)
                    flash("Dataset image uploaded successfully.", "success")
                except DuplicateImageError:
                    flash(
                        "Duplicate image detected. The selected image already exists in the dataset and was not added again.",
                        "warning",
                    )
            else:
                result = save_dataset_files(files, class_name)
                _flash_upload_result(result)
        except ValueError as exc:
            flash(str(exc), "danger")
        except Exception as exc:
            flash(f"Upload failed: {exc}", "danger")
        return redirect(url_for("dataset.list_dataset"))
    return render_template(
        "dataset_upload.html",
        class_labels=current_app.config["CLASS_LABELS"],
        upload_max_size_mb=current_app.config.get("UPLOAD_MAX_SIZE_MB", 1024),
        upload_max_size_bytes=_upload_max_bytes(),
        upload_chunk_size_mb=current_app.config.get("UPLOAD_CHUNK_SIZE_MB", 20),
        upload_settings=load_upload_settings(),
    )


@dataset_bp.route("/upload-settings", methods=["POST"])
@admin_required
def save_dataset_upload_settings():
    payload = {
        "max_upload_size_mb": request.form.get("max_upload_size_mb", type=int),
        "max_image_dimension": request.form.get("max_image_dimension", type=int),
        "jpeg_quality": request.form.get("jpeg_quality", type=int),
        "output_format": request.form.get("output_format", "JPEG"),
        "keep_raw_files": request.form.get("keep_raw_files") == "on",
        "auto_optimize_on_upload": request.form.get("auto_optimize_on_upload") == "on",
        "generate_model_ready_images": request.form.get("generate_model_ready_images") == "on",
    }
    cleaned = {k: v for k, v in payload.items() if v is not None}
    save_upload_settings(cleaned)
    apply_upload_settings_to_config(current_app.config)
    flash("Pengaturan upload dataset berhasil disimpan.", "success")
    return redirect(url_for("dataset.upload_dataset"))


@dataset_bp.route("/upload-zip", methods=["POST"])
@admin_required
def upload_dataset_zip():
    max_bytes = _upload_max_bytes()
    content_length = request.content_length or 0
    if content_length > max_bytes:
        return _upload_too_large_response()

    zip_file = request.files.get("dataset_zip")
    if not zip_file or not zip_file.filename.lower().endswith(".zip"):
        message = "Silakan unggah file dataset .zip yang valid."
        if _wants_json_response():
            return jsonify({"success": False, "message": message}), 400
        flash(message, "danger")
        return redirect(url_for("dataset.upload_dataset"))

    zip_file.seek(0, os.SEEK_END)
    file_size = zip_file.tell()
    zip_file.seek(0)
    if file_size > max_bytes:
        return _upload_too_large_response()

    temp_dir = current_app.config.get("DATASET_TMP_DIR") or os.path.join(
        current_app.config["DATASET_DIR"], "tmp"
    )
    os.makedirs(temp_dir, exist_ok=True)
    temp_zip = os.path.join(temp_dir, f"dataset_upload_{uuid.uuid4().hex}.zip")
    try:
        zip_file.save(temp_zip)
        summary = import_dataset_zip(temp_zip)
        if _wants_json_response():
            _store_zip_import_flash(summary)
            return jsonify(_zip_import_json_response(summary))
        _flash_zip_import_summary(summary)
    except ValueError as exc:
        message = str(exc)
        if _wants_json_response():
            return jsonify({"success": False, "message": message}), 400
        flash(message, "danger")
    except Exception as exc:
        message = f"Import ZIP gagal: {exc}"
        if _wants_json_response():
            return jsonify({"success": False, "message": message}), 500
        flash(message, "danger")
    finally:
        if os.path.exists(temp_zip):
            try:
                os.remove(temp_zip)
            except OSError:
                pass
    return redirect(url_for("dataset.list_dataset"))


@dataset_bp.route("/upload-zip/chunk", methods=["POST"])
@admin_required
def upload_dataset_zip_chunk():
    """Optional chunked upload endpoint for very large ZIP archives."""
    session_id = request.form.get("session_id", "").strip()
    chunk_index = request.form.get("chunk_index", type=int)
    total_chunks = request.form.get("total_chunks", type=int)
    chunk_file = request.files.get("chunk")

    if chunk_index is None or total_chunks is None or not chunk_file:
        return jsonify({"success": False, "message": "Data chunk tidak lengkap."}), 400

    store = get_chunk_upload_store()
    try:
        if not session_id:
            session_id = store.start_session(request.form.get("filename", "dataset.zip"), total_chunks)
        received = store.save_chunk(session_id, chunk_index, chunk_file.read())
        return jsonify(
            {
                "success": True,
                "session_id": session_id,
                "received_chunks": received,
                "total_chunks": total_chunks,
            }
        )
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400


@dataset_bp.route("/upload-zip/finalize", methods=["POST"])
@admin_required
def finalize_dataset_zip_chunk():
    session_id = request.form.get("session_id", "").strip()
    if not session_id:
        return jsonify({"success": False, "message": "Session upload tidak ditemukan."}), 400

    store = get_chunk_upload_store()
    try:
        temp_zip = store.assemble(session_id)
        summary = import_dataset_zip(temp_zip)
        _store_zip_import_flash(summary)
        return jsonify(_zip_import_json_response(summary))
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "message": f"Import ZIP gagal: {exc}"}), 500


@dataset_bp.route("/delete/<int:item_id>", methods=["POST"])
@admin_required
def delete_dataset(item_id):
    item = Dataset.query.get_or_404(item_id)
    abs_path = resolve_dataset_image_path(item)
    if abs_path and os.path.exists(abs_path):
        try:
            os.remove(abs_path)
        except OSError:
            pass
    db.session.delete(item)
    db.session.commit()
    flash("Dataset image deleted successfully.", "success")
    return redirect(request.referrer or url_for("dataset.list_dataset"))


@dataset_bp.route("/split", methods=["POST"])
@admin_required
def split_dataset():
    try:
        result = split_and_copy_dataset()
        flash(
            "Dataset processed successfully. Training, validation, and testing folders are ready. "
            f"Train: {result['train']}, Validation: {result['validation']}, Test: {result['test']}.",
            "success",
        )
    except ValueError as exc:
        flash(str(exc), "warning")
    except Exception:
        flash(
            "Dataset processing failed. Please check folder permissions and dataset structure.",
            "danger",
        )
    return redirect(url_for("dataset.list_dataset"))


@dataset_bp.route("/clear-split", methods=["POST"])
@admin_required
def clear_split():
    try:
        for dir_key in ("TRAIN_DIR", "VALIDATION_DIR", "TEST_DIR"):
            root = current_app.config[dir_key]
            if os.path.exists(root):
                shutil.rmtree(root)
            os.makedirs(root, exist_ok=True)
        flash("Processed split folders cleared successfully.", "success")
    except Exception:
        flash("Failed to clear processed split folders. Please check folder permissions.", "danger")
    return redirect(url_for("dataset.list_dataset"))


@dataset_bp.route("/audit/labels", methods=["POST"])
@admin_required
def run_label_audit_route():
    try:
        report = run_label_audit()
        flash(
            f"Label audit complete. {report.get('mismatch_count', 0)} possible mismatch(es) found.",
            "warning" if report.get("mismatch_count") else "success",
        )
    except Exception as exc:
        flash(f"Label audit failed: {exc}", "danger")
    return redirect(url_for("dataset.list_dataset"))


@dataset_bp.route("/audit/label-mismatch", methods=["POST"])
@admin_required
def run_label_mismatch_route():
    try:
        report = generate_label_mismatch_report()
        flash(
            f"Label mismatch scan complete. {report.get('mismatch_count', 0)} possible mislabel(s) found.",
            "warning" if report.get("mismatch_count") else "success",
        )
    except Exception as exc:
        flash(f"Label mismatch scan failed: {exc}", "danger")
    return redirect(url_for("dataset.list_dataset"))


@dataset_bp.route("/audit/duplicates", methods=["POST"])
@admin_required
def run_duplicate_audit_route():
    try:
        report = run_duplicate_audit()
        leakage = report.get("cross_split_leakage_detected")
        flash(
            f"Duplicate audit complete. Exact duplicates: {report.get('exact_duplicate_count', 0)}. "
            f"Near duplicates: {report.get('near_duplicate_count', 0)}.",
            "warning" if leakage else "success",
        )
        if leakage:
            flash("Cross-split duplicate leakage detected. Rebuild the dataset split.", "danger")
    except Exception as exc:
        flash(f"Duplicate audit failed: {exc}", "danger")
    return redirect(url_for("dataset.list_dataset"))


@dataset_bp.route("/audit/predictions", methods=["POST"])
@admin_required
def run_prediction_audit_route():
    try:
        split = request.form.get("split", "test")
        use_tta = request.form.get("use_tta") in ("1", "on", "true")
        report = audit_predictions_on_dataset(split=split, use_tta=use_tta)
        acc = report.get("accuracy", 0) * 100
        flash(
            f"Prediction audit on {split}: accuracy {acc:.1f}%, "
            f"{report.get('wrong_predictions', 0)} wrong prediction(s).",
            "success" if acc >= 85 else "warning",
        )
    except Exception as exc:
        flash(f"Prediction audit failed: {exc}", "danger")
    return redirect(url_for("dataset.list_dataset"))


@dataset_bp.route("/optimize", methods=["POST"])
@admin_required
def optimize_dataset_images_route():
    if is_optimization_running(current_app.config):
        flash("Image optimization is already running. Please wait for it to finish.", "warning")
        return redirect(url_for("dataset.list_dataset"))

    started = start_optimization_background(current_app)
    if started:
        flash("Image optimization started. Original images will not be modified.", "info")
    else:
        flash("Could not start image optimization. Please try again.", "danger")
    return redirect(url_for("dataset.list_dataset"))


@dataset_bp.route("/optimization/status")
@login_required
def optimization_status():
    status = load_optimization_status(current_app.config)
    report = load_optimization_report(current_app.config)
    return jsonify({"status": status, "report": report})


@dataset_bp.route("/optimization/report")
@login_required
def optimization_report():
    report = load_optimization_report(current_app.config)
    if not report:
        return jsonify({"error": "No optimization report found."}), 404
    return jsonify(report)
