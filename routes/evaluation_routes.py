import json
import os
from pathlib import Path

from flask import Blueprint, abort, current_app, flash, jsonify, redirect, render_template, request, send_file, url_for

from models import EvaluasiModel, db
from models.evaluasi_model import TIPE_RINGKASAN, buat_ringkasan_evaluasi
from routes.utils import admin_required, login_required
from services.evaluation_service import evaluate_model, generate_evaluation_charts
from services.evaluation_ui_service import (
    analyze_misclassification_summary,
    build_accuracy_formula_display,
    build_active_model_eval_display,
    build_confusion_insight,
    build_confusion_interpretation,
    backfill_class_accuracy,
    build_dataset_eval_display,
    build_error_dashboard_cards,
    build_macro_metric_cards,
    build_performance_insights,
    build_eval_summary,
    build_per_class_rows,
    enrich_misclassified_row,
    interpret_training_history,
    merge_recommendations,
    metric_card_hint,
    metric_description,
    metric_tone,
    weak_class_summary,
    display_class,
    resolve_eval_class_labels,
)
from services.evaluation_correction_service import (
    load_evaluation_corrections,
    mark_as_reviewed,
    merge_correction_into_row,
    move_image_to_class,
    save_label_correction,
)
from services.evaluation_dataset_service import resolve_evaluation_counts
from services.evaluation_image_service import (
    persist_rebuilt_previews,
    rebuild_evaluation_previews,
    serve_evaluation_image_path,
)
from services.eval_dependency_service import check_eval_dependencies, check_keras_eval_dependencies
from services.external_assets_service import should_use_keras_model
from services.metrics_service import ACCURACY_TARGET, target_status
from services.model_version_service import (
    list_model_versions,
    model_version_to_dict,
    register_active_tm_model,
    register_model_version,
)
from services.teachable_machine_service import is_teachable_machine_available, get_tm_display_info
from services.prediction_audit_service import load_misclassification_report, load_confusion_pair_report

evaluation_bp = Blueprint("evaluation", __name__, url_prefix="/evaluation")


def _load_eval_json():
    path = current_app.config["EVALUATION_RESULT_PATH"]
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _file_exists(path: str) -> bool:
    return bool(path) and os.path.isfile(path)


def _build_downloads() -> dict:
    cfg = current_app.config
    eval_dir = cfg["EVALUATION_FOLDER"]
    return {
        "csv": _file_exists(os.path.join(eval_dir, "classification_report.csv")),
        "json": _file_exists(cfg["EVALUATION_RESULT_PATH"]),
        "misclassification": _file_exists(cfg["MISCLASSIFICATION_REPORT_PATH"]),
        "confusion_matrix": _file_exists(os.path.join(eval_dir, "confusion_matrix.png")),
        "training_accuracy": _file_exists(os.path.join(eval_dir, "training_accuracy.png")),
        "validation_accuracy": _file_exists(os.path.join(eval_dir, "validation_accuracy.png")),
        "training_loss": _file_exists(os.path.join(eval_dir, "training_loss.png")),
        "validation_loss": _file_exists(os.path.join(eval_dir, "validation_loss.png")),
        "accuracy_curves": _file_exists(os.path.join(eval_dir, "accuracy_curves.png")),
        "loss_curves": _file_exists(os.path.join(eval_dir, "loss_curves.png")),
        "precision_per_class": _file_exists(os.path.join(eval_dir, "precision_per_class.png")),
        "recall_per_class": _file_exists(os.path.join(eval_dir, "recall_per_class.png")),
        "f1_per_class": _file_exists(os.path.join(eval_dir, "f1_per_class.png")),
        "per_class_metrics_comparison": _file_exists(os.path.join(eval_dir, "per_class_metrics_comparison.png")),
    }


def _ensure_evaluation_charts(eval_json: dict, class_labels: list) -> None:
    """Regenerate chart PNGs from saved JSON when files are missing."""
    if not eval_json:
        return

    downloads = _build_downloads()
    chart_keys = (
        "accuracy_curves",
        "loss_curves",
        "training_accuracy",
        "validation_accuracy",
        "training_loss",
        "validation_loss",
        "precision_per_class",
        "recall_per_class",
        "f1_per_class",
        "per_class_metrics_comparison",
    )
    if all(downloads.get(key) for key in chart_keys):
        return

    cfg = current_app.config
    history_file = eval_json.get("history") or {}
    if not history_file and os.path.isfile(cfg["TRAINING_HISTORY_PATH"]):
        try:
            with open(cfg["TRAINING_HISTORY_PATH"], "r", encoding="utf-8") as handle:
                history_file = json.load(handle)
        except (OSError, json.JSONDecodeError):
            history_file = {}

    generate_evaluation_charts(
        cfg["EVALUATION_FOLDER"],
        history_file=history_file,
        labels=class_labels,
        per_class=eval_json.get("per_class", {}),
        classification_report=eval_json.get("classification_report", {}),
        plot_dpi=int(cfg.get("EVAL_PLOT_DPI", 100)),
    )


@evaluation_bp.route("/")
@login_required
def evaluation_page():
    latest = EvaluasiModel.query.filter_by(tipe_record=TIPE_RINGKASAN).order_by(EvaluasiModel.dibuat_pada.desc()).first()
    eval_json = _load_eval_json()
    class_labels = current_app.config.get("CLASS_LABELS", [])
    eval_class_labels = resolve_eval_class_labels(current_app.config)
    if eval_json:
        eval_json = backfill_class_accuracy(eval_json, class_labels)
        _ensure_evaluation_charts(eval_json, eval_class_labels)

    eval_deps_ok = True
    eval_deps_message = ""
    if should_use_keras_model():
        eval_deps_ok, _, eval_deps_message = check_keras_eval_dependencies()
    elif is_teachable_machine_available():
        eval_deps_ok, _, eval_deps_message = check_eval_dependencies()

    report_rows = {}
    if eval_json.get("classification_report"):
        report_rows = eval_json["classification_report"]
    elif latest and latest.classification_report:
        try:
            report_rows = json.loads(latest.classification_report)
        except Exception:
            report_rows = {}

    acc = eval_json.get("accuracy") or (latest.accuracy if latest else None)
    prec = eval_json.get("precision") or (latest.precision_score if latest else None)
    rec = eval_json.get("recall") or (latest.recall_score if latest else None)
    f1 = eval_json.get("f1_score") or (latest.f1_score if latest else None)

    target = target_status(acc, ACCURACY_TARGET)
    f1_target = target_status(f1, ACCURACY_TARGET)
    eval_summary = build_eval_summary(acc, f1, ACCURACY_TARGET)

    misclassification = load_misclassification_report()
    if not misclassification and eval_json.get("misclassified_images"):
        misclassification = {
            "misclassified_count": eval_json.get("misclassified_count", 0),
            "misclassified_images": eval_json.get("misclassified_images", []),
        }

    raw_misclassified = misclassification.get("misclassified_images", []) if misclassification else []
    test_dir = current_app.config["TEST_DIR"]
    optimized_test_dir = os.path.join(current_app.config["OPTIMIZED_MODEL_DIR"], "test")

    def _static_preview_url(static_rel: str) -> str:
        return url_for("static", filename=static_rel)

    def _test_image_url(rel: str) -> str:
        return url_for("evaluation.test_image", path=rel)

    misclassified_rows = []
    for idx, row in enumerate(raw_misclassified):
        enriched = enrich_misclassified_row(
            row,
            row_id=idx,
            test_dir=test_dir,
            optimized_test_dir=optimized_test_dir,
            image_url_builder=_test_image_url,
            static_url_builder=_static_preview_url,
        )
        misclassified_rows.append(merge_correction_into_row(enriched))
    misclassified_rows.sort(key=lambda r: (not r.get("is_high_confidence_error"), -(r.get("confidence_pct") or 0)))

    confusion_pairs = load_confusion_pair_report()
    if not confusion_pairs.get("pairs") and eval_json.get("confused_pairs"):
        confusion_pairs = {"pairs": eval_json.get("confused_pairs", [])}

    pairs = confusion_pairs.get("pairs", [])
    for pair in pairs:
        pair.setdefault("true_display", display_class(pair.get("true_class")))
        pair.setdefault("predicted_display", display_class(pair.get("predicted_class")))

    per_class_rows = build_per_class_rows(
        eval_json.get("per_class", {}),
        report_rows,
        ACCURACY_TARGET,
        confusion_matrix=eval_json.get("confusion_matrix"),
        class_labels=eval_class_labels,
    )
    recommendations = merge_recommendations(eval_json.get("recommendations", []))
    history = eval_json.get("history", {})
    curve_notes = interpret_training_history(history)
    confusion_info = build_confusion_interpretation(eval_json.get("confusion_matrix", []))
    performance_insights = build_performance_insights(history, eval_summary, confusion_info)
    misc_summary = analyze_misclassification_summary(misclassified_rows, pairs)
    confusion_insight = build_confusion_insight(pairs, misc_summary)

    eval_counts = resolve_evaluation_counts(current_app.config, eval_json)
    test_samples = eval_counts.get("test_samples")
    if test_samples is None:
        test_samples = eval_json.get("test_samples")
    correct_predictions = eval_json.get("correct_predictions")
    incorrect_predictions = eval_json.get("incorrect_predictions")
    if correct_predictions is None and test_samples is not None and eval_json.get("misclassified_count") is not None:
        correct_predictions = test_samples - eval_json.get("misclassified_count", 0)
        incorrect_predictions = eval_json.get("misclassified_count", 0)

    error_dashboard_cards = build_error_dashboard_cards(
        test_samples=test_samples,
        correct_predictions=correct_predictions,
        incorrect_predictions=incorrect_predictions,
        accuracy_pct=eval_summary.get("accuracy_pct"),
        misc_summary=misc_summary,
        misclassified_rows=misclassified_rows,
    )
    class_summary_text = weak_class_summary(per_class_rows)
    downloads = _build_downloads()

    metric_cards = [
        {"key": "accuracy", "label": "Akurasi", "icon": "bi-bullseye", "value": acc, "tone": metric_tone(acc), "hint": metric_card_hint("accuracy", acc), "desc": metric_description("accuracy")},
        {"key": "precision", "label": "Presisi", "icon": "bi-check2-circle", "value": prec, "tone": metric_tone(prec), "hint": metric_card_hint("precision", prec), "desc": metric_description("precision")},
        {"key": "recall", "label": "Recall", "icon": "bi-search", "value": rec, "tone": metric_tone(rec), "hint": metric_card_hint("recall", rec), "desc": metric_description("recall")},
        {"key": "f1", "label": "F1-Score", "icon": "bi-diagram-2", "value": f1, "tone": metric_tone(f1), "hint": metric_card_hint("f1", f1), "desc": metric_description("f1")},
    ]

    has_any_download = any(downloads.values())

    model_versions = [model_version_to_dict(v) for v in list_model_versions(8)]
    active_version = next((v for v in model_versions if v.get("is_active")), None)
    tm_info = get_tm_display_info() if is_teachable_machine_available() else None

    cm = eval_json.get("confusion_matrix") or []
    class_labels_display = [display_class(label) for label in eval_class_labels]
    model_eval = build_active_model_eval_display(current_app.config, eval_json)
    accuracy_formula = build_accuracy_formula_display(correct_predictions, test_samples, acc)
    dataset_eval = build_dataset_eval_display(eval_json.get("dataset_summary"))
    for key in (
        "total_dataset_raw",
        "total_dataset_processed",
        "holdout_percent",
        "train_images",
        "validation_images",
    ):
        if eval_counts.get(key) is not None:
            dataset_eval[key] = eval_counts[key]
    dataset_eval["split_valid"] = eval_counts.get("split_valid", dataset_eval.get("split_valid", True))
    dataset_eval["count_warnings"] = list(dict.fromkeys(
        (dataset_eval.get("warnings") or []) + (eval_counts.get("warnings") or [])
    ))
    if eval_counts.get("total_dataset_processed"):
        dataset_eval["available"] = True
    macro_metric_cards = build_macro_metric_cards(eval_json)
    cm_normalized = eval_json.get("confusion_matrix_normalized") or []

    return render_template(
        "evaluation.html",
        latest=latest,
        eval_json=eval_json,
        report_rows=report_rows,
        accuracy_target=ACCURACY_TARGET,
        target_status=target,
        f1_target_status=f1_target,
        eval_summary=eval_summary,
        weak_classes=eval_json.get("weak_classes", []),
        recommendations=recommendations,
        per_class_rows=per_class_rows,
        misclassification=misclassification,
        misclassified_rows=misclassified_rows,
        misc_summary=misc_summary,
        confusion_insight=confusion_insight,
        error_dashboard_cards=error_dashboard_cards,
        confusion_pairs=confusion_pairs,
        pairs=pairs[:10],
        curve_notes=curve_notes,
        confusion_info=confusion_info,
        performance_insights=performance_insights,
        class_summary_text=class_summary_text,
        metric_cards=metric_cards,
        downloads=downloads,
        has_any_download=has_any_download,
        has_results=bool(latest or eval_json),
        evaluated_at=eval_json.get("evaluated_at"),
        test_samples=test_samples,
        correct_predictions=correct_predictions,
        incorrect_predictions=incorrect_predictions,
        evaluation_split=eval_json.get("evaluation_split", "test_holdout"),
        active_engine=eval_json.get("active_engine"),
        tm_info=tm_info,
        model_versions=model_versions,
        active_model_version=active_version,
        confusion_matrix=cm,
        confusion_matrix_labels=class_labels_display,
        confusion_matrix_normalized=cm_normalized,
        dataset_eval=dataset_eval,
        macro_metric_cards=macro_metric_cards,
        eval_deps_ok=eval_deps_ok,
        eval_deps_message=eval_deps_message,
        model_eval=model_eval,
        accuracy_formula=accuracy_formula,
        eval_counts=eval_counts,
    )


@evaluation_bp.route("/run", methods=["POST"])
@admin_required
def run_evaluation():
    from services.db_compat import get_active_model_id, save_evaluation_details

    if is_teachable_machine_available() and not should_use_keras_model():
        ok, _, message = check_eval_dependencies()
        if not ok:
            flash(f"Evaluasi gagal: {message}", "danger")
            return redirect(url_for("evaluation.evaluation_page"))
    elif should_use_keras_model():
        ok, _, message = check_keras_eval_dependencies()
        if not ok:
            flash(f"Evaluasi gagal: {message}", "danger")
            return redirect(url_for("evaluation.evaluation_page"))

    try:
        if should_use_keras_model():
            from services.calibration_service import ensure_model_calibrated, should_recalibrate_on_eval

            if should_recalibrate_on_eval(current_app.config):
                ensure_model_calibrated(current_app.config)
        result = evaluate_model()
        history = result.get("history", {})
        model_name = result.get("model_name") or os.path.basename(
            current_app.config.get("EXTERNAL_MODEL_PATH")
            or current_app.config.get("MODEL_PATH", "khat_best.keras")
        )
        if is_teachable_machine_available() and not should_use_keras_model():
            register_active_tm_model(
                evaluation_accuracy=result.get("accuracy"),
                notes=f"Evaluated on {result.get('test_samples', 0)} test holdout images.",
            )
            tm = get_tm_display_info()
            model_name = tm["model_name"]
        elif should_use_keras_model():
            register_model_version(
                model_name=model_name,
                model_source="External Keras H5" if current_app.config.get("USE_EXTERNAL_MODEL") else "CNN Transfer Learning",
                model_runtime="TensorFlow Keras",
                evaluation_accuracy=result.get("accuracy"),
                notes=json.dumps({
                    "path": current_app.config.get("EXTERNAL_MODEL_PATH") or current_app.config.get("MODEL_PATH"),
                    "test_samples": result.get("test_samples"),
                    "evaluated_at": result.get("evaluated_at"),
                }),
            )
        evaluation = buat_ringkasan_evaluasi(
            model_name=model_name,
            accuracy=result["accuracy"],
            precision_score=result["precision"],
            recall_score=result["recall"],
            f1_score=result["f1_score"],
            confusion_matrix=json.dumps(result["confusion_matrix"]),
            classification_report=json.dumps(result["classification_report"]),
            training_accuracy=history.get("accuracy", [None])[-1] if history.get("accuracy") else None,
            validation_accuracy=history.get("val_accuracy", [None])[-1] if history.get("val_accuracy") else None,
            training_loss=history.get("loss", [None])[-1] if history.get("loss") else None,
            validation_loss=history.get("val_loss", [None])[-1] if history.get("val_loss") else None,
        )
        evaluation.id_model = get_active_model_id()
        evaluation.total_gambar = result.get("test_samples")
        evaluation.prediksi_benar = result.get("correct_predictions")
        evaluation.prediksi_salah = result.get("incorrect_predictions") or result.get("misclassified_count")
        db.session.add(evaluation)
        db.session.flush()
        save_evaluation_details(evaluation, result.get("misclassified_images") or [], commit=False)
        db.session.commit()
        flash("Evaluasi model selesai.", "success")
        if result.get("target_achieved"):
            flash("Target evaluasi tercapai — akurasi test di atas 85%.", "success")
        else:
            flash("Target evaluasi belum tercapai — akurasi test di bawah 85%.", "warning")
        for weak in result.get("weak_classes", [])[:3]:
            flash(weak.get("message", ""), "warning")
    except ImportError as exc:
        flash(
            "Evaluasi gagal: dependensi TensorFlow belum terpasang. "
            "Jalankan installer\\install_eval_deps.bat atau perintah: "
            ".\\.venv\\Scripts\\pip.exe install tensorflow==2.20.0 tensorflowjs==4.22.0 h5py",
            "danger",
        )
    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        if "tensorflow" in lowered or "tensorflowjs" in lowered:
            flash(f"Evaluasi gagal: {message}", "danger")
        elif any(token in lowered for token in ("resource_exhausted", "oom", "out of memory", "memoryerror")):
            flash(
                "Evaluasi gagal karena memori tidak cukup. "
                "Tutup aplikasi lain, set EVAL_BATCH_SIZE=8 dan EVAL_USE_TTA=0 di .env, lalu jalankan ulang.",
                "danger",
            )
        elif "test dataset is empty" in lowered or "holdout" in lowered:
            flash(f"Evaluasi gagal: {message}", "danger")
        else:
            flash(f"Evaluasi gagal: {exc}", "danger")
    return redirect(url_for("evaluation.evaluation_page"))


@evaluation_bp.route("/test-image")
@login_required
def test_image():
    rel = (request.args.get("path") or "").strip().replace("\\", "/")
    if not rel or ".." in rel:
        abort(404)

    resolved = serve_evaluation_image_path(rel, config=current_app.config)
    if resolved and resolved.is_file():
        return send_file(resolved)

    abort(404)


@evaluation_bp.route("/media/evaluation/<path:filename>")
@login_required
def serve_evaluation_media(filename):
    """Serve evaluation/dataset images from allowed folders only."""
    safe_name = (filename or "").replace("\\", "/").strip()
    if not safe_name or ".." in safe_name:
        abort(404)

    suffix = Path(safe_name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        abort(404)

    resolved = serve_evaluation_image_path(safe_name, config=current_app.config)
    if resolved and resolved.is_file():
        return send_file(resolved)

    abort(404)


@evaluation_bp.route("/rebuild-previews", methods=["POST"])
@admin_required
def rebuild_previews():
    misclassification = load_misclassification_report()
    eval_json = _load_eval_json()
    raw_rows = []
    if misclassification:
        raw_rows = misclassification.get("misclassified_images", [])
    elif eval_json.get("misclassified_images"):
        raw_rows = eval_json.get("misclassified_images", [])

    if not raw_rows:
        flash("No evaluation error records found to rebuild.", "warning")
        return redirect(url_for("evaluation.evaluation_page"))

    result = rebuild_evaluation_previews(raw_rows)
    persist_rebuilt_previews(result["records"])
    flash(
        f"Evaluation image previews rebuilt: {result['fixed']} available, {result['unavailable']} unavailable.",
        "success" if result["fixed"] else "warning",
    )
    return redirect(url_for("evaluation.evaluation_page"))


@evaluation_bp.route("/error/mark-reviewed", methods=["POST"])
@login_required
def mark_error_reviewed():
    payload = request.get_json(silent=True) or {}
    image_rel = payload.get("image_rel") or request.form.get("image_rel")
    note = payload.get("note") or request.form.get("note") or ""
    if not image_rel:
        return jsonify({"ok": False, "message": "Missing image reference."}), 400
    record = mark_as_reviewed(image_rel, note=note)
    return jsonify({"ok": True, "record": record})


@evaluation_bp.route("/error/save-correction", methods=["POST"])
@login_required
def save_error_correction():
    payload = request.get_json(silent=True) or {}
    image_rel = payload.get("image_rel") or request.form.get("image_rel")
    correct_class = payload.get("correct_class") or request.form.get("correct_class")
    note = payload.get("note") or request.form.get("note") or ""
    action = payload.get("action") or request.form.get("action") or "save_to_dataset"
    if not image_rel or not correct_class:
        return jsonify({"ok": False, "message": "Missing image reference or correction class."}), 400
    try:
        if action == "move_to_class":
            record = move_image_to_class(image_rel, correct_class, note=note)
        else:
            record = save_label_correction(image_rel, correct_class, note=note)
        return jsonify({"ok": True, "record": record})
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400


@evaluation_bp.route("/download/<kind>")
@login_required
def download_report(kind):
    cfg = current_app.config
    mapping = {
        "csv": (os.path.join(cfg["EVALUATION_FOLDER"], "classification_report.csv"), "classification_report.csv"),
        "json": (cfg["EVALUATION_RESULT_PATH"], "evaluation_result.json"),
        "misclassification": (cfg["MISCLASSIFICATION_REPORT_PATH"], "misclassification_report.json"),
        "confusion-matrix": (os.path.join(cfg["EVALUATION_FOLDER"], "confusion_matrix.png"), "confusion_matrix.png"),
    }
    if kind not in mapping:
        abort(404)
    path, name = mapping[kind]
    if not os.path.isfile(path):
        flash("File laporan belum tersedia.", "warning")
        return redirect(url_for("evaluation.evaluation_page"))
    return send_file(path, as_attachment=True, download_name=name)
