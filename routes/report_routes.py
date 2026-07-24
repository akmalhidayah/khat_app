import csv
import io
import json
import os
from datetime import datetime

from flask import Blueprint, current_app, make_response, render_template, session, url_for

from models import ClassificationResult, Dataset, EvaluasiModel
from models.evaluasi_model import TIPE_RINGKASAN
from routes.utils import login_required
from services.dataset_service import class_distribution
from services.evaluation_ui_service import build_per_class_rows
from services.metrics_service import ACCURACY_TARGET
from services.model_context_service import get_active_model_path, get_model_context, is_teachable_machine_active
from services.report_ui_service import (
    build_confusion_matrix_table,
    build_conclusion,
    build_dataset_balance_note,
    build_distribution_rows,
    build_eval_metric_rows,
    build_executive_summary,
    build_history_report_summary,
    build_recommendations,
    build_research_insights,
    build_target_card,
    build_technical_model_info,
    build_training_metric_rows,
    detect_class_imbalance,
    serialize_report_row,
)
from services.algorithm_calculation_service import build_algorithm_report_summary
from services.teachable_machine_service import get_tm_display_info, is_teachable_machine_available

report_bp = Blueprint("report", __name__, url_prefix="/report")


def _count_images(root_dir: str) -> int:
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


def _image_url(image_path: str) -> str:
    if not image_path:
        return ""
    static_rel = image_path.replace("static/", "", 1) if image_path.startswith("static/") else image_path
    return url_for("static", filename=static_rel)


@report_bp.route("/")
@login_required
def report_page():
    latest_eval = EvaluasiModel.query.filter_by(tipe_record=TIPE_RINGKASAN).order_by(EvaluasiModel.dibuat_pada.desc()).first()
    all_history = ClassificationResult.query.order_by(ClassificationResult.dibuat_pada.desc()).all()

    model_ctx = get_model_context()
    is_tm = is_teachable_machine_active(model_ctx)
    tm_info = get_tm_display_info() if is_teachable_machine_available() else None
    arch_label = (
        tm_info.get("model_architecture", "Teachable Machine Image Model")
        if is_tm and tm_info
        else model_ctx.get("model_architecture", "EfficientNetB0 Transfer Learning")
    )
    model_filename = (
        tm_info.get("model_name", "tm-my-image-model")
        if is_tm and tm_info
        else os.path.basename(get_active_model_path())
    )
    technical_info = build_technical_model_info(tm_info, model_ctx)

    eval_json = {}
    eval_path = current_app.config.get("EVALUATION_RESULT_PATH")
    if eval_path and os.path.isfile(eval_path):
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                eval_json = json.load(f)
        except (json.JSONDecodeError, OSError):
            eval_json = {}

    report_rows = eval_json.get("classification_report", {})
    if latest_eval and latest_eval.classification_report and not report_rows:
        try:
            report_rows = json.loads(latest_eval.classification_report)
        except (json.JSONDecodeError, TypeError):
            report_rows = {}

    per_class_rows = build_per_class_rows(
        eval_json.get("per_class", {}),
        report_rows,
        ACCURACY_TARGET,
        confusion_matrix=eval_json.get("confusion_matrix"),
        class_labels=current_app.config.get("CLASS_LABELS"),
    )
    confusion_data = build_confusion_matrix_table(
        eval_json.get("confusion_matrix") or (
            json.loads(latest_eval.confusion_matrix) if latest_eval and latest_eval.confusion_matrix else None
        ),
        current_app.config["CLASS_LABELS"],
    )
    training_metric_rows = build_training_metric_rows(latest_eval)

    dist_raw = class_distribution()
    distribution = {label: dist_raw.get(label, 0) for label in current_app.config["CLASS_LABELS"]}
    total_dataset = Dataset.query.count()
    is_imbalanced = detect_class_imbalance(distribution)

    history_summary = build_history_report_summary(all_history)
    algorithm_summary = build_algorithm_report_summary(all_history)
    target_card = build_target_card(latest_eval)
    eval_rows = build_eval_metric_rows(latest_eval)
    recommendations = build_recommendations(
        latest_eval, is_imbalanced, model_ctx.get("training_mode_key"), model_ctx.get("model_architecture_key")
    )
    conclusion = build_conclusion(target_card)
    executive_summary = build_executive_summary(
        arch_label, total_dataset, latest_eval, history_summary, target_card, is_tm=is_tm
    )
    research_insights = build_research_insights(
        target_card, is_imbalanced, distribution, latest_eval, history_summary
    )
    distribution_rows = build_distribution_rows(distribution)
    dataset_balance_note = build_dataset_balance_note(distribution, is_imbalanced)

    train_count = _count_images(current_app.config["TRAIN_DIR"])
    validation_count = _count_images(current_app.config["VALIDATION_DIR"])
    test_count = _count_images(current_app.config["TEST_DIR"])
    processed_count = train_count + validation_count + test_count
    dataset_status = (
        "Siap"
        if train_count > 0 and validation_count > 0 and test_count > 0
        else "Perlu diproses"
    )

    valid_history_rows = [
        serialize_report_row(r, _image_url(r.image_path))
        for r in history_summary["valid_khat_records"]
    ]
    rejected_rows = [
        serialize_report_row(r, _image_url(r.image_path))
        for r in history_summary["latest_rejected"]
    ]

    eval_date = _fmt_eval_date(latest_eval.created_at if latest_eval else None)

    return render_template(
        "report.html",
        total_dataset=total_dataset,
        total_history=history_summary["total"],
        distribution=distribution,
        is_imbalanced=is_imbalanced,
        latest_eval=latest_eval,
        valid_history_rows=valid_history_rows,
        rejected_rows=rejected_rows,
        history_summary=history_summary,
        algorithm_summary=algorithm_summary,
        train_count=train_count,
        validation_count=validation_count,
        test_count=test_count,
        total_classes=len(current_app.config["CLASS_LABELS"]),
        generated_at=datetime.now(),
        model_ctx=model_ctx,
        arch_label=arch_label,
        model_filename=model_filename,
        target_card=target_card,
        eval_rows=eval_rows,
        recommendations=recommendations,
        conclusion=conclusion,
        executive_summary=executive_summary,
        eval_date=eval_date,
        accuracy_target=ACCURACY_TARGET,
        is_tm=is_tm,
        tm_info=tm_info,
        technical_info=technical_info,
        per_class_rows=per_class_rows,
        confusion_data=confusion_data,
        training_metric_rows=training_metric_rows,
        research_insights=research_insights,
        distribution_rows=distribution_rows,
        dataset_balance_note=dataset_balance_note,
        processed_count=processed_count,
        dataset_status=dataset_status,
        generated_by=session.get("name", "Administrator"),
    )


def _fmt_eval_date(value):
    if not value:
        return "—"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


@report_bp.route("/export/summary.csv")
@login_required
def export_summary_csv():
    latest_eval = EvaluasiModel.query.filter_by(tipe_record=TIPE_RINGKASAN).order_by(EvaluasiModel.dibuat_pada.desc()).first()
    all_history = ClassificationResult.query.all()
    model_ctx = get_model_context()
    arch_label = model_ctx.get("model_architecture", "EfficientNetB0 Transfer Learning")
    history_summary = build_history_report_summary(all_history)
    target_card = build_target_card(latest_eval)
    total_dataset = Dataset.query.count()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Section", "Metric", "Value"])
    writer.writerow(["Model", "Architecture", arch_label])
    writer.writerow(["Target", "Accuracy Target", f"{target_card['target_pct']}%"])
    if latest_eval:
        writer.writerow(["Evaluation", "Accuracy", f"{target_card['accuracy_pct']}%"])
        writer.writerow(["Evaluation", "F1-Score", f"{target_card['f1_pct']}%"])
        writer.writerow(["Evaluation", "Status", target_card["status_label"]])
    writer.writerow(["Dataset", "Total Images", total_dataset])
    writer.writerow(["History", "Total Records", history_summary["total"]])
    writer.writerow(["History", "Valid Khat", history_summary["valid_khat_count"]])
    writer.writerow(["History", "Rejected Non-Khat", history_summary["rejected_count"]])
    writer.writerow(["History", "Uncertain", history_summary["uncertain_count"]])
    writer.writerow(["Report", "Generated At", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    response = make_response(output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = "attachment; filename=research_summary_report.csv"
    return response
