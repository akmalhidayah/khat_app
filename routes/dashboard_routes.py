import json
import os

from flask import Blueprint, current_app, flash, redirect, render_template, url_for

from models import ClassificationResult, Dataset, EvaluasiModel
from models.evaluasi_model import TIPE_RINGKASAN
from routes.utils import login_required
from services.dashboard_ui_service import (
    build_dashboard_insights,
    build_dataset_insights,
    workflow_steps,
)
from services.dataset_service import class_distribution
from services.metrics_service import ACCURACY_TARGET, target_status
from services.history_ui_service import is_valid_khat_prediction
from services.khat_detector_service import is_detector_available, get_thresholds

dashboard_bp = Blueprint("dashboard", __name__)


def _load_training_history():
    path = current_app.config["TRAINING_HISTORY_PATH"]
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _load_eval_json():
    path = current_app.config["EVALUATION_RESULT_PATH"]
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


@dashboard_bp.route("/")
@dashboard_bp.route("/dashboard")
@login_required
def dashboard():
    if current_app.config.get("APP_SIMPLE_MODE"):
        from flask import session
        if session.get("role") != "admin":
            return redirect(url_for("classification.classify_image"))
    total_dataset = Dataset.query.count()
    total_classes = len(current_app.config["CLASS_LABELS"])
    all_history = ClassificationResult.query.order_by(ClassificationResult.dibuat_pada.desc()).all()
    total_history = len(all_history)
    latest_result = all_history[0] if all_history else None
    latest_eval = EvaluasiModel.query.filter_by(tipe_record=TIPE_RINGKASAN).order_by(EvaluasiModel.dibuat_pada.desc()).first()
    eval_json = _load_eval_json()
    train_hist = _load_training_history()
    config = train_hist.get("config", {})

    test_accuracy = eval_json.get("accuracy") or (latest_eval.accuracy if latest_eval else None)
    train_accuracy = train_hist.get("final_training_accuracy") or (
        train_hist.get("accuracy", [None])[-1] if train_hist.get("accuracy") else None
    )
    val_accuracy = train_hist.get("final_validation_accuracy") or config.get("best_validation_accuracy") or (
        train_hist.get("val_accuracy", [None])[-1] if train_hist.get("val_accuracy") else None
    )
    precision = eval_json.get("precision") or (latest_eval.precision_score if latest_eval else None)
    recall = eval_json.get("recall") or (latest_eval.recall_score if latest_eval else None)
    f1 = eval_json.get("f1_score") or (latest_eval.f1_score if latest_eval else None)

    target_test = target_status(test_accuracy, ACCURACY_TARGET)
    recent_classifications = all_history[:5]

    distribution_raw = class_distribution()
    distribution = {label: distribution_raw.get(label, 0) for label in current_app.config["CLASS_LABELS"]}
    avg_confidence = None
    if total_history:
        confidences = [row.confidence for row in recent_classifications if row.confidence is not None]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)

    valid_khat_predictions = sum(1 for row in all_history if is_valid_khat_prediction(row))
    rejected_non_khat = sum(1 for row in all_history if (row.input_status or "") == "non_khat")
    unrecognized_inputs = sum(1 for row in all_history if (row.input_status or "") == "unrecognized")
    uncertain_inputs = sum(1 for row in all_history if (row.input_status or "") == "uncertain")
    latest_rejected = next(
        (row for row in all_history if (row.input_status or "") in ("non_khat", "uncertain", "unrecognized")),
        None,
    )
    detector_meta = {}
    meta_path = current_app.config.get("KHAT_DETECTOR_METADATA_PATH")
    if meta_path and os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                detector_meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    detector_eval = {}
    eval_path = current_app.config.get("KHAT_DETECTOR_EVALUATION_PATH")
    if eval_path and os.path.isfile(eval_path):
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                detector_eval = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    khat_thresholds = get_thresholds()
    dataset_insights = build_dataset_insights(
        distribution,
        class_labels=current_app.config["CLASS_LABELS"],
    )
    dashboard_insights = build_dashboard_insights(
        test_accuracy=test_accuracy,
        accuracy_target=ACCURACY_TARGET,
        target_achieved=target_test.get("achieved", False),
        dataset_insights=dataset_insights,
        detector_available=is_detector_available(),
        khat_thresholds=khat_thresholds,
    )
    wf_steps = workflow_steps(
        total_dataset=total_dataset,
        has_training=bool(train_accuracy is not None or train_hist),
        has_evaluation=bool(test_accuracy is not None or latest_eval),
        total_predictions=total_history,
    )

    return render_template(
        "dashboard.html",
        total_dataset=total_dataset,
        total_classes=total_classes,
        total_history=total_history,
        latest_accuracy=test_accuracy,
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        test_accuracy=test_accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        accuracy_target=ACCURACY_TARGET,
        target_test=target_test,
        latest_result=latest_result,
        recent_classifications=recent_classifications,
        distribution=distribution,
        avg_confidence=avg_confidence,
        training_mode=config.get("training_mode"),
        valid_khat_predictions=valid_khat_predictions,
        rejected_non_khat=rejected_non_khat,
        unrecognized_inputs=unrecognized_inputs,
        uncertain_inputs=uncertain_inputs,
        latest_rejected=latest_rejected,
        detector_available=is_detector_available(),
        detector_meta=detector_meta,
        detector_eval=detector_eval,
        khat_thresholds=khat_thresholds,
        dataset_insights=dataset_insights,
        dashboard_insights=dashboard_insights,
        workflow_steps=wf_steps,
    )
