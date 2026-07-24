import csv
import io
import json
import os
import uuid
from datetime import datetime
from typing import Optional

from flask import Blueprint, current_app, flash, make_response, redirect, render_template, request, session, url_for
from PIL import Image
from werkzeug.utils import secure_filename

from models import ClassificationResult, db
from routes.utils import login_required
from services.algorithm_detail_service import build_classification_detail_context, resolve_detail_riwayat_id
from services.classification_flow_service import allowed_classification_file, run_user_classification
from services.class_mapping_service import (
    format_trained_class_list,
    resolve_trained_class_labels,
    trained_class_display_names,
)
from services.khat_recognition_messages import STATUS_LABEL, flash_for_prediction, flash_message, should_treat_as_unrecognized
from services.dataset_path_service import safe_relpath
from services.dataset_service import allowed_file
from services.khat_detector_service import (
    DETECTION_DECISION_LABELS,
    DETECTION_STATUS_LABELS,
    get_thresholds,
    is_detector_available,
    resolve_detection,
)
from services.khat_characteristics_service import get_characteristics_catalog
from services.history_ui_service import (
    CLASS_LABELS,
    apply_confidence_filter,
    build_history_record,
    build_history_summary,
    build_prediction_insights,
    serialize_records_for_export,
)
from services.model_context_service import get_active_model_path, get_model_context, get_model_display_label, get_model_short_name
from services.prediction_service import predict_image
from services.dataset_similarity_service import check_dataset_similarity, merge_similarity_review
from services.algorithm_calculation_service import (
    apply_algorithm_fields_to_row,
    build_algorithm_calculation,
    safe_build_algorithm_calculation,
)
from services.result_interpretation_service import (
    apply_stage1_combined_decision,
    build_image_metadata,
    build_result_analysis,
    compute_validation_fields,
    normalize_manual_expected_class,
)
from services.classification_correction_service import (
    mark_misclassification,
    save_correction_to_dataset,
)
from services.teachable_machine_service import get_tm_display_info, is_teachable_machine_available

classification_bp = Blueprint("classification", __name__, url_prefix="/classification")


def _validation_from_prediction(
    safe_name: str,
    prediction: dict,
    manual_expected_class: Optional[str] = None,
) -> dict:
    scores = prediction.get("scores") or {}
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_score = sorted_scores[0][1] if sorted_scores else 0
    second = sorted_scores[1] if len(sorted_scores) > 1 else None
    top_pct = top_score * 100 if top_score <= 1 else top_score
    second_pct = (second[1] * 100 if second[1] <= 1 else second[1]) if second else 0
    margin_pct = round(top_pct - second_pct, 2)
    fields = compute_validation_fields(
        safe_name,
        prediction.get("predicted_class"),
        prediction.get("confidence"),
        margin_pct=margin_pct,
        top_2_class=second[0] if second else None,
        top_2_score=second[1] if second else None,
        manual_expected_class=manual_expected_class,
    )
    return apply_stage1_combined_decision(
        fields,
        detection_status=prediction.get("detection_status"),
        input_status=prediction.get("input_status"),
        detection_explanation=prediction.get("detection_explanation"),
    )


def _model_fields_for_save() -> dict:
    if is_teachable_machine_available():
        info = get_tm_display_info()
        return {
            "model_source": info["model_source"],
            "model_name": info["model_name"],
            "model_runtime": info["model_runtime"],
            "model_input_size": info["model_input_size"],
        }
    model_ctx = get_model_context()
    short = get_model_short_name(model_ctx)
    return {
        "model_source": "Keras Transfer Learning",
        "model_name": short,
        "model_runtime": "TensorFlow Keras",
        "model_input_size": "224x224",
    }


def _save_classification_row(
    prediction: dict,
    rel_path: str,
    original_filename: str,
    abs_path: Optional[str] = None,
    manual_expected_class: Optional[str] = None,
    preprocessing_mode: str = "standard",
) -> ClassificationResult:
    scores = prediction.get("scores") or {}
    safe_name = secure_filename(original_filename)
    manual_key = normalize_manual_expected_class(manual_expected_class)
    validation_fields = _validation_from_prediction(
        safe_name,
        prediction,
        manual_expected_class=manual_key,
    )

    similarity = None
    if abs_path and os.path.isfile(abs_path):
        try:
            similarity = check_dataset_similarity(abs_path, current_app.config)
        except Exception:
            similarity = None

    conf = prediction.get("confidence")
    conf_pct = (conf * 100 if conf is not None and conf <= 1 else conf) if conf is not None else None
    if similarity and similarity.get("similarity_score") is not None:
        validation_fields = merge_similarity_review(validation_fields, similarity, conf_pct)

    model_fields = _model_fields_for_save()
    row = ClassificationResult(
        filename=safe_name,
        uploaded_filename=safe_name,
        image_path=rel_path,
        predicted_class=prediction.get("predicted_class"),
        confidence=prediction.get("confidence"),
        confidence_score=prediction.get("confidence"),
        naskhi_score=scores.get("naskhi", 0) if scores else 0,
        diwani_score=scores.get("diwani", 0) if scores else 0,
        diwani_jali_score=scores.get("diwani_jali", 0) if scores else 0,
        tsuluts_score=scores.get("tsuluts", 0) if scores else 0,
        probability_naskhi=scores.get("naskhi", 0) if scores else 0,
        probability_diwani=scores.get("diwani", 0) if scores else 0,
        probability_diwani_jali=scores.get("diwani_jali", 0) if scores else 0,
        probability_tsuluts=scores.get("tsuluts", 0) if scores else 0,
        expected_class=validation_fields["expected_class"],
        validation_status=validation_fields["validation_status"],
        review_status=validation_fields["review_status"],
        confidence_label=validation_fields.get("confidence_label"),
        final_decision=validation_fields["final_decision"],
        explanation_text=validation_fields["explanation_text"],
        model_name=model_fields["model_name"],
        model_source=model_fields["model_source"],
        model_runtime=model_fields["model_runtime"],
        model_input_size=model_fields["model_input_size"],
        top_2_class=validation_fields.get("top_2_class"),
        top_2_score=validation_fields.get("top_2_score"),
        input_status=prediction.get("input_status", "khat"),
        is_khat=prediction.get("is_khat", True) or prediction.get("stage2_ran", False),
        khat_probability=prediction.get("khat_probability"),
        non_khat_probability=prediction.get("non_khat_probability"),
        rejection_reason=prediction.get("rejection_reason"),
        top2_margin=validation_fields.get("top_2_margin") or prediction.get("top2_margin"),
        reliability_level=validation_fields.get("confidence_label") or validation_fields["review_status"],
        detection_status=prediction.get("detection_status"),
        detection_decision=prediction.get("detection_decision"),
        manual_review_required=validation_fields["manual_review_required"],
        similarity_status=similarity.get("similarity_status") if similarity else None,
        similarity_score=similarity.get("similarity_score") if similarity else None,
        nearest_dataset_image=similarity.get("nearest_dataset_image") if similarity else None,
        nearest_dataset_class=similarity.get("nearest_dataset_class") if similarity else None,
        similarity_risk_level=similarity.get("risk_level") if similarity else None,
        similarity_message=similarity.get("similarity_message") if similarity else None,
        similarity_recommendation=similarity.get("similarity_recommendation") if similarity else None,
        manual_expected_class=manual_key,
        source_type=validation_fields.get("expected_source"),
        preprocessing_mode=preprocessing_mode,
        known_class_status=validation_fields.get("known_class_status"),
        input_source="upload",
        expected_class_source=validation_fields.get("expected_source"),
    )
    db.session.add(row)
    db.session.flush()

    image_meta = build_image_metadata(abs_path) if abs_path and os.path.isfile(abs_path) else {}
    if image_meta.get("width"):
        row.original_width = image_meta["width"]
    if image_meta.get("height"):
        row.original_height = image_meta["height"]
    row.processed_width = 224
    row.processed_height = 224

    algorithm_calc = None
    try:
        analysis_preview = build_result_analysis(row, saved=False, tta_used=bool(prediction.get("tta_used")))
        algorithm_calc = build_algorithm_calculation(
            row,
            analysis=analysis_preview,
            tta_used=bool(prediction.get("tta_used")),
            input_source="upload",
        )
        apply_algorithm_fields_to_row(row, algorithm_calc, image_meta=image_meta)
    except Exception:
        pass

    try:
        from services.db_compat import finalize_classification_save, get_active_model_id

        row.id_model = get_active_model_id()
        if session.get("user_id"):
            row.id_pengguna = session.get("user_id")
        scores_payload = scores or {}
        if algorithm_calc:
            row.softmax_scores_json = json.dumps(scores_payload, ensure_ascii=False)
        algo_fields = {
            "original_width": row.original_width,
            "original_height": row.original_height,
            "processed_width": row.processed_width or 224,
            "processed_height": row.processed_height or 224,
            "model_input_size": row.model_input_size or "224x224",
            "tensor_shape": "[1, 224, 224, 3]",
            "preprocessing_steps_json": row.preprocessing_steps_json,
            "khat_probability": row.khat_probability,
            "non_khat_probability": row.non_khat_probability,
            "stage1_decision": row.detection_decision,
            "confidence_level": row.confidence_label,
            "known_class_status": row.known_class_status,
            "formula_summary_json": json.dumps(algorithm_calc.get("formulas", []), ensure_ascii=False)
            if algorithm_calc and algorithm_calc.get("formulas")
            else None,
            "calculation_notes": row.calculation_notes,
        }
        finalize_classification_save(row, scores=scores_payload, algorithm_fields=algo_fields)
    except Exception:
        pass

    db.session.commit()
    return row


def _refresh_algorithm_calculation_fields(
    row: ClassificationResult,
    abs_path: Optional[str] = None,
    tta_used: bool = False,
) -> None:
    try:
        image_meta = build_image_metadata(abs_path) if abs_path and os.path.isfile(abs_path) else {}
        analysis_preview = build_result_analysis(row, saved=True, tta_used=tta_used)
        algorithm_calc = build_algorithm_calculation(
            row,
            analysis=analysis_preview,
            tta_used=tta_used,
            input_source=getattr(row, "input_source", None) or "upload",
        )
        apply_algorithm_fields_to_row(row, algorithm_calc, image_meta=image_meta)
    except Exception:
        pass


def _image_url_for_row(result: ClassificationResult):
    if not result.image_path:
        return None
    static_rel = result.image_path.replace("static/", "", 1) if result.image_path.startswith("static/") else result.image_path
    return url_for("static", filename=static_rel)


def _detection_context(result: ClassificationResult) -> dict:
    thresholds = get_thresholds()
    khat_prob = float(result.khat_probability or 0)
    resolved = resolve_detection(khat_prob) if result.khat_probability is not None else {}
    status = result.detection_status or result.input_status or resolved.get("detection_status") or "khat"
    stage2_allowed = (
        (result.input_status or "khat") == "khat"
        and result.input_status != "unrecognized"
        and bool(result.predicted_class)
    )
    borderline_pct = round(thresholds["borderline"] * 100, 2)
    accept_pct = round(thresholds["accept"] * 100, 2)
    return {
        "khat_pct": round((result.khat_probability or 0) * 100, 2),
        "non_khat_pct": round((result.non_khat_probability or 0) * 100, 2),
        "detection_status": status,
        "detection_status_label": DETECTION_STATUS_LABELS.get(
            status,
            status.replace("_", " ").title(),
        ),
        "detection_decision": result.detection_decision,
        "detection_decision_label": DETECTION_DECISION_LABELS.get(
            result.detection_decision or "",
            result.detection_decision or "—",
        ),
        "accept_threshold": thresholds["accept"],
        "reject_threshold": thresholds["reject"],
        "borderline_threshold": thresholds["borderline"],
        "stage2_permission": "Enabled with caution" if status in ("borderline_khat", "uncertain_khat", "uncertain") else (
            "Enabled" if stage2_allowed else "Blocked"
        ),
        "stage2_allowed": stage2_allowed,
        "manual_review_required": result.manual_review_required,
        "is_borderline": status == "borderline_khat",
        "is_uncertain_khat": status in ("uncertain_khat", "uncertain"),
        "borderline_range_label": f"{borderline_pct}%–{accept_pct - 0.01:.2f}%",
        "banner_title": resolved.get("title"),
        "banner_message": resolved.get("message"),
        "detection_explanation": resolved.get("explanation_text"),
    }


@classification_bp.route("/predict", methods=["POST"])
@login_required
def predict_classification():
    return classify_image()


@classification_bp.route("/", methods=["GET", "POST"])
@login_required
def classify_image():
    simple_mode = current_app.config.get("APP_SIMPLE_MODE", False)

    if request.method == "POST":
        file = request.files.get("image")
        if not file or not file.filename:
            flash("Pilih citra kaligrafi terlebih dahulu.", "danger")
            return redirect(url_for("classification.classify_image"))

        if simple_mode:
            if not allowed_classification_file(file.filename):
                flash("Hanya file JPG, JPEG, dan PNG yang diperbolehkan.", "danger")
                return redirect(url_for("classification.classify_image"))
            user_id = session.get("user_id")
            if not user_id:
                flash("Sesi login tidak valid. Silakan masuk kembali.", "warning")
                return redirect(url_for("auth.login"))
            try:
                row, prediction = run_user_classification(user_id, file)
                session["last_prediction_id"] = row.id
                if prediction.get("input_status") in ("unrecognized", "non_khat", "uncertain"):
                    flash(flash_for_prediction(prediction, format_trained_class_list()), "warning")
                else:
                    flash("Klasifikasi berhasil.", "success")
                return redirect(url_for("classification.classification_result"))
            except ValueError as exc:
                flash(str(exc), "danger")
                return redirect(url_for("classification.classify_image"))
            except Exception as exc:
                if should_treat_as_unrecognized(exc):
                    flash(flash_message(format_trained_class_list()), "warning")
                else:
                    flash(f"Klasifikasi gagal: {exc}", "danger")
                return redirect(url_for("classification.classify_image"))

        if not allowed_file(file.filename):
            flash("Only JPG, JPEG, PNG, WEBP are allowed.", "danger")
            return redirect(url_for("classification.classify_image"))

        ext = file.filename.rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(upload_path)
        try:
            Image.open(upload_path).verify()
        except Exception:
            os.remove(upload_path)
            flash("Uploaded file is not a valid image.", "danger")
            return redirect(url_for("classification.classify_image"))
        rel_path = safe_relpath(upload_path, current_app.config["BASE_DIR"])

        try:
            use_tta = request.form.get("use_tta") in ("1", "on", "true", "yes")
            manuscript_mode = request.form.get("manuscript_mode") in ("1", "on", "true", "yes")
            preprocessing_mode = "manuscript" if manuscript_mode else "standard"
            manual_expected = request.form.get("manual_expected_class", "unknown")
            prediction = predict_image(
                upload_path,
                use_tta=use_tta,
                preprocessing_mode=preprocessing_mode,
            )
            row = _save_classification_row(
                prediction,
                rel_path,
                file.filename,
                abs_path=upload_path,
                manual_expected_class=manual_expected,
                preprocessing_mode=preprocessing_mode,
            )
            session["last_prediction_id"] = row.id
            session["last_prediction_tta"] = use_tta
            session["last_preprocessing_mode"] = preprocessing_mode
            session["last_manual_expected_class"] = manual_expected
            if prediction.get("input_status") in ("non_khat", "unrecognized", "uncertain"):
                flash(flash_for_prediction(prediction, format_trained_class_list()), "warning")
            elif prediction.get("detection_status") == "moderate_khat":
                flash("Khat detected with moderate confidence. Manual review is recommended.", "info")
            return redirect(url_for("classification.classification_result"))
        except FileNotFoundError as exc:
            if os.path.exists(upload_path):
                os.remove(upload_path)
            flash(f"Classification model not ready: {exc}", "warning")
            return redirect(url_for("classification.classify_image"))
        except Exception as exc:
            if os.path.exists(upload_path):
                os.remove(upload_path)
            if should_treat_as_unrecognized(exc):
                flash(flash_message(format_trained_class_list()), "warning")
            else:
                flash(f"Klasifikasi gagal: {exc}", "danger")
            return redirect(url_for("classification.classify_image"))

    model_ctx = get_model_context()
    thresholds = get_thresholds()
    tm_info = get_tm_display_info() if is_teachable_machine_available() and not simple_mode else None
    active_path = model_ctx.get("active_model_path") if model_ctx.get("model_exists") else get_active_model_path()
    display_model_name = tm_info["model_name"] if tm_info else os.path.basename(active_path or "CNN Model")
    active_model = None
    if simple_mode:
        from services.db_compat import get_active_model
        active_model = get_active_model()
    return render_template(
        "classify.html",
        model_exists=model_ctx.get("model_exists", False) or bool(active_model),
        model_name=display_model_name,
        model_ctx=model_ctx,
        detector_available=is_detector_available(),
        khat_thresholds=thresholds,
        simple_mode=simple_mode,
        active_model=active_model,
        class_labels=trained_class_display_names(),
        trained_class_labels=resolve_trained_class_labels(),
        trained_class_list=format_trained_class_list(),
    )


@classification_bp.route("/api/model-info")
@login_required
def model_info_api():
    """Return active model metadata for UI verification and browser-side TM loader."""
    model_ctx = get_model_context()
    tm_active = is_teachable_machine_available()
    payload = {
        "active_engine": "teachable_machine" if tm_active else "keras",
        "model_exists": bool(model_ctx.get("model_exists")),
        "model_architecture": model_ctx.get("model_architecture"),
        "model_architecture_key": model_ctx.get("model_architecture_key"),
        "trained_classes": resolve_trained_class_labels(),
        "trained_class_labels": trained_class_display_names(),
        "trained_class_list": format_trained_class_list(),
    }
    if tm_active:
        info = get_tm_display_info()
        payload.update({
            "model_source": info["model_source"],
            "model_name": info["model_name"],
            "model_runtime": info["model_runtime"],
            "model_input_size": info["model_input_size"],
            "model_type": info["model_type"],
            "model_json_url": info["model_json_url"],
            "model_metadata_url": info["model_metadata_url"],
            "model_weights_url": info["model_weights_url"],
            "labels": info["labels_raw"] or info["labels"],
            "labels_display": info["labels_display"],
            "image_size": info["model_input_size_px"],
            "teachable_machine": True,
        })
    else:
        payload.update({
            "model_source": "Keras Transfer Learning",
            "model_name": get_model_short_name(model_ctx),
            "model_runtime": "TensorFlow Keras",
            "model_input_size": "224x224",
            "active_model_path": model_ctx.get("active_model_path"),
            "teachable_machine": False,
        })
    return payload


@classification_bp.route("/result")
@login_required
def classification_result():
    row_id = session.get("last_prediction_id")
    result = ClassificationResult.query.get(row_id) if row_id else None
    if not result:
        flash("No classification result found.", "warning")
        return redirect(url_for("classification.classify_image"))

    detection = _detection_context(result)
    image_url = _image_url_for_row(result)
    show_rejected = not result.predicted_class or result.input_status in ("non_khat", "unrecognized")

    if show_rejected:
        from services.prediction_service import _stage2_thresholds
        return render_template(
            "result_rejected.html",
            result=result,
            image_url=image_url,
            input_status=result.input_status or "non_khat",
            detection=detection,
            stage2_thresholds=_stage2_thresholds(),
            trained_class_list=format_trained_class_list(),
            status_label=STATUS_LABEL if result.input_status == "unrecognized" else None,
        )

    analysis = build_result_analysis(result, saved=True, tta_used=session.get("last_prediction_tta", False))
    algorithm_calc = safe_build_algorithm_calculation(
        result,
        analysis=analysis,
        detection=detection,
        tta_used=session.get("last_prediction_tta", False),
        input_source=getattr(result, "input_source", None) or "upload",
    )
    return render_template(
        "result.html",
        result=result,
        analysis=analysis,
        algorithm_calc=algorithm_calc,
        image_url=image_url,
        detection=detection,
        khat_chars=analysis.get("khat_characteristics"),
        simple_mode=current_app.config.get("APP_SIMPLE_MODE", False),
        detail_url=url_for("classification.classification_detail", item_id=result.id),
    )


@classification_bp.route("/perhitungan-algoritma")
@login_required
def classification_detail_latest():
    """Menu sidebar: buka detail perhitungan klasifikasi terakhir milik user."""
    user_id = session.get("user_id")
    row_id = resolve_detail_riwayat_id(user_id, session.get("last_prediction_id"))
    if not row_id:
        return render_template(
            "classification_detail.html",
            detail={
                "available": False,
                "riwayat_id": None,
                "message": (
                    "Belum ada hasil klasifikasi. Unggah gambar di menu "
                    "Klasifikasi Gambar terlebih dahulu untuk melihat detail perhitungan."
                ),
            },
            detail_json={},
        )
    detail, error = build_classification_detail_context(row_id, user_id=user_id)
    if error == "not_found":
        flash("Data riwayat klasifikasi tidak ditemukan.", "warning")
        return redirect(url_for("classification.history"))
    if error == "forbidden":
        flash("Anda tidak memiliki akses ke detail perhitungan ini.", "danger")
        return redirect(url_for("classification.history"))
    return render_template(
        "classification_detail.html",
        detail=detail,
        detail_json=detail if detail.get("available") else {},
    )


@classification_bp.route("/detail/<int:item_id>")
@login_required
def classification_detail(item_id):
    user_id = session.get("user_id")
    detail, error = build_classification_detail_context(item_id, user_id=user_id)
    if error == "not_found":
        flash("Data riwayat klasifikasi tidak ditemukan.", "warning")
        return redirect(url_for("classification.history"))
    if error == "forbidden":
        flash("Anda tidak memiliki akses ke detail perhitungan ini.", "danger")
        return redirect(url_for("classification.history"))
    return render_template(
        "classification_detail.html",
        detail=detail,
        detail_json=detail if detail.get("available") else {},
    )


@classification_bp.route("/continue/<int:item_id>", methods=["POST"])
@login_required
def continue_classification(item_id):
    result = ClassificationResult.query.get_or_404(item_id)
    if result.input_status != "uncertain":
        flash("Continue classification is only available for uncertain inputs.", "warning")
        return redirect(url_for("classification.classification_result"))

    abs_path = os.path.join(current_app.config["BASE_DIR"], result.image_path)
    if not os.path.isfile(abs_path):
        flash("Uploaded image file is missing.", "danger")
        return redirect(url_for("classification.history"))

    try:
        use_tta = session.get("last_prediction_tta", False)
        prediction = predict_image(abs_path, use_tta=use_tta, force_classify=True)
        validation_fields = _validation_from_prediction(
            result.uploaded_filename or result.filename or "",
            prediction,
            manual_expected_class=result.manual_expected_class,
        )
        scores = prediction.get("scores") or {}

        similarity = None
        try:
            similarity = check_dataset_similarity(abs_path, current_app.config)
        except Exception:
            similarity = None

        conf = prediction.get("confidence")
        conf_pct = (conf * 100 if conf is not None and conf <= 1 else conf) if conf is not None else None
        if similarity and similarity.get("similarity_score") is not None:
            validation_fields = merge_similarity_review(validation_fields, similarity, conf_pct)

        result.predicted_class = prediction.get("predicted_class")
        result.confidence = prediction.get("confidence")
        result.confidence_score = prediction.get("confidence")
        result.naskhi_score = scores.get("naskhi", 0)
        result.diwani_score = scores.get("diwani", 0)
        result.diwani_jali_score = scores.get("diwani_jali", 0)
        result.tsuluts_score = scores.get("tsuluts", 0)
        result.probability_naskhi = scores.get("naskhi", 0)
        result.probability_diwani = scores.get("diwani", 0)
        result.probability_diwani_jali = scores.get("diwani_jali", 0)
        result.probability_tsuluts = scores.get("tsuluts", 0)
        result.expected_class = validation_fields["expected_class"]
        result.validation_status = validation_fields["validation_status"]
        result.review_status = validation_fields["review_status"]
        result.confidence_label = validation_fields.get("confidence_label")
        result.final_decision = validation_fields["final_decision"]
        result.explanation_text = validation_fields["explanation_text"]
        result.top_2_class = validation_fields.get("top_2_class")
        result.top_2_score = validation_fields.get("top_2_score")
        model_fields = _model_fields_for_save()
        result.model_name = model_fields["model_name"]
        result.model_source = model_fields["model_source"]
        result.model_runtime = model_fields["model_runtime"]
        result.model_input_size = model_fields["model_input_size"]
        result.is_khat = True
        result.detection_decision = "continue"
        result.manual_review_required = validation_fields["manual_review_required"]
        result.reliability_level = validation_fields["review_status"]
        result.top2_margin = validation_fields.get("top_2_margin") or prediction.get("top2_margin")
        result.rejection_reason = None
        result.similarity_status = similarity.get("similarity_status") if similarity else None
        result.similarity_score = similarity.get("similarity_score") if similarity else None
        result.nearest_dataset_image = similarity.get("nearest_dataset_image") if similarity else None
        result.nearest_dataset_class = similarity.get("nearest_dataset_class") if similarity else None
        result.similarity_risk_level = similarity.get("risk_level") if similarity else None
        result.similarity_message = similarity.get("similarity_message") if similarity else None
        result.similarity_recommendation = similarity.get("similarity_recommendation") if similarity else None
        result.known_class_status = validation_fields.get("known_class_status")
        _refresh_algorithm_calculation_fields(result, abs_path, tta_used=use_tta)
        db.session.commit()
        session["last_prediction_id"] = result.id
        flash("Classification continued with manual review flag.", "info")
    except Exception as exc:
        flash(f"Could not continue classification: {exc}", "danger")
    return redirect(url_for("classification.classification_result"))


@classification_bp.route("/reclassify/<int:item_id>", methods=["POST"])
@login_required
def reclassify_result(item_id):
    result = ClassificationResult.query.get_or_404(item_id)
    abs_path = os.path.join(current_app.config["BASE_DIR"], result.image_path)
    if not os.path.isfile(abs_path):
        flash("Uploaded image file is missing.", "danger")
        return redirect(url_for("classification.classification_result"))

    try:
        use_tta = session.get("last_prediction_tta", False)
        force_manuscript = request.form.get("force_manuscript") in ("1", "on", "true", "yes")
        manuscript_mode = force_manuscript or request.form.get("manuscript_mode") in ("1", "on", "true", "yes")
        preprocessing_mode = "manuscript" if manuscript_mode else (result.preprocessing_mode or "standard")
        prediction = predict_image(abs_path, use_tta=use_tta, preprocessing_mode=preprocessing_mode)
        validation_fields = _validation_from_prediction(
            result.uploaded_filename or result.filename or "",
            prediction,
            manual_expected_class=result.manual_expected_class,
        )
        scores = prediction.get("scores") or {}
        similarity = None
        try:
            similarity = check_dataset_similarity(abs_path, current_app.config)
        except Exception:
            similarity = None
        conf = prediction.get("confidence")
        conf_pct = (conf * 100 if conf is not None and conf <= 1 else conf) if conf is not None else None
        if similarity and similarity.get("similarity_score") is not None:
            validation_fields = merge_similarity_review(validation_fields, similarity, conf_pct)

        result.predicted_class = prediction.get("predicted_class")
        result.confidence = prediction.get("confidence")
        result.confidence_score = prediction.get("confidence")
        result.naskhi_score = scores.get("naskhi", 0)
        result.diwani_score = scores.get("diwani", 0)
        result.diwani_jali_score = scores.get("diwani_jali", 0)
        result.tsuluts_score = scores.get("tsuluts", 0)
        result.probability_naskhi = scores.get("naskhi", 0)
        result.probability_diwani = scores.get("diwani", 0)
        result.probability_diwani_jali = scores.get("diwani_jali", 0)
        result.probability_tsuluts = scores.get("tsuluts", 0)
        result.expected_class = validation_fields["expected_class"]
        result.validation_status = validation_fields["validation_status"]
        result.review_status = validation_fields["review_status"]
        result.confidence_label = validation_fields.get("confidence_label")
        result.final_decision = validation_fields["final_decision"]
        result.explanation_text = validation_fields["explanation_text"]
        result.top_2_class = validation_fields.get("top_2_class")
        result.top_2_score = validation_fields.get("top_2_score")
        result.manual_review_required = validation_fields["manual_review_required"]
        result.reliability_level = validation_fields["review_status"]
        result.top2_margin = validation_fields.get("top_2_margin") or prediction.get("top2_margin")
        result.preprocessing_mode = preprocessing_mode
        result.known_class_status = validation_fields.get("known_class_status")
        result.similarity_status = similarity.get("similarity_status") if similarity else result.similarity_status
        result.similarity_score = similarity.get("similarity_score") if similarity else result.similarity_score
        result.nearest_dataset_image = similarity.get("nearest_dataset_image") if similarity else result.nearest_dataset_image
        result.nearest_dataset_class = similarity.get("nearest_dataset_class") if similarity else result.nearest_dataset_class
        result.similarity_risk_level = similarity.get("risk_level") if similarity else result.similarity_risk_level
        result.similarity_message = similarity.get("similarity_message") if similarity else result.similarity_message
        result.similarity_recommendation = similarity.get("similarity_recommendation") if similarity else result.similarity_recommendation
        _refresh_algorithm_calculation_fields(result, abs_path, tta_used=use_tta)
        db.session.commit()
        session["last_prediction_id"] = result.id
        session["last_preprocessing_mode"] = preprocessing_mode
        flash("Klasifikasi ulang selesai.", "info")
    except Exception as exc:
        flash(f"Could not reclassify: {exc}", "danger")
    return redirect(url_for("classification.classification_result"))


@classification_bp.route("/correct/<int:item_id>", methods=["POST"])
@login_required
def save_correction(item_id):
    result = ClassificationResult.query.get_or_404(item_id)
    correct_class = request.form.get("correct_class", "").strip()
    notes = request.form.get("correction_notes", "").strip()
    if not correct_class:
        flash("Pilih kelas koreksi yang benar.", "warning")
        return redirect(url_for("classification.classification_result"))
    try:
        save_correction_to_dataset(result, correct_class, notes)
        flash("Koreksi disimpan ke dataset (raw + processed).", "success")
    except Exception as exc:
        flash(f"Gagal menyimpan koreksi: {exc}", "danger")
    return redirect(url_for("classification.classification_result"))


@classification_bp.route("/mark-misclassification/<int:item_id>", methods=["POST"])
@login_required
def mark_wrong_prediction(item_id):
    result = ClassificationResult.query.get_or_404(item_id)
    notes = request.form.get("correction_notes", "").strip()
    try:
        mark_misclassification(result, notes)
        flash("Hasil ditandai sebagai salah prediksi.", "info")
    except Exception as exc:
        flash(f"Gagal menandai: {exc}", "danger")
    return redirect(url_for("classification.classification_result"))


@classification_bp.route("/history")
@login_required
def history():
    class_filter = request.args.get("class_name", "").strip()
    status_filter = request.args.get("input_status", "").strip()
    confidence_filter = request.args.get("confidence_level", "").strip()
    search_query = request.args.get("q", "").strip().lower()
    date_from = request.args.get("date_from", "").strip()
    date_to = request.args.get("date_to", "").strip()

    query = ClassificationResult.query.order_by(ClassificationResult.dibuat_pada.desc())
    if current_app.config.get("APP_SIMPLE_MODE") and session.get("user_id"):
        query = query.filter_by(id_pengguna=session.get("user_id"))
    if class_filter:
        query = query.filter_by(predicted_class=class_filter)
    if status_filter:
        query = query.filter_by(input_status=status_filter)

    records = query.all()
    all_query = ClassificationResult.query.order_by(ClassificationResult.dibuat_pada.desc())
    if current_app.config.get("APP_SIMPLE_MODE") and session.get("user_id"):
        all_query = all_query.filter_by(id_pengguna=session.get("user_id"))
    all_records = all_query.all()

    if search_query:
        records = [r for r in records if search_query in (r.filename or "").lower()]

    if date_from:
        try:
            from_dt = datetime.strptime(date_from, "%Y-%m-%d")
            records = [r for r in records if r.created_at and r.created_at.date() >= from_dt.date()]
        except ValueError:
            pass
    if date_to:
        try:
            to_dt = datetime.strptime(date_to, "%Y-%m-%d")
            records = [r for r in records if r.created_at and r.created_at.date() <= to_dt.date()]
        except ValueError:
            pass

    if confidence_filter == "high":
        records = apply_confidence_filter(records, "high")
    elif confidence_filter == "medium":
        records = apply_confidence_filter(records, "medium")
    elif confidence_filter == "low":
        records = apply_confidence_filter(records, "low")

    model_ctx = get_model_context()
    arch_label = model_ctx.get("model_architecture", "EfficientNetB0 Transfer Learning")
    mode_label = model_ctx.get("training_mode_label", "—")
    summary = build_history_summary(all_records)
    insight_source = records if (class_filter or status_filter or confidence_filter or search_query or date_from or date_to) else all_records
    prediction_insights = build_prediction_insights(insight_source)

    history_records = [
        build_history_record(
            row,
            _image_url_for_row(row),
            url_for("classification.delete_history", item_id=row.id),
            arch_label,
            mode_label,
        )
        for row in records
    ]

    filter_labels = []
    if class_filter:
        filter_labels.append(f"Kelas: {CLASS_LABELS.get(class_filter, class_filter)}")
    if status_filter:
        status_names = {
            "khat": "Khat Valid",
            "non_khat": "Non-Khat",
            "uncertain": "Tidak Pasti",
            "unrecognized": STATUS_LABEL,
        }
        filter_labels.append(f"Status: {status_names.get(status_filter, status_filter)}")
    if confidence_filter:
        conf_names = {"high": "High Confidence", "medium": "Medium Confidence", "low": "Low Confidence"}
        filter_labels.append(conf_names.get(confidence_filter, confidence_filter))
    if search_query:
        filter_labels.append(f'File: "{search_query}"')
    if date_from:
        filter_labels.append(f"Dari: {date_from}")
    if date_to:
        filter_labels.append(f"Sampai: {date_to}")

    return render_template(
        "history.html",
        records=records,
        history_records_json=history_records,
        class_filter=class_filter,
        status_filter=status_filter,
        confidence_filter=confidence_filter,
        search_query=search_query,
        date_from=date_from,
        date_to=date_to,
        model_ctx=model_ctx,
        arch_label=arch_label,
        class_labels=CLASS_LABELS,
        total_predictions=summary["total"],
        valid_khat_count=summary["valid_khat"],
        rejected_non_khat_count=summary["rejected"],
        uncertain_count=summary["uncertain"],
        status_counts=summary["status_counts"],
        class_counts=summary["class_counts"],
        avg_confidence=summary["avg_confidence"],
        most_predicted=summary["most_predicted"],
        most_predicted_display=summary["most_predicted_display"],
        latest_prediction_date=summary["latest_prediction_date"],
        prediction_insights=prediction_insights,
        filter_labels=filter_labels,
        filtered_count=len(records),
        khat_characteristics_catalog=get_characteristics_catalog(),
    )


@classification_bp.route("/history/delete/<int:item_id>", methods=["POST"])
@login_required
def delete_history(item_id):
    qs = request.query_string.decode("utf-8")
    target = url_for("classification.history")

    if qs:
        target = f"{target}?{qs}"

    # Cari berdasarkan primary key terbaru.
    item = ClassificationResult.query.filter_by(
        id=item_id
    ).first()

    # Dukungan untuk ID lama hasil impor database.
    if item is None:
        item = ClassificationResult.query.filter_by(
            id_legacy=item_id
        ).first()

    if item is None:
        flash(
            "Riwayat klasifikasi tidak ditemukan "
            "atau sudah dihapus.",
            "warning",
        )
        return redirect(target)

    image_path = item.image_path or ""

    try:
        # Audit koreksi dipertahankan, tetapi relasinya dilepas.
        from models import LogKoreksi

        LogKoreksi.query.filter_by(
            id_riwayat=item.id
        ).update(
            {"id_riwayat": None},
            synchronize_session=False,
        )

        db.session.delete(item)
        db.session.commit()

    except Exception:
        db.session.rollback()

        current_app.logger.exception(
            "Gagal menghapus riwayat klasifikasi id=%s",
            item_id,
        )

        flash(
            "Riwayat klasifikasi gagal dihapus.",
            "danger",
        )
        return redirect(target)

    # Hapus file hanya setelah transaksi DB berhasil.
    if image_path:
        base_dir = os.path.realpath(
            current_app.config["BASE_DIR"]
        )

        abs_path = os.path.realpath(
            os.path.join(
                base_dir,
                image_path.lstrip("/"),
            )
        )

        if (
            abs_path.startswith(base_dir + os.sep)
            and os.path.isfile(abs_path)
        ):
            try:
                os.remove(abs_path)
            except OSError:
                current_app.logger.warning(
                    "Riwayat terhapus, tetapi file "
                    "gagal dihapus: %s",
                    abs_path,
                )

    flash(
        "Riwayat klasifikasi berhasil dihapus.",
        "success",
    )

    return redirect(target)


def _filtered_history_records():
    """Apply same filters as history page for export."""
    class_filter = request.args.get("class_name", "").strip()
    status_filter = request.args.get("input_status", "").strip()
    confidence_filter = request.args.get("confidence_level", "").strip()
    search_query = request.args.get("q", "").strip().lower()
    date_from = request.args.get("date_from", "").strip()
    date_to = request.args.get("date_to", "").strip()

    query = ClassificationResult.query.order_by(ClassificationResult.dibuat_pada.desc())
    if current_app.config.get("APP_SIMPLE_MODE") and session.get("user_id"):
        query = query.filter_by(id_pengguna=session.get("user_id"))
    if class_filter:
        query = query.filter_by(predicted_class=class_filter)
    if status_filter:
        query = query.filter_by(input_status=status_filter)
    records = query.all()

    if search_query:
        records = [r for r in records if search_query in (r.filename or "").lower()]
    if date_from:
        try:
            from_dt = datetime.strptime(date_from, "%Y-%m-%d")
            records = [r for r in records if r.created_at and r.created_at.date() >= from_dt.date()]
        except ValueError:
            pass
    if date_to:
        try:
            to_dt = datetime.strptime(date_to, "%Y-%m-%d")
            records = [r for r in records if r.created_at and r.created_at.date() <= to_dt.date()]
        except ValueError:
            pass
    if confidence_filter == "high":
        records = apply_confidence_filter(records, "high")
    elif confidence_filter == "medium":
        records = apply_confidence_filter(records, "medium")
    elif confidence_filter == "low":
        records = apply_confidence_filter(records, "low")
    return records


@classification_bp.route("/history/export/<string:fmt>")
@login_required
def export_history(fmt):
    fmt = (fmt or "").lower()
    if fmt not in ("csv", "json"):
        flash("Unsupported export format.", "warning")
        return redirect(url_for("classification.history"))

    model_ctx = get_model_context()
    arch_label = model_ctx.get("model_architecture", "EfficientNetB0 Transfer Learning")
    mode_label = model_ctx.get("training_mode_label", "—")
    records = _filtered_history_records()

    if not records:
        flash("Tidak ada data riwayat untuk diekspor.", "warning")
        qs = request.query_string.decode("utf-8")
        target = url_for("classification.history")
        if qs:
            target = f"{target}?{qs}"
        return redirect(target)

    payload = serialize_records_for_export(records, arch_label, mode_label)

    if fmt == "json":
        response = make_response(json.dumps(payload, indent=2, ensure_ascii=False))
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        response.headers["Content-Disposition"] = "attachment; filename=classification_history.json"
        return response

    output = io.StringIO()
    fieldnames = [
        "id",
        "uploaded_filename",
        "filename",
        "expected_class",
        "predicted_class",
        "confidence_score",
        "probability_diwani",
        "probability_diwani_jali",
        "probability_naskhi",
        "probability_tsuluts",
        "validation_status",
        "review_status",
        "input_status",
        "status",
        "khat_probability",
        "non_khat_probability",
        "confidence",
        "reliability_level",
        "model_architecture",
        "model_training_mode",
        "created_at",
        "confidence_band",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in payload:
        writer.writerow(row)

    response = make_response(output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = "attachment; filename=classification_history.csv"
    return response
