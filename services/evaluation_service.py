import csv
import json
import os
from datetime import datetime

import numpy as np
from flask import current_app

from services.metrics_service import ACCURACY_TARGET, target_status, weak_classes, training_recommendations


def _plot_dual_curves(train_data, val_data, title, ylabel, path, ylim_max=None, dpi=100):
    """Plot training and validation series on one chart."""
    import matplotlib.pyplot as plt

    if not train_data and not val_data:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=dpi)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    if train_data:
        ax.plot(
            list(range(1, len(train_data) + 1)),
            train_data,
            label="Training",
            linewidth=2.8,
            color="#0f766e",
            marker="o",
            markersize=4,
        )
    if val_data:
        ax.plot(
            list(range(1, len(val_data) + 1)),
            val_data,
            label="Validation",
            linewidth=2.8,
            color="#0891b2",
            marker="s",
            markersize=4,
        )
    ax.set_title(title, fontsize=13, fontweight="600", color="#0f172a", pad=14)
    ax.set_xlabel("Epoch", fontsize=11, color="#475569", labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, color="#475569", labelpad=8)
    if ylabel.lower() == "accuracy":
        ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.28, color="#cbd5e1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#64748b", labelsize=10)
    fig.tight_layout(pad=1.6)
    fig.savefig(path, dpi=dpi, facecolor="#ffffff", edgecolor="none")
    plt.close(fig)


def _plot_and_save(data, label, path, ylabel=None, dpi=100):
    import matplotlib.pyplot as plt

    if not data:
        return
    color = "#0f766e" if "Training" in label and "Validation" not in label else "#0891b2"
    if "Loss" in label:
        color = "#64748b"

    fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    epochs = list(range(1, len(data) + 1))
    ax.plot(epochs, data, label=label, linewidth=2.6, color=color, marker="o", markersize=4, alpha=0.95)
    ax.set_xlabel("Epoch", fontsize=11, color="#475569")
    ax.set_ylabel(ylabel or label, fontsize=11, color="#475569")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.28, linestyle="-", linewidth=0.7, color="#cbd5e1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#64748b", labelsize=9)
    fig.tight_layout(pad=1.4)
    fig.savefig(path, dpi=dpi, facecolor="#ffffff", edgecolor="none")
    plt.close(fig)


def _plot_per_class_metric(labels, values, metric_label, title, path, dpi=100):
    """Bar chart for one metric (precision or recall) across classes."""
    import matplotlib.pyplot as plt
    from services.evaluation_ui_service import CLASS_CHART_COLORS, display_class, format_metric_pct

    if not labels or not values:
        return

    display_labels = [display_class(label) for label in labels]
    bar_colors = [CLASS_CHART_COLORS.get(label, "#0f766e") for label in labels]

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=dpi)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    x_pos = list(range(len(labels)))
    bars = ax.bar(x_pos, values, color=bar_colors, width=0.62, edgecolor="#ffffff", linewidth=1.2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_labels, rotation=28, ha="right", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(metric_label, fontsize=11, color="#475569", labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="600", color="#0f172a", pad=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val * 100:.0f}%"))
    ax.grid(True, axis="y", alpha=0.28, color="#cbd5e1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#64748b", labelsize=9)

    for bar, value in zip(bars, values):
        if value is None:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            format_metric_pct(value),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#334155",
            fontweight="600",
        )

    fig.tight_layout(pad=1.6)
    fig.savefig(path, dpi=dpi, facecolor="#ffffff", edgecolor="none")
    plt.close(fig)


def _plot_per_class_comparison(labels, per_class, path, dpi=100):
    """Grouped bar chart comparing precision, recall, and F1-score per class."""
    import matplotlib.pyplot as plt
    import numpy as np
    from services.evaluation_ui_service import display_class, format_metric_pct

    if not labels or not per_class:
        return

    precision_vals = []
    recall_vals = []
    f1_vals = []
    for label in labels:
        row = per_class.get(label, {})
        precision_vals.append(float(row.get("precision", 0) or 0))
        recall_vals.append(float(row.get("recall", 0) or 0))
        f1_key = "f1_score" if "f1_score" in row else "f1-score"
        f1_vals.append(float(row.get(f1_key, 0) or 0))

    display_labels = [display_class(label) for label in labels]
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    bars_p = ax.bar(x - width, precision_vals, width, label="Precision", color="#0f766e", edgecolor="#ffffff")
    bars_r = ax.bar(x, recall_vals, width, label="Recall", color="#0891b2", edgecolor="#ffffff")
    bars_f = ax.bar(x + width, f1_vals, width, label="F1-Score", color="#7c3aed", edgecolor="#ffffff")

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=28, ha="right", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Skor", fontsize=11, color="#475569", labelpad=8)
    ax.set_title(
        "Perbandingan Precision, Recall, dan F1-Score per Kelas",
        fontsize=13,
        fontweight="600",
        color="#0f172a",
        pad=14,
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val * 100:.0f}%"))
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.28, color="#cbd5e1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#64748b", labelsize=9)

    for bars in (bars_p, bars_r, bars_f):
        for bar in bars:
            height = bar.get_height()
            if height <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                format_metric_pct(height),
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#334155",
            )

    fig.tight_layout(pad=1.6)
    fig.savefig(path, dpi=dpi, facecolor="#ffffff", edgecolor="none")
    plt.close(fig)


def generate_evaluation_charts(
    eval_dir,
    *,
    history_file=None,
    labels=None,
    per_class=None,
    classification_report=None,
    plot_dpi=100,
):
    """Generate evaluation chart PNGs from saved training history and per-class metrics."""
    import matplotlib
    matplotlib.use("Agg")

    os.makedirs(eval_dir, exist_ok=True)

    hist_series = {}
    if isinstance(history_file, dict):
        hist_series = history_file.get("history", history_file)
        if not isinstance(hist_series, dict):
            hist_series = {}

    if hist_series:
        train_acc = hist_series.get("accuracy", [])
        val_acc = hist_series.get("val_accuracy", [])
        train_loss = hist_series.get("loss", [])
        val_loss = hist_series.get("val_loss", [])
        _plot_dual_curves(
            train_acc,
            val_acc,
            "Model Accuracy",
            "Accuracy",
            os.path.join(eval_dir, "accuracy_curves.png"),
            dpi=plot_dpi,
        )
        _plot_dual_curves(
            train_loss,
            val_loss,
            "Model Loss",
            "Loss",
            os.path.join(eval_dir, "loss_curves.png"),
            dpi=plot_dpi,
        )
        _plot_and_save(
            train_acc,
            "Training Accuracy",
            os.path.join(eval_dir, "training_accuracy.png"),
            "Accuracy",
            dpi=plot_dpi,
        )
        _plot_and_save(
            val_acc,
            "Validation Accuracy",
            os.path.join(eval_dir, "validation_accuracy.png"),
            "Accuracy",
            dpi=plot_dpi,
        )
        _plot_and_save(
            train_loss,
            "Training Loss",
            os.path.join(eval_dir, "training_loss.png"),
            "Loss",
            dpi=plot_dpi,
        )
        _plot_and_save(
            val_loss,
            "Validation Loss",
            os.path.join(eval_dir, "validation_loss.png"),
            "Loss",
            dpi=plot_dpi,
        )

    effective_labels = list(labels or [])
    effective_per_class = dict(per_class or {})
    if classification_report and effective_labels:
        for label in effective_labels:
            if label in effective_per_class:
                continue
            row = classification_report.get(label, {})
            if not row:
                continue
            effective_per_class[label] = {
                "precision": row.get("precision", 0),
                "recall": row.get("recall", 0),
                "f1_score": row.get("f1-score", row.get("f1_score", 0)),
            }

    if effective_per_class:
        from services.evaluation_ui_service import resolve_per_class_chart_order

        chart_labels = resolve_per_class_chart_order(effective_per_class, effective_labels)
        precision_vals = [
            float(effective_per_class.get(label, {}).get("precision", 0) or 0)
            for label in chart_labels
        ]
        recall_vals = [
            float(effective_per_class.get(label, {}).get("recall", 0) or 0)
            for label in chart_labels
        ]
        f1_vals = []
        for label in chart_labels:
            row = effective_per_class.get(label, {})
            f1_key = "f1_score" if "f1_score" in row else "f1-score"
            f1_vals.append(float(row.get(f1_key, 0) or 0))
        _plot_per_class_metric(
            chart_labels,
            precision_vals,
            "Precision",
            "Precision per Kelas",
            os.path.join(eval_dir, "precision_per_class.png"),
            dpi=plot_dpi,
        )
        _plot_per_class_metric(
            chart_labels,
            recall_vals,
            "Recall",
            "Recall per Kelas",
            os.path.join(eval_dir, "recall_per_class.png"),
            dpi=plot_dpi,
        )
        _plot_per_class_metric(
            chart_labels,
            f1_vals,
            "F1-Score",
            "F1-Score per Kelas",
            os.path.join(eval_dir, "f1_per_class.png"),
            dpi=plot_dpi,
        )
        _plot_per_class_comparison(
            chart_labels,
            effective_per_class,
            os.path.join(eval_dir, "per_class_metrics_comparison.png"),
            dpi=plot_dpi,
        )


def _plot_confusion_matrix(cm, labels, path, dpi=100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    cm_arr = np.array(cm, dtype=int)
    n = cm_arr.shape[0]
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=dpi)
    fig.patch.set_facecolor("#ffffff")

    display_labels = [lbl.replace("_", " ").title() for lbl in labels]
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="YlGnBu")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count", fontsize=10, color="#475569")
    cbar.ax.tick_params(labelsize=9)

    thresh = cm_arr.max() / 2.0 if cm_arr.size else 0
    for i in range(n):
        for j in range(n):
            val = int(cm_arr[i, j])
            is_diag = i == j
            if val == 0:
                label = "—"
                text_color = "#cbd5e1"
                fontweight = "normal"
                fontsize = 12
            else:
                label = str(val)
                text_color = "white" if val > thresh else "#0f172a"
                fontweight = "bold" if is_diag else "normal"
                fontsize = 15 if is_diag else 13
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=fontsize,
                fontweight=fontweight,
            )
        rect = Rectangle(
            (i - 0.48, i - 0.48),
            0.96,
            0.96,
            fill=False,
            edgecolor="#0f766e",
            linewidth=2.8,
        )
        ax.add_patch(rect)

    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(display_labels, rotation=38, ha="right", fontsize=11)
    ax.set_yticklabels(display_labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, color="#334155", labelpad=12)
    ax.set_ylabel("True Label", fontsize=12, color="#334155", labelpad=12)
    ax.set_title("Confusion Matrix (Test Set)", fontsize=14, fontweight="700", color="#0f172a", pad=16)
    fig.tight_layout(pad=2.0)
    fig.savefig(path, dpi=dpi, facecolor="#ffffff", edgecolor="none")
    plt.close(fig)


def evaluate_model():
    from services.teachable_machine_service import is_teachable_machine_available
    from services.external_assets_service import should_use_keras_model

    if is_teachable_machine_available() and not should_use_keras_model():
        from services.evaluation_tm_service import evaluate_tm_model
        return evaluate_tm_model()

    from services.evaluation_pipeline_service import (
        evaluate_with_production_pipeline,
        should_use_production_eval_pipeline,
    )

    if should_use_production_eval_pipeline():
        return evaluate_with_production_pipeline()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from services.dataset_readiness_service import is_valid_trained_model
    from services.evaluation_dataset_service import (
        build_dataset_summary,
        iter_test_images,
        resolve_evaluation_test_dir,
        validate_test_holdout_before_eval,
    )
    from services.evaluation_metrics_service import compute_classification_metrics
    from services.model_context_service import get_active_model_path, get_model_context
    from services.preprocessing_service import get_architecture_preprocess_fn, preprocess_pil_image
    from services.model_builder_service import load_trained_classifier
    from services.training_utils import get_keras, save_json
    from services.class_mapping_service import validate_class_mapping

    model_path = get_active_model_path()
    if not is_valid_trained_model(model_path):
        raise FileNotFoundError("Model file not found. Train model first.")

    ctx = get_model_context()
    architecture = ctx.get("model_architecture_key", "efficientnetb0")
    preprocess_input = get_architecture_preprocess_fn(architecture)

    class_indices_path = current_app.config.get("CLASS_INDICES_PATH")
    eval_class_labels = list(current_app.config["CLASS_LABELS"])
    if class_indices_path and os.path.isfile(class_indices_path):
        try:
            with open(class_indices_path, "r", encoding="utf-8") as handle:
                indices_data = json.load(handle)
            saved = indices_data.get("class_indices", indices_data)
            if isinstance(saved, dict) and saved:
                eval_class_labels = sorted(saved.keys(), key=lambda key: int(saved[key]))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    num_classes = len(eval_class_labels)

    validate_class_mapping(
        config=current_app.config,
        raise_on_mismatch=not current_app.config.get("USE_EXTERNAL_MODEL"),
    )

    get_keras()
    from PIL import Image
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    cfg = current_app.config
    batch_size = int(cfg.get("EVAL_BATCH_SIZE", 64))
    plot_dpi = int(cfg.get("EVAL_PLOT_DPI", 100))
    use_tta = bool(cfg.get("EVAL_USE_TTA", False))
    use_ensemble = bool(cfg.get("EVAL_USE_ENSEMBLE", False))
    defer_previews = bool(cfg.get("EVAL_DEFER_PREVIEWS", True))

    def _pad_preprocess(img_array):
        pil = Image.fromarray(img_array.astype("uint8"))
        if architecture in ("keras_h5", "teachable_machine_h5", "external_h5"):
            from PIL import ImageOps
            rgb = ImageOps.exif_transpose(pil).convert("RGB")
            array = np.array(rgb.resize((224, 224)), dtype=np.float32)
            return (array / 127.5) - 1.0
        processed = preprocess_pil_image(pil, (224, 224))
        return preprocess_input(np.array(processed, dtype=np.float32))

    eval_gen = ImageDataGenerator(preprocessing_function=_pad_preprocess)
    test_dir, test_source = resolve_evaluation_test_dir(cfg)
    test_data = eval_gen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        class_mode="categorical",
        classes=eval_class_labels,
        batch_size=batch_size,
        shuffle=False,
    )
    if test_data.samples == 0:
        raise ValueError("Test dataset is empty. Split dataset before evaluation.")

    is_keras_tm = architecture in ("keras_h5", "teachable_machine_h5", "external_h5")

    if is_keras_tm:
        validate_test_holdout_before_eval([], cfg, test_count=test_data.samples)
        from collections import Counter

        inv_labels = {idx: name for name, idx in test_data.class_indices.items()}
        test_items = [
            (os.path.join(test_dir, name), inv_labels[idx])
            for name, idx in zip(test_data.filenames, test_data.classes)
        ]
        actual_counts = Counter(inv_labels[int(idx)] for idx in test_data.classes)
        dataset_summary = build_dataset_summary(
            test_dir,
            test_items,
            eval_class_labels,
            cfg,
            actual_counts=dict(actual_counts),
        )
    else:
        test_items = iter_test_images(test_dir, eval_class_labels)
        validate_test_holdout_before_eval(test_items, cfg)
        dataset_summary = build_dataset_summary(
            test_dir,
            test_items,
            eval_class_labels,
            cfg,
        )
    dataset_summary["test_source_key"] = test_source

    from services.calibration_service import (
        load_calibrator,
        apply_calibrator,
        predict_keras_test_probs,
        predict_architecture_test_probs,
        load_finetune_checkpoint_model,
    )
    from services.model_cache_service import get_cached_keras_model

    calibrator = load_calibrator(cfg)

    if is_keras_tm:
        raw_model = get_cached_keras_model(model_path)
        extra_models = []
        if use_ensemble:
            ensemble_member = load_finetune_checkpoint_model(cfg, raw_model)
            if ensemble_member is not None:
                extra_models.append(ensemble_member)
        if use_tta:
            probs = predict_keras_test_probs(
                raw_model,
                test_items,
                eval_class_labels,
                use_hflip_tta=True,
                extra_models=extra_models or None,
            )
        else:
            probs = raw_model.predict(test_data, verbose=0)
            if extra_models:
                test_data.reset()
                for extra in extra_models:
                    probs = probs + extra.predict(test_data, verbose=0)
                    test_data.reset()
                probs = probs / (1 + len(extra_models))
        y_true = test_data.classes
        labels = list(test_data.class_indices.keys())
        inv_labels = {idx: name for name, idx in test_data.class_indices.items()}
        filenames = test_data.filenames
    else:
        model = load_trained_classifier(model_path, architecture, num_classes)
        extra_models = []
        if use_ensemble:
            ensemble_member = load_finetune_checkpoint_model(cfg, model)
            if ensemble_member is not None:
                extra_models.append(ensemble_member)
        if use_tta:
            probs = predict_architecture_test_probs(
                model,
                test_items,
                eval_class_labels,
                architecture=architecture,
                use_hflip_tta=True,
                extra_models=extra_models or None,
            )
        else:
            probs = model.predict(test_data, verbose=0)
            if extra_models:
                test_data.reset()
                for extra in extra_models:
                    probs = probs + extra.predict(test_data, verbose=0)
                    test_data.reset()
                probs = probs / (1 + len(extra_models))
        y_true = test_data.classes
        labels = list(test_data.class_indices.keys())
        inv_labels = {idx: name for name, idx in test_data.class_indices.items()}
        filenames = test_data.filenames

    if calibrator is not None:
        probs = apply_calibrator(probs, calibrator, model_class_order=labels)
    y_pred = np.argmax(probs, axis=1)

    metrics = compute_classification_metrics(y_true, y_pred, labels)
    sample_count = int(len(y_true))
    accuracy = metrics["accuracy"]
    precision_w = metrics["precision"]
    recall_w = metrics["recall"]
    f1_w = metrics["f1_score"]
    precision_m = metrics["precision_macro"]
    recall_m = metrics["recall_macro"]
    f1_m = metrics["f1_macro"]
    cm = metrics["confusion_matrix"]
    report_dict = metrics["classification_report"]

    misclassified = []
    pair_counts = {}
    for i in range(len(y_true)):
        true_idx = int(y_true[i])
        pred_idx = int(y_pred[i])
        true_label = inv_labels.get(true_idx, labels[true_idx])
        pred_label = inv_labels.get(pred_idx, labels[pred_idx])
        prob_row = probs[i]
        sorted_idx = np.argsort(prob_row)[::-1]
        top_conf = float(prob_row[sorted_idx[0]]) * 100
        second_conf = float(prob_row[sorted_idx[1]]) * 100 if len(sorted_idx) > 1 else 0.0
        margin = round(top_conf - second_conf, 2)
        if true_idx != pred_idx:
            from services.evaluation_ui_service import classify_error_type

            error_meta = classify_error_type(top_conf, margin, true_label=true_label, predicted_label=pred_label)
            pair_key = (true_label, pred_label)
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
            from services.evaluation_image_service import attach_preview_metadata_only, attach_preview_to_record

            preview_fn = attach_preview_metadata_only if defer_previews else attach_preview_to_record
            misclassified.append(preview_fn({
                "image_path": os.path.join(test_dir, filenames[i]) if i < len(filenames) else "",
                "filename": filenames[i] if i < len(filenames) else "",
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": float(prob_row[pred_idx]),
                "confidence_pct": round(top_conf, 2),
                "top2_class": inv_labels.get(int(sorted_idx[1]), labels[int(sorted_idx[1])]) if len(sorted_idx) > 1 else None,
                "top2_probability": float(prob_row[sorted_idx[1]]) if len(sorted_idx) > 1 else None,
                "margin_pct": margin,
                "error_type": error_meta["key"],
                "error_type_label": error_meta["label_id"],
                "error_hint": error_meta["hint"],
                "is_high_confidence_error": error_meta["key"] == "high_confidence",
                "probability_distribution": {
                    inv_labels.get(j, labels[j]): round(float(prob_row[j]), 4) for j in range(len(prob_row))
                },
            }))

    confused_pairs = [
        {
            "true_class": k[0],
            "predicted_class": k[1],
            "count": v,
            "percentage": round(v / len(misclassified) * 100, 2) if misclassified else 0,
            "example_images": [m["image_path"] for m in misclassified if m["true_label"] == k[0] and m["predicted_label"] == k[1]][:5],
        }
        for k, v in sorted(pair_counts.items(), key=lambda x: -x[1])
    ]

    history_file = {}
    if os.path.exists(current_app.config["TRAINING_HISTORY_PATH"]):
        with open(current_app.config["TRAINING_HISTORY_PATH"], "r", encoding="utf-8") as f:
            history_file = json.load(f)

    hist_series = history_file.get("history", history_file) if isinstance(history_file, dict) else {}
    if not isinstance(hist_series, dict):
        hist_series = {}

    eval_dir = current_app.config["EVALUATION_FOLDER"]
    os.makedirs(eval_dir, exist_ok=True)

    generate_evaluation_charts(
        eval_dir,
        history_file=history_file,
        labels=labels,
        per_class=metrics["per_class"],
        classification_report=report_dict,
        plot_dpi=plot_dpi,
    )

    _plot_confusion_matrix(cm, labels, os.path.join(eval_dir, "confusion_matrix.png"), dpi=plot_dpi)

    csv_path = os.path.join(eval_dir, "classification_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["class", "precision", "recall", "f1-score", "support", "class_accuracy"])
        for label in labels:
            row = report_dict.get(label, {})
            writer.writerow([
                label,
                row.get("precision", ""),
                row.get("recall", ""),
                row.get("f1-score", ""),
                row.get("support", ""),
                metrics["per_class_accuracy"].get(label, ""),
            ])
        for avg_key in ("macro avg", "weighted avg"):
            row = report_dict.get(avg_key, {})
            writer.writerow([
                avg_key,
                row.get("precision", ""),
                row.get("recall", ""),
                row.get("f1-score", ""),
                row.get("support", ""),
            ])

    weak = weak_classes(report_dict)
    target = target_status(accuracy)
    recommendations = training_recommendations(
        history_file.get("final_validation_accuracy")
        or history_file.get("config", {}).get("best_validation_accuracy")
        or (hist_series.get("val_accuracy") or [None])[-1],
        accuracy,
        history_file.get("config", {}).get("training_mode_key"),
    )
    if dataset_summary.get("warnings"):
        recommendations = list(dataset_summary["warnings"][:2]) + list(recommendations or [])

    result = {
        "accuracy": float(accuracy),
        "precision": float(precision_w),
        "recall": float(recall_w),
        "f1_score": float(f1_w),
        "precision_macro": float(precision_m),
        "recall_macro": float(recall_m),
        "f1_macro": float(f1_m),
        "cohen_kappa": metrics["cohen_kappa"],
        "confusion_matrix": cm,
        "confusion_matrix_normalized": metrics["confusion_matrix_normalized"],
        "classification_report": report_dict,
        "history": history_file,
        "evaluated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "test_samples": sample_count,
        "accuracy_target": ACCURACY_TARGET,
        "target_achieved": target["achieved"],
        "target_status": target,
        "weak_classes": weak,
        "recommendations": recommendations,
        "per_class": metrics["per_class"],
        "per_class_accuracy": metrics["per_class_accuracy"],
        "misclassified_count": len(misclassified),
        "misclassified_images": misclassified[:500],
        "confused_pairs": confused_pairs,
        "correct_predictions": sample_count - len(misclassified),
        "incorrect_predictions": len(misclassified),
        "evaluation_split": "test_holdout",
        "dataset_summary": dataset_summary,
        "active_engine": "keras_h5" if architecture == "keras_h5" else "keras",
        "model_architecture": ctx.get("model_architecture") or architecture,
        "model_path": model_path,
        "invalid_predictions": metrics["invalid_predictions"],
    }
    with open(current_app.config["EVALUATION_RESULT_PATH"], "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    save_json(current_app.config["MISCLASSIFICATION_REPORT_PATH"], {
        "generated_at": result["evaluated_at"],
        "split": "test",
        "total": sample_count,
        "misclassified_count": len(misclassified),
        "misclassified_images": misclassified[:500],
    })
    save_json(current_app.config["CONFUSION_PAIR_REPORT_PATH"], {
        "generated_at": result["evaluated_at"],
        "split": "test",
        "pairs": confused_pairs,
    })
    return result
