"""CNN VGG16 Transfer Learning calculation page — full dataset & all classes."""

from flask import Blueprint, Response, current_app, render_template, url_for

from routes.utils import login_required
from services.cnn_calculation_page_service import build_cnn_calculation_page_context

cnn_calculation_bp = Blueprint("cnn_calculation", __name__)


@cnn_calculation_bp.route("/perhitungan-cnn-vgg16")
@cnn_calculation_bp.route("/cnn-calculation")
@login_required
def cnn_calculation_page():
    context = build_cnn_calculation_page_context(current_app.config)
    context["algo_doc_url"] = url_for("algorithm_calculation.algorithm_calculation_page")
    context["model_management_url"] = url_for("model_management.model_management_page")
    context["dataset_url"] = url_for("dataset.list_dataset")
    context["evaluation_url"] = url_for("evaluation.evaluation_page")
    return render_template("cnn_calculation.html", **context)


@cnn_calculation_bp.route("/cnn-calculation/download")
@login_required
def download_cnn_calculation():
    ctx = build_cnn_calculation_page_context(current_app.config)
    totals = ctx["dataset_totals"]
    split_math = ctx.get("split_math") or {}
    training = ctx["training_summary"]
    lines = [
        "PERHITUNGAN CNN VGG16 TRANSFER LEARNING — Arabic Khat AI",
        "=" * 58,
        "",
        "Dataset (80% Latih / 20% Uji)",
        f"- Total dataset: {totals['processed'] or totals['raw']} gambar",
        f"- Data latih (80%): {split_math.get('train_data_count', totals['train'] + totals['validation'])} "
        f"(train {totals['train']} + val {totals['validation']})",
        f"- Data uji (20%): {totals['test']} gambar",
        "",
        "Arsitektur",
        f"- Model: {ctx['architecture_label']}",
        f"- Input: {ctx['input_size']}",
        f"- Kelas output: {ctx['num_classes']}",
        f"- Fine-tune layers (Fase 2): {ctx['fine_tune_layers']} layer terakhir VGG16",
        "",
        "Ringkasan Dataset (Semua Kelas)",
        f"- Raw: {totals['raw']} gambar",
        f"- Train: {totals['train']} · Validation: {totals['validation']} · Test: {totals['test']}",
        "",
        "Distribusi per Kelas",
    ]
    for row in ctx["class_rows"]:
        lines.append(
            f"- {row['display']}: raw={row['raw_count']}, train={row['train_count']}, "
            f"val={row['validation_count']}, test={row['test_count']}, "
            f"weight={row['class_weight']} ({row.get('weight_formula', '')})"
        )

    tm = ctx.get("training_math") or {}
    lines.extend(
        [
            "",
            "Perhitungan Iterasi Training",
            f"- Batch size: {tm.get('batch_size')}",
            f"- Steps/epoch train: {tm.get('steps_per_epoch_train')}",
            f"- {tm.get('formula_steps', '')}",
            f"- {tm.get('formula_updates', '')}",
        ]
    )

    eval_m = ctx.get("evaluation_metrics") or {}
    if eval_m.get("available"):
        lines.extend(
            [
                "",
                "Evaluasi Test Set",
                f"- {eval_m.get('accuracy_formula', '')}",
            ]
        )
        for erow in eval_m.get("per_class_rows") or []:
            lines.append(
                f"  · {erow['display']}: test={erow['test_count']}, "
                f"akurasi={erow.get('class_accuracy_pct') or '—'}%"
            )

    lines.extend(
        [
            "",
            "Fase Transfer Learning",
        ]
    )
    for phase in ctx["transfer_phases"]:
        lines.append(f"- Fase {phase['phase']}: {phase['title']} ({phase['epochs']} epoch, lr={phase['learning_rate']})")

    lines.extend(
        [
        "",
        "Rumus Utama",
        f"- {ctx['softmax_block']['formula']}",
        f"- {ctx['softmax_block']['percent']}",
        f"- Loss: {ctx['softmax_block']['loss']}",
        "- Prediksi: argmax(softmax output)",
        "- Preprocessing: vgg16.preprocess_input — mean BGR [103.939, 116.779, 123.68]",
        "- Class weight: w_i = N / (K × n_i)",
        "- Precision_i = TP_i / (TP_i + FP_i)",
        "- Recall_i = TP_i / (TP_i + FN_i)",
        "- F1_i = 2 × P_i × R_i / (P_i + R_i)",
        "",
            "Ringkasan Training",
            f"- Mode: {training.get('training_mode')}",
            f"- Arsitektur terlatih: {training.get('trained_architecture_label')}",
            f"- Batch size: {training.get('batch_size')}",
            f"- Best val accuracy: {training.get('best_val_accuracy') or '—'}%",
            f"- Test accuracy: {training.get('test_accuracy') or '—'}%",
            "",
            "Dokumen dihasilkan otomatis dari Arabic Khat AI.",
        ]
    )
    body = "\n".join(lines)
    return Response(
        body,
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="perhitungan-cnn-vgg16.txt"'},
    )
