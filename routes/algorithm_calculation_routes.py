"""Standalone Perhitungan Algoritma documentation page."""

from flask import Blueprint, Response, current_app, render_template, url_for

from routes.utils import login_required
from services.algorithm_calculation_page_service import build_algorithm_calculation_page_context

algorithm_calculation_bp = Blueprint("algorithm_calculation", __name__)


@algorithm_calculation_bp.route("/algorithm-calculation")
@algorithm_calculation_bp.route("/perhitungan-algoritma")
@login_required
def algorithm_calculation_page():
    context = build_algorithm_calculation_page_context(current_app.config)
    context["report_url"] = url_for("report.report_page")
    return render_template("algorithm_calculation.html", **context)


@algorithm_calculation_bp.route("/algorithm-calculation/download")
@login_required
def download_algorithm_explanation():
    context = build_algorithm_calculation_page_context(current_app.config)
    model = context["model_info"]
    eval_sum = context["evaluation_summary"]
    sample = context["latest_prediction"]
    lines = [
        "PERHITUNGAN ALGORITMA — Arabic Khat AI",
        "=" * 50,
        "",
        "Ringkasan Model",
        f"- Sumber: {model.get('source')}",
        f"- Runtime: {model.get('runtime')}",
        f"- Input: {model.get('input_size')}",
        f"- Kelas: {', '.join(model.get('labels') or [])}",
        "",
        "Rumus Utama",
        "- Probabilitas (%) = Softmax Score × 100",
        "- Confidence = Top-1 Probability",
        "- Top-2 Margin = Top-1 Score − Top-2 Score",
        "- Predicted Class = argmax(probability scores)",
        "",
        "Threshold Confidence",
        "- ≥ 85%: Strong Prediction",
        "- 70%–84.99%: Moderate Prediction",
        "- 60%–69.99%: Weak Prediction",
        "- < 60%: Low Confidence",
        "",
        "Validasi Khat / Non-Khat (Tahap 1)",
        f"- Diterima sebagai Khat: Khat Probability ≥ {context['thresholds']['khat_accept']}%",
        f"- Review manual: {context['thresholds']['khat_reject']}% – {context['thresholds']['khat_accept'] - 0.01:.2f}%",
        f"- Ditolak Non-Khat: Khat Probability < {context['thresholds']['khat_reject']}%",
        "",
    ]
    if eval_sum.get("available"):
        ds = context.get("dataset_summary") or {}
        sm = ds.get("split_math") or {}
        lines.extend(
            [
                "Evaluasi Model (Test Holdout)",
                f"- Akurasi: {eval_sum.get('accuracy_pct')}%",
                f"- Presisi (weighted): {eval_sum.get('precision_pct')}%",
                f"- Recall (weighted): {eval_sum.get('recall_pct')}%",
                f"- F1-Score (weighted): {eval_sum.get('f1_pct')}%",
                f"- F1-Score (macro): {eval_sum.get('f1_macro_pct')}%",
                f"- Total uji: {eval_sum.get('test_samples')}",
                f"- Benar: {eval_sum.get('correct_predictions')} · Salah: {eval_sum.get('incorrect_predictions')}",
                "",
                "Pembagian Dataset",
                f"- Total diproses: {ds.get('processed_total') or sm.get('processed')}",
                f"- Train: {ds.get('train_count') or sm.get('train_count')}",
                f"- Validation: {ds.get('validation_count') or sm.get('validation_count')}",
                f"- Test holdout: {ds.get('test_count') or sm.get('test_count')}",
                "",
            ]
        )
    else:
        lines.append("Evaluasi model belum tersedia. Jalankan evaluasi terlebih dahulu.\n")

    lines.extend(
        [
            f"Contoh Prediksi ({sample.get('sample_label', 'Contoh')})",
            f"- File: {sample.get('filename')}",
            f"- Prediksi: {sample.get('predicted_display')}",
            f"- Confidence: {sample.get('confidence_pct')}%",
            f"- Top-2 Margin: {sample.get('margin_pct')}%",
            f"- Keputusan: {sample.get('final_decision')}",
            "",
            "Interpretasi Akademik",
            (
                f"Sistem klasifikasi Khat Arab menggunakan model citra berbasis CNN "
                f"({model.get('architecture_label') or model.get('source')}). "
                f"Dataset {context['dataset_summary'].get('processed_total', '—')} citra "
                f"dibagi train/validation/test. Citra masukan diproses menjadi ukuran "
                f"224 × 224 piksel, kemudian model menghasilkan distribusi probabilitas "
                f"untuk empat kelas khat. Kelas dengan probabilitas tertinggi ditetapkan "
                f"sebagai prediksi utama. Evaluasi pada test holdout: akurasi "
                f"{eval_sum.get('accuracy_pct', '—')}%."
            ),
            "",
            "Dokumen ini dihasilkan otomatis dari Arabic Khat AI.",
        ]
    )
    body = "\n".join(lines)
    return Response(
        body,
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="perhitungan-algoritma.txt"'},
    )
