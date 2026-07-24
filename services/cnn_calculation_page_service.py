"""Context builder for CNN VGG16 Transfer Learning calculation page (full dataset & classes)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from flask import current_app

from models import GambarDataset, KelasKhat
from services.dataset_readiness_service import count_images_per_class, get_dataset_readiness
from services.dataset_service import class_distribution
from services.evaluation_ui_service import display_class
from services.metrics_service import ACCURACY_TARGET
from services.model_builder_service import ARCHITECTURES, FINE_TUNE_LAYERS
from services.model_context_service import load_model_metadata, load_training_config


def _load_json(path: Optional[str]) -> Dict:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _compute_balanced_weights(counts: Dict[str, int], labels: List[str]) -> Dict[str, float]:
    """sklearn balanced class weight per slug."""
    try:
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
    except ImportError:
        return {label: 1.0 for label in labels}

    pairs = [(label, counts.get(label, 0)) for label in labels if counts.get(label, 0) > 0]
    if not pairs:
        return {label: 1.0 for label in labels}

    slugs, values = zip(*pairs)
    classes = np.arange(len(slugs))
    labels_idx = []
    for idx, count in enumerate(values):
        labels_idx.extend([idx] * count)
    if not labels_idx:
        return {label: 1.0 for label in labels}

    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.array(labels_idx))
    result = {label: 1.0 for label in labels}
    for slug, weight in zip(slugs, weights):
        result[slug] = round(float(weight), 4)
    return result


def _split_counts(config: Dict, labels: List[str]) -> Dict[str, Dict[str, int]]:
    splits = {}
    for key, dir_key in (
        ("train", "TRAIN_DIR"),
        ("validation", "VALIDATION_DIR"),
        ("test", "TEST_DIR"),
        ("raw", "RAW_DATASET_DIR"),
    ):
        root = config.get(dir_key, "")
        splits[key] = count_images_per_class(root, labels) if root else {label: 0 for label in labels}
    return splits


def _load_split_report(config: Dict) -> Dict[str, Any]:
    return _load_json(config.get("DATASET_SPLIT_REPORT_PATH"))


def build_split_math_summary(dataset_totals: Dict[str, int]) -> Dict[str, Any]:
    raw = int(dataset_totals.get("raw") or 0)
    train = int(dataset_totals.get("train") or 0)
    val = int(dataset_totals.get("validation") or 0)
    test = int(dataset_totals.get("test") or 0)
    processed = int(dataset_totals.get("processed") or 0) or (train + val + test)
    train_val = train + val
    k = 4  # classes — informational

    train_data_pct = round(train_val / processed * 100, 1) if processed else 0.0
    test_data_pct = round(test / processed * 100, 1) if processed else 0.0
    train_share_of_latih = round(train / train_val * 100, 1) if train_val else 0.0
    val_share_of_latih = round(val / train_val * 100, 1) if train_val else 0.0

    return {
        "raw": raw,
        "processed": processed,
        "train_count": train,
        "validation_count": val,
        "test_count": test,
        "train_data_count": train_val,
        "train_data_pct": train_data_pct,
        "test_data_count": test,
        "test_data_pct": test_data_pct,
        "cards": [
            {"label": "Total Dataset (N)", "value": processed or raw, "desc": "Seluruh gambar setelah import"},
            {"label": "Data Latih (80%)", "value": train_val, "desc": f"Train + validation = {train} + {val}"},
            {"label": "Data Uji (20%)", "value": test, "desc": "Holdout test — tidak dipakai training"},
        ],
        "formulas": [
            {
                "title": "Pembagian utama 80 / 20",
                "expression": f"N_test = 0,20 × N = 0,20 × {processed} ≈ {test}",
                "detail": f"Data uji (holdout) = {test} gambar ({test_data_pct}%). Tidak pernah dipakai saat training.",
            },
            {
                "title": "Data latih (80%)",
                "expression": f"N_latih = N − N_test = {processed} − {test} = {train_val}",
                "detail": f"Digunakan untuk training: {train_data_pct}% dari total dataset.",
            },
            {
                "title": "Sub-split internal data latih",
                "expression": (
                    f"N_train = {train} ({train_share_of_latih}% dari latih) · "
                    f"N_val = {val} ({val_share_of_latih}% dari latih)"
                ),
                "detail": (
                    "Dari 80% data latih: ±85% menjadi folder train dan ±15% menjadi validation "
                    "(stratified per kelas). Validation dipakai memantau overfitting, bukan untuk evaluasi final."
                ),
            },
            {
                "title": "Stratified per kelas",
                "expression": "n_test,i / N_test ≈ n_train,i / N_train ≈ proporsi kelas i",
                "detail": f"Setiap kelas ({k} kelas) di-split dengan rasio yang sama agar distribusi kelas terjaga.",
            },
        ],
    }


def build_vgg16_spatial_dimensions() -> List[Dict[str, str]]:
    """Output spatial size after each VGG16 block (input 224×224)."""
    return [
        {
            "block": "Input",
            "operation": "RGB citra kaligrafi",
            "formula": "H × W = 224 × 224",
            "output": "224 × 224 × 3",
        },
        {
            "block": "Conv1 + Conv2 + MaxPool",
            "operation": "2× Conv3×3 (64 filter) + MaxPool 2×2",
            "formula": "224 → Conv → 224 → Conv → 224 → Pool → 112",
            "output": "112 × 112 × 64",
        },
        {
            "block": "Conv3 + Conv4 + MaxPool",
            "operation": "2× Conv3×3 (128 filter) + MaxPool",
            "formula": "112 → Pool → 56",
            "output": "56 × 56 × 128",
        },
        {
            "block": "Conv5–7 + MaxPool",
            "operation": "3× Conv3×3 (256 filter) + MaxPool",
            "formula": "56 → Pool → 28",
            "output": "28 × 28 × 256",
        },
        {
            "block": "Conv8–10 + MaxPool",
            "operation": "3× Conv3×3 (512 filter) + MaxPool",
            "formula": "28 → Pool → 14",
            "output": "14 × 14 × 512",
        },
        {
            "block": "Conv11–13 + MaxPool",
            "operation": "3× Conv3×3 (512 filter) + MaxPool",
            "formula": "14 → Pool → 7",
            "output": "7 × 7 × 512",
        },
        {
            "block": "GlobalAveragePooling2D",
            "operation": "Rata-rata per channel",
            "formula": "GAP(F) = (1/49) Σ F_{i,j,c} → vektor 512",
            "output": "512",
        },
    ]


def build_vgg16_parameter_breakdown(num_classes: int) -> Dict[str, Any]:
    """Approximate trainable vs frozen parameter counts for VGG16 head."""
    gap_dim = 512
    dense1 = gap_dim * 512 + 512
    dense2 = 512 * 256 + 256
    dense_out = 256 * num_classes + num_classes
    head_trainable = dense1 + dense2 + dense_out
    backbone_total = 14_714_688
    fine_tune_layers = FINE_TUNE_LAYERS.get("vgg16", 4)
    fine_tune_approx = int(backbone_total * fine_tune_layers / 13)

    return {
        "backbone_total": f"{backbone_total:,}".replace(",", "."),
        "head_trainable_phase1": f"{head_trainable:,}".replace(",", "."),
        "fine_tune_approx_phase2": f"{fine_tune_approx:,}".replace(",", "."),
        "phase1_frozen": f"{backbone_total:,}".replace(",", "."),
        "phase1_trainable": f"{head_trainable:,}".replace(",", "."),
        "output_params": dense_out,
        "rows": [
            {
                "component": "VGG16 Backbone (ImageNet)",
                "params": f"≈ {backbone_total / 1e6:.1f}M",
                "phase1": "Dibekukan (frozen)",
                "phase2": f"{fine_tune_layers} layer terakhir dibuka",
            },
            {
                "component": "GlobalAveragePooling2D + BatchNorm",
                "params": "≈ 2K",
                "phase1": "Trainable",
                "phase2": "Trainable",
            },
            {
                "component": "Dense(512) + Dropout(0.5)",
                "params": f"≈ {dense1 / 1000:.0f}K",
                "phase1": "Trainable",
                "phase2": "Trainable",
            },
            {
                "component": "Dense(256) + Dropout(0.3)",
                "params": f"≈ {dense2 / 1000:.0f}K",
                "phase1": "Trainable",
                "phase2": "Trainable",
            },
            {
                "component": f"Dense({num_classes}, Softmax)",
                "params": f"≈ {dense_out}",
                "phase1": "Trainable",
                "phase2": "Trainable",
            },
        ],
        "formulas": [
            "Params Dense = (n_in × n_out) + n_out",
            f"Dense(512): ({gap_dim} × 512) + 512 = {dense1:,}",
            f"Dense(256): (512 × 256) + 256 = {dense2:,}",
            f"Dense({num_classes}): (256 × {num_classes}) + {num_classes} = {dense_out:,}",
        ],
    }


def build_preprocessing_detail() -> Dict[str, Any]:
    return {
        "input_size": "224 × 224 × 3",
        "steps": [
            {
                "step": "1",
                "title": "Resize & padding",
                "formula": "Citra → 224×224 RGB (aspect ratio dipertahankan dengan padding)",
                "detail": "Dataset train/val/test disimpan sebagai JPEG 224×224 padded.",
            },
            {
                "step": "2",
                "title": "Channel order BGR",
                "formula": "X_BGR = X_RGB[..., ::-1]",
                "detail": "VGG16 ImageNet menggunakan urutan channel Blue-Green-Red.",
            },
            {
                "step": "3",
                "title": "Mean subtraction (ImageNet)",
                "formula": "X' = X_BGR − μ,  μ = [103.939, 116.779, 123.68]",
                "detail": "Nilai mean ImageNet dikurangkan per channel setelah konversi BGR.",
            },
        ],
        "note": "Preprocessing identik pada training dan inferensi agar distribusi input konsisten.",
    }


def build_loss_worked_example(labels: List[str]) -> Dict[str, Any]:
    """Illustrative one-hot crossentropy for predicted class index 0."""
    import math

    display = [display_class(slug) for slug in labels]
    y_true = [1.0] + [0.0] * (len(labels) - 1)
    probs = [0.72, 0.12, 0.08, 0.08][: len(labels)]
    terms = [y_true[i] * math.log(max(probs[i], 1e-12)) for i in range(len(labels))]
    loss = -sum(terms)
    rows = [
        {
            "class": display[i],
            "y_true": y_true[i],
            "y_hat": probs[i],
            "term": round(-terms[i], 4),
        }
        for i in range(len(labels))
    ]
    return {
        "rows": rows,
        "loss": round(loss, 4),
        "formula": "L = −Σ y_true,i · log(ŷ_i)",
        "calculation": f"L = −log({probs[0]}) ≈ {round(loss, 4)} (contoh kelas benar = {display[0]})",
        "weighted_note": "Dengan class weight: L_w = w_k · L, w_k = N / (K × n_k).",
    }


def build_metric_formula_reference(
    eval_metrics: Dict[str, Any],
    class_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    blocks = [
        {
            "name": "Akurasi (Accuracy)",
            "formula": "Accuracy = TP_total / N_test = Σ benar / N_test",
            "detail": "Proporsi prediksi benar dari seluruh gambar test holdout.",
        },
        {
            "name": "Presisi per kelas",
            "formula": "Precision_i = TP_i / (TP_i + FP_i)",
            "detail": "Dari semua prediksi kelas i, berapa persen benar.",
        },
        {
            "name": "Recall per kelas",
            "formula": "Recall_i = TP_i / (TP_i + FN_i)",
            "detail": "Dari semua sampel kelas i yang sebenarnya, berapa persen terdeteksi.",
        },
        {
            "name": "F1-Score per kelas",
            "formula": "F1_i = 2 × (Precision_i × Recall_i) / (Precision_i + Recall_i)",
            "detail": "Harmonic mean presisi dan recall — seimbang untuk kelas minoritas.",
        },
        {
            "name": "Akurasi kelas (Class Accuracy)",
            "formula": "ClassAcc_i = CM[i,i] / Support_i",
            "detail": "Diagonal confusion matrix dibagi support kelas i pada test set.",
        },
    ]
    if eval_metrics.get("available"):
        blocks[0]["example"] = eval_metrics.get("accuracy_formula")
    per_class_examples = []
    for row in eval_metrics.get("per_class_rows") or []:
        if row.get("correct") is not None and row.get("test_count"):
            per_class_examples.append(
                {
                    "class": row["display"],
                    "class_accuracy": (
                        f"ClassAcc = {row['correct']} / {row['test_count']} "
                        f"= {row.get('class_accuracy_pct')}%"
                    ),
                    "f1": f"F1 = {row.get('f1_pct')}%" if row.get("f1_pct") is not None else None,
                }
            )
    return {"blocks": blocks, "per_class_examples": per_class_examples}


def build_adam_and_update_formulas(training_math: Dict[str, Any]) -> List[Dict[str, str]]:
    lr1 = "1e-4"
    lr2 = "1e-5"
    return [
        {
            "title": "Mini-batch Gradient Descent",
            "formula": "θ ← θ − η · ∇_θ L_batch",
            "detail": f"Satu update bobot per batch ({training_math.get('batch_size', 32)} citra).",
        },
        {
            "title": "Optimizer Adam (Fase 1)",
            "formula": f"Adam(lr={lr1}) — adaptive learning rate per parameter",
            "detail": "Menggabungkan momentum dan RMSprop untuk konvergensi stabil pada head classifier.",
        },
        {
            "title": "Optimizer Adam (Fase 2)",
            "formula": f"Adam(lr={lr2}) — learning rate lebih kecil saat fine-tune backbone",
            "detail": "Mencegah perubahan drastis pada bobot ImageNet yang sudah terlatih.",
        },
        {
            "title": "Steps per epoch",
            "formula": training_math.get("formula_steps", "steps = ⌈N_train / batch⌉"),
            "detail": training_math.get("samples_per_epoch", ""),
        },
        {
            "title": "Total update bobot",
            "formula": training_math.get("formula_updates", ""),
            "detail": (
                f"Fase 1: {training_math.get('phase1_epochs')} epoch × "
                f"{training_math.get('steps_per_epoch_train')} steps = "
                f"{training_math.get('weight_updates_phase1')} update · "
                f"Fase 2: {training_math.get('phase2_epochs')} epoch × "
                f"{training_math.get('steps_per_epoch_train')} steps = "
                f"{training_math.get('weight_updates_phase2')} update"
            ),
        },
    ]


def build_transfer_learning_concept() -> List[Dict[str, str]]:
    return [
        {
            "term": "Transfer Learning",
            "definition": (
                "Memanfaatkan bobot VGG16 yang sudah dilatih pada ImageNet (1,2 juta citra, 1000 kelas) "
                "sebagai feature extractor, lalu melatih head baru untuk 4 kelas kaligrafi Arab."
            ),
        },
        {
            "term": "Feature Extraction (Fase 1)",
            "definition": (
                "Backbone dibekukan — hanya Dense + Softmax yang belajar mapping fitur visual "
                "(tepi, tekstur, pola) ke kelas Khat."
            ),
        },
        {
            "term": "Fine-Tuning (Fase 2)",
            "definition": (
                "Beberapa layer konvolusi terakhir VGG16 dibuka dengan lr kecil agar filter "
                "menyesuaikan detail spesifik kaligrafi tanpa melupakan representasi umum."
            ),
        },
        {
            "term": "Categorical Crossentropy",
            "definition": (
                "Fungsi loss untuk klasifikasi multi-kelas one-hot: mengukur seberapa jauh "
                "distribusi softmax ŷ dari label y_true."
            ),
        },
    ]


def build_dataset_split_steps(
    class_rows: List[Dict[str, Any]],
    totals: Dict[str, int],
    split_report: Dict[str, Any],
    config: Optional[Dict] = None,
) -> List[Dict[str, str]]:
    if config is None:
        config = current_app.config

    dataset_source = config.get("EXTERNAL_DATASET_DIR") or config.get("RAW_DATASET_DIR") or config.get("DATASET_DIR") or "folder dataset"
    processed = totals.get("processed") or sum(
        r["train_count"] + r["validation_count"] + r["test_count"] for r in class_rows
    )
    train_n = totals.get("train") or 0
    val_n = totals.get("validation") or 0
    test_n = totals.get("test") or 0
    ratio = split_report.get("effective_ratio") or "68/12/20 train/validation/test"
    holdout = split_report.get("test_holdout_percent", 20)
    train_val = train_n + val_n
    train_data_pct = round(train_val / processed * 100, 1) if processed else 0
    test_pct = round(test_n / processed * 100, 1) if processed else 0

    return [
        {
            "step": "1",
            "title": "Kumpulkan Dataset Raw",
            "formula": f"N_raw = Σ kelas = {totals.get('raw', 0)} gambar",
            "detail": (
                "Semua citra kaligrafi dari folder sumber eksternal "
                f"({dataset_source}) "
                "dikelompokkan per kelas: naskhi, diwani, diwani_jali, tsuluts."
            ),
        },
        {
            "step": "2",
            "title": "Total Dataset Siap Split",
            "formula": f"N = {processed} gambar",
            "detail": (
                f"Total dataset yang dibagi: {processed} gambar. "
                + ("Deduplikasi aktif — " if split_report.get("deduplication_applied") else "")
                + "Setiap gambar masuk tepat satu folder split (train, validation, atau test)."
            ),
        },
        {
            "step": "3",
            "title": "Pembagian Utama 80% Latih / 20% Uji",
            "formula": f"N_uji = 0,20 × N = 0,20 × {processed} = {test_n} gambar",
            "detail": (
                f"Data uji (holdout) = {test_n} gambar ({test_pct}%). "
                f"Tidak pernah dipakai training maupun validasi. "
                f"Data latih = N − N_uji = {processed} − {test_n} = {train_val} ({train_data_pct}%)."
            ),
        },
        {
            "step": "4",
            "title": "Sub-split Data Latih → Train & Validation",
            "formula": f"N_train = {train_n} · N_val = {val_n} (dari {train_val} data latih)",
            "detail": (
                f"Dari {train_data_pct}% data latih: ±85% → train ({train_n}), ±15% → validation ({val_n}). "
                "Validation hanya untuk memantau overfitting selama epoch, bukan evaluasi final."
            ),
        },
        {
            "step": "5",
            "title": "Rasio Efektif & Stratified",
            "formula": f"{ratio} (train / validation / test dari N)",
            "detail": (
                f"Train = {train_n} ({round(train_n / processed * 100, 1) if processed else 0}%) · "
                f"Val = {val_n} ({round(val_n / processed * 100, 1) if processed else 0}%) · "
                f"Test = {test_n} ({test_pct}%). "
                "Stratified: proporsi setiap kelas dipertahankan di setiap split."
            ),
        },
    ]


def build_class_split_detail_rows(class_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for row in class_rows:
        total = row["total_split"] or 1
        train_pct = round(row["train_count"] / total * 100, 1)
        val_pct = round(row["validation_count"] / total * 100, 1)
        test_pct = round(row["test_count"] / total * 100, 1)
        n_train = row["train_count"]
        n_i = n_train or 1
        n_total = sum(r["train_count"] for r in class_rows) or 1
        k = len(class_rows)
        weight_manual = round(n_total / (k * n_i), 4) if n_train > 0 else None
        rows.append(
            {
                **row,
                "split_train_pct": train_pct,
                "split_val_pct": val_pct,
                "split_test_pct": test_pct,
                "weight_manual": weight_manual,
                "weight_formula": (
                    f"w_{row['slug']} = {n_total} / ({k} × {n_train}) = {weight_manual}"
                    if weight_manual is not None
                    else "—"
                ),
            }
        )
    return rows


def get_vgg16_tensor_pipeline(num_classes: int) -> List[Dict[str, str]]:
    """Output shape after each major stage (224×224 RGB input)."""
    return [
        {"stage": "Input", "operation": "Citra RGB dari dataset", "shape": "224 × 224 × 3", "params": "—"},
        {"stage": "Preprocess", "operation": "vgg16.preprocess_input (BGR, mean ImageNet)", "shape": "224 × 224 × 3", "params": "—"},
        {"stage": "Block 1–2", "operation": "Conv3×3 + ReLU + MaxPool", "shape": "56 × 56 × 128", "params": "~0.9M (ImageNet)"},
        {"stage": "Block 3–4", "operation": "Conv3×3 + ReLU + MaxPool", "shape": "28 × 28 × 256", "params": "~2.6M"},
        {"stage": "Block 5", "operation": "Conv3×3 + ReLU + MaxPool", "shape": "14 × 14 × 512", "params": "~7.1M"},
        {"stage": "Block 6–7", "operation": "Conv3×3 + ReLU + MaxPool", "shape": "7 × 7 × 512", "params": "~14.7M total backbone"},
        {"stage": "GAP", "operation": "GlobalAveragePooling2D", "shape": "512", "params": "0"},
        {"stage": "BN + Dense", "operation": "BatchNorm → Dense(512, ReLU) → Dropout(0.5)", "shape": "512", "params": "~262K"},
        {"stage": "Dense", "operation": "Dense(256, ReLU) → Dropout(0.3)", "shape": "256", "params": "~131K"},
        {"stage": "Output", "operation": f"Dense({num_classes}, Softmax)", "shape": str(num_classes), "params": f"~{256 * num_classes + num_classes}"},
    ]


def build_training_math(
    training_summary: Dict[str, Any],
    dataset_totals: Dict[str, int],
) -> Dict[str, Any]:
    batch_size = int(training_summary.get("batch_size") or 32)
    train_n = int(dataset_totals.get("train") or training_summary.get("train_count") or 0)
    val_n = int(dataset_totals.get("validation") or training_summary.get("validation_count") or 0)
    test_n = int(dataset_totals.get("test") or training_summary.get("test_count") or 0)
    phase1 = int(training_summary.get("phase1_epochs") or 15)
    phase2 = int(training_summary.get("phase2_epochs") or 10)

    import math

    steps_train = math.ceil(train_n / batch_size) if train_n and batch_size else 0
    steps_val = math.ceil(val_n / batch_size) if val_n and batch_size else 0
    updates_p1 = steps_train * phase1
    updates_p2 = steps_train * phase2

    return {
        "batch_size": batch_size,
        "train_samples": train_n,
        "validation_samples": val_n,
        "test_samples": test_n,
        "steps_per_epoch_train": steps_train,
        "steps_per_epoch_val": steps_val,
        "phase1_epochs": phase1,
        "phase2_epochs": phase2,
        "weight_updates_phase1": updates_p1,
        "weight_updates_phase2": updates_p2,
        "total_weight_updates": updates_p1 + updates_p2,
        "formula_steps": f"steps/epoch = ⌈N_train / batch⌉ = ⌈{train_n} / {batch_size}⌉ = {steps_train}",
        "formula_updates": (
            f"Total update bobot ≈ ({phase1} + {phase2}) × {steps_train} "
            f"= {updates_p1 + updates_p2} batch gradient descent"
        ),
        "samples_per_epoch": f"{steps_train} × {batch_size} ≈ {steps_train * batch_size} (dengan padding batch terakhir)",
    }


def build_worked_softmax_example(class_rows: List[Dict[str, Any]], labels: List[str]) -> Dict[str, Any]:
    """Numerical softmax example using illustrative logits for first class."""
    import math

    display = [display_class(slug) for slug in labels]
    logits = [2.1, 0.4, -0.3, 1.2][: len(labels)]
    exp_vals = [math.exp(z) for z in logits]
    denom = sum(exp_vals)
    probs = [round(e / denom, 4) for e in exp_vals]
    pred_idx = probs.index(max(probs))
    rows = [
        {
            "class": display[i],
            "slug": labels[i],
            "logit_z": logits[i],
            "exp_z": round(exp_vals[i], 4),
            "probability": probs[i],
            "percent": round(probs[i] * 100, 2),
            "is_predicted": i == pred_idx,
        }
        for i in range(len(labels))
    ]
    return {
        "rows": rows,
        "denominator": round(denom, 4),
        "formula": "P(kelas_i) = exp(z_i) / Σ exp(z_j)",
        "calculation": " + ".join(f"exp({z})" for z in logits) + f" = {round(denom, 2)}",
        "predicted_class": display[pred_idx],
        "note": "Contoh numerik ilustratif — logit aktual dihasilkan oleh Dense+Softmax pada inferensi.",
    }


def build_evaluation_metrics(config: Dict, class_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    eval_json = _load_json(config.get("EVALUATION_RESULT_PATH"))
    if not eval_json:
        return {"available": False}

    per_class = eval_json.get("per_class") or eval_json.get("classification_report") or {}
    cm = eval_json.get("confusion_matrix") or []
    rows = []
    for idx, row in enumerate(class_rows):
        slug = row["slug"]
        stats = per_class.get(slug) or {}
        if not stats or not isinstance(stats, dict):
            stats = {}
        cm_row = cm[idx] if idx < len(cm) else []
        correct = int(cm_row[idx]) if cm_row and idx < len(cm_row) else None
        support = int(stats.get("support") or row["test_count"] or 0)
        class_acc = round(correct / support * 100, 2) if correct is not None and support else None
        f1_val = stats.get("f1_score", stats.get("f1-score"))
        rows.append(
            {
                "display": row["display"],
                "slug": slug,
                "test_count": row["test_count"],
                "support_eval": support,
                "precision_pct": round(float(stats.get("precision", 0)) * 100, 2) if stats.get("precision") is not None else None,
                "recall_pct": round(float(stats.get("recall", 0)) * 100, 2) if stats.get("recall") is not None else None,
                "f1_pct": round(float(f1_val) * 100, 2) if f1_val is not None else None,
                "class_accuracy_pct": class_acc,
                "correct": correct,
            }
        )

    return {
        "available": True,
        "accuracy_pct": round(float(eval_json["accuracy"]) * 100, 2) if eval_json.get("accuracy") is not None else None,
        "test_samples": eval_json.get("test_samples"),
        "correct_predictions": eval_json.get("correct_predictions"),
        "incorrect_predictions": eval_json.get("incorrect_predictions"),
        "evaluated_at": eval_json.get("evaluated_at"),
        "per_class_rows": rows,
        "accuracy_formula": (
            f"Akurasi = benar / total = {eval_json.get('correct_predictions')} / {eval_json.get('test_samples')}"
            f" = {round(float(eval_json.get('accuracy', 0)) * 100, 2)}%"
            if eval_json.get("correct_predictions") is not None and eval_json.get("test_samples")
            else "Akurasi = Σ prediksi benar / N_test"
        ),
    }


def build_full_pipeline_overview() -> List[Dict[str, str]]:
    return [
        {"num": "01", "title": "Dataset & Split 80/20", "anchor": "sec-dataset", "desc": "2030 gambar · latih 80% · uji 20%"},
        {"num": "02", "title": "Class Weight", "anchor": "sec-weights", "desc": "Bobot kelas seimbang pada loss"},
        {"num": "03", "title": "Preprocessing", "anchor": "sec-preprocess", "desc": "Resize 224×224 · mean ImageNet"},
        {"num": "04", "title": "Arsitektur VGG16", "anchor": "sec-architecture", "desc": "Backbone + classifier head"},
        {"num": "05", "title": "Dimensi Konvolusi", "anchor": "sec-conv-math", "desc": "Ukuran feature map per block"},
        {"num": "06", "title": "Parameter Model", "anchor": "sec-params", "desc": "Frozen vs trainable per fase"},
        {"num": "07", "title": "Transfer Learning", "anchor": "sec-transfer", "desc": "Dua fase training"},
        {"num": "08", "title": "Iterasi & Adam", "anchor": "sec-training-math", "desc": "Batch · epoch · update bobot"},
        {"num": "09", "title": "Forward Pass", "anchor": "sec-forward", "desc": "Conv → GAP → Dense → Softmax"},
        {"num": "10", "title": "Loss & Softmax", "anchor": "sec-softmax-example", "desc": "Crossentropy + contoh numerik"},
        {"num": "11", "title": "Evaluasi & Metrik", "anchor": "sec-evaluation", "desc": "Akurasi · P/R/F1 pada test set"},
        {"num": "12", "title": "Ringkasan", "anchor": "sec-training", "desc": "Hasil training aktual"},
    ]


def build_class_dataset_rows(config: Optional[Dict] = None) -> List[Dict[str, Any]]:
    if config is None:
        config = current_app.config
    labels = list(config.get("CLASS_LABELS", []))
    splits = _split_counts(config, labels)
    db_counts = class_distribution()
    train_counts = splits["train"]
    total_train = sum(train_counts.values()) or 1

    weights = _compute_balanced_weights(train_counts, labels)
    saved_weights = _load_json(config.get("CLASS_WEIGHTS_PATH")).get("class_weights") or {}

    rows: List[Dict[str, Any]] = []
    for idx, slug in enumerate(labels):
        train_n = train_counts.get(slug, 0)
        raw_n = splits["raw"].get(slug, 0) or db_counts.get(slug, 0)
        val_n = splits["validation"].get(slug, 0)
        test_n = splits["test"].get(slug, 0)
        total_split = train_n + val_n + test_n
        pct = round((train_n / total_train) * 100, 2) if total_train else 0.0
        weight = saved_weights.get(str(idx), saved_weights.get(idx, weights.get(slug, 1.0)))
        rows.append(
            {
                "index": idx,
                "slug": slug,
                "display": display_class(slug),
                "raw_count": raw_n,
                "train_count": train_n,
                "validation_count": val_n,
                "test_count": test_n,
                "total_split": total_split,
                "db_count": db_counts.get(slug, 0),
                "train_pct": pct,
                "class_weight": round(float(weight), 4) if weight is not None else weights.get(slug, 1.0),
                "weight_formula": (
                    f"w = N / (K × n_{slug}) = {total_train} / ({len(labels)} × {train_n})"
                    if train_n > 0
                    else "—"
                ),
            }
        )
    return rows


def _sample_images_per_class(config: Dict, labels: List[str]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    base = config.get("BASE_DIR", "")
    for slug in labels:
        kelas = KelasKhat.query.filter_by(slug=slug).first()
        row = None
        if kelas:
            row = (
                GambarDataset.query.filter_by(id_kelas=kelas.id)
                .order_by(GambarDataset.dibuat_pada.desc())
                .first()
            )
        rel = row.path_asli if row else None
        abs_path = os.path.join(base, rel) if rel else None
        samples.append(
            {
                "slug": slug,
                "display": display_class(slug),
                "filename": row.nama_file_asli if row else "—",
                "path": rel or "",
                "has_image": bool(rel and abs_path and os.path.isfile(abs_path)),
            }
        )
    return samples


def get_vgg16_architecture_layers(num_classes: int) -> List[Dict[str, str]]:
    fine_tune = FINE_TUNE_LAYERS.get("vgg16", 4)
    return [
        {"name": "Input Layer", "detail": "224 × 224 × 3 (RGB)", "phase": "Preprocessing"},
        {"name": "VGG16 Backbone (ImageNet)", "detail": "13 Conv blocks + 5 MaxPool — weights ImageNet", "phase": "Transfer Learning"},
        {"name": f"Frozen Layers (Fase 1)", "detail": f"Semua backbone dibekukan — hanya head yang dilatih", "phase": "Fase 1"},
        {"name": "GlobalAveragePooling2D", "detail": "Feature map → vektor fitur", "phase": "Classifier Head"},
        {"name": "BatchNormalization", "detail": "Stabilisasi distribusi aktivasi", "phase": "Classifier Head"},
        {"name": "Dense (512, ReLU)", "detail": "Fully connected layer 1", "phase": "Classifier Head"},
        {"name": "Dropout (0.5)", "detail": "Regularisasi", "phase": "Classifier Head"},
        {"name": "Dense (256, ReLU)", "detail": "Fully connected layer 2", "phase": "Classifier Head"},
        {"name": "Dropout (0.3)", "detail": "Regularisasi", "phase": "Classifier Head"},
        {
            "name": f"Dense ({num_classes}, Softmax)",
            "detail": f"Output probabilitas {num_classes} kelas Khat",
            "phase": "Output",
        },
        {
            "name": f"Fine-Tune ({fine_tune} layer terakhir)",
            "detail": f"{fine_tune} layer akhir VGG16 dibuka (trainable) pada Fase 2",
            "phase": "Fase 2",
        },
    ]


def get_transfer_learning_phases(config: Dict, num_classes: int) -> List[Dict[str, Any]]:
    train_cfg = load_training_config(config)
    meta = load_model_metadata(config)
    phase1 = train_cfg.get("phase1_epochs") or 15
    phase2 = train_cfg.get("phase2_epochs") or 10
    lr1 = train_cfg.get("learning_rate_phase1") or train_cfg.get("learning_rate") or 1e-4
    lr2 = train_cfg.get("learning_rate_phase2") or 1e-5
    fine_tune = FINE_TUNE_LAYERS.get("vgg16", 4)
    return [
        {
            "phase": 1,
            "title": "Feature Extraction (Backbone Dibekukan)",
            "epochs": phase1,
            "learning_rate": lr1,
            "trainable": "Hanya classifier head (Dense + Softmax)",
            "loss": "Categorical Crossentropy",
            "optimizer": f"Adam (lr={lr1})",
            "description": (
                "Bobot VGG16 ImageNet tidak diubah. Model hanya mempelajari mapping fitur visual "
                f"ke {num_classes} kelas kaligrafi Arab melalui head classifier."
            ),
        },
        {
            "phase": 2,
            "title": "Fine-Tuning (Partial Unfreeze)",
            "epochs": phase2,
            "learning_rate": lr2,
            "trainable": f"{fine_tune} layer terakhir backbone VGG16 + classifier head",
            "loss": "Categorical Crossentropy",
            "optimizer": f"Adam (lr={lr2})",
            "description": (
                f"{fine_tune} layer konvolusi terakhir VGG16 dibuka dengan learning rate lebih kecil "
                "agar adaptasi spesifik domain Khat tidak merusak representasi ImageNet."
            ),
        },
    ]


def get_cnn_forward_steps(num_classes: int) -> List[Dict[str, str]]:
    labels_display = num_classes
    return [
        {
            "step": "1",
            "title": "Input Citra",
            "formula": "X ∈ ℝ^{224×224×3}",
            "detail": "Citra RGB dari dataset (raw / train / validation / test).",
        },
        {
            "step": "2",
            "title": "VGG16 Preprocessing",
            "formula": "X' = vgg16.preprocess_input(X)",
            "detail": "Normalisasi channel sesuai ImageNet (mean BGR subtraction).",
        },
        {
            "step": "3",
            "title": "Konvolusi + Pooling",
            "formula": "F = VGG16(X')",
            "detail": "Ekstraksi fitur hierarkis: tepi → tekstur → pola kaligrafi.",
        },
        {
            "step": "4",
            "title": "Global Average Pooling",
            "formula": "v = GAP(F)",
            "detail": "Feature map diratakan menjadi vektor fitur 1D.",
        },
        {
            "step": "5",
            "title": "Fully Connected + Softmax",
            "formula": f"ŷ = softmax(W·v + b), ŷ ∈ ℝ^{num_classes}",
            "detail": f"Vektor probabilitas untuk {labels_display} kelas; Σ ŷ_i = 1.",
        },
        {
            "step": "6",
            "title": "Prediksi Kelas",
            "formula": "ŷ_class = argmax(ŷ)",
            "detail": "Kelas dengan probabilitas tertinggi menjadi prediksi akhir.",
        },
    ]


def get_softmax_formula_block(num_classes: int) -> Dict[str, str]:
    return {
        "formula": "P(kelas_i) = exp(z_i) / Σ_j exp(z_j)",
        "percent": "Probabilitas (%) = P(kelas_i) × 100",
        "loss": "Loss = −Σ y_true · log(ŷ)  (Categorical Crossentropy)",
        "note": f"Output layer berdimensi {num_classes} (satu logit per kelas aktif).",
    }


def build_training_summary(config: Dict) -> Dict[str, Any]:
    train_cfg = load_training_config(config)
    meta = load_model_metadata(config)
    eval_json = _load_json(config.get("EVALUATION_RESULT_PATH"))
    history = _load_json(config.get("TRAINING_HISTORY_PATH"))
    hist_vals = history.get("history", {}) if history else {}
    val_acc_hist = hist_vals.get("val_accuracy") or []
    best_val = train_cfg.get("best_validation_accuracy") or meta.get("validation_accuracy")
    if best_val is None and val_acc_hist:
        best_val = max(val_acc_hist)

    return {
        "architecture": "vgg16",
        "architecture_label": ARCHITECTURES["vgg16"],
        "trained_architecture": train_cfg.get("model_architecture_key") or meta.get("architecture") or "—",
        "trained_architecture_label": train_cfg.get("model_architecture") or meta.get("architecture_label") or "—",
        "training_mode": train_cfg.get("training_mode") or meta.get("training_mode_label") or "Research Accuracy",
        "training_date": train_cfg.get("training_date") or meta.get("training_date") or "—",
        "batch_size": train_cfg.get("batch_size") or meta.get("batch_size") or 32,
        "epochs_total": train_cfg.get("actual_epochs_completed") or train_cfg.get("epochs"),
        "phase1_epochs": train_cfg.get("phase1_epochs"),
        "phase2_epochs": train_cfg.get("phase2_epochs"),
        "two_phase": train_cfg.get("two_phase_training", True),
        "train_count": train_cfg.get("train_count"),
        "validation_count": train_cfg.get("validation_count"),
        "test_count": train_cfg.get("test_count"),
        "best_val_accuracy": round(float(best_val) * 100, 2) if best_val is not None else None,
        "test_accuracy": round(float(eval_json["accuracy"]) * 100, 2) if eval_json.get("accuracy") is not None else None,
        "class_weights_enabled": train_cfg.get("class_weights_enabled", True),
        "has_training_history": bool(val_acc_hist),
        "final_train_loss": round(float(hist_vals["loss"][-1]), 4) if hist_vals.get("loss") else None,
        "final_val_loss": round(float(hist_vals["val_loss"][-1]), 4) if hist_vals.get("val_loss") else None,
    }


def build_cnn_calculation_page_context(config: Optional[Dict] = None) -> Dict[str, Any]:
    if config is None:
        config = current_app.config

    labels = list(config.get("CLASS_LABELS", []))
    num_classes = len(labels)
    readiness = get_dataset_readiness(config)
    class_rows = build_class_dataset_rows(config)
    class_rows = build_class_split_detail_rows(class_rows)
    total_raw = sum(r["raw_count"] for r in class_rows)
    total_train = sum(r["train_count"] for r in class_rows)
    total_val = sum(r["validation_count"] for r in class_rows)
    total_test = sum(r["test_count"] for r in class_rows)
    total_split = total_train + total_val + total_test
    dataset_totals = {
        "raw": total_raw or readiness.get("raw_count", 0),
        "train": total_train or readiness.get("train_count", 0),
        "validation": total_val or readiness.get("validation_count", 0),
        "test": total_test or readiness.get("test_count", 0),
        "processed": total_split or readiness.get("processed_count", 0),
        "split_ready": readiness.get("is_ready", False),
    }
    split_report = _load_split_report(config)
    training_summary = build_training_summary(config)
    training_math = build_training_math(training_summary, dataset_totals)
    evaluation_metrics = build_evaluation_metrics(config, class_rows)

    return {
        "architecture_key": "vgg16",
        "architecture_label": ARCHITECTURES["vgg16"],
        "class_labels": labels,
        "class_display_labels": [display_class(l) for l in labels],
        "num_classes": num_classes,
        "class_rows": class_rows,
        "dataset_totals": dataset_totals,
        "split_math": build_split_math_summary(dataset_totals),
        "split_report": split_report,
        "dataset_split_steps": build_dataset_split_steps(class_rows, dataset_totals, split_report, config),
        "preprocessing_detail": build_preprocessing_detail(),
        "architecture_layers": get_vgg16_architecture_layers(num_classes),
        "parameter_breakdown": build_vgg16_parameter_breakdown(num_classes),
        "spatial_dimensions": build_vgg16_spatial_dimensions(),
        "tensor_pipeline": get_vgg16_tensor_pipeline(num_classes),
        "transfer_phases": get_transfer_learning_phases(config, num_classes),
        "transfer_concepts": build_transfer_learning_concept(),
        "forward_steps": get_cnn_forward_steps(num_classes),
        "softmax_block": get_softmax_formula_block(num_classes),
        "training_summary": training_summary,
        "training_math": training_math,
        "adam_formulas": build_adam_and_update_formulas(training_math),
        "softmax_example": build_worked_softmax_example(class_rows, labels),
        "loss_example": build_loss_worked_example(labels),
        "evaluation_metrics": evaluation_metrics,
        "metric_formulas": build_metric_formula_reference(evaluation_metrics, class_rows),
        "pipeline_overview": build_full_pipeline_overview(),
        "samples_per_class": _sample_images_per_class(config, labels),
        "fine_tune_layers": FINE_TUNE_LAYERS.get("vgg16", 4),
        "input_size": "224 × 224 × 3",
        "accuracy_target_pct": int(ACCURACY_TARGET * 100),
        "uses_external_model": bool(config.get("USE_EXTERNAL_MODEL")),
    }
