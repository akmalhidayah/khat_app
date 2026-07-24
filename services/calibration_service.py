"""Per-class logit bias calibration from validation holdout."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np


def _load_class_order(class_indices_path: str, fallback: List[str]) -> List[str]:
    if not os.path.isfile(class_indices_path):
        return list(fallback)
    try:
        with open(class_indices_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        indices = data.get("class_indices", data)
        return sorted(indices.keys(), key=lambda key: int(indices[key]))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return list(fallback)


def fit_logit_bias(
    y_true: np.ndarray,
    logits: np.ndarray,
    max_iter: int = 400,
) -> np.ndarray:
    num_classes = logits.shape[1]

    def _accuracy(bias: np.ndarray) -> float:
        preds = np.argmax(logits + bias.reshape(1, -1), axis=1)
        return float(np.mean(preds == y_true))

    try:
        from scipy.optimize import minimize

        result = minimize(
            lambda b: -_accuracy(b),
            np.zeros(num_classes, dtype=np.float64),
            method="Powell",
            options={"maxiter": 250, "ftol": 1e-6},
        )
        if result.success and _accuracy(result.x) > 0:
            return result.x.astype(np.float64)
    except Exception:
        pass

    bias = np.zeros(num_classes, dtype=np.float64)
    best_acc = -1.0
    for span in (1.4, 0.6, 0.25):
        grid = np.linspace(-span, span, 29)
        for _ in range(max_iter):
            improved = False
            for cls in range(num_classes):
                local_best = bias[cls]
                local_acc = best_acc
                for delta in grid:
                    trial = bias.copy()
                    trial[cls] = delta
                    preds = np.argmax(logits + trial, axis=1)
                    acc = float(np.mean(preds == y_true))
                    if acc > local_acc:
                        local_acc = acc
                        local_best = delta
                if local_acc > best_acc + 1e-9:
                    best_acc = local_acc
                    bias[cls] = local_best
                    improved = True
            if not improved:
                break
    return bias


def fit_linear_calibrator(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, np.ndarray]:
    """Fit W @ logits + b + temperature to maximize accuracy on holdout logits."""
    num_classes = logits.shape[1]
    logits = np.asarray(logits, dtype=np.float64)

    def _accuracy(params: np.ndarray) -> float:
        temperature = max(float(params[0]), 0.05)
        matrix = params[1 : 1 + num_classes * num_classes].reshape(num_classes, num_classes)
        bias = params[1 + num_classes * num_classes :]
        adjusted = (logits @ matrix) / temperature + bias
        preds = np.argmax(adjusted, axis=1)
        return float(np.mean(preds == y_true))

    x0 = np.concatenate(
        [
            np.array([1.0], dtype=np.float64),
            np.eye(num_classes, dtype=np.float64).reshape(-1),
            np.zeros(num_classes, dtype=np.float64),
        ]
    )

    best_acc = _accuracy(x0)
    best = x0.copy()

    try:
        from scipy.optimize import differential_evolution, minimize

        result = minimize(
            lambda p: -_accuracy(p),
            x0,
            method="Powell",
            options={"maxiter": 600, "ftol": 1e-7},
        )
        if _accuracy(result.x) >= best_acc:
            best = result.x
            best_acc = _accuracy(result.x)

        dim = len(x0)
        bounds = [(0.05, 4.0)] + [(-2.5, 2.5)] * (dim - 1)
        de = differential_evolution(
            lambda p: -_accuracy(p),
            bounds,
            maxiter=450,
            seed=42,
            polish=True,
            workers=1,
        )
        if _accuracy(de.x) >= best_acc:
            best = de.x
            best_acc = _accuracy(de.x)
    except Exception:
        pass

    temperature = max(float(best[0]), 0.05)
    matrix = best[1 : 1 + num_classes * num_classes].reshape(num_classes, num_classes)
    bias = best[1 + num_classes * num_classes :]
    return {
        "temperature": np.array([temperature], dtype=np.float64),
        "matrix": matrix,
        "bias": bias,
        "fit_accuracy": np.array([best_acc]),
    }


def _collect_logits(
    model,
    config,
    class_order: List[str],
    include_train: bool,
) -> tuple[np.ndarray, np.ndarray]:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from services.preprocessing_service import get_architecture_preprocess_fn

    meta_path = config.get("MODEL_METADATA_PATH", "")
    arch = "keras_h5"
    if meta_path and os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            arch = meta.get("model_architecture_key") or meta.get("architecture") or arch
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    if arch in ("keras_h5", "teachable_machine_h5", "external_h5"):
        def _preprocess(img):
            return (img / 127.5) - 1.0
    else:
        preprocess_fn = get_architecture_preprocess_fn(arch)

        def _preprocess(img):
            return preprocess_fn(img)

    logits_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    def _consume(directory: str, extra_models: Optional[List] = None) -> None:
        gen = ImageDataGenerator(preprocessing_function=_preprocess).flow_from_directory(
            directory,
            target_size=(224, 224),
            class_mode="categorical",
            classes=class_order,
            batch_size=16,
            shuffle=False,
        )
        seen = 0
        for batch_images, batch_labels in gen:
            batch_probs = model.predict(batch_images, verbose=0)
            if extra_models:
                for extra in extra_models:
                    batch_probs = batch_probs + extra.predict(batch_images, verbose=0)
                batch_probs = batch_probs / (1 + len(extra_models))
            logits_list.append(np.log(np.clip(batch_probs, 1e-8, 1.0)))
            labels_list.append(np.argmax(batch_labels, axis=1))
            seen += len(batch_labels)
            if seen >= gen.samples:
                break

    extra_models: List = []
    checkpoint_model = load_finetune_checkpoint_model(config, model)
    if checkpoint_model is not None:
        extra_models.append(checkpoint_model)

    _consume(config["VALIDATION_DIR"], extra_models)
    if include_train:
        train_dir = os.path.join(config.get("PROCESSED_BALANCED_DIR", ""), "train")
        if not os.path.isdir(train_dir):
            train_dir = config["TRAIN_DIR"]
        _consume(train_dir, extra_models)

    logits = np.vstack(logits_list)
    y_true = np.concatenate(labels_list)
    count = min(logits.shape[0], y_true.shape[0])
    return logits[:count], y_true[:count]


def _validation_argmax_acc(
    val_logits: np.ndarray,
    matrix: np.ndarray,
    temperature: float,
    bias: np.ndarray,
    val_y: np.ndarray,
) -> float:
    adjusted = (val_logits @ matrix) / temperature + bias
    return float(np.mean(np.argmax(adjusted, axis=1) == val_y))


def _tune_pairwise_bias_on_val(
    val_logits: np.ndarray,
    val_y: np.ndarray,
    matrix: np.ndarray,
    temperature: float,
    bias: np.ndarray,
    class_order: List[str],
    pairs: List[tuple[str, str, float]],
) -> np.ndarray:
    tuned = bias.copy()
    for class_a, class_b, counter_scale in pairs:
        if class_a not in class_order or class_b not in class_order:
            continue
        a_idx = class_order.index(class_a)
        b_idx = class_order.index(class_b)
        best_acc = _validation_argmax_acc(val_logits, matrix, temperature, tuned, val_y)
        for bump in np.linspace(0.0, 0.22, 45):
            trial = tuned.copy()
            trial[a_idx] += bump
            trial[b_idx] -= bump * counter_scale
            acc = _validation_argmax_acc(val_logits, matrix, temperature, trial, val_y)
            if acc > best_acc:
                best_acc = acc
                tuned = trial
    return tuned


def calibrate_keras_model(
    config,
    model,
    class_order: Optional[List[str]] = None,
    include_train: bool = False,
) -> Dict:
    class_order = class_order or _load_class_order(
        config.get("CLASS_INDICES_PATH", ""),
        config.get("CLASS_LABELS", []),
    )
    from services.evaluation_dataset_service import iter_test_images

    logits, y_true = _collect_logits(model, config, class_order, include_train)
    val_count = len(iter_test_images(config["VALIDATION_DIR"], class_order))
    val_logits = logits[:val_count]
    val_y = y_true[:val_count]
    raw_probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    raw_probs = raw_probs / np.sum(raw_probs, axis=1, keepdims=True)
    before = float(np.mean(np.argmax(raw_probs, axis=1) == y_true))
    calibrator = fit_linear_calibrator(y_true, logits)
    bias = calibrator["bias"].copy()
    matrix = calibrator["matrix"]
    temperature = float(calibrator["temperature"][0])

    if "diwani_jali" in class_order:
        bias = _tune_pairwise_bias_on_val(
            val_logits,
            val_y,
            matrix,
            temperature,
            bias,
            class_order,
            [
                ("diwani_jali", "tsuluts", 0.55),
                ("diwani_jali", "diwani", 0.45),
                ("diwani", "tsuluts", 0.50),
            ],
        )

    adjusted = (logits @ matrix) / temperature + bias
    val_adjusted = (val_logits @ matrix) / temperature + bias
    after = float(np.mean(np.argmax(adjusted, axis=1) == y_true))

    best_margin = _tune_borderline_margin(val_adjusted, val_y, class_order)

    payload = {
        "class_order": class_order,
        "calibration_type": "linear",
        "temperature": temperature,
        "matrix": matrix.tolist(),
        "bias": {class_order[i]: float(bias[i]) for i in range(len(class_order))},
        "borderline_margin": best_margin,
        "validation_accuracy_before": before,
        "validation_accuracy_after": after,
    }
    path = os.path.join(config["MODEL_DIR"], "logit_calibration.json")
    os.makedirs(config["MODEL_DIR"], exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _apply_borderline_margin(probs: np.ndarray, class_order: List[str], margin: float) -> np.ndarray:
    if margin <= 0 or "diwani_jali" not in class_order:
        return probs
    jali_idx = class_order.index("diwani_jali")
    adjusted = probs.copy()
    rival_pairs = []
    if "tsuluts" in class_order:
        rival_pairs.append(class_order.index("tsuluts"))
    if "diwani" in class_order:
        rival_pairs.append(class_order.index("diwani"))
    for i in range(adjusted.shape[0]):
        row = adjusted[i]
        pred_idx = int(np.argmax(row))
        if pred_idx not in rival_pairs:
            continue
        gap = float(row[pred_idx] - row[jali_idx])
        if gap <= margin:
            row = row.copy()
            row[jali_idx] = row[pred_idx] + 1e-4
            adjusted[i] = row / row.sum()
    return adjusted


def _tune_borderline_margin(
    calibrated_logits: np.ndarray,
    y_true: np.ndarray,
    class_order: List[str],
) -> float:
    probs = _softmax_rows(calibrated_logits)
    best_margin = 0.0
    best_acc = float(np.mean(np.argmax(probs, axis=1) == y_true))
    for margin in np.linspace(0.0, 0.18, 37):
        trial = _apply_borderline_margin(probs, class_order, float(margin))
        acc = float(np.mean(np.argmax(trial, axis=1) == y_true))
        if acc > best_acc:
            best_acc = acc
            best_margin = float(margin)
    return best_margin


def load_calibrator(config) -> Optional[Dict]:
    path = os.path.join(config["MODEL_DIR"], "logit_calibration.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        class_order = data.get("class_order") or list(config.get("CLASS_LABELS", []))
        bias_map = data.get("bias", {})
        bias = np.array([float(bias_map.get(cls, 0.0)) for cls in class_order], dtype=np.float32)
        matrix = np.array(data.get("matrix"), dtype=np.float32) if data.get("matrix") else None
        temperature = float(data.get("temperature", 1.0))
        return {
            "class_order": class_order,
            "bias": bias,
            "matrix": matrix,
            "temperature": temperature,
            "calibration_type": data.get("calibration_type", "bias"),
            "borderline_margin": float(data.get("borderline_margin") or 0.0),
        }
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def load_logit_bias(config) -> Optional[np.ndarray]:
    calibrator = load_calibrator(config)
    if calibrator is None:
        return None
    return calibrator.get("bias")


def _reorder_probability_columns(
    probs: np.ndarray,
    source_order: List[str],
    target_order: List[str],
) -> np.ndarray:
    """Permute probability columns from source_order into target_order."""
    if not source_order or not target_order or list(source_order) == list(target_order):
        return probs
    source_index = {label: index for index, label in enumerate(source_order)}
    columns = []
    for label in target_order:
        if label in source_index:
            columns.append(probs[:, source_index[label]])
        else:
            columns.append(np.zeros(probs.shape[0], dtype=probs.dtype))
    reordered = np.column_stack(columns)
    row_sums = np.sum(reordered, axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return reordered / row_sums


def apply_calibrator(
    probs: np.ndarray,
    calibrator: Optional[Dict],
    model_class_order: Optional[List[str]] = None,
) -> np.ndarray:
    """Apply logit calibrator.

    When ``model_class_order`` differs from the calibrator's stored ``class_order``,
    probability columns are remapped first so bias/matrix attach to the correct
    class names (avoids silent accuracy loss on external Teachable Machine order).
    """
    if not calibrator:
        return probs
    cal_order = list(calibrator.get("class_order") or [])
    working = np.asarray(probs, dtype=np.float64)
    remap_back = False
    if model_class_order and cal_order and list(model_class_order) != cal_order:
        working = _reorder_probability_columns(working, list(model_class_order), cal_order)
        remap_back = True

    eps = 1e-8
    logits = np.log(np.clip(working, eps, 1.0))
    temperature = max(float(calibrator.get("temperature") or 1.0), 0.05)
    matrix = calibrator.get("matrix")
    bias = calibrator.get("bias")
    if matrix is not None:
        logits = logits @ matrix.astype(np.float64)
    logits = logits / temperature
    if bias is not None and len(bias) == logits.shape[1]:
        logits = logits + bias.reshape(1, -1)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    working = exp / np.sum(exp, axis=1, keepdims=True)
    margin = float(calibrator.get("borderline_margin") or 0.0)
    if margin > 0 and cal_order:
        working = _apply_borderline_margin(working, cal_order, margin)

    if remap_back and model_class_order:
        working = _reorder_probability_columns(working, cal_order, list(model_class_order))
    return working.astype(np.float32, copy=False)


def apply_logit_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return apply_calibrator(probs, {"bias": bias, "temperature": 1.0, "matrix": None})


def ensure_model_calibrated(config, model_path: Optional[str] = None) -> Dict:
    """Refresh calibrator for the active Keras model (validation holdout only)."""
    from services.accuracy_boost_service import _model_class_order, _resolve_external_source
    from services.model_builder_service import load_trained_classifier
    from services.model_context_service import load_model_metadata

    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    class_order = _model_class_order(config)
    path = model_path or _resolve_external_source(config) or config.get("MODEL_PATH")
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Model not found for calibration: {path}")
    meta = load_model_metadata(config)
    architecture = meta.get("model_architecture_key") or meta.get("architecture") or "efficientnetb0"
    num_classes = len(class_order)
    model = load_trained_classifier(path, architecture, num_classes)
    return calibrate_keras_model(config, model, class_order, include_train=False)


def should_recalibrate_on_eval(config) -> bool:
    """Return True only when calibration file is missing or older than the active model."""
    if not config.get("EVAL_SKIP_CALIBRATION_ON_RUN", True):
        return True
    cal_path = config.get("LOGIT_CALIBRATION_PATH") or os.path.join(
        config.get("MODEL_DIR", ""), "logit_calibration.json"
    )
    if not cal_path or not os.path.isfile(cal_path):
        return True
    model_path = config.get("EXTERNAL_MODEL_PATH") or config.get("MODEL_PATH")
    if not model_path or not os.path.isfile(model_path):
        return False
    try:
        return os.path.getmtime(model_path) > os.path.getmtime(cal_path)
    except OSError:
        return True


def predict_keras_test_probs_batched(
    model,
    test_dir: str,
    class_order: List[str],
    batch_size: int = 64,
    extra_models: Optional[List] = None,
) -> np.ndarray:
    """Fast batched inference on the test split (no per-image TTA)."""
    from PIL import Image, ImageOps
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    def _preprocess(img_array):
        pil = Image.fromarray(img_array.astype("uint8"))
        rgb = ImageOps.exif_transpose(pil).convert("RGB")
        array = np.array(rgb.resize((224, 224)), dtype=np.float32)
        return (array / 127.5) - 1.0

    gen = ImageDataGenerator(preprocessing_function=_preprocess)
    flow = gen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        class_mode="categorical",
        classes=class_order,
        batch_size=batch_size,
        shuffle=False,
    )
    if flow.samples == 0:
        raise ValueError("Test dataset is empty.")

    models = [model] + list(extra_models or [])
    total_probs = None
    for active in models:
        preds = active.predict(flow, verbose=0)
        total_probs = preds if total_probs is None else total_probs + preds
        if len(models) > 1:
            flow.reset()
    return total_probs / len(models)


def predict_architecture_test_probs(
    model,
    test_items,
    class_order,
    architecture: str = "efficientnetb0",
    use_hflip_tta: bool = True,
    extra_models: Optional[List] = None,
) -> np.ndarray:
    """Predict test probabilities with architecture-specific preprocessing and TTA."""
    from PIL import Image, ImageEnhance, ImageOps
    from services.preprocessing_service import get_architecture_preprocess_fn, preprocess_pil_image

    preprocess_fn = get_architecture_preprocess_fn(architecture)
    models = [model] + list(extra_models or [])
    rows: List[np.ndarray] = []
    for image_path, _label in test_items:
        pil = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        variants = [pil]
        if use_hflip_tta:
            variants.extend([
                pil.transpose(Image.FLIP_LEFT_RIGHT),
                ImageEnhance.Brightness(pil).enhance(0.93),
                ImageEnhance.Brightness(pil).enhance(1.07),
                ImageEnhance.Contrast(pil).enhance(1.06),
                ImageEnhance.Sharpness(pil).enhance(1.08),
            ])
        tensors = []
        for frame in variants:
            processed = preprocess_pil_image(frame, (224, 224))
            array = preprocess_fn(np.array(processed, dtype=np.float32))
            tensors.append(array)
        batch = np.stack(tensors)
        model_probs = [active.predict(batch, verbose=0).mean(axis=0) for active in models]
        rows.append(np.mean(model_probs, axis=0))
    return np.vstack(rows)


def predict_keras_test_probs(
    model,
    test_items,
    class_order,
    use_hflip_tta: bool = True,
    extra_models: Optional[List] = None,
) -> np.ndarray:
    """Predict test probabilities with optional horizontal-flip TTA and model ensemble."""
    from PIL import Image, ImageEnhance, ImageOps

    models = [model] + list(extra_models or [])
    rows: List[np.ndarray] = []
    for image_path, _label in test_items:
        pil = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        variants = [pil]
        if use_hflip_tta:
            variants.extend([
                pil.transpose(Image.FLIP_LEFT_RIGHT),
                ImageEnhance.Brightness(pil).enhance(0.93),
                ImageEnhance.Brightness(pil).enhance(1.07),
                ImageEnhance.Contrast(pil).enhance(1.06),
                ImageEnhance.Sharpness(pil).enhance(1.08),
            ])
        tensors = []
        for frame in variants:
            img = frame.resize((224, 224))
            array = np.array(img, dtype=np.float32)
            tensors.append((array / 127.5) - 1.0)
        batch = np.stack(tensors)
        model_probs = [active.predict(batch, verbose=0).mean(axis=0) for active in models]
        rows.append(np.mean(model_probs, axis=0))
    return np.vstack(rows)


def load_finetune_checkpoint_model(config, base_model):
    """Load optional fine-tune checkpoint as a second ensemble member."""
    import tensorflow as tf

    checkpoint = os.path.join(config["MODEL_DIR"], "khat_finetune_checkpoint.weights.h5")
    if not checkpoint or not os.path.isfile(checkpoint) or os.path.getsize(checkpoint) < 1000:
        return None
    try:
        clone = tf.keras.models.clone_model(base_model)
        clone.build(base_model.input_shape)
        clone.set_weights(base_model.get_weights())
        clone.load_weights(checkpoint)
        return clone
    except Exception:
        return None

