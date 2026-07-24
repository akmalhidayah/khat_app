"""Fine-tune / retrain pipeline to reach the 85% accuracy research target."""

from __future__ import annotations

import gc
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from services.class_mapping_service import ensure_canonical_class_indices, load_saved_class_indices
from services.metrics_service import ACCURACY_TARGET
from services.training_utils import get_keras, save_json

WEAK_CLASS_BOOST = {
    "diwani": 2.4,
    "diwani_jali": 2.0,
    "tsuluts": 1.55,
    "naskhi": 1.4,
}

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def _is_image(name: str) -> bool:
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _resolve_train_dir(config: Dict[str, Any]) -> str:
    balanced = os.path.join(config["PROCESSED_BALANCED_DIR"], "train")
    if os.path.isdir(balanced):
        for label in config["CLASS_LABELS"]:
            cls_dir = os.path.join(balanced, label)
            if os.path.isdir(cls_dir) and any(_is_image(f) for f in os.listdir(cls_dir)):
                return balanced
    return config["TRAIN_DIR"]


def _model_class_order(config: Dict[str, Any]) -> List[str]:
    saved = load_saved_class_indices(config)
    if saved:
        return sorted(saved.keys(), key=lambda key: saved[key])
    return list(config["CLASS_LABELS"])


def _class_weights(train_dir: str, class_order: List[str]) -> Dict[int, float]:
    indices = {name: idx for idx, name in enumerate(class_order)}
    labels: List[int] = []
    for cls in class_order:
        cls_dir = os.path.join(train_dir, cls)
        count = 0
        if os.path.isdir(cls_dir):
            count = sum(1 for f in os.listdir(cls_dir) if _is_image(f))
        labels.extend([indices[cls]] * count)

    if not labels:
        return {}

    unique = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=unique, y=labels)
    class_weight = {int(class_id): float(weight) for class_id, weight in zip(unique, weights)}
    for cls_name, boost in WEAK_CLASS_BOOST.items():
        idx = indices.get(cls_name)
        if idx is not None and idx in class_weight:
            class_weight[idx] *= boost
    return class_weight


def _tm_preprocess_batch(images, labels):
    import tensorflow as tf

    images = tf.cast(images, tf.float32)
    images = (images / 127.5) - 1.0
    return images, labels


def _efficientnet_preprocess_batch(images, labels, preprocess_fn):
    import tensorflow as tf

    images = tf.cast(images, tf.float32)
    images = preprocess_fn(images)
    return images, labels


def _build_generators(
    train_dir: str,
    validation_dir: str,
    class_order: List[str],
    batch_size: int,
    preprocessing: str,
    preprocess_fn=None,
):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if preprocessing == "teachable_machine":
        def _preprocess(img):
            return (img / 127.5) - 1.0
    else:
        def _preprocess(img):
            return preprocess_fn(img)

    train_gen = ImageDataGenerator(
        preprocessing_function=_preprocess,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.10,
        shear_range=0.02,
        horizontal_flip=False,
        brightness_range=(0.92, 1.08),
        fill_mode="nearest",
    )
    val_gen = ImageDataGenerator(preprocessing_function=_preprocess)

    train_flow = train_gen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        class_mode="categorical",
        classes=class_order,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )
    val_flow = val_gen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        class_mode="categorical",
        classes=class_order,
        batch_size=batch_size,
        shuffle=False,
    )
    train_count = train_flow.samples
    val_count = val_flow.samples
    return train_flow, val_flow, train_count, val_count


def _resolve_preprocessing_mode(config: Dict[str, Any]) -> str:
    """Match training/eval preprocessing from saved model metadata."""
    if config.get("USE_EXTERNAL_MODEL") and config.get("EXTERNAL_MODEL_PATH", "").lower().endswith(".h5"):
        labels_path = config.get("EXTERNAL_LABELS_PATH") or ""
        if labels_path and os.path.isfile(labels_path):
            return "teachable_machine"

    meta_path = config.get("MODEL_METADATA_PATH", "")
    if meta_path and os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            preprocessing = (meta.get("preprocessing") or "").lower()
            arch = (meta.get("model_architecture_key") or meta.get("architecture") or "").lower()
            if preprocessing in ("teachable_machine", "keras_h5", "external_h5"):
                return "teachable_machine"
            if preprocessing in ("vgg16", "efficientnetb0", "mobilenetv2"):
                return preprocessing
            if arch in ("keras_h5", "teachable_machine_h5", "external_h5"):
                return "teachable_machine"
            if arch in ("vgg16", "efficientnetb0", "mobilenetv2"):
                return arch
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return "teachable_machine"


def _prepare_dataset(config: Dict[str, Any], *, rebalance: bool = True) -> Dict[str, Any]:
    """Build augmented balanced train set without removing any source images."""
    from services.dataset_balancing_service import build_balanced_train_set

    if not rebalance:
        return {}
    return build_balanced_train_set(config)


def _unfreeze_tail(model, tail_layers: int = 24) -> None:
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-tail_layers:]:
        layer.trainable = True


def _resolve_external_source(config: Dict[str, Any]) -> str:
    ext_dir = (config.get("EXTERNAL_MODEL_DIR") or "").strip()
    ext_file = (config.get("EXTERNAL_MODEL_FILE") or "keras_model.h5").strip()
    if ext_dir:
        candidate = Path(ext_dir) / ext_file
        if candidate.is_file():
            return str(candidate)
    path = config.get("EXTERNAL_MODEL_PATH") or config.get("MODEL_PATH") or ""
    return path if path and os.path.isfile(path) else ""


def fine_tune_external_model(
    config: Dict[str, Any],
    source_model_path: Optional[str] = None,
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    phase1_epochs: int = 8,
) -> Dict[str, Any]:
    """Fine-tune the external Keras model on the balanced local split (dataset preserved)."""
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    get_keras()
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers.legacy import Adam
    from services.preprocessing_service import get_architecture_preprocess_fn

    source_model_path = source_model_path or _resolve_external_source(config)
    if not source_model_path or not os.path.isfile(source_model_path):
        raise FileNotFoundError(f"External model not found: {source_model_path}")

    preprocessing = _resolve_preprocessing_mode(config)
    preprocess_fn = None
    if preprocessing != "teachable_machine":
        preprocess_fn = get_architecture_preprocess_fn(preprocessing)

    _prepare_dataset(config)
    train_dir = _resolve_train_dir(config)
    validation_dir = config["VALIDATION_DIR"]
    class_order = _model_class_order(config)
    class_weight = _class_weights(train_dir, class_order)

    train_flow, val_flow, _, _ = _build_generators(
        train_dir,
        validation_dir,
        class_order,
        batch_size,
        preprocessing=preprocessing,
        preprocess_fn=preprocess_fn,
    )

    model = tf.keras.models.load_model(source_model_path)

    model_output_dir = config.get("MODEL_DIR") or str(Path(config["BASE_DIR"]) / "model")
    internal_best = str(Path(model_output_dir) / "khat_best.keras")
    internal_latest = str(Path(model_output_dir) / "khat_latest.keras")
    checkpoint = str(Path(model_output_dir) / "khat_finetune_checkpoint.weights.h5")
    os.makedirs(model_output_dir, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(
            checkpoint,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    # Phase 1 — train classification head only
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-4:]:
        layer.trainable = True
    model.compile(
        optimizer=Adam(learning_rate=learning_rate * 4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    phase1 = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=min(phase1_epochs, epochs),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Phase 2 — fine-tune tail of backbone
    _unfreeze_tail(model, tail_layers=36)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    remaining = max(1, epochs - len(phase1.history.get("accuracy", [])))
    phase2 = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=remaining,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
        initial_epoch=len(phase1.history.get("accuracy", [])),
    )

    if os.path.isfile(checkpoint) and os.path.getsize(checkpoint) > 1000:
        model.load_weights(checkpoint)

    ext_model_file = "khat_best.keras"
    primary_save = internal_best
    model.save(primary_save)
    model.save(internal_latest)
    if config.get("EXTERNAL_MODEL_DIR"):
        try:
            ext_copy = str(Path(config["EXTERNAL_MODEL_DIR"]) / (config.get("EXTERNAL_MODEL_FILE") or "keras_model.h5"))
            model.save(ext_copy)
        except OSError:
            pass

    config["EXTERNAL_MODEL_PATH"] = primary_save
    config["MODEL_PATH"] = primary_save
    config["BEST_MODEL_PATH"] = primary_save
    config["LATEST_MODEL_PATH"] = primary_save

    hist = {
        "accuracy": phase1.history.get("accuracy", []) + phase2.history.get("accuracy", []),
        "val_accuracy": phase1.history.get("val_accuracy", []) + phase2.history.get("val_accuracy", []),
    }
    best_val = float(max(hist["val_accuracy"])) if hist.get("val_accuracy") else None
    class_indices = {name: idx for idx, name in enumerate(class_order)}
    arch_key = "keras_h5" if preprocessing == "teachable_machine" else preprocessing
    save_json(config["CLASS_INDICES_PATH"], {
        "class_indices": class_indices,
        "class_names": class_order,
        "source": "fine_tuned_external",
    })
    save_json(config["MODEL_METADATA_PATH"], {
        "architecture": arch_key,
        "architecture_label": "Fine-Tuned Keras Model",
        "model_architecture_key": arch_key,
        "training_mode": "research",
        "training_mode_label": "Fine-Tuned External Model",
        "training_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "model_path": primary_save,
        "validation_accuracy": best_val,
        "class_indices": class_indices,
        "preprocessing": preprocessing,
        "source": "fine_tuned",
    })
    save_json(config["CLASS_WEIGHTS_PATH"], {"class_weights": class_weight, "enabled": True})

    from services.model_cache_service import clear_model_cache
    clear_model_cache()

    tf.keras.backend.clear_session()
    gc.collect()

    return {
        "strategy": "fine_tune_external",
        "source_model": source_model_path,
        "best_model_path": primary_save,
        "best_validation_accuracy": best_val,
        "epochs_run": len(hist.get("accuracy", [])),
        "class_order": class_order,
        "preprocessing": preprocessing,
    }


def retrain_transfer_model(
    config: Dict[str, Any],
    epochs: int = 25,
    batch_size: int = 16,
    architecture: str = "efficientnetb0",
) -> Dict[str, Any]:
    """Full two-phase transfer learning using the in-app training pipeline."""
    from services.training_service import train_model

    _prepare_dataset(config)
    return train_model(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-4,
        training_mode="research",
        model_architecture=architecture,
        use_optimized_images=True,
        cache_to_memory=False,
        auto_batch_fallback=True,
        enable_class_weights=True,
        enable_augmentation=True,
    )


def _use_internal_model_paths(config: Dict[str, Any], model_path: str) -> None:
    """Route prediction/evaluation to the fine-tuned in-project model."""
    config["EXTERNAL_MODEL_PATH"] = model_path
    config["MODEL_PATH"] = model_path
    config["BEST_MODEL_PATH"] = model_path
    config["LATEST_MODEL_PATH"] = model_path
    config["USE_EXTERNAL_MODEL"] = True


def evaluate_current_model(config: Dict[str, Any]) -> Dict[str, Any]:
    from services.evaluation_service import evaluate_model

    return evaluate_model()


def switch_to_internal_model(env_path: Optional[str] = None) -> None:
    """Persist fine-tuned model as the active Keras model for the app."""
    env_path = env_path or str(Path(__file__).resolve().parents[1] / ".env")
    if not os.path.isfile(env_path):
        return
    model_dir = str(Path(__file__).resolve().parents[1] / "model")
    text = Path(env_path).read_text(encoding="utf-8")
    text = re.sub(r"^USE_EXTERNAL_MODEL=.*$", "USE_EXTERNAL_MODEL=1", text, flags=re.MULTILINE)
    text = re.sub(r"^USE_TEACHABLE_MACHINE=.*$", "USE_TEACHABLE_MACHINE=0", text, flags=re.MULTILINE)
    text = re.sub(r"^EXTERNAL_MODEL_DIR=.*$", f"EXTERNAL_MODEL_DIR={model_dir}", text, flags=re.MULTILINE)
    text = re.sub(r"^EXTERNAL_MODEL_FILE=.*$", "EXTERNAL_MODEL_FILE=khat_best.keras", text, flags=re.MULTILINE)
    Path(env_path).write_text(text, encoding="utf-8")


def _calibrate_active_model(config: Dict[str, Any], model_path: str, class_order: List[str]) -> Dict:
    import tensorflow as tf

    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    model = tf.keras.models.load_model(model_path)
    return calibrate_keras_model(config, model, class_order)


def run_accuracy_improvement(
    config: Dict[str, Any],
    strategy: str = "auto",
    min_accuracy: float = ACCURACY_TARGET,
    *,
    fine_tune_epochs: int = 25,
    skip_fine_tune: bool = False,
) -> Dict[str, Any]:
    """Fine-tune, calibrate, and evaluate — preserves all dataset images."""
    from services.calibration_service import ensure_model_calibrated
    from services.model_cache_service import clear_model_cache

    results: Dict[str, Any] = {"strategies": [], "target": min_accuracy}
    source_path = _resolve_external_source(config)
    if not source_path:
        raise FileNotFoundError("External Keras model not found.")

    config["EVAL_USE_TTA"] = True
    config["EVAL_USE_ENSEMBLE"] = True

    baseline = evaluate_current_model(config)
    results["baseline_accuracy"] = baseline.get("accuracy")
    results["strategies"].append({
        "strategy": "baseline_eval",
        "accuracy": baseline.get("accuracy"),
        "f1_macro": baseline.get("f1_macro"),
    })

    active_path = source_path
    if (
        not skip_fine_tune
        and strategy in ("auto", "fine_tune", "full")
        and (baseline.get("accuracy") or 0) < min_accuracy
    ):
        tune = fine_tune_external_model(
            config,
            source_model_path=source_path,
            epochs=fine_tune_epochs,
            batch_size=16,
            learning_rate=2e-5,
        )
        active_path = tune.get("best_model_path") or source_path
        results["strategies"].append({"strategy": "fine_tune_external", **tune})
        clear_model_cache()

    _use_internal_model_paths(config, active_path)
    cal = ensure_model_calibrated(config, active_path)
    results["strategies"].append({"strategy": "calibrate", **cal})
    clear_model_cache()

    eval_result = evaluate_current_model(config)
    results["evaluation"] = eval_result
    results["accuracy"] = eval_result.get("accuracy")
    results["success"] = (eval_result.get("accuracy") or 0) >= min_accuracy
    results["improved"] = (eval_result.get("accuracy") or 0) > (baseline.get("accuracy") or 0)

    if (
        not results["success"]
        and strategy in ("auto", "full", "retrain")
    ):
        train_result = retrain_transfer_model(
            config,
            epochs=fine_tune_epochs,
            batch_size=16,
            architecture="efficientnetb0",
        )
        retrain_path = train_result.get("best_model_path") or train_result.get("model_path")
        if retrain_path and os.path.isfile(retrain_path):
            active_path = retrain_path
            _use_internal_model_paths(config, active_path)
            results["strategies"].append({"strategy": "retrain_transfer", **train_result})
            clear_model_cache()
            cal = ensure_model_calibrated(config, active_path)
            results["strategies"].append({"strategy": "calibrate_after_retrain", **cal})
            clear_model_cache()
            eval_result = evaluate_current_model(config)
            results["evaluation"] = eval_result
            results["accuracy"] = eval_result.get("accuracy")
            results["success"] = (eval_result.get("accuracy") or 0) >= min_accuracy
            results["improved"] = (eval_result.get("accuracy") or 0) > (baseline.get("accuracy") or 0)

    results["model_switched"] = active_path != source_path
    results["model_path"] = active_path
    return results
