import gc
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from flask import current_app
from sklearn.utils.class_weight import compute_class_weight

from services.class_mapping_service import ensure_canonical_class_indices, validate_class_mapping
from services.metrics_service import ACCURACY_TARGET, training_recommendations
from services.model_builder_service import (
    ARCHITECTURES,
    FINE_TUNE_LAYERS,
    build_classifier,
    resolve_architecture,
    save_classifier_bundle,
    unfreeze_last_layers,
)
from services.preprocessing_service import get_architecture_preprocess_fn
from services.training_status_service import (
    mark_training_completed,
    mark_training_failed,
    mark_training_started,
    update_training_status,
)
from services.training_utils import get_keras, save_json
from services.training_validation_service import ensure_training_dataset_ready

DISPLAY_NAMES = {
    "naskhi": "Naskhi",
    "diwani": "Diwani",
    "diwani_jali": "Diwani Jali",
    "tsuluts": "Tsuluts",
}

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ULTRA_TRAIN_PER_CLASS = 25
ULTRA_VAL_PER_CLASS = 10
QUICK_TRAIN_PER_CLASS = 50
QUICK_VAL_PER_CLASS = 15

MODE_LABELS = {
    "ultra_fast": "Ultra Fast Demo",
    "fast": "Fast Demo",
    "research": "Research Accuracy",
    "normal": "Research Accuracy",
}


def _is_image_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _dir_has_images(directory: str, class_labels: List[str]) -> bool:
    if not os.path.isdir(directory):
        return False
    for class_name in class_labels:
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            if any(_is_image_file(f) for f in os.listdir(class_dir)):
                return True
    return False


def _resolve_train_dir(config, training_mode: str, use_optimized: bool = True) -> str:
    if use_optimized:
        optimized_train = os.path.join(config["OPTIMIZED_MODEL_DIR"], "train")
        if _dir_has_images(optimized_train, config["CLASS_LABELS"]):
            return optimized_train

    if training_mode in ("research", "normal"):
        balanced = os.path.join(config["PROCESSED_BALANCED_DIR"], "train")
        if _dir_has_images(balanced, config["CLASS_LABELS"]):
            return balanced

    return config["TRAIN_DIR"]


def _resolve_validation_dir(config, use_optimized: bool = True) -> str:
    if use_optimized:
        optimized_val = os.path.join(config["OPTIMIZED_MODEL_DIR"], "validation")
        if _dir_has_images(optimized_val, config["CLASS_LABELS"]):
            return optimized_val
    return config["VALIDATION_DIR"]


def _subset_limits(training_mode: str) -> Tuple[Optional[int], Optional[int]]:
    if training_mode == "ultra_fast":
        return ULTRA_TRAIN_PER_CLASS, ULTRA_VAL_PER_CLASS
    if training_mode == "fast":
        return QUICK_TRAIN_PER_CLASS, QUICK_VAL_PER_CLASS
    return None, None


def _count_images_per_class(directory: str, class_labels: List[str]) -> Dict[str, int]:
    counts = {}
    for cls in class_labels:
        cls_dir = os.path.join(directory, cls)
        if not os.path.isdir(cls_dir):
            counts[cls] = 0
            continue
        counts[cls] = sum(
            1 for f in os.listdir(cls_dir)
            if _is_image_file(f) and os.path.isfile(os.path.join(cls_dir, f))
        )
    return counts


def _make_subset_directory(source_dir: str, class_labels: List[str], max_per_class: Optional[int], seed: int = 123) -> str:
    import shutil
    import tempfile

    if not max_per_class:
        return source_dir

    rng = random.Random(seed)
    subset_root = tempfile.mkdtemp(prefix="khat_subset_")
    for cls in class_labels:
        src_cls = os.path.join(source_dir, cls)
        dst_cls = os.path.join(subset_root, cls)
        os.makedirs(dst_cls, exist_ok=True)
        if not os.path.isdir(src_cls):
            continue
        files = [f for f in os.listdir(src_cls) if _is_image_file(f)]
        if len(files) > max_per_class:
            files = rng.sample(files, max_per_class)
        for fname in files:
            shutil.copy2(os.path.join(src_cls, fname), os.path.join(dst_cls, fname))
    return subset_root


def _directory_has_extra_classes(directory: str, class_labels: List[str]) -> bool:
    if not os.path.isdir(directory):
        return False
    allowed = set(class_labels)
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isdir(path) and name not in allowed:
            return True
    return False


def _build_generator_datasets(
    train_dir: str,
    validation_dir: str,
    class_labels: List[str],
    batch_size: int,
    architecture: str,
    training_mode: str,
):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    preprocess_fn = get_architecture_preprocess_fn(architecture)
    rot = 12 if training_mode in ("research", "normal") else 8 if training_mode == "fast" else 0
    zoom = 0.12 if training_mode in ("research", "normal") else 0.08 if training_mode == "fast" else 0
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=rot,
        width_shift_range=0.04 if training_mode != "ultra_fast" else 0,
        height_shift_range=0.04 if training_mode != "ultra_fast" else 0,
        zoom_range=zoom,
        horizontal_flip=training_mode != "ultra_fast",
        brightness_range=(0.92, 1.08) if training_mode in ("research", "normal") else None,
    )
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    train_flow = train_gen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        class_mode="categorical",
        classes=class_labels,
        batch_size=batch_size,
        shuffle=True,
        seed=123,
    )
    val_flow = val_gen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        class_mode="categorical",
        classes=class_labels,
        batch_size=batch_size,
        shuffle=False,
    )
    class_indices = dict(train_flow.class_indices)
    train_labels = []
    per_class_counts = _count_images_per_class(train_dir, class_labels)
    for cls, count in per_class_counts.items():
        train_labels.extend([class_indices[cls]] * count)
    return (
        train_flow,
        val_flow,
        train_labels,
        class_indices,
        train_flow.samples,
        val_flow.samples,
    )


def _build_directory_datasets(
    train_dir: str,
    validation_dir: str,
    class_labels: List[str],
    batch_size: int,
    architecture: str,
    training_mode: str,
    cache_to_memory: bool = False,
):
    import tensorflow as tf

    preprocess_fn = get_architecture_preprocess_fn(architecture)

    if _directory_has_extra_classes(train_dir, class_labels) or _directory_has_extra_classes(
        validation_dir, class_labels
    ):
        return _build_generator_datasets(
            train_dir, validation_dir, class_labels, batch_size, architecture, training_mode
        )

    autotune = tf.data.AUTOTUNE
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_labels,
        image_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
        seed=123,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_labels,
        image_size=(224, 224),
        batch_size=batch_size,
        shuffle=False,
    )

    class_indices = {name: idx for idx, name in enumerate(class_labels)}
    per_class_counts = _count_images_per_class(train_dir, class_labels)
    train_labels = []
    for cls, count in per_class_counts.items():
        train_labels.extend([class_indices[cls]] * count)

    train_count = sum(per_class_counts.values())
    val_count = sum(_count_images_per_class(validation_dir, class_labels).values())

    augmentation_layers = []
    if training_mode != "ultra_fast":
        rot = 0.04 if training_mode == "fast" else 0.05
        zoom = (-0.05, 0.05) if training_mode == "fast" else (-0.10, 0.10)
        contrast = 0.08 if training_mode == "fast" else 0.10
        augmentation_layers = [
            tf.keras.layers.RandomRotation(rot),
            tf.keras.layers.RandomZoom(zoom),
            tf.keras.layers.RandomContrast(contrast),
        ]
        if training_mode in ("research", "normal"):
            augmentation_layers.append(tf.keras.layers.RandomTranslation(0.04, 0.04))
            augmentation_layers.append(tf.keras.layers.RandomBrightness(0.08))

    def _preprocess_batch(images, labels):
        images = tf.cast(images, tf.float32)
        images = preprocess_fn(images)
        return images, labels

    def _augment_batch(images, labels):
        for layer in augmentation_layers:
            images = layer(images, training=True)
        return images, labels

    train_ds = train_ds.map(_preprocess_batch, num_parallel_calls=autotune)
    if training_mode != "ultra_fast":
        train_ds = train_ds.map(_augment_batch, num_parallel_calls=autotune)
    val_ds = val_ds.map(_preprocess_batch, num_parallel_calls=autotune)

    if cache_to_memory:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    return train_ds, val_ds, train_labels, class_indices, train_count, val_count


WEAK_CLASS_WEIGHT_BOOST = {
    "diwani_jali": 1.75,
    "diwani": 1.45,
    "naskhi": 1.25,
}


def _class_weights_from_labels(labels: List[int], class_indices: Optional[dict] = None) -> dict:
    if not labels:
        return {}
    unique = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=unique, y=labels)
    class_weight = {int(class_id): float(weight) for class_id, weight in zip(unique, weights)}
    if class_indices:
        index_to_name = {idx: name for name, idx in class_indices.items()}
        for class_id in list(class_weight.keys()):
            cls_name = index_to_name.get(class_id)
            boost = WEAK_CLASS_WEIGHT_BOOST.get(cls_name)
            if boost:
                class_weight[class_id] *= boost
    return class_weight


def _progress_callback(total_epochs: int, phase: str, phase_progress: Tuple[int, int]):
    from tensorflow.keras.callbacks import Callback

    start_pct, end_pct = phase_progress

    class TrainingProgressCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            pct = start_pct + int(((epoch + 1) / max(total_epochs, 1)) * (end_pct - start_pct))
            update_training_status(
                "running",
                f"{phase}: epoch {epoch + 1}/{total_epochs}",
                current_stage=phase,
                phase=phase,
                current_epoch=epoch + 1,
                total_epochs=total_epochs,
                progress_percent=min(pct, end_pct),
                training_phase=phase,
                training_accuracy=float(logs.get("accuracy", 0)),
                validation_accuracy=float(logs.get("val_accuracy", 0)),
                training_loss=float(logs.get("loss", 0)),
                validation_loss=float(logs.get("val_loss", 0)),
                current_accuracy=float(logs.get("accuracy", 0)),
                current_val_accuracy=float(logs.get("val_accuracy", 0)),
                current_loss=float(logs.get("loss", 0)),
                current_val_loss=float(logs.get("val_loss", 0)),
            )

    return TrainingProgressCallback()


def _checkpoint_weights_path(model_path: str) -> str:
    base, _ = os.path.splitext(model_path)
    return f"{base}_checkpoint.weights.h5"


def _make_callbacks(
    total_epochs: int,
    phase: str,
    phase_progress: Tuple[int, int],
    checkpoint_weights_path: str,
    patience: int,
    *,
    monitor: str = "val_loss",
):
    from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    callbacks = [
        _progress_callback(total_epochs, phase, phase_progress),
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            mode="max" if monitor == "val_accuracy" else "min",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]
    if checkpoint_weights_path:
        callbacks.append(
            ModelCheckpoint(
                checkpoint_weights_path,
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=True,
                mode="max",
                verbose=1,
            )
        )
    csv_path = current_app.config.get("TRAINING_CSV_LOG_PATH")
    if csv_path:
        callbacks.append(CSVLogger(csv_path, append=False))
    return callbacks


def _save_class_indices(class_indices: dict, path: str, class_labels: list) -> None:
    ensure_canonical_class_indices(path, class_labels)


def _save_best_model_metadata(config_payload: dict, config, test_accuracy=None) -> None:
    if not config_payload.get("best_model_updated"):
        return
    payload = {
        "architecture": config_payload.get("model_architecture_key"),
        "architecture_label": config_payload.get("model_architecture"),
        "validation_accuracy": config_payload.get("best_validation_accuracy"),
        "test_accuracy": test_accuracy,
        "training_mode": config_payload.get("training_mode_key"),
        "training_mode_label": config_payload.get("training_mode"),
        "training_date": config_payload.get("training_date"),
        "dataset_version": {
            "train_count": config_payload.get("train_count"),
            "validation_count": config_payload.get("validation_count"),
            "test_count": config_payload.get("test_count"),
        },
        "class_mapping": config_payload.get("class_names"),
        "best_model_path": config_payload.get("best_model_path"),
    }
    save_json(config["BEST_MODEL_METADATA_PATH"], payload)


def _save_training_history_file(history, path: str, config_payload: dict) -> dict:
    hist = history.history
    summary = {
        "config": config_payload,
        "history": {key: [float(v) for v in values] for key, values in hist.items()},
        "accuracy": [float(v) for v in hist.get("accuracy", [])],
        "val_accuracy": [float(v) for v in hist.get("val_accuracy", [])],
        "loss": [float(v) for v in hist.get("loss", [])],
        "val_loss": [float(v) for v in hist.get("val_loss", [])],
        "final_training_accuracy": float(hist["accuracy"][-1]) if hist.get("accuracy") else None,
        "final_validation_accuracy": float(hist["val_accuracy"][-1]) if hist.get("val_accuracy") else None,
        "final_training_loss": float(hist["loss"][-1]) if hist.get("loss") else None,
        "final_validation_loss": float(hist["val_loss"][-1]) if hist.get("val_loss") else None,
        "best_validation_accuracy": float(max(hist["val_accuracy"])) if hist.get("val_accuracy") else None,
    }
    save_json(path, summary)
    return summary


def _save_model_metadata(config_payload: dict, config) -> None:
    arch_key = config_payload.get("model_architecture_key")
    metadata = {
        "architecture": arch_key,
        "architecture_label": config_payload.get("model_architecture"),
        "model_architecture_key": arch_key,
        "training_mode": config_payload.get("training_mode_key"),
        "training_mode_label": config_payload.get("training_mode"),
        "validation_accuracy": config_payload.get("best_validation_accuracy"),
        "training_date": config_payload.get("training_date"),
        "model_path": config_payload.get("model_path"),
        "best_model_path": config_payload.get("best_model_path"),
        "latest_model_path": config_payload.get("latest_model_path"),
        "batch_size": config_payload.get("batch_size"),
        "class_weights_enabled": config_payload.get("class_weights_enabled", True),
        "preprocessing": arch_key if arch_key in ("vgg16", "efficientnetb0", "mobilenetv2") else "teachable_machine",
        "source": "vgg16_research_training" if arch_key == "vgg16" else "transfer_learning",
    }
    save_json(config["MODEL_METADATA_PATH"], metadata)


def _read_best_val_accuracy(best_path: str) -> Optional[float]:
    if not os.path.isfile(best_path):
        return None
    try:
        import json
        history_path = current_app.config["TRAINING_HISTORY_PATH"]
        if os.path.isfile(history_path):
            with open(history_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data.get("best_validation_accuracy") or data.get("config", {}).get("best_validation_accuracy")
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        pass
    return None


def _fit_phase(model, train_ds, val_ds, epochs, class_weight, callbacks):
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )


def _apply_mode_settings(epochs: int, batch_size: int, training_mode: str) -> Tuple[int, int, int, List[int], int]:
    requested = int(epochs)
    requested_batch = max(int(batch_size), 1)
    fallback = [32, 16, 8, 4, 2]
    batch_candidates = [requested_batch] + [b for b in fallback if b < requested_batch]
    if requested_batch not in fallback and requested_batch not in batch_candidates[1:]:
        batch_candidates = [requested_batch] + [b for b in fallback if b != requested_batch]
    if training_mode == "ultra_fast":
        return requested, 1, 0, batch_candidates, 5
    if training_mode == "fast":
        phase_epochs = min(max(requested, 3), 5)
        return requested, phase_epochs, 0, batch_candidates, 5
    phase1 = min(max(requested, 15), 20)
    phase2 = min(max(requested // 2, 10), 20)
    return requested, phase1, phase2, batch_candidates, 7


def _cleanup_memory():
    import tensorflow as tf
    tf.keras.backend.clear_session()
    gc.collect()


def train_model(
    epochs=12,
    batch_size=16,
    learning_rate=1e-4,
    training_mode="fast",
    model_architecture="efficientnetb0",
    fast_mode=False,
    quick_subset=False,
    use_optimized_images=True,
    cache_to_memory=False,
    auto_batch_fallback=True,
    enable_class_weights=True,
    enable_augmentation=True,
):
    if training_mode == "normal":
        training_mode = "research"
    if training_mode not in MODE_LABELS:
        training_mode = "fast"

    start_time = time.time()
    get_keras()
    config = current_app.config
    class_labels = list(config["CLASS_LABELS"])
    architecture = resolve_architecture(model_architecture, training_mode)
    validation_dir = _resolve_validation_dir(config, use_optimized_images)
    best_path = config["BEST_MODEL_PATH"]
    checkpoint_weights_path = _checkpoint_weights_path(best_path)
    latest_path = config["LATEST_MODEL_PATH"]
    legacy_path = config["MODEL_PATH"]

    os.makedirs(config["MODEL_DIR"], exist_ok=True)
    requested_epochs, phase1_epochs, phase2_epochs, batch_candidates, es_patience = _apply_mode_settings(
        epochs, batch_size, training_mode
    )
    if not auto_batch_fallback:
        batch_candidates = [int(batch_size)]

    mark_training_started(
        message="Validating dataset",
        current_stage="Dataset validation",
        phase="Dataset validation",
        total_epochs=phase1_epochs + phase2_epochs,
        training_mode=training_mode,
        progress_percent=0,
    )

    subset_dirs = []
    batch_reduced = False

    try:
        validation_report = ensure_training_dataset_ready(config)
        test_count = validation_report.get("test_count", 0)

        validate_class_mapping(config=config, loader_class_names=class_labels)

        if training_mode in ("research", "normal"):
            from services.dataset_balancing_service import build_balanced_train_set

            if not config.get("PRESERVE_DATASET_IMAGES"):
                from services.dataset_quality_service import clean_raw_dataset

                update_training_status("running", "Cleaning dataset", current_stage="Dataset cleaning", phase="Preprocessing", progress_percent=5)
                try:
                    clean_raw_dataset(config)
                except Exception:
                    pass

            update_training_status("running", "Building balanced training set", current_stage="Balancing", phase="Preprocessing", progress_percent=10)
            try:
                build_balanced_train_set(config)
            except Exception:
                pass

        update_training_status("running", "Loading dataset", current_stage="Loading dataset", phase="Preprocessing", progress_percent=15)
        train_dir = _resolve_train_dir(config, training_mode, use_optimized_images)

        train_max, val_max = _subset_limits(training_mode)
        if train_max:
            train_dir = _make_subset_directory(train_dir, class_labels, train_max)
            subset_dirs.append(train_dir)
        if val_max:
            validation_dir = _make_subset_directory(validation_dir, class_labels, val_max)
            subset_dirs.append(validation_dir)

        last_error = None
        history = None
        model = None
        class_indices = None
        train_samples = val_samples = 0
        used_batch = batch_size
        class_weight = {}
        phase1_lr = learning_rate
        phase2_lr = learning_rate / 10.0
        is_research = training_mode in ("research", "normal")

        for index, batch_size_try in enumerate(batch_candidates):
            if index > 0:
                batch_reduced = True
            try:
                _cleanup_memory()
                train_ds, val_ds, train_labels, class_indices, train_samples, val_samples = _build_directory_datasets(
                    train_dir,
                    validation_dir,
                    class_labels,
                    batch_size_try,
                    architecture,
                    training_mode if enable_augmentation else "ultra_fast",
                    cache_to_memory=cache_to_memory,
                )
                class_weight = (
                    _class_weights_from_labels(train_labels, class_indices) if enable_class_weights else {}
                )
                used_batch = batch_size_try

                update_training_status(
                    "running",
                    "Building model",
                    current_stage="Model building",
                    phase="Model building",
                    progress_percent=25,
                    model_architecture=ARCHITECTURES.get(architecture, architecture),
                )

                if is_research:
                    model, _ = build_classifier(architecture, len(class_labels), phase1_lr, trainable_backbone=False)
                    cb1 = _make_callbacks(
                        phase1_epochs,
                        "Feature Extraction",
                        (30, 60),
                        checkpoint_weights_path,
                        es_patience,
                        monitor="val_accuracy",
                    )
                    h1 = _fit_phase(model, train_ds, val_ds, phase1_epochs, class_weight, cb1)

                    update_training_status(
                        "running",
                        "Fine-tuning model",
                        current_stage="Fine-Tuning",
                        phase="Fine-Tuning",
                        progress_percent=60,
                    )
                    unfreeze_last_layers(model, architecture, FINE_TUNE_LAYERS.get(architecture, 4))
                    model.compile(
                        optimizer=__import__("tensorflow").keras.optimizers.legacy.Adam(learning_rate=phase2_lr),
                        loss="categorical_crossentropy",
                        metrics=["accuracy"],
                    )
                    cb2 = _make_callbacks(
                        phase2_epochs,
                        "Fine-Tuning",
                        (60, 90),
                        checkpoint_weights_path,
                        es_patience,
                        monitor="val_accuracy",
                    )
                    h2 = _fit_phase(model, train_ds, val_ds, phase2_epochs, class_weight, cb2)

                    combined = {k: h1.history.get(k, []) + h2.history.get(k, []) for k in set(h1.history) | set(h2.history)}

                    class CombinedHistory:
                        history = combined
                    history = CombinedHistory()
                else:
                    model, _ = build_classifier(architecture, len(class_labels), learning_rate, trainable_backbone=False)
                    total = phase1_epochs if phase1_epochs else 1
                    cb = _make_callbacks(
                        total, "Training", (30, 90), checkpoint_weights_path if is_research else "", es_patience
                    )
                    history = _fit_phase(model, train_ds, val_ds, total, class_weight, cb)
                break
            except Exception as exc:
                last_error = exc
                err_name = type(exc).__name__
                if auto_batch_fallback and (
                    "ResourceExhausted" in err_name or isinstance(exc, MemoryError) or "OOM" in str(exc)
                ):
                    continue
                raise

        if model is None:
            raise MemoryError("Training failed due to insufficient memory.") from last_error

        update_training_status("running", "Saving model", current_stage="Saving model", phase="Saving", progress_percent=95)

        if os.path.isfile(checkpoint_weights_path) and os.path.getsize(checkpoint_weights_path) > 1000:
            model.load_weights(checkpoint_weights_path)

        save_classifier_bundle(model, latest_path)

        hist = history.history
        new_best_val = float(max(hist["val_accuracy"])) if hist.get("val_accuracy") else None
        previous_best_val = _read_best_val_accuracy(best_path)

        if new_best_val is not None and (previous_best_val is None or new_best_val >= previous_best_val):
            save_classifier_bundle(model, best_path)
            best_kept = True
        elif os.path.isfile(best_path):
            best_kept = False
        else:
            save_classifier_bundle(model, best_path)
            best_kept = True

        save_classifier_bundle(model, legacy_path)

        _save_class_indices(class_indices, config["CLASS_INDICES_PATH"], class_labels)
        if class_weight:
            save_json(config["CLASS_WEIGHTS_PATH"], {"class_weights": class_weight, "enabled": enable_class_weights})

        duration_seconds = time.time() - start_time
        actual_epochs = len(hist.get("accuracy", []))
        best_val = new_best_val
        final_val = float(hist["val_accuracy"][-1]) if hist.get("val_accuracy") else None
        target_achieved = best_val is not None and best_val >= ACCURACY_TARGET

        config_payload = {
            "epochs": requested_epochs,
            "phase1_epochs": phase1_epochs,
            "phase2_epochs": phase2_epochs,
            "actual_epochs_completed": actual_epochs,
            "batch_size": used_batch,
            "batch_size_requested": batch_size,
            "batch_size_reduced": batch_reduced,
            "learning_rate_phase1": phase1_lr,
            "learning_rate_phase2": phase2_lr,
            "learning_rate": learning_rate,
            "training_mode": MODE_LABELS[training_mode],
            "training_mode_key": training_mode,
            "model_architecture": ARCHITECTURES.get(architecture, architecture),
            "model_architecture_key": architecture,
            "class_weights": class_weight,
            "class_weights_enabled": enable_class_weights,
            "quick_subset": training_mode in ("ultra_fast", "fast"),
            "two_phase_training": is_research,
            "use_optimized_images": use_optimized_images,
            "cache_to_memory": cache_to_memory,
            "augmentation_enabled": enable_augmentation,
            "training_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "model_path": legacy_path,
            "latest_model_path": latest_path,
            "best_model_path": best_path,
            "best_model_updated": best_kept,
            "train_count": train_samples,
            "validation_count": val_samples,
            "test_count": test_count,
            "class_names": class_labels,
            "duration_seconds": round(duration_seconds, 2),
            "accuracy_target": ACCURACY_TARGET,
            "target_achieved": target_achieved,
            "best_validation_accuracy": best_val,
            "recommendations": training_recommendations(final_val, None, training_mode),
        }

        summary = _save_training_history_file(history, config["TRAINING_HISTORY_PATH"], config_payload)
        _save_model_metadata(config_payload, config)
        _save_best_model_metadata(config_payload, config)

        completion_msg = "Training completed successfully."
        if batch_reduced:
            completion_msg += " Batch size was automatically reduced due to memory limitations."
        if target_achieved:
            completion_msg += " Target achieved on validation."
        else:
            completion_msg += " Validation below 85% target — run evaluation on the test set."

        mark_training_completed(
            completion_msg,
            best_path if best_kept and os.path.isfile(best_path) else latest_path,
            progress_percent=100,
            phase="Completed",
            target_achieved=target_achieved,
            best_validation_accuracy=best_val,
            batch_size_reduced=batch_reduced,
        )

        return {
            "epochs": requested_epochs,
            "phase1_epochs": phase1_epochs,
            "phase2_epochs": phase2_epochs,
            "actual_epochs_completed": actual_epochs,
            "batch_size": used_batch,
            "batch_size_reduced": batch_reduced,
            "learning_rate": learning_rate,
            "training_mode": training_mode,
            "model_architecture": architecture,
            "train_samples": train_samples,
            "validation_samples": val_samples,
            "final_train_accuracy": summary["final_training_accuracy"],
            "final_val_accuracy": summary["final_validation_accuracy"],
            "best_val_accuracy": best_val,
            "target_achieved": target_achieved,
            "final_train_loss": summary["final_training_loss"],
            "final_val_loss": summary["final_validation_loss"],
            "model_path": legacy_path,
            "best_model_path": best_path,
            "latest_model_path": latest_path,
            "duration_seconds": duration_seconds,
            "recommendations": config_payload["recommendations"],
        }
    except Exception as exc:
        mark_training_failed(str(exc), progress_percent=0)
        raise
    finally:
        for path in subset_dirs:
            try:
                import shutil
                shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass
        _cleanup_memory()
