"""End-to-end VGG16 research pipeline: clean, split, train, evaluate, publish."""

from __future__ import annotations

import gc
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from services.class_mapping_service import ensure_canonical_class_indices
from services.metrics_service import ACCURACY_TARGET
from services.training_utils import save_json


def _log(message: str, log_path: Optional[str] = None) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}] {message}"
    print(line, flush=True)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def prepare_dataset(config: Dict[str, Any], log_path: Optional[str] = None) -> Dict[str, Any]:
    """Clean raw data, rebuild stratified split, balance train set, validate labels."""
    from services.dataset_balancing_service import build_balanced_train_set
    from services.dataset_path_service import get_dataset_inventory, resolve_class_folder
    from services.dataset_quality_service import clean_raw_dataset
    from services.dataset_restore_service import merge_rejected_into_class
    from services.dataset_service import split_and_copy_dataset_from_filesystem
    from services.training_validation_service import validate_training_dataset

    rejected_root = config.get("REJECTED_DATASET_DIR") or os.path.join(config["DATASET_DIR"], "rejected")
    _log("Step 0/4: Restoring images from rejected folder (if any)...", log_path)
    restore_report = {}
    for label in config.get("CLASS_LABELS") or []:
        class_dir = resolve_class_folder(config["RAW_DATASET_DIR"], label)
        rejected_class = os.path.join(rejected_root, label)
        restore_report[label] = merge_rejected_into_class(
            class_dir,
            rejected_class,
            target_count=700,
        )
    _log(f"  restore merge: {restore_report}", log_path)

    _log("Step 1/4: Cleaning raw dataset (corrupt + exact duplicates only)...", log_path)
    clean_report = clean_raw_dataset(config, phash_threshold=-1, remove_duplicates=False)
    _log(
        f"  removed corrupt={clean_report.get('corrupted_removed', 0)}, "
        f"duplicates={clean_report.get('duplicates_removed', 0)}",
        log_path,
    )

    _log("Step 2/4: Rebuilding stratified train/validation/test split...", log_path)
    split_counts = split_and_copy_dataset_from_filesystem(
        config,
        deduplicate_hashes=False,
        phash_threshold=-1,
    )
    _log(f"  split counts: {split_counts}", log_path)

    _log("Step 3/4: Building balanced training set with augmentation...", log_path)
    balance_report = build_balanced_train_set(config)
    _log(f"  balanced train total={balance_report.get('total', balance_report)}", log_path)

    _log("Step 4/4: Validating dataset integrity and class folders...", log_path)
    validation = validate_training_dataset(config, fast=True)
    if not validation.get("valid"):
        raise ValueError(f"Dataset validation failed: {validation.get('errors', [])[:3]}")

    inventory = get_dataset_inventory(config, force_refresh=True)
    return {
        "restore_report": restore_report,
        "clean_report": clean_report,
        "split_counts": split_counts,
        "balance_report": balance_report,
        "validation": {
            "train_count": validation.get("train_count"),
            "validation_count": validation.get("validation_count"),
            "test_count": validation.get("test_count"),
            "corrupt_files": len(validation.get("corrupt_files") or []),
        },
        "inventory": inventory,
    }


def publish_trained_model(
    config: Dict[str, Any],
    source_path: str,
    training_result: Dict[str, Any],
    architecture: str = "efficientnetb0",
) -> str:
    """Copy best model to external dir as keras_model.h5 and refresh metadata."""
    from services.model_builder_service import ARCHITECTURES

    arch = (architecture or "efficientnetb0").lower()
    arch_label = ARCHITECTURES.get(arch, arch)

    from services.model_cache_service import clear_model_cache

    model_dir = (config.get("EXTERNAL_MODEL_DIR") or config.get("MODEL_DIR") or "").strip()
    if not model_dir:
        raise ValueError("EXTERNAL_MODEL_DIR is not configured.")

    os.makedirs(model_dir, exist_ok=True)
    ext_file = (config.get("EXTERNAL_MODEL_FILE") or "keras_model.h5").strip()
    primary_path = str(Path(model_dir) / ext_file)

    if os.path.abspath(source_path) != os.path.abspath(primary_path):
        shutil.copy2(source_path, primary_path)

    class_labels = list(config.get("CLASS_LABELS") or [])
    ensure_canonical_class_indices(config["CLASS_INDICES_PATH"], class_labels)

    best_val = training_result.get("best_val_accuracy")
    metadata = {
        "architecture": arch,
        "architecture_label": arch_label,
        "model_architecture_key": arch,
        "training_mode": "research",
        "training_mode_label": "Research Accuracy",
        "training_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "model_path": primary_path,
        "best_model_path": primary_path,
        "latest_model_path": primary_path,
        "validation_accuracy": best_val,
        "preprocessing": arch,
        "source": "research_pipeline",
    }
    save_json(config["MODEL_METADATA_PATH"], metadata)

    config["EXTERNAL_MODEL_PATH"] = primary_path
    config["MODEL_PATH"] = primary_path
    config["BEST_MODEL_PATH"] = primary_path
    config["LATEST_MODEL_PATH"] = primary_path
    config["USE_EXTERNAL_MODEL"] = True
    clear_model_cache()
    return primary_path


def train_vgg16_research(
    config: Dict[str, Any],
    *,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    architecture: str = "efficientnetb0",
    log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Memory-efficient two-phase transfer learning using ImageDataGenerator."""
    import time

    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")

    from services.preprocessing_service import get_architecture_preprocess_fn
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers.legacy import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    from services.accuracy_boost_service import _class_weights, _resolve_train_dir
    from services.model_builder_service import (
        ARCHITECTURES,
        build_classifier,
        save_classifier_bundle,
        unfreeze_last_layers,
    )
    from services.training_utils import get_keras

    architecture = (architecture or "efficientnetb0").lower()
    preprocess_input = get_architecture_preprocess_fn(architecture)
    arch_label = ARCHITECTURES.get(architecture, architecture)

    _log(
        f"Training {arch_label} research mode (epochs={epochs}, batch={batch_size}, lr={learning_rate})...",
        log_path,
    )

    start = time.time()
    get_keras()
    train_dir = _resolve_train_dir(config)
    validation_dir = config["VALIDATION_DIR"]
    class_labels = list(config["CLASS_LABELS"])
    class_weight = _class_weights(train_dir, class_labels)

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=12,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.10,
        brightness_range=(0.92, 1.08),
        horizontal_flip=False,
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_flow = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        class_mode="categorical",
        classes=class_labels,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )
    val_flow = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        class_mode="categorical",
        classes=class_labels,
        batch_size=batch_size,
        shuffle=False,
    )

    phase1_epochs = min(max(int(epochs), 12), 18)
    phase2_epochs = min(max(int(epochs) // 2, 8), 15)
    phase2_lr = learning_rate / 10.0

    model_dir = config["MODEL_DIR"]
    os.makedirs(model_dir, exist_ok=True)
    best_path = config["BEST_MODEL_PATH"]
    latest_path = config["LATEST_MODEL_PATH"]
    legacy_path = config["MODEL_PATH"]
    checkpoint_path = os.path.join(model_dir, f"{architecture}_research_checkpoint.weights.h5")

    model, _ = build_classifier(architecture, len(class_labels), learning_rate, trainable_backbone=False)
    callbacks_phase1 = [
        EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.35, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1,
        ),
    ]
    history1 = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=phase1_epochs,
        class_weight=class_weight,
        callbacks=callbacks_phase1,
        verbose=1,
    )

    unfreeze_last_layers(model, architecture, 12 if architecture == "vgg16" else 20)
    model.compile(
        optimizer=Adam(learning_rate=phase2_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    train_flow.reset()
    val_flow.reset()
    callbacks_phase2 = [
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.35, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1,
        ),
    ]
    history2 = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=phase2_epochs,
        class_weight=class_weight,
        callbacks=callbacks_phase2,
        verbose=1,
    )

    if os.path.isfile(checkpoint_path) and os.path.getsize(checkpoint_path) > 1000:
        model.load_weights(checkpoint_path)

    save_classifier_bundle(model, latest_path)
    save_classifier_bundle(model, best_path)
    save_classifier_bundle(model, legacy_path)

    hist1 = history1.history
    hist2 = history2.history
    val_scores = list(hist1.get("val_accuracy", [])) + list(hist2.get("val_accuracy", []))
    best_val = float(max(val_scores)) if val_scores else None

    from services.class_mapping_service import ensure_canonical_class_indices
    from services.training_service import _save_training_history_file

    class_indices = {name: idx for idx, name in enumerate(class_labels)}
    ensure_canonical_class_indices(config["CLASS_INDICES_PATH"], class_labels)

    config_payload = {
        "epochs": epochs,
        "phase1_epochs": phase1_epochs,
        "phase2_epochs": phase2_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "learning_rate_phase2": phase2_lr,
        "training_mode": "Research Accuracy",
        "training_mode_key": "research",
        "model_architecture": arch_label,
        "model_architecture_key": architecture,
        "training_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "model_path": legacy_path,
        "best_model_path": best_path,
        "latest_model_path": latest_path,
        "train_count": train_flow.samples,
        "validation_count": val_flow.samples,
        "best_validation_accuracy": best_val,
        "target_achieved": bool(best_val and best_val >= ACCURACY_TARGET),
        "two_phase_training": True,
        "duration_seconds": round(time.time() - start, 2),
    }

    combined = {k: hist1.get(k, []) + hist2.get(k, []) for k in set(hist1) | set(hist2)}

    class CombinedHistory:
        history = combined

    _save_training_history_file(CombinedHistory(), config["TRAINING_HISTORY_PATH"], config_payload)
    save_json(config["MODEL_METADATA_PATH"], {
        "architecture": architecture,
        "architecture_label": arch_label,
        "model_architecture_key": architecture,
        "training_mode": "research",
        "validation_accuracy": best_val,
        "training_date": config_payload["training_date"],
        "model_path": legacy_path,
        "preprocessing": architecture,
        "source": "research_pipeline",
    })

    import tensorflow as tf

    tf.keras.backend.clear_session()
    gc.collect()

    _log(
        f"  best_val_accuracy={best_val}, duration={config_payload['duration_seconds']}s",
        log_path,
    )
    return {
        "best_val_accuracy": best_val,
        "model_path": legacy_path,
        "best_model_path": best_path,
        "duration_seconds": config_payload["duration_seconds"],
    }


def evaluate_published_model(config: Dict[str, Any], log_path: Optional[str] = None) -> Dict[str, Any]:
    from services.evaluation_service import evaluate_model

    _log("Running holdout test evaluation...", log_path)
    result = evaluate_model()
    metrics = {
        "accuracy": result.get("accuracy"),
        "precision": result.get("precision"),
        "recall": result.get("recall"),
        "f1_score": result.get("f1_score"),
        "f1_macro": result.get("f1_macro"),
        "test_samples": result.get("test_samples"),
    }
    _log(f"  evaluation metrics: {json.dumps(metrics, default=str)}", log_path)

    metadata = {
        "test_accuracy": result.get("accuracy"),
        "precision": result.get("precision"),
        "recall": result.get("recall"),
        "f1_macro": result.get("f1_macro"),
        "evaluated_at": result.get("evaluated_at"),
    }
    meta_path = config.get("MODEL_METADATA_PATH")
    if meta_path and os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
        except (OSError, json.JSONDecodeError):
            existing = {}
        existing.update(metadata)
        save_json(meta_path, existing)

    return result


def metrics_meet_target(result: Dict[str, Any], target: float = ACCURACY_TARGET) -> bool:
    keys = ("accuracy", "precision", "recall", "f1_score")
    values = [float(result.get(key) or 0) for key in keys]
    return all(value >= target for value in values)


def run_vgg16_research_pipeline(
    config: Dict[str, Any],
    *,
    target_accuracy: float = ACCURACY_TARGET,
    max_attempts: int = 2,
    log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Full automated pipeline until evaluation metrics exceed target."""
    if log_path is None:
        log_path = str(Path(config.get("BASE_DIR", ".")) / "logs" / "vgg16_research_pipeline.log")

    summary: Dict[str, Any] = {
        "target": target_accuracy,
        "attempts": [],
        "success": False,
        "log_path": log_path,
    }

    _log("=== VGG16 Research Pipeline START ===", log_path)
    prep = prepare_dataset(config, log_path=log_path)
    summary["dataset"] = prep

    hyperparams = [
        {"epochs": 20, "batch_size": 4, "learning_rate": 1e-4, "architecture": "efficientnetb0"},
        {"epochs": 15, "batch_size": 2, "learning_rate": 5e-5, "architecture": "efficientnetb0"},
    ]

    for attempt in range(1, max_attempts + 1):
        params = hyperparams[min(attempt - 1, len(hyperparams) - 1)]
        _log(f"=== Training attempt {attempt}/{max_attempts} params={params} ===", log_path)

        training = train_vgg16_research(config, log_path=log_path, **params)
        model_path = training.get("best_model_path") or training.get("model_path")
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError("Training completed but no model file was saved.")

        published = publish_trained_model(
            config,
            model_path,
            training,
            architecture=params.get("architecture", "efficientnetb0"),
        )

        try:
            from services.calibration_service import ensure_model_calibrated

            _log("Calibrating model on validation holdout...", log_path)
            ensure_model_calibrated(config, published)
        except Exception as exc:
            _log(f"Calibration skipped: {exc}", log_path)

        evaluation = evaluate_published_model(config, log_path=log_path)
        attempt_summary = {
            "attempt": attempt,
            "hyperparams": params,
            "training": {
                "best_val_accuracy": training.get("best_val_accuracy"),
                "duration_seconds": training.get("duration_seconds"),
            },
            "published_model": published,
            "evaluation": {
                "accuracy": evaluation.get("accuracy"),
                "precision": evaluation.get("precision"),
                "recall": evaluation.get("recall"),
                "f1_score": evaluation.get("f1_score"),
            },
        }
        summary["attempts"].append(attempt_summary)

        if metrics_meet_target(evaluation, target_accuracy):
            summary["success"] = True
            summary["final_evaluation"] = evaluation
            summary["model_path"] = published
            _log("=== TARGET ACHIEVED ===", log_path)
            break

        _log(
            f"Attempt {attempt} below target "
            f"(acc={evaluation.get('accuracy'):.4f}, target={target_accuracy})",
            log_path,
        )
        gc.collect()

    if not summary["success"] and summary["attempts"]:
        summary["final_evaluation"] = summary["attempts"][-1]["evaluation"]
        summary["model_path"] = summary["attempts"][-1]["published_model"]

    _log(f"=== Pipeline END success={summary['success']} ===", log_path)
    return summary
