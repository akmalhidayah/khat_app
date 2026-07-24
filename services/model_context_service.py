"""Load active model path and training/evaluation context for UI and prediction."""

import json
import os
from typing import Dict, Optional

from flask import current_app

from services.dataset_readiness_service import is_valid_trained_model
from services.metrics_service import ACCURACY_TARGET, target_status
from services.teachable_machine_service import get_tm_model_context, is_teachable_machine_available
from services.external_assets_service import should_use_keras_model


MODE_DISPLAY = {
    "ultra_fast": "Demo Model",
    "fast": "Fast Experimental Model",
    "research": "Research Model",
    "normal": "Research Model",
}

DEMO_MODES = {"ultra_fast", "fast"}


def _candidate_model_paths(config) -> list:
    paths = []
    if config.get("USE_EXTERNAL_MODEL") and config.get("EXTERNAL_MODEL_PATH"):
        paths.append(config["EXTERNAL_MODEL_PATH"])
    local_best = os.path.join(config.get("BASE_DIR", ""), "model", "khat_best.keras")
    local_latest = os.path.join(config.get("BASE_DIR", ""), "model", "khat_latest.keras")
    if os.path.isfile(local_best):
        paths.append(local_best)
    if os.path.isfile(local_latest):
        paths.append(local_latest)
    paths.extend(
        [
            config.get("BEST_MODEL_PATH"),
            config.get("LATEST_MODEL_PATH"),
            config.get("MODEL_PATH"),
            config.get("LEGACY_BEST_MODEL_PATH"),
        ]
    )
    return paths


def get_active_model_path(config=None) -> str:
    if config is None:
        config = current_app.config
    for path in _candidate_model_paths(config):
        if path and is_valid_trained_model(path):
            return path
    return config["MODEL_PATH"]


def load_model_metadata(config=None) -> Dict:
    if config is None:
        config = current_app.config
    path = config.get("MODEL_METADATA_PATH")
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def load_training_config(config=None) -> Dict:
    if config is None:
        config = current_app.config
    path = config.get("TRAINING_HISTORY_PATH")
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("config", {}) if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def load_evaluation_summary(config=None) -> Dict:
    if config is None:
        config = current_app.config
    path = config.get("EVALUATION_RESULT_PATH")
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def get_model_context(config=None) -> Dict:
    if config is None:
        config = current_app.config

    if is_teachable_machine_available(config) and not should_use_keras_model(config):
        return get_tm_model_context(config)

    train_cfg = load_training_config(config)
    meta = load_model_metadata(config)
    eval_data = load_evaluation_summary(config)
    if config.get("USE_EXTERNAL_MODEL") and meta.get("model_architecture_key"):
        mode_key = meta.get("training_mode", "external")
        arch_key = meta.get("model_architecture_key", "keras_h5")
    else:
        mode_key = train_cfg.get("training_mode_key", meta.get("training_mode", "unknown"))
        arch_key = train_cfg.get("model_architecture_key", meta.get("architecture", "efficientnetb0"))
    val_acc = train_cfg.get("best_validation_accuracy") or train_cfg.get("final_validation_accuracy") or meta.get("validation_accuracy")
    test_acc = eval_data.get("accuracy")
    f1_macro = eval_data.get("f1_macro")

    is_demo = mode_key in DEMO_MODES
    mode_label = MODE_DISPLAY.get(mode_key, train_cfg.get("training_mode", "Unknown"))
    target_val = target_status(val_acc, ACCURACY_TARGET)
    target_test = target_status(test_acc, ACCURACY_TARGET)
    target_f1 = target_status(f1_macro, ACCURACY_TARGET)

    active_path = get_active_model_path(config)
    using_best = active_path == config.get("BEST_MODEL_PATH") and os.path.isfile(active_path)

    return {
        "active_model_path": active_path,
        "using_best_model": using_best,
        "model_exists": is_valid_trained_model(active_path),
        "model_source": "External Keras H5" if config.get("USE_EXTERNAL_MODEL") else "CNN Transfer Learning",
        "model_runtime": "TensorFlow Keras",
        "training_mode_key": mode_key,
        "training_mode_label": mode_label,
        "is_demo_model": is_demo,
        "is_research_model": mode_key in ("research", "normal"),
        "model_architecture": (
            meta.get("architecture_label")
            if config.get("USE_EXTERNAL_MODEL")
            else train_cfg.get("model_architecture", meta.get("architecture_label", "EfficientNetB0 Transfer Learning"))
        ),
        "model_architecture_key": arch_key,
        "epochs": train_cfg.get("epochs"),
        "phase1_epochs": train_cfg.get("phase1_epochs"),
        "phase2_epochs": train_cfg.get("phase2_epochs"),
        "train_count": train_cfg.get("train_count"),
        "validation_count": train_cfg.get("validation_count"),
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc,
        "f1_macro": f1_macro,
        "target_validation": target_val,
        "target_test": target_test,
        "target_f1": target_f1,
        "target_achieved": bool(target_test.get("achieved")),
        "training_date": train_cfg.get("training_date") or meta.get("training_date"),
        "two_phase_training": train_cfg.get("two_phase_training", False),
        "demo_prediction_only": is_demo,
        "batch_size": train_cfg.get("batch_size"),
        "class_weights_enabled": train_cfg.get("class_weights_enabled", True),
    }


def is_teachable_machine_active(model_ctx: Optional[Dict] = None) -> bool:
    if model_ctx is None:
        model_ctx = get_model_context()
    return model_ctx.get("model_architecture_key") == "teachable_machine"


def get_model_short_name(model_ctx: Optional[Dict] = None) -> str:
    if model_ctx is None:
        model_ctx = get_model_context()
    if is_teachable_machine_active(model_ctx):
        return "Teachable Machine"
    arch = model_ctx.get("model_architecture") or "EfficientNetB0 Transfer Learning"
    return arch.split(" Transfer")[0].split(" Image")[0].strip()


def get_model_subtitle(model_ctx: Optional[Dict] = None) -> str:
    if model_ctx is None:
        model_ctx = get_model_context()
    if is_teachable_machine_active(model_ctx):
        return "Arabic Calligraphy Khat Classification with Teachable Machine Image Model"
    short = get_model_short_name(model_ctx)
    return f"Arabic Calligraphy Khat Classification with {short} Transfer Learning"


def get_brand_chip_label(model_ctx: Optional[Dict] = None) -> str:
    if model_ctx is None:
        model_ctx = get_model_context()
    if is_teachable_machine_active(model_ctx):
        return "Arabic Khat AI"
    return f"{get_model_short_name(model_ctx)} AI"


def get_model_display_label(model_ctx: Optional[Dict] = None) -> str:
    if model_ctx is None:
        model_ctx = get_model_context()
    if is_teachable_machine_active(model_ctx):
        return model_ctx.get("model_architecture", "Teachable Machine Image Model")
    short = get_model_short_name(model_ctx)
    return f"Current Research Model ({short})"
