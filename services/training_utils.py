import json
import os
import traceback
from datetime import datetime
from typing import Any, Dict

import numpy as np


def get_logs_dir(base_dir: str) -> str:
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def log_training_error(base_dir: str, message: str = "") -> str:
    log_path = os.path.join(get_logs_dir(base_dir), "training_error.log")
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{datetime.now().isoformat()}] TRAINING ERROR\n")
        if message:
            log_file.write(f"{message}\n")
        log_file.write(traceback.format_exc())
        log_file.write("\n")
    return log_path


def read_last_training_error(base_dir: str) -> str:
    log_path = os.path.join(base_dir, "logs", "training_error.log")
    if not os.path.isfile(log_path):
        return ""
    try:
        with open(log_path, "r", encoding="utf-8") as log_file:
            content = log_file.read().strip()
        if not content:
            return ""
        return content.split("TRAINING ERROR")[-1].strip()[:500]
    except OSError:
        return ""


def to_serializable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def save_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(to_serializable(payload), outfile, indent=2)


def load_json(path: str, default=None):
    if default is None:
        default = {}
    if not path or not os.path.isfile(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError, TypeError):
        return default


def import_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is not installed or cannot be imported. "
            "Install it with: pip install tensorflow"
        ) from exc

    if not hasattr(tf, "keras"):
        raise ImportError(
            "TensorFlow installation is incomplete (tensorflow.keras is missing). "
            "Reinstall with: pip uninstall tensorflow keras -y && pip install tensorflow==2.15.0"
        )
    return tf


def get_keras():
    tf = import_tensorflow()
    return tf.keras
