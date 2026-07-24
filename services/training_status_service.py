import json
import os
from datetime import datetime
from typing import Dict, Optional

from flask import current_app

from services.training_utils import save_json

STATUS_FILENAME = "training_status.json"


def _status_path(config=None) -> str:
    if config is None:
        config = current_app.config
    return os.path.join(config["MODEL_DIR"], STATUS_FILENAME)


def load_training_status(config=None) -> Dict:
    path = _status_path(config)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def update_training_status(status: str, message: str, config=None, **extra) -> Dict:
    if config is None:
        config = current_app.config

    current = load_training_status(config)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    payload = {
        "status": status,
        "message": message,
        "started_at": current.get("started_at") or now,
        "finished_at": None if status == "running" else now,
        "current_stage": extra.get("current_stage", current.get("current_stage")),
        "current_epoch": extra.get("current_epoch", current.get("current_epoch")),
        "total_epochs": extra.get("total_epochs", current.get("total_epochs")),
        "model_path": extra.get("model_path", current.get("model_path")),
    }
    if status == "running":
        payload["started_at"] = current.get("started_at") or now
        payload["finished_at"] = None
    if status in ("completed", "failed"):
        payload["finished_at"] = now

    for key, value in extra.items():
        if key not in payload and value is not None:
            payload[key] = value

    save_json(_status_path(config), payload)
    return payload


def mark_training_started(config=None, **extra) -> Dict:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    payload = {
        "status": "running",
        "message": extra.get("message", "Training is running"),
        "started_at": now,
        "finished_at": None,
        "current_stage": extra.get("current_stage", "preparing"),
        "current_epoch": 0,
        "total_epochs": extra.get("total_epochs"),
    }
    for key, value in extra.items():
        if key not in payload and value is not None:
            payload[key] = value
    save_json(_status_path(config), payload)
    return payload


def mark_training_completed(message: str, model_path: str, config=None, **extra) -> Dict:
    return update_training_status(
        "completed",
        message,
        config=config,
        current_stage="Completed",
        model_path=model_path,
        progress_percent=extra.get("progress_percent", 100),
        **{k: v for k, v in extra.items() if k != "progress_percent"},
    )


def mark_training_failed(message: str, config=None, **extra) -> Dict:
    return update_training_status(
        "failed", message, config=config, current_stage="Failed", **extra
    )


def reconcile_training_status(
    training_status: Dict,
    model_exists: bool,
    has_history: bool,
    config=None,
) -> Dict:
    """Fix stale 'running' state when model and history already exist."""
    if config is None:
        config = current_app.config

    model_path = config["MODEL_PATH"]

    if model_exists and has_history:
        if training_status.get("status") != "completed":
            return mark_training_completed(
                "Training completed successfully.",
                model_path,
                config=config,
            )
        return training_status

    if training_status.get("status") == "running" and model_exists and os.path.isfile(model_path):
        history_path = config.get("TRAINING_HISTORY_PATH")
        if history_path and os.path.isfile(history_path):
            try:
                model_mtime = os.path.getmtime(model_path)
                history_mtime = os.path.getmtime(history_path)
                started = training_status.get("started_at", "")
                if model_mtime and history_mtime:
                    return mark_training_completed(
                        "Training completed successfully.",
                        model_path,
                        config=config,
                    )
            except OSError:
                pass

    if not training_status and model_exists:
        return mark_training_completed(
            "Training completed successfully.",
            model_path,
            config=config,
        )

    return training_status


def resolve_button_mode(
    readiness: Dict,
    model_status: Dict,
    has_history: bool,
    training_status: Dict,
) -> str:
    if not readiness.get("is_ready"):
        return "prepare"

    # Trained model always wins over stale "running" frontend/backend state.
    if model_status.get("exists") and has_history:
        return "retrain"

    status = training_status.get("status", "idle")
    if status == "running":
        return "running"
    if status == "failed":
        return "retry"
    if model_status.get("exists"):
        return "retrain"
    return "start"


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "Not recorded"
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"
    if minutes:
        return f"{minutes} minute{'s' if minutes != 1 else ''} {secs} second{'s' if secs != 1 else ''}"
    return f"{secs} second{'s' if secs != 1 else ''}"
