import json
import os
from typing import Dict, List, Optional

from flask import current_app


ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MIN_MODEL_BYTES = 10_000


def _is_image_file(filename: str, allowed_extensions: Optional[set] = None) -> bool:
    allowed = allowed_extensions or ALLOWED_IMAGE_EXTENSIONS
    if "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in allowed


def count_images_in_dir(root_dir: str, allowed_extensions: Optional[set] = None) -> int:
    if not os.path.isdir(root_dir):
        return 0
    total = 0
    for _, _, files in os.walk(root_dir):
        total += sum(1 for name in files if _is_image_file(name, allowed_extensions))
    return total


def count_images_per_class(root_dir: str, class_labels: List[str], allowed_extensions: Optional[set] = None) -> Dict[str, int]:
    from services.dataset_path_service import count_images_in_folder, resolve_class_folder

    allowed = allowed_extensions or ALLOWED_IMAGE_EXTENSIONS
    counts = {label: 0 for label in class_labels}
    if not os.path.isdir(root_dir):
        return counts

    for label in class_labels:
        class_dir = resolve_class_folder(root_dir, label)
        counts[label] = count_images_in_folder(class_dir, allowed)
    return counts


def is_valid_trained_model(path: str) -> bool:
    if not path or not os.path.isfile(path):
        return False
    try:
        if os.path.getsize(path) < MIN_MODEL_BYTES:
            return False
        with open(path, "rb") as model_file:
            header = model_file.read(8)
        if header[:2] == b"PK":
            return True
        if header.startswith(b"\x89HDF\r\n\x1a"):
            return True
        return False
    except OSError:
        return False


def resolve_trained_model_path(model_dir: str, keras_path: str, config=None) -> Optional[str]:
    if config and config.get("USE_EXTERNAL_MODEL") and config.get("EXTERNAL_MODEL_PATH"):
        path = config["EXTERNAL_MODEL_PATH"]
        return path if is_valid_trained_model(path) else None
    candidates = [
        os.path.join(model_dir, "khat_best.keras"),
        os.path.join(model_dir, "khat_latest.keras"),
        keras_path,
        os.path.join(model_dir, "khat_vgg16_best.keras"),
        os.path.join(model_dir, "khat_vgg16_model.h5"),
    ]
    for path in candidates:
        if is_valid_trained_model(path):
            return path
    return None


def has_training_history(history_path: str) -> bool:
    if not os.path.isfile(history_path):
        return False
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("history") and data["history"].get("accuracy"):
            return True
        return isinstance(data, dict) and bool(data.get("accuracy"))
    except (json.JSONDecodeError, OSError):
        return False


def has_evaluation_result(eval_path: str) -> bool:
    if not os.path.isfile(eval_path):
        return False
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, dict) and bool(data)
    except (json.JSONDecodeError, OSError):
        return False


def load_training_history(history_path: str) -> dict:
    summary = load_training_summary(history_path)
    if summary.get("history"):
        return summary["history"]
    if summary.get("accuracy"):
        return {k: summary[k] for k in ("accuracy", "val_accuracy", "loss", "val_loss") if k in summary}
    return {}


def load_training_summary(history_path: str) -> dict:
    if not os.path.isfile(history_path):
        return {}
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except (json.JSONDecodeError, OSError):
        return {}


def get_model_status(model_dir: str, model_path: str, history_path: str, eval_path: str, config=None) -> dict:
    resolved_path = resolve_trained_model_path(model_dir, model_path, config=config)
    history_exists = has_training_history(history_path)
    eval_exists = has_evaluation_result(eval_path)
    modified_at = None
    if resolved_path:
        modified_at = os.path.getmtime(resolved_path)

    return {
        "exists": resolved_path is not None,
        "path": resolved_path,
        "filename": os.path.basename(resolved_path) if resolved_path else None,
        "modified_at": modified_at,
        "has_history": history_exists,
        "has_evaluation": eval_exists,
        "label": "Trained model available" if resolved_path else "Not trained",
        "note": (
            f"Model file: {os.path.basename(resolved_path)}"
            if resolved_path
            else "No trained model file found."
        ),
    }


def build_split_preview(raw_class_counts: Dict[str, int]) -> dict:
    total = sum(raw_class_counts.values())
    est_train = int(total * 0.70)
    est_val = int(total * 0.15)
    est_test = max(total - est_train - est_val, 0)
    per_class = []
    for label, count in raw_class_counts.items():
        if count <= 0:
            continue
        cls_train = int(count * 0.70)
        cls_val = int(count * 0.15)
        cls_test = max(count - cls_train - cls_val, 0)
        per_class.append(
            {
                "label": label,
                "total": count,
                "train": cls_train,
                "validation": cls_val,
                "test": cls_test,
            }
        )
    return {
        "total": total,
        "train": est_train,
        "validation": est_val,
        "test": est_test,
        "per_class": per_class,
    }


def get_dataset_readiness(config=None) -> dict:
    if config is None:
        config = current_app.config

    if config.get("USE_EXTERNAL_DATASET"):
        from services.dataset_path_service import get_dataset_inventory

        inv = get_dataset_inventory(config)
        raw_class_counts = dict(inv.get("per_class") or {})
        raw_count = int(inv.get("raw_total") or 0)
        train_count = int(inv.get("train") or 0)
        validation_count = int(inv.get("validation") or 0)
        test_count = int(inv.get("test") or 0)
        processed_count = int(inv.get("processed_total") or 0)
    else:
        allowed = config.get("ALLOWED_EXTENSIONS", ALLOWED_IMAGE_EXTENSIONS)
        class_labels = config["CLASS_LABELS"]
        raw_dir = config["RAW_DATASET_DIR"]
        train_dir = config["TRAIN_DIR"]
        validation_dir = config["VALIDATION_DIR"]
        test_dir = config["TEST_DIR"]

        raw_class_counts = count_images_per_class(raw_dir, class_labels, allowed)
        raw_count = sum(raw_class_counts.values())
        if raw_count == 0:
            raw_count = count_images_in_dir(raw_dir, allowed)
        train_count = count_images_in_dir(train_dir, allowed)
        validation_count = count_images_in_dir(validation_dir, allowed)
        test_count = count_images_in_dir(test_dir, allowed)
        processed_count = train_count + validation_count + test_count

    is_ready = train_count > 0 and validation_count > 0 and test_count > 0

    if is_ready:
        status = "Ready"
        status_message = "Dataset ready for training"
    elif raw_count > 0:
        status = "Not processed"
        status_message = "Raw dataset imported but not split yet"
    else:
        status = "Empty"
        status_message = "No dataset images found"

    return {
        "raw_count": raw_count,
        "train_count": train_count,
        "validation_count": validation_count,
        "test_count": test_count,
        "processed_count": processed_count,
        "is_ready": is_ready,
        "status": status,
        "status_message": status_message,
        "raw_class_counts": raw_class_counts,
        "split_preview": build_split_preview(raw_class_counts),
    }


def build_pipeline_status(readiness: dict, model_status: dict) -> List[dict]:
    split_ready = readiness["is_ready"]
    processed_count = readiness["processed_count"]
    model_exists = model_status["exists"]
    eval_exists = model_status["has_evaluation"]

    if not split_ready:
        training_status = "locked"
        training_label = "Locked"
    elif model_exists:
        training_status = "ready"
        training_label = "Completed"
    else:
        training_status = "waiting"
        training_label = "Waiting"

    if model_exists:
        saving_status = "ready"
        saving_label = "Saved"
    elif split_ready:
        saving_status = "waiting"
        saving_label = "Waiting"
    else:
        saving_status = "locked"
        saving_label = "Locked"

    if eval_exists:
        eval_status = "ready"
        eval_label = "Completed"
    elif model_exists:
        eval_status = "waiting"
        eval_label = "Waiting"
    else:
        eval_status = "locked"
        eval_label = "Locked"

    return [
        {
            "step": 1,
            "title": "Dataset Split",
            "description": "Train/validation/test folders prepared.",
            "icon": "bi-diagram-2",
            "status": "ready" if split_ready else "pending",
            "status_label": "Completed" if split_ready else "Pending",
        },
        {
            "step": 2,
            "title": "Preprocessing",
            "description": "Resize and normalize input tensors.",
            "icon": "bi-sliders",
            "status": "ready" if processed_count > 0 else "pending",
            "status_label": "Completed" if processed_count > 0 else "Pending",
        },
        {
            "step": 3,
            "title": "Augmentation",
            "description": "Improve model robustness and diversity.",
            "icon": "bi-magic",
            "status": "ready" if split_ready else "pending",
            "status_label": "Applied" if split_ready else "Pending",
        },
        {
            "step": 4,
            "title": "Training",
            "description": "Transfer learning and fine-tuning.",
            "icon": "bi-cpu",
            "status": training_status,
            "status_label": training_label,
        },
        {
            "step": 5,
            "title": "Model Saving",
            "description": "Store best checkpoint automatically.",
            "icon": "bi-hdd",
            "status": saving_status,
            "status_label": saving_label,
        },
        {
            "step": 6,
            "title": "Evaluation",
            "description": "Assess performance on holdout set.",
            "icon": "bi-clipboard-check",
            "status": eval_status,
            "status_label": eval_label,
        },
    ]


def build_readiness_cards(readiness: dict, model_status: dict) -> List[dict]:
    raw_count = readiness["raw_count"]
    train_count = readiness["train_count"]
    validation_count = readiness["validation_count"]
    test_count = readiness["test_count"]

    return [
        {
            "label": "Raw Dataset",
            "value": raw_count,
            "unit": "Images",
            "note": "Imported calligraphy samples" if raw_count > 0 else "No raw images found",
            "badge": "ready" if raw_count > 0 else "unavailable",
            "badge_text": "Imported" if raw_count > 0 else "Empty",
            "icon": "bi-images",
        },
        {
            "label": "Training Images",
            "value": train_count,
            "unit": "Images",
            "note": "Ready for model fit" if train_count > 0 else "Missing",
            "badge": "ready" if train_count > 0 else "pending",
            "badge_text": "Ready" if train_count > 0 else "Missing",
            "icon": "bi-folder2-open",
        },
        {
            "label": "Validation Images",
            "value": validation_count,
            "unit": "Images",
            "note": "Generalization check" if validation_count > 0 else "Missing",
            "badge": "ready" if validation_count > 0 else "pending",
            "badge_text": "Ready" if validation_count > 0 else "Missing",
            "icon": "bi-check2-square",
        },
        {
            "label": "Testing Images",
            "value": test_count,
            "unit": "Images",
            "note": "Final performance set" if test_count > 0 else "Missing",
            "badge": "ready" if test_count > 0 else "pending",
            "badge_text": "Ready" if test_count > 0 else "Missing",
            "icon": "bi-clipboard-data",
        },
        {
            "label": "Model Status",
            "value": "Trained" if model_status["exists"] else "Not trained",
            "unit": "",
            "note": model_status["note"],
            "badge": "ready" if model_status["exists"] else "unavailable",
            "badge_text": "Trained" if model_status["exists"] else "Not trained",
            "icon": "bi-cpu",
        },
    ]
