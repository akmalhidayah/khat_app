"""Register and track model versions for academic reproducibility."""

import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import current_app

from models import ModelVersion, db
from services.dataset_readiness_service import count_images_per_class
from services.teachable_machine_service import clear_tm_model_cache, get_tm_display_info, is_teachable_machine_available, load_tm_metadata

REQUIRED_TM_FILES = ("model.json", "metadata.json", "weights.bin")


def _load_latest_evaluation_accuracy(config=None) -> Optional[float]:
    cfg = config or current_app.config
    path = cfg.get("EVALUATION_RESULT_PATH")
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        accuracy = data.get("accuracy")
        return float(accuracy) if accuracy is not None else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def load_latest_evaluation_accuracy(config=None) -> Optional[float]:
    return _load_latest_evaluation_accuracy(config)


def _build_tm_version_meta(base: Optional[Dict[str, Any]] = None, text: Optional[str] = None) -> Dict[str, Any]:
    meta = dict(base or {})
    if is_teachable_machine_available():
        info = get_tm_display_info()
        tm_meta = load_tm_metadata()
        raw_labels = tm_meta.get("labels") or []
        meta.setdefault("labels", raw_labels)
        meta.setdefault("labels_normalized", info.get("labels") or [])
        meta.setdefault("input_size", int(tm_meta.get("imageSize") or 224))
        meta.setdefault("model_name", info.get("model_name"))
    if text:
        meta.setdefault("text", text)
    return meta


def _resolve_labels_display(row: ModelVersion, meta: Dict[str, Any]) -> str:
    labels = meta.get("labels") or meta.get("labels_normalized")
    if not labels and row.labels_json:
        try:
            parsed = json.loads(row.labels_json)
            labels = parsed if isinstance(parsed, list) else None
        except (json.JSONDecodeError, TypeError):
            labels = None
    if not labels and row.is_active and row.model_source == "Teachable Machine" and is_teachable_machine_available():
        labels = load_tm_metadata().get("labels") or []
    if isinstance(labels, list):
        return ", ".join(str(label) for label in labels) if labels else "—"
    return str(labels) if labels else "—"


def _resolve_input_size_display(row: ModelVersion, meta: Dict[str, Any]) -> str:
    input_size = meta.get("input_size")
    if input_size is None and row.input_width:
        input_size = row.input_width
    if input_size is None and row.is_active and row.model_source == "Teachable Machine" and is_teachable_machine_available():
        input_size = int(load_tm_metadata().get("imageSize") or 224)
    if input_size:
        return f"{input_size} × {input_size} px"
    return "224 × 224 px"


def _resolve_evaluation_accuracy(row: ModelVersion) -> Optional[float]:
    if row.evaluation_accuracy is not None:
        return row.evaluation_accuracy

    from models.evaluasi_model import EvaluasiModel, TIPE_RINGKASAN

    latest_eval = (
        EvaluasiModel.query.filter_by(id_model=row.id, tipe_record=TIPE_RINGKASAN)
        .order_by(EvaluasiModel.dibuat_pada.desc())
        .first()
    )
    if latest_eval and latest_eval.akurasi is not None:
        return float(latest_eval.akurasi)

    if row.is_active and row.model_source == "Teachable Machine":
        return _load_latest_evaluation_accuracy()
    return None


def _next_version(source: str) -> str:
    latest = (
        ModelVersion.query.filter_by(sumber_model=source)
        .order_by(ModelVersion.id.desc())
        .first()
    )
    if not latest:
        return "v1.0"
    try:
        num = float(latest.model_version.lstrip("v"))
        return f"v{num + 0.1:.1f}"
    except ValueError:
        return f"v{latest.id + 1}.0"


def parse_version_notes(notes: Optional[str]) -> Dict[str, Any]:
    if not notes:
        return {}
    try:
        data = json.loads(notes)
        return data if isinstance(data, dict) else {"text": notes}
    except (json.JSONDecodeError, TypeError):
        return {"text": notes}


def update_version_notes(version_id: int, payload: Dict[str, Any]) -> Optional[ModelVersion]:
    row = ModelVersion.query.get(version_id)
    if not row:
        return None
    row.notes = json.dumps(payload)
    db.session.commit()
    return row


def register_model_version(
    model_name: str,
    model_source: str,
    model_runtime: Optional[str] = None,
    evaluation_accuracy: Optional[float] = None,
    notes: Optional[str] = None,
    training_date: Optional[datetime] = None,
    set_active: bool = True,
) -> ModelVersion:
    config = current_app.config
    counts = count_images_per_class(config["RAW_DATASET_DIR"], config["CLASS_LABELS"])
    total_images = sum(counts.values())

    if set_active:
        ModelVersion.query.filter_by(aktif=True).update({"aktif": False})

    row = ModelVersion(
        model_name=model_name,
        model_source=model_source,
        model_version=_next_version(model_source),
        model_runtime=model_runtime,
        training_date=training_date or datetime.utcnow(),
        num_classes=len(config["CLASS_LABELS"]),
        dataset_images=total_images,
        evaluation_accuracy=evaluation_accuracy,
        is_active=set_active,
    )
    if notes:
        row.notes = notes
    db.session.add(row)
    db.session.commit()
    return row


def register_active_tm_model(evaluation_accuracy: Optional[float] = None, notes: Optional[str] = None) -> Optional[ModelVersion]:
    if not is_teachable_machine_available():
        return None
    info = get_tm_display_info()
    metadata = load_tm_metadata()
    ts = metadata.get("timeStamp")
    training_date = None
    if ts:
        try:
            training_date = datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            training_date = None

    existing = (
        ModelVersion.query.filter_by(nama_model=info["model_name"], sumber_model=info["model_source"], aktif=True)
        .order_by(ModelVersion.id.desc())
        .first()
    )
    accuracy = evaluation_accuracy
    if accuracy is None:
        accuracy = _load_latest_evaluation_accuracy()

    note_text = None
    if notes:
        try:
            parsed_notes = json.loads(notes)
            if isinstance(parsed_notes, dict):
                notes_payload = _build_tm_version_meta(parsed_notes)
            else:
                note_text = str(notes)
                notes_payload = _build_tm_version_meta(text=note_text)
        except (json.JSONDecodeError, TypeError):
            note_text = notes
            notes_payload = _build_tm_version_meta(text=note_text)
    else:
        notes_payload = _build_tm_version_meta(text="Teachable Machine model registered from static/model/")

    if existing:
        merged_meta = _build_tm_version_meta(parse_version_notes(existing.notes), text=notes_payload.get("text"))
        merged_meta.update({key: value for key, value in notes_payload.items() if key != "text" or "text" not in merged_meta})
        if note_text:
            merged_meta["text"] = note_text
        existing.notes = json.dumps(merged_meta)
        if accuracy is not None:
            existing.evaluation_accuracy = accuracy
        if training_date:
            existing.training_date = training_date
        labels = merged_meta.get("labels") or []
        if labels:
            existing.labels_json = json.dumps(labels)
            existing.input_width = int(merged_meta.get("input_size") or 224)
            existing.input_height = int(merged_meta.get("input_size") or 224)
        db.session.commit()
        return existing

    return register_model_version(
        model_name=info["model_name"],
        model_source=info["model_source"],
        model_runtime=info["model_runtime"],
        evaluation_accuracy=accuracy,
        notes=json.dumps(notes_payload),
        training_date=training_date,
        set_active=True,
    )


def get_active_model_version() -> Optional[ModelVersion]:
    return ModelVersion.query.filter_by(aktif=True).order_by(ModelVersion.id.desc()).first()


def list_model_versions(limit: int = 20) -> List[ModelVersion]:
    return ModelVersion.query.order_by(ModelVersion.dibuat_pada.desc()).limit(limit).all()


def _version_status(row: ModelVersion, meta: Dict[str, Any]) -> str:
    if row.is_active:
        return "active"
    if meta.get("archived"):
        return "archived"
    return "inactive"


def sync_active_model_version_metrics() -> Optional[ModelVersion]:
    """Persist TM labels and latest evaluation accuracy on the active model version."""
    if not is_teachable_machine_available():
        return None
    return register_active_tm_model()


def model_version_to_dict(row: ModelVersion) -> Dict:
    meta = parse_version_notes(row.notes)
    resolved_accuracy = _resolve_evaluation_accuracy(row)
    return {
        "id": row.id,
        "model_name": row.model_name,
        "model_source": row.model_source,
        "model_version": row.model_version,
        "model_runtime": row.model_runtime,
        "training_date": row.training_date.strftime("%Y-%m-%d %H:%M") if row.training_date else None,
        "upload_date": row.created_at.strftime("%Y-%m-%d %H:%M") if row.created_at else None,
        "num_classes": row.num_classes,
        "dataset_images": row.dataset_images,
        "evaluation_accuracy": round(resolved_accuracy * 100, 2) if resolved_accuracy is not None else None,
        "notes": row.notes,
        "labels": _resolve_labels_display(row, meta),
        "input_size": _resolve_input_size_display(row, meta),
        "status": _version_status(row, meta),
        "is_active": row.is_active,
        "created_at": row.created_at.strftime("%Y-%m-%d %H:%M") if row.created_at else None,
        "can_activate": bool(meta.get("snapshot_dir")) and not row.is_active,
        "can_delete": not row.is_active,
    }


def _restore_snapshot(snapshot_dir: str) -> None:
    if not snapshot_dir or not os.path.isdir(snapshot_dir):
        raise ValueError("Snapshot model tidak ditemukan untuk versi ini.")
    missing = [name for name in REQUIRED_TM_FILES if not os.path.isfile(os.path.join(snapshot_dir, name))]
    if missing:
        raise ValueError(f"Snapshot tidak lengkap: {', '.join(missing)}")

    model_dir = current_app.config["TEACHABLE_MODEL_DIR"]
    os.makedirs(model_dir, exist_ok=True)
    for name in REQUIRED_TM_FILES:
        shutil.copy2(os.path.join(snapshot_dir, name), os.path.join(model_dir, name))
    clear_tm_model_cache()


def activate_model_version(version_id: int) -> ModelVersion:
    row = ModelVersion.query.get(version_id)
    if not row:
        raise ValueError("Versi model tidak ditemukan.")
    if row.is_active:
        return row

    meta = parse_version_notes(row.notes)
    snapshot_dir = meta.get("snapshot_dir")
    if snapshot_dir:
        _restore_snapshot(snapshot_dir)

    ModelVersion.query.filter_by(aktif=True).update({"aktif": False})
    row.aktif = True
    meta["archived"] = False
    row.notes = json.dumps(meta)
    db.session.commit()
    return row


def archive_model_version(version_id: int) -> ModelVersion:
    row = ModelVersion.query.get(version_id)
    if not row:
        raise ValueError("Versi model tidak ditemukan.")
    if row.is_active:
        raise ValueError("Model aktif tidak dapat diarsipkan. Aktifkan versi lain terlebih dahulu.")

    meta = parse_version_notes(row.notes)
    meta["archived"] = True
    row.notes = json.dumps(meta)
    db.session.commit()
    return row


def delete_model_version(version_id: int) -> None:
    row = ModelVersion.query.get(version_id)
    if not row:
        raise ValueError("Versi model tidak ditemukan.")
    if row.is_active:
        raise ValueError("Model aktif tidak dapat dihapus.")

    meta = parse_version_notes(row.notes)
    snapshot_dir = meta.get("snapshot_dir")
    if snapshot_dir and os.path.isdir(snapshot_dir):
        shutil.rmtree(snapshot_dir, ignore_errors=True)

    db.session.delete(row)
    db.session.commit()
