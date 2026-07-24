#!/usr/bin/env python
"""Export best research checkpoint, fine-tune lightly, calibrate, and evaluate."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def _export_checkpoint(config, architecture: str, checkpoint_name: str) -> str:
    import tensorflow as tf

    from services.model_builder_service import ARCHITECTURES, build_classifier, save_classifier_bundle

    checkpoint = Path(config["MODEL_DIR"]) / checkpoint_name
    if not checkpoint.is_file():
        return config.get("MODEL_PATH") or ""

    class_labels = list(config["CLASS_LABELS"])
    model, _ = build_classifier(architecture, len(class_labels), 1e-4, trainable_backbone=False)
    model.load_weights(str(checkpoint))

    out = str(Path(config["MODEL_DIR"]) / (config.get("EXTERNAL_MODEL_FILE") or "keras_model.h5"))
    save_classifier_bundle(model, out)
    tf.keras.backend.clear_session()
    return out


def main() -> int:
    from app import create_app
    from services.accuracy_boost_service import fine_tune_external_model, run_accuracy_improvement
    from services.evaluation_service import evaluate_model
    from services.model_cache_service import clear_model_cache
    from services.training_utils import save_json

    app = create_app()
    with app.app_context():
        cfg = app.config
        model_dir = Path(cfg["MODEL_DIR"])

        candidates = [
            ("efficientnetb0", "efficientnetb0_research_checkpoint.weights.h5"),
            ("vgg16", "vgg16_research_checkpoint.weights.h5"),
        ]
        model_path = cfg.get("MODEL_PATH") or ""
        used_arch = "keras_h5"
        for arch, ckpt in candidates:
            ckpt_path = model_dir / ckpt
            if ckpt_path.is_file() and ckpt_path.stat().st_size > 1000:
                model_path = _export_checkpoint(cfg, arch, ckpt)
                used_arch = arch
                print(f"Exported {arch} checkpoint -> {model_path}")
                break

        if not model_path or not os.path.isfile(model_path):
            model_path = cfg.get("EXTERNAL_MODEL_PATH") or cfg.get("MODEL_PATH")
            print(f"Using existing model: {model_path}")

        cfg["MODEL_PATH"] = model_path
        cfg["EXTERNAL_MODEL_PATH"] = model_path
        cfg["BEST_MODEL_PATH"] = model_path
        clear_model_cache()

        save_json(cfg["MODEL_METADATA_PATH"], {
            "architecture": used_arch,
            "model_architecture_key": used_arch,
            "architecture_label": used_arch.replace("_", " ").title(),
            "model_path": model_path,
            "preprocessing": used_arch if used_arch != "keras_h5" else "teachable_machine",
            "source": "checkpoint_export",
        })

        baseline = evaluate_model()
        print("BASELINE", json.dumps({
            "accuracy": baseline.get("accuracy"),
            "precision": baseline.get("precision"),
            "recall": baseline.get("recall"),
            "f1_score": baseline.get("f1_score"),
        }, indent=2))

        try:
            tune = fine_tune_external_model(cfg, epochs=12, batch_size=8, learning_rate=1e-5)
            print("FINE_TUNE", json.dumps(tune, indent=2, default=str))
        except Exception as exc:
            print(f"FINE_TUNE skipped: {exc}")

        clear_model_cache()
        try:
            run_accuracy_improvement(cfg, strategy="calibrate")
        except Exception as exc:
            print(f"CALIBRATION skipped: {exc}")

        final = evaluate_model()
        summary = {
            "model_path": model_path,
            "architecture": used_arch,
            "baseline_accuracy": baseline.get("accuracy"),
            "final_accuracy": final.get("accuracy"),
            "precision": final.get("precision"),
            "recall": final.get("recall"),
            "f1_score": final.get("f1_score"),
            "target": cfg.get("ACCURACY_TARGET", 0.85),
        }
        print("FINAL", json.dumps(summary, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
