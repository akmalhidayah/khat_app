#!/usr/bin/env python
"""Evaluate, fine-tune, calibrate, and re-evaluate the active Keras model."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("USE_TEACHABLE_MACHINE", "0")


def main() -> int:
    from app import create_app
    from services.accuracy_boost_service import fine_tune_external_model, run_accuracy_improvement
    from services.evaluation_service import evaluate_model

    app = create_app()
    with app.app_context():
        baseline = evaluate_model()
        print("BASELINE", json.dumps({
            "accuracy": baseline.get("accuracy"),
            "precision": baseline.get("precision"),
            "recall": baseline.get("recall"),
            "f1_macro": baseline.get("f1_macro"),
        }, indent=2))

        tune = fine_tune_external_model(
            app.config,
            epochs=20,
            batch_size=16,
            learning_rate=2e-5,
        )
        print("FINE_TUNE", json.dumps({
            "best_validation_accuracy": tune.get("best_validation_accuracy"),
            "epochs_run": tune.get("epochs_run"),
            "best_model_path": tune.get("best_model_path"),
        }, indent=2, default=str))

        calibrate = run_accuracy_improvement(app.config, strategy="calibrate")
        print("CALIBRATION", json.dumps({
            "success": calibrate.get("success"),
            "accuracy": calibrate.get("accuracy"),
            "target": calibrate.get("target"),
        }, indent=2, default=str))

        final = evaluate_model()
        summary = {
            "baseline_accuracy": baseline.get("accuracy"),
            "final_accuracy": final.get("accuracy"),
            "baseline_f1": baseline.get("f1_macro"),
            "final_f1": final.get("f1_macro"),
            "precision": final.get("precision"),
            "recall": final.get("recall"),
            "target": app.config.get("ACCURACY_TARGET", 0.85),
            "model_path": app.config.get("EXTERNAL_MODEL_PATH") or app.config.get("MODEL_PATH"),
            "improved": (final.get("accuracy") or 0) > (baseline.get("accuracy") or 0),
        }
        print("SUMMARY", json.dumps(summary, indent=2, default=str))
        return 0 if (final.get("accuracy") or 0) >= (baseline.get("accuracy") or 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
