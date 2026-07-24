#!/usr/bin/env python
"""Fine-tune / retrain to reach >=85% test accuracy."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep external paths readable; internal paths used when saving fine-tuned model.
os.environ.setdefault("USE_TEACHABLE_MACHINE", "0")

from app import create_app
from services.accuracy_boost_service import run_accuracy_improvement


def main() -> int:
    strategy = os.environ.get("ACCURACY_STRATEGY", "auto")
    skip_fine_tune = os.environ.get("SKIP_FINE_TUNE", "0").strip().lower() in ("1", "true", "yes")
    app = create_app()
    with app.app_context():
        try:
            result = run_accuracy_improvement(
                app.config,
                strategy=strategy,
                fine_tune_epochs=25,
                skip_fine_tune=skip_fine_tune,
            )
        except Exception as exc:
            print(json.dumps({"success": False, "error": str(exc)}, indent=2))
            return 1
        print(json.dumps({
            "success": result.get("success"),
            "accuracy": result.get("accuracy"),
            "target": result.get("target"),
            "model_switched": result.get("model_switched"),
            "strategies": [
                {k: v for k, v in s.items() if k != "train_result"}
                for s in result.get("strategies", [])
            ],
        }, indent=2, ensure_ascii=False, default=str))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
