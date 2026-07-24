#!/usr/bin/env python
"""Calibrate + evaluate external Keras model."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import create_app
from services.accuracy_boost_service import run_accuracy_improvement


def main() -> int:
    app = create_app()
    with app.app_context():
        result = run_accuracy_improvement(app.config)
        print(json.dumps({
            "success": result.get("success"),
            "accuracy": result.get("accuracy"),
            "target": result.get("target"),
            "strategies": result.get("strategies"),
        }, indent=2, default=str))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
