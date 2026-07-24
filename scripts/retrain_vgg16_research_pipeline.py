#!/usr/bin/env python
"""Run the full VGG16 research pipeline to reach 85% evaluation metrics."""

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
os.environ.setdefault("USE_EXTERNAL_DATASET", "1")
os.environ.setdefault("USE_EXTERNAL_MODEL", "1")


def main() -> int:
    from app import create_app
    from services.metrics_service import ACCURACY_TARGET
    from services.vgg16_research_pipeline_service import run_vgg16_research_pipeline

    app = create_app()
    with app.app_context():
        summary = run_vgg16_research_pipeline(
            app.config,
            target_accuracy=ACCURACY_TARGET,
            max_attempts=2,
        )
        print(json.dumps(summary, indent=2, default=str))
        return 0 if summary.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
