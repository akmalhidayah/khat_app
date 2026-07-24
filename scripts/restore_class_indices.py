"""Restore class indices from external labels.txt."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import create_app
from services.external_assets_service import (
    build_class_indices_from_labels,
    write_class_indices_file,
)


def main() -> int:
    app = create_app()
    with app.app_context():
        labels_path = app.config.get("EXTERNAL_LABELS_PATH") or str(
            Path(app.config.get("EXTERNAL_MODEL_DIR", "")) / app.config.get("EXTERNAL_LABELS_FILE", "labels.txt")
        )
        class_labels = list(app.config["CLASS_LABELS"])
        indices = build_class_indices_from_labels(labels_path, class_labels)
        write_class_indices_file(app.config["CLASS_INDICES_PATH"], indices, class_labels)
        print(json.dumps(indices, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
