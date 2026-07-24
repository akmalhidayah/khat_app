#!/usr/bin/env python
"""Merge rejected dataset images back into class folders (700 per class)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import create_app
from services.dataset_restore_service import restore_dataset_to_target


def main() -> int:
    app = create_app()
    with app.app_context():
        result = restore_dataset_to_target(app.config, target_per_class=700, resplit=True)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
