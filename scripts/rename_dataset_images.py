#!/usr/bin/env python
"""Rename dataset images to class-based names (e.g. diwani-1.jpg)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import create_app
from services.dataset_rename_service import rename_dataset_images


def main() -> int:
    app = create_app()
    with app.app_context():
        result = rename_dataset_images(app.config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
