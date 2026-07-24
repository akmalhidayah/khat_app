#!/usr/bin/env python
"""Apply external dataset (D:\\dataset_kaligrafi) and Keras model (D:\\model_treaning) to the web app."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import create_app
from services.external_assets_service import apply_external_assets


def main() -> int:
    app = create_app()
    with app.app_context():
        result = apply_external_assets(app.config, import_dataset=True, split_dataset=True, clean_before_split=False)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
