#!/usr/bin/env python3
"""Report exact and perceptual duplicates across dataset splits; never mutates by default."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path) -> str:
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        gray = ImageOps.grayscale(ImageOps.exif_transpose(image))
        pixels = list(gray.resize((9, 8), Image.Resampling.LANCZOS).getdata())
    bits = [pixels[y * 9 + x] > pixels[y * 9 + x + 1] for y in range(8) for x in range(8)]
    return f"{sum(int(bit) << index for index, bit in enumerate(bits)):016x}"


def distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def build_report(dataset: Path, threshold: int = 4) -> dict:
    rows = []
    for split in ("train", "validation", "test"):
        root = dataset / split
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                try:
                    rows.append(
                        {"path": str(path), "split": split, "sha256": sha256(path), "dhash": dhash(path)}
                    )
                except (OSError, ValueError):
                    continue
    exact = defaultdict(list)
    for row in rows:
        exact[row["sha256"]].append(row)
    exact_groups = [group for group in exact.values() if len(group) > 1]
    perceptual = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            if left["sha256"] == right["sha256"]:
                continue
            dist = distance(left["dhash"], right["dhash"])
            if dist <= threshold:
                perceptual.append({"left": left, "right": right, "distance": dist})
    leakage_exact = [g for g in exact_groups if len({r["split"] for r in g}) > 1]
    leakage_near = [
        pair for pair in perceptual if pair["left"]["split"] != pair["right"]["split"]
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset),
        "files_scanned": len(rows),
        "dhash_distance_threshold": threshold,
        "exact_duplicate_groups": exact_groups,
        "perceptual_duplicate_pairs": perceptual,
        "cross_split_exact_leakage": leakage_exact,
        "cross_split_perceptual_leakage": leakage_near,
        "safe_to_use_test_metrics": bool(rows) and not leakage_exact and not leakage_near,
        "status": "complete" if rows else "not_run_no_dataset_files",
        "mutated_files": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--output", default="model/dataset_leakage_report")
    parser.add_argument("--distance", type=int, default=4)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        raise SystemExit("--apply is intentionally unsupported until a reviewed move policy is supplied")
    report = build_report(Path(args.dataset), args.distance)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = (
        "# Dataset Leakage Report\n\n"
        f"- Status: {report['status']}\n"
        f"- Files scanned: {report['files_scanned']}\n"
        f"- Exact cross-split groups: {len(report['cross_split_exact_leakage'])}\n"
        f"- Near-duplicate cross-split pairs: {len(report['cross_split_perceptual_leakage'])}\n"
        f"- Test metrics safe from detected leakage: {report['safe_to_use_test_metrics']}\n"
        "\nNo file was moved or deleted.\n"
    )
    output.with_suffix(".md").write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
