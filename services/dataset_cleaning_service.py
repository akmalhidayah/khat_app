"""Detect exact and near-duplicate images, including cross-split leakage."""

import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple

from flask import current_app

from services.preprocessing_service import image_hash, perceptual_hash
from services.training_utils import load_json, save_json

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
PHASH_THRESHOLD = 5


def _hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _iter_split_images(config) -> List[Dict]:
    rows = []
    for split in ("raw", "train", "validation", "test"):
        root = config.get(f"{split.upper()}_DIR") if split != "raw" else config["RAW_DATASET_DIR"]
        if split == "train":
            root = config["TRAIN_DIR"]
        elif split == "validation":
            root = config["VALIDATION_DIR"]
        elif split == "test":
            root = config["TEST_DIR"]
        elif split == "raw":
            root = config["RAW_DATASET_DIR"]

        if not os.path.isdir(root):
            continue
        for cls in config["CLASS_LABELS"]:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for name in os.listdir(cls_dir):
                if "." not in name or name.rsplit(".", 1)[1].lower() not in ALLOWED_EXTENSIONS:
                    continue
                path = os.path.join(cls_dir, name)
                if os.path.isfile(path):
                    rows.append({
                        "split": split,
                        "class_name": cls,
                        "filename": name,
                        "path": path,
                    })
    return rows


def run_duplicate_audit(config=None, phash_threshold: int = PHASH_THRESHOLD) -> Dict:
    if config is None:
        config = current_app.config

    images = _iter_split_images(config)
    exact_groups: Dict[str, List[Dict]] = defaultdict(list)
    near_groups: List[Dict] = []
    cross_split_exact: List[Dict] = []
    cross_split_near: List[Dict] = []

    phash_index: List[Tuple[str, Dict]] = []

    for item in images:
        try:
            fhash = image_hash(item["path"])
            item["file_hash"] = fhash
            exact_groups[fhash].append(item)
            phash = perceptual_hash(item["path"])
            item["phash"] = phash
            phash_index.append((phash, item))
        except Exception:
            item["error"] = "unreadable"

    exact_duplicate_count = 0
    duplicate_groups = []
    for fhash, group in exact_groups.items():
        if len(group) < 2:
            continue
        exact_duplicate_count += len(group) - 1
        duplicate_groups.append({
            "type": "exact",
            "hash": fhash,
            "count": len(group),
            "images": [
                {"split": g["split"], "class_name": g["class_name"], "path": g["path"]}
                for g in group
            ],
        })
        splits = {g["split"] for g in group}
        if len(splits) > 1 and ("test" in splits or "validation" in splits):
            cross_split_exact.append({
                "hash": fhash,
                "splits": sorted(splits),
                "images": duplicate_groups[-1]["images"],
                "recommended_action": "Remove duplicate from test/validation or rebuild split.",
            })

    near_duplicate_count = 0
    seen_near: Set[Tuple[str, str]] = set()
    for i, (phash_a, item_a) in enumerate(phash_index):
        for phash_b, item_b in phash_index[i + 1:]:
            if _hamming(phash_a, phash_b) > phash_threshold:
                continue
            key = tuple(sorted([item_a["path"], item_b["path"]]))
            if key in seen_near:
                continue
            seen_near.add(key)
            near_duplicate_count += 1
            entry = {
                "type": "near",
                "hamming_distance": _hamming(phash_a, phash_b),
                "images": [
                    {"split": item_a["split"], "class_name": item_a["class_name"], "path": item_a["path"]},
                    {"split": item_b["split"], "class_name": item_b["class_name"], "path": item_b["path"]},
                ],
            }
            near_groups.append(entry)
            splits = {item_a["split"], item_b["split"]}
            if len(splits) > 1 and ("test" in splits or "validation" in splits):
                cross_split_near.append({
                    **entry,
                    "recommended_action": "Rebuild split to prevent leakage.",
                })

    recommendations = []
    if cross_split_exact:
        recommendations.append(
            f"Found {len(cross_split_exact)} exact duplicate group(s) across train/val/test. Rebuild split."
        )
    if cross_split_near:
        recommendations.append(
            f"Found {len(cross_split_near)} near-duplicate pair(s) across splits. Review before evaluation."
        )
    if exact_duplicate_count:
        recommendations.append("Remove exact duplicates within raw dataset to improve training quality.")

    report = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_images_scanned": len(images),
        "exact_duplicate_count": exact_duplicate_count,
        "near_duplicate_count": near_duplicate_count,
        "duplicate_groups": duplicate_groups[:200],
        "near_duplicate_groups": near_groups[:200],
        "cross_split_exact": cross_split_exact[:100],
        "cross_split_near": cross_split_near[:100],
        "cross_split_leakage_detected": bool(cross_split_exact or cross_split_near),
        "recommended_actions": recommendations,
    }
    save_json(config["DUPLICATE_REPORT_PATH"], report)
    return report


def load_duplicate_report(config=None) -> Dict:
    if config is None:
        config = current_app.config
    return load_json(config.get("DUPLICATE_REPORT_PATH"), {})
