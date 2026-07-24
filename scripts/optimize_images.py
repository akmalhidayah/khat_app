#!/usr/bin/env python3
"""Command-line image optimization for Arabic Khat dataset folders."""

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import Config
from services.image_optimization_service import (
    generate_optimization_report,
    optimize_all_dataset_folders,
    optimize_dataset_images,
)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_config() -> dict:
    return {
        "BASE_DIR": Config.BASE_DIR,
        "DATASET_DIR": Config.DATASET_DIR,
        "MODEL_DIR": Config.MODEL_DIR,
        "OPTIMIZED_DIR": Config.OPTIMIZED_DIR,
        "OPTIMIZED_DISPLAY_DIR": Config.OPTIMIZED_DISPLAY_DIR,
        "OPTIMIZED_MODEL_DIR": Config.OPTIMIZED_MODEL_DIR,
        "CLASS_LABELS": Config.CLASS_LABELS,
    }


def main():
    parser = argparse.ArgumentParser(description="Optimize Arabic Khat dataset images.")
    parser.add_argument(
        "--source",
        default="all",
        help="Source folder relative to dataset/ (e.g. raw, train) or 'all' for every dataset folder.",
    )
    parser.add_argument("--output", default="dataset/optimized", help="Output base folder.")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality for display images.")
    parser.add_argument("--max-size", type=int, default=1024, help="Max dimension for display images.")
    parser.add_argument("--model-size", type=int, default=224, help="Model-ready square size.")
    parser.add_argument("--model-quality", type=int, default=90, help="JPEG quality for model images.")
    parser.add_argument("--convert-jpg", action="store_true", default=True, help="Convert output to JPEG.")
    parser.add_argument("--trim-border", default="false", help="Trim near-white borders for model images (true/false).")
    parser.add_argument("--force", action="store_true", help="Overwrite existing optimized files.")
    parser.add_argument("--display-only", action="store_true", help="Only create display versions.")
    parser.add_argument("--model-only", action="store_true", help="Only create model-ready versions.")
    args = parser.parse_args()

    config = build_config()
    trim_border = _parse_bool(args.trim_border)
    output_base = args.output if os.path.isabs(args.output) else os.path.join(Config.BASE_DIR, args.output)
    display_root = os.path.join(output_base, "display")
    model_root = os.path.join(output_base, "model")

    create_display = not args.model_only
    create_model = not args.display_only

    if args.source == "all":
        report = optimize_all_dataset_folders(
            config,
            quality=args.quality,
            max_size=args.max_size,
            model_size=args.model_size,
            model_quality=args.model_quality,
            trim_border=trim_border,
            force=args.force,
        )
        print("Image optimization completed.")
        print(f"  Scanned: {report['total_scanned']}")
        print(f"  Optimized: {report['optimized']}")
        print(f"  Skipped: {report['skipped']}")
        print(f"  Original: {report['original_size_mb']} MB")
        print(f"  Optimized: {report['optimized_size_mb']} MB")
        print(f"  Saved: {report['saved_mb']} MB ({report['compression_percent']}%)")
        print(f"  Report: {os.path.join(Config.MODEL_DIR, 'image_optimization_report.json')}")
        return

    source = args.source if os.path.isabs(args.source) else os.path.join(Config.DATASET_DIR, args.source)
    if not os.path.isdir(source):
        print(f"Error: source folder not found: {source}", file=sys.stderr)
        sys.exit(1)

    rel_name = os.path.relpath(source, Config.DATASET_DIR)
    stats = optimize_dataset_images(
        source_dir=source,
        output_dir=output_base,
        max_size=args.max_size,
        quality=args.quality,
        model_size=args.model_size,
        model_quality=args.model_quality,
        convert_to_jpg=args.convert_jpg,
        trim_border=trim_border,
        force=args.force,
        create_display=create_display,
        create_model=create_model,
        display_output_dir=os.path.join(display_root, rel_name) if create_display else None,
        model_output_dir=os.path.join(model_root, rel_name) if create_model else None,
    )
    report = generate_optimization_report([stats], config, output_base)
    print("Image optimization completed.")
    print(f"  Source: {source}")
    print(f"  Display output: {os.path.join(display_root, rel_name)}")
    print(f"  Model output: {os.path.join(model_root, rel_name)}")
    print(f"  Optimized: {report['optimized']} / {report['total_scanned']}")
    print(f"  Saved: {report['saved_mb']} MB ({report['compression_percent']}%)")


if __name__ == "__main__":
    main()
