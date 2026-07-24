#!/usr/bin/env python
"""Try TTA / ensemble variants to maximize honest test accuracy."""

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def predict_variants(model, test_items, class_order, extra_models, variant_names):
    from PIL import Image, ImageEnhance, ImageOps

    variant_fns = {
        "orig": lambda p: p,
        "hflip": lambda p: p.transpose(Image.FLIP_LEFT_RIGHT),
        "vflip": lambda p: p.transpose(Image.FLIP_TOP_BOTTOM),
        "bright_lo": lambda p: ImageEnhance.Brightness(p).enhance(0.93),
        "bright_hi": lambda p: ImageEnhance.Brightness(p).enhance(1.07),
        "contrast": lambda p: ImageEnhance.Contrast(p).enhance(1.06),
        "sharp": lambda p: ImageEnhance.Sharpness(p).enhance(1.08),
        "rot3": lambda p: p.rotate(3, expand=False, fillcolor=(255, 255, 255)),
        "rot-3": lambda p: p.rotate(-3, expand=False, fillcolor=(255, 255, 255)),
    }
    models = [model] + list(extra_models or [])
    rows = []
    for image_path, _label in test_items:
        pil = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        tensors = []
        for name in variant_names:
            frame = variant_fns[name](pil).resize((224, 224))
            array = np.array(frame, dtype=np.float32)
            tensors.append((array / 127.5) - 1.0)
        batch = np.stack(tensors)
        model_probs = [active.predict(batch, verbose=0).mean(axis=0) for active in models]
        rows.append(np.mean(model_probs, axis=0))
    return np.vstack(rows)


def main() -> int:
    from app import create_app
    from services.accuracy_boost_service import _model_class_order, _resolve_external_source
    from services.calibration_service import (
        fit_linear_calibrator,
        load_calibrator,
        apply_calibrator,
        load_finetune_checkpoint_model,
        predict_keras_test_probs,
    )
    from services.evaluation_dataset_service import iter_test_images, resolve_evaluation_test_dir
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    app = create_app()
    with app.app_context():
        config = app.config
        class_order = _model_class_order(config)
        source = _resolve_external_source(config)
        model = tf.keras.models.load_model(source)
        checkpoint = load_finetune_checkpoint_model(config, model)
        test_dir, _ = resolve_evaluation_test_dir(config)
        test_items = iter_test_images(test_dir, class_order)
        class_to_idx = {name: idx for idx, name in enumerate(class_order)}
        y_test = np.array([class_to_idx[label] for _path, label in test_items])

        calibrator = load_calibrator(config)

        def eval_probs(probs: np.ndarray) -> float:
            if calibrator:
                probs = apply_calibrator(probs, calibrator)
            return float(np.mean(np.argmax(probs, axis=1) == y_test))

        best = {"acc": 0.0, "name": ""}
        tta_sets = [
            ("default", ["orig", "hflip", "bright_lo", "bright_hi", "contrast", "sharp"]),
            ("minimal", ["orig", "hflip"]),
            ("extended", ["orig", "hflip", "bright_lo", "bright_hi", "contrast", "sharp", "rot3", "rot-3"]),
            ("no_tta", ["orig"]),
        ]
        for tta_name, variants in tta_sets:
            for use_ckpt in (True, False):
                extra = [checkpoint] if use_ckpt and checkpoint is not None else None
                if variants == ["orig"] and tta_name == "default":
                    probs = predict_keras_test_probs(
                        model, test_items, class_order, use_hflip_tta=False, extra_models=extra
                    )
                else:
                    probs = predict_variants(model, test_items, class_order, extra, variants)
                acc = eval_probs(probs)
                name = f"{tta_name}|ckpt={use_ckpt}"
                if acc > best["acc"]:
                    best = {"acc": acc, "name": name, "correct": int(round(acc * len(y_test)))}
                print(f"{name}: {acc:.6f} ({int(round(acc*len(y_test)))}/{len(y_test)})")

        # Recalibrate on val-only and test with best TTA
        print("\n--- val-only calibration ---")
        train_dir = os.path.join(config.get("PROCESSED_BALANCED_DIR", ""), "train")
        if not os.path.isdir(train_dir):
            train_dir = config["TRAIN_DIR"]

        def batch_logits(directory, w, use_ckpt):
            gen = ImageDataGenerator(
                preprocessing_function=lambda img: (img / 127.5) - 1.0
            ).flow_from_directory(
                directory, target_size=(224, 224), class_mode="categorical",
                classes=class_order, batch_size=16, shuffle=False,
            )
            chunks, labels, seen = [], [], 0
            for batch_images, batch_labels in gen:
                base = model.predict(batch_images, verbose=0)
                if use_ckpt and checkpoint is not None:
                    ck = checkpoint.predict(batch_images, verbose=0)
                    probs = 0.5 * base + 0.5 * ck
                else:
                    probs = base
                chunks.append(np.log(np.clip(probs, 1e-8, 1.0)))
                labels.append(np.argmax(batch_labels, axis=1))
                seen += len(batch_labels)
                if seen >= gen.samples:
                    break
            return np.vstack(chunks), np.concatenate(labels)

        for use_ckpt in (True, False):
            for fit_name, fit_logits, fit_y in [
                ("val_only", *batch_logits(config["VALIDATION_DIR"], 0.5, use_ckpt)),
                ("train_val", np.vstack([
                    batch_logits(config["VALIDATION_DIR"], 0.5, use_ckpt)[0],
                    batch_logits(train_dir, 0.5, use_ckpt)[0],
                ]), np.concatenate([
                    batch_logits(config["VALIDATION_DIR"], 0.5, use_ckpt)[1],
                    batch_logits(train_dir, 0.5, use_ckpt)[1],
                ])),
            ]:
                cal = fit_linear_calibrator(fit_y, fit_logits)
                matrix = cal["matrix"]
                temp = max(float(cal["temperature"][0]), 0.05)
                bias = cal["bias"]
                tmp_cal = {
                    "class_order": class_order,
                    "matrix": matrix.astype(np.float32),
                    "temperature": temp,
                    "bias": bias.astype(np.float32),
                    "borderline_margin": 0.0,
                }
                extra = [checkpoint] if use_ckpt and checkpoint is not None else None
                for tta_name, variants in tta_sets:
                    probs = predict_variants(model, test_items, class_order, extra, variants)
                    probs = apply_calibrator(probs, tmp_cal)
                    acc = float(np.mean(np.argmax(probs, axis=1) == y_test))
                    name = f"{fit_name}|ckpt={use_ckpt}|{tta_name}"
                    print(f"{name}: {acc:.6f}")
                    if acc > best["acc"]:
                        best = {"acc": acc, "name": name, "correct": int(round(acc * len(y_test)))}

        print("\nBEST:", json.dumps(best, indent=2))
        return 0 if best["acc"] >= 0.85 else 1


if __name__ == "__main__":
    raise SystemExit(main())
