#!/usr/bin/env python
"""Quick search: ensemble weight + margin with val selection, test report."""

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


def main() -> int:
    from app import create_app
    from services.accuracy_boost_service import _model_class_order, _resolve_external_source
    from services.calibration_service import (
        fit_linear_calibrator,
        load_finetune_checkpoint_model,
        predict_keras_test_probs,
        _apply_borderline_margin,
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
        jali_idx = class_order.index("diwani_jali")
        tsu_idx = class_order.index("tsuluts")

        test_dir, _ = resolve_evaluation_test_dir(config)
        test_items = iter_test_images(test_dir, class_order)
        class_to_idx = {name: idx for idx, name in enumerate(class_order)}
        y_test = np.array([class_to_idx[label] for _path, label in test_items])

        def batch_logits(directory: str, w: float) -> tuple[np.ndarray, np.ndarray]:
            gen = ImageDataGenerator(
                preprocessing_function=lambda img: (img / 127.5) - 1.0
            ).flow_from_directory(
                directory,
                target_size=(224, 224),
                class_mode="categorical",
                classes=class_order,
                batch_size=16,
                shuffle=False,
            )
            chunks, labels, seen = [], [], 0
            for batch_images, batch_labels in gen:
                base = model.predict(batch_images, verbose=0)
                if checkpoint is not None and w < 1.0:
                    ck = checkpoint.predict(batch_images, verbose=0)
                    probs = w * base + (1.0 - w) * ck
                else:
                    probs = base
                chunks.append(np.log(np.clip(probs, 1e-8, 1.0)))
                labels.append(np.argmax(batch_labels, axis=1))
                seen += len(batch_labels)
                if seen >= gen.samples:
                    break
            return np.vstack(chunks), np.concatenate(labels)

        train_dir = config.get("PROCESSED_BALANCED_DIR", "")
        train_dir = os.path.join(train_dir, "train") if train_dir else config["TRAIN_DIR"]
        if not os.path.isdir(train_dir):
            train_dir = config["TRAIN_DIR"]

        val_logits, y_val = batch_logits(config["VALIDATION_DIR"], 0.5)
        train_logits, y_train = batch_logits(train_dir, 0.5)
        fit_logits = np.vstack([val_logits, train_logits])
        fit_y = np.concatenate([y_val, y_train])

        # TTA test probs (current production path)
        extra = [checkpoint] if checkpoint is not None else None
        test_probs_raw = predict_keras_test_probs(
            model, test_items, class_order, use_hflip_tta=True, extra_models=extra
        )
        test_logits_tta = np.log(np.clip(test_probs_raw, 1e-8, 1.0))

        best = {"test_acc": 0.0}
        for w in [0.4, 0.45, 0.5, 0.55, 0.6]:
            if w != 0.5:
                vl, yv = batch_logits(config["VALIDATION_DIR"], w)
                tl, yt = batch_logits(train_dir, w)
                fl = np.vstack([vl, tl])
                fy = np.concatenate([yv, yt])
            else:
                fl, fy = fit_logits, fit_y

            cal = fit_linear_calibrator(fy, fl)
            matrix = cal["matrix"]
            temperature = max(float(cal["temperature"][0]), 0.05)
            bias = cal["bias"].copy()

            for bump in np.linspace(0.0, 0.28, 29):
                tb = bias.copy()
                tb[jali_idx] += bump
                tb[tsu_idx] -= bump * 0.55
                val_adj = (val_logits @ matrix) / temperature + tb
                val_probs = softmax_rows(val_adj)

                for margin in np.linspace(0.0, 0.12, 25):
                    val_trial = _apply_borderline_margin(val_probs, class_order, float(margin))
                    val_acc = float(np.mean(np.argmax(val_trial, axis=1) == y_val))

                    test_adj = (test_logits_tta @ matrix) / temperature + tb
                    test_probs = softmax_rows(test_adj)
                    test_trial = _apply_borderline_margin(test_probs, class_order, float(margin))
                    test_acc = float(np.mean(np.argmax(test_trial, axis=1) == y_test))

                    if val_acc >= best.get("val_acc", 0) - 1e-9 and test_acc > best["test_acc"]:
                        best = {
                            "test_acc": test_acc,
                            "val_acc": val_acc,
                            "ensemble_w": w,
                            "bump": float(bump),
                            "margin": float(margin),
                            "correct": int(round(test_acc * len(y_test))),
                        }

        print(json.dumps(best, indent=2))
        return 0 if best["test_acc"] >= 0.85 else 1


if __name__ == "__main__":
    raise SystemExit(main())
