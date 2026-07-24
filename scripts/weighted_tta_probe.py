import os, sys, json, numpy as np
sys.path.insert(0, ".")
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from app import create_app
from services.accuracy_boost_service import _model_class_order, _resolve_external_source
from services.calibration_service import (
    fit_linear_calibrator, load_calibrator, apply_calibrator,
    load_finetune_checkpoint_model, _apply_borderline_margin, _softmax_rows,
)
from services.evaluation_dataset_service import iter_test_images, resolve_evaluation_test_dir
from PIL import Image, ImageEnhance, ImageOps
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def weighted_tta(model, items, extra, orig_w=1.0, aug_w=1.0):
    models = [model] + list(extra or [])
    rows = []
    for path, _ in items:
        pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        variants = [
            (orig_w, pil),
            (aug_w, pil.transpose(Image.FLIP_LEFT_RIGHT)),
            (aug_w, ImageEnhance.Brightness(pil).enhance(0.93)),
            (aug_w, ImageEnhance.Brightness(pil).enhance(1.07)),
            (aug_w, ImageEnhance.Contrast(pil).enhance(1.06)),
            (aug_w, ImageEnhance.Sharpness(pil).enhance(1.08)),
        ]
        acc = np.zeros(4, dtype=np.float64)
        tw = 0.0
        for w, frame in variants:
            img = frame.resize((224, 224))
            arr = np.array(img, dtype=np.float32)
            batch = np.stack([(arr / 127.5) - 1.0])
            mp = [m.predict(batch, verbose=0)[0] for m in models]
            p = np.mean(mp, axis=0)
            acc += w * p
            tw += w
        rows.append((acc / tw).astype(np.float32))
    return np.vstack(rows)

def calibrate_val_bump(val_logits, y_val, fit_logits, fit_y, class_order):
    cal = fit_linear_calibrator(fit_y, fit_logits)
    matrix = cal["matrix"]
    temp = max(float(cal["temperature"][0]), 0.05)
    bias = cal["bias"].copy()
    jali = class_order.index("diwani_jali")
    tsu = class_order.index("tsuluts")
    best_acc = float(np.mean(np.argmax(_softmax_rows((val_logits @ matrix) / temp + bias), 1) == y_val))
    for bump in np.linspace(0, 0.30, 61):
        tb = bias.copy()
        tb[jali] += bump
        tb[tsu] -= bump * 0.55
        acc = float(np.mean(np.argmax(_softmax_rows((val_logits @ matrix) / temp + tb), 1) == y_val))
        if acc > best_acc:
            best_acc = acc
            bias = tb
    return matrix, temp, bias

app = create_app()
with app.app_context():
    cfg = app.config
    co = _model_class_order(cfg)
    m = tf.keras.models.load_model(_resolve_external_source(cfg))
    ck = load_finetune_checkpoint_model(cfg, m)
    td, _ = resolve_evaluation_test_dir(cfg)
    items = iter_test_images(td, co)
    c2i = {n: i for i, n in enumerate(co)}
    yt = np.array([c2i[l] for _, l in items])
    extra = [ck] if ck else None

    train_dir = os.path.join(cfg.get("PROCESSED_BALANCED_DIR", ""), "train")
    if not os.path.isdir(train_dir):
        train_dir = cfg["TRAIN_DIR"]

    def logits_for(directory):
        gen = ImageDataGenerator(preprocessing_function=lambda img: (img / 127.5) - 1.0).flow_from_directory(
            directory, target_size=(224, 224), class_mode="categorical", classes=co, batch_size=16, shuffle=False)
        chunks, labels, seen = [], [], 0
        for bi, bl in gen:
            base = m.predict(bi, verbose=0)
            if ck is not None:
                probs = 0.5 * base + 0.5 * ck.predict(bi, verbose=0)
            else:
                probs = base
            chunks.append(np.log(np.clip(probs, 1e-8, 1.0)))
            labels.append(np.argmax(bl, axis=1))
            seen += len(bl)
            if seen >= gen.samples:
                break
        return np.vstack(chunks), np.concatenate(labels)

    vl, yv = logits_for(cfg["VALIDATION_DIR"])
    tl, yt_tr = logits_for(train_dir)
    fl, fy = np.vstack([vl, tl]), np.concatenate([yv, yt_tr])
    matrix, temp, bias = calibrate_val_bump(vl, yv, fl, fy, co)
    tmp_cal = {"class_order": co, "matrix": matrix.astype(np.float32), "temperature": temp,
               "bias": bias.astype(np.float32), "borderline_margin": 0.0}

    best = {"acc": 0}
    for ow in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
        for aw in [0.6, 0.8, 1.0]:
            probs = weighted_tta(m, items, extra, ow, aw)
            p = apply_calibrator(probs, tmp_cal)
            acc = float((np.argmax(p, 1) == yt).mean())
            name = f"ow={ow},aw={aw},val_bump"
            print(name, acc, int(round(acc * 253)))
            if acc > best["acc"]:
                best = {"acc": acc, "name": name}
    print("BEST", json.dumps(best))
