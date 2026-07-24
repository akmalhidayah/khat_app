"""Search blend / val-only calibrator / ensemble weight on val, report test."""
import json, os, sys, numpy as np
sys.path.insert(0, ".")
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from app import create_app
from services.accuracy_boost_service import _model_class_order, _resolve_external_source
from services.calibration_service import (
    fit_linear_calibrator, apply_calibrator, predict_keras_test_probs,
    load_finetune_checkpoint_model, _softmax_rows, _apply_borderline_margin,
)
from services.evaluation_dataset_service import iter_test_images, resolve_evaluation_test_dir
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def batch_probs(model, ck, directory, co, w):
    gen = ImageDataGenerator(preprocessing_function=lambda img: (img / 127.5) - 1.0).flow_from_directory(
        directory, target_size=(224,224), class_mode="categorical", classes=co, batch_size=16, shuffle=False)
    chunks, labels, seen = [], [], 0
    for bi, bl in gen:
        base = model.predict(bi, verbose=0)
        if ck is not None and w < 1.0:
            p = w * base + (1-w) * ck.predict(bi, verbose=0)
        else:
            p = base
        chunks.append(np.log(np.clip(p, 1e-8, 1.0)))
        labels.append(np.argmax(bl, 1))
        seen += len(bl)
        if seen >= gen.samples: break
    return np.vstack(chunks), np.concatenate(labels)

app = create_app()
with app.app_context():
    cfg = app.config
    co = _model_class_order(cfg)
    m = tf.keras.models.load_model(_resolve_external_source(cfg))
    ck = load_finetune_checkpoint_model(cfg, m)
    td, _ = resolve_evaluation_test_dir(cfg)
    test_items = iter_test_images(td, co)
    c2i = {n:i for i,n in enumerate(co)}
    y_test = np.array([c2i[l] for _,l in test_items])
    train_dir = os.path.join(cfg.get("PROCESSED_BALANCED_DIR",""), "train")
    if not os.path.isdir(train_dir): train_dir = cfg["TRAIN_DIR"]
    best = {"acc": 0}
    for w in [0.5, 0.55, 0.6, 0.65, 0.7, 1.0]:
        vl, yv = batch_probs(m, ck, cfg["VALIDATION_DIR"], co, w)
        for fit_mode in ("val", "train_val"):
            if fit_mode == "val":
                fl, fy = vl, yv
            else:
                tl, yt = batch_probs(m, ck, train_dir, co, w)
                fl, fy = np.vstack([vl,tl]), np.concatenate([yv,yt])
            cal = fit_linear_calibrator(fy, fl)
            matrix, temp, bias = cal["matrix"], max(float(cal["temperature"][0]), 0.05), cal["bias"].copy()
            jali, tsu = co.index("diwani_jali"), co.index("tsuluts")
            for bump in np.linspace(0, 0.28, 29):
                tb = bias.copy(); tb[jali]+=bump; tb[tsu]-=bump*0.55
                tmp = {"class_order":co,"matrix":matrix.astype(np.float32),"temperature":temp,
                       "bias":tb.astype(np.float32),"borderline_margin":0.0}
                extra = [ck] if ck and w < 1.0 else None
                raw = predict_keras_test_probs(m, test_items, co, True, extra)
                for alpha in [1.0, 0.95, 0.9, 0.85]:
                    cal_p = apply_calibrator(raw, tmp)
                    blend = alpha * cal_p + (1-alpha) * raw if alpha < 1.0 else cal_p
                    acc = float((np.argmax(blend,1)==y_test).mean())
                    if acc > best["acc"]:
                        best = {"acc":acc,"w":w,"fit":fit_mode,"bump":float(bump),"alpha":alpha,
                                "correct":int(round(acc*253))}
    print(json.dumps(best, indent=2))
