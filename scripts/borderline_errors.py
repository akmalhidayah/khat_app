import os, sys, json, numpy as np
sys.path.insert(0, ".")
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from app import create_app
from services.accuracy_boost_service import _model_class_order, _resolve_external_source
from services.calibration_service import (
    load_calibrator, apply_calibrator, predict_keras_test_probs, load_finetune_checkpoint_model,
)
from services.evaluation_dataset_service import iter_test_images, resolve_evaluation_test_dir
import tensorflow as tf

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
    probs = predict_keras_test_probs(m, items, co, True, [ck] if ck else None)
    cal = load_calibrator(cfg)
    p = apply_calibrator(probs, cal)
    pred = np.argmax(p, 1)
    close = []
    for i, (path, label) in enumerate(items):
        ti, pi = c2i[label], pred[i]
        if ti == pi:
            continue
        gap = float(p[i, ti] - p[i, pi])
        close.append({
            "file": os.path.basename(path),
            "true": label,
            "pred": co[pi],
            "true_prob": float(p[i, ti]),
            "pred_prob": float(p[i, pi]),
            "gap": gap,
        })
    close.sort(key=lambda x: x["gap"], reverse=True)
    print(json.dumps(close[:15], indent=2))
    print("total_wrong", len(close))
