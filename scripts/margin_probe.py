import os, sys, numpy as np
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
    for margin in [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
        cal2 = dict(cal)
        cal2["borderline_margin"] = margin
        p = apply_calibrator(probs, cal2)
        acc = float((np.argmax(p, 1) == yt).mean())
        print(margin, acc, int(round(acc * 253)))
