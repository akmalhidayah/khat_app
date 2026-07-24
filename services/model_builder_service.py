"""Model builders for VGG16, EfficientNetB0, and MobileNetV2 transfer learning."""

import os
from typing import Tuple

ARCHITECTURES = {
    "vgg16": "VGG16 Transfer Learning",
    "efficientnetb0": "EfficientNetB0 Transfer Learning",
    "mobilenetv2": "MobileNetV2 Transfer Learning",
}

DEFAULT_FAST_ARCHITECTURE = "mobilenetv2"
DEFAULT_RESEARCH_ARCHITECTURE = "efficientnetb0"

FINE_TUNE_LAYERS = {
    "vgg16": 15,
    "efficientnetb0": 20,
    "mobilenetv2": 20,
}


def resolve_architecture(architecture: str, training_mode: str) -> str:
    arch = (architecture or "").lower()
    if arch in ARCHITECTURES:
        return arch
    if training_mode in ("ultra_fast", "fast"):
        return DEFAULT_FAST_ARCHITECTURE
    return DEFAULT_RESEARCH_ARCHITECTURE


def build_classifier(
    architecture: str,
    num_classes: int,
    learning_rate: float,
    trainable_backbone: bool = False,
    fine_tune_layers: int = 0,
    optimizer_name: str = "adam",
):
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import Adam

    architecture = (architecture or DEFAULT_RESEARCH_ARCHITECTURE).lower()

    if architecture == "efficientnetb0":
        from tensorflow.keras.applications import EfficientNetB0
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    elif architecture == "mobilenetv2":
        from tensorflow.keras.applications import MobileNetV2
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    else:
        from tensorflow.keras.applications import VGG16
        base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    base.trainable = trainable_backbone
    if fine_tune_layers > 0 and hasattr(base, "layers"):
        for layer in base.layers:
            layer.trainable = False
        for layer in base.layers[-fine_tune_layers:]:
            layer.trainable = True

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    if architecture in ("efficientnetb0", "mobilenetv2"):
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
    else:
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=base.input, outputs=outputs)
    if optimizer_name.lower() == "adamw":
        try:
            optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
        except AttributeError:
            optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model._khat_backbone = base
    return model, architecture


def _find_backbone(model):
    backbone = getattr(model, "_khat_backbone", None)
    if backbone is not None:
        return backbone
    import tensorflow as tf

    for layer in model.layers:
        name = (layer.name or "").lower()
        if any(key in name for key in ("efficientnet", "vgg16", "mobilenet")):
            return layer
        if isinstance(layer, tf.keras.Model) and len(layer.layers) > 20:
            return layer
    raise ValueError("Could not locate backbone model for fine-tuning.")


def companion_weights_path(model_path: str) -> str:
    base, _ = os.path.splitext(model_path)
    return f"{base}.weights.h5"


def load_trained_classifier(
    model_path: str,
    architecture: str,
    num_classes: int,
    learning_rate: float = 1e-4,
):
    """Load a trained classifier using Keras 3 first.

    The production EfficientNet model is stored in modern `.keras`
    format. Legacy TensorFlow/TF-Keras loading is retained as fallback
    for older H5 models and companion weight files.
    """

    load_errors = []

    # Modern full-model format: load through standalone Keras 3.
    if model_path and model_path.lower().endswith(".keras"):
        try:
            import keras

            return keras.models.load_model(
                model_path,
                compile=False,
            )
        except Exception as exc:
            load_errors.append(
                f"Keras 3 full model: {exc}"
            )

    # Legacy H5 support must be configured before TensorFlow import.
    if model_path and model_path.lower().endswith(".h5"):
        os.environ.setdefault(
            "TF_USE_LEGACY_KERAS",
            "1",
        )

    try:
        import tensorflow as tf

        return tf.keras.models.load_model(
            model_path,
            compile=False,
        )
    except Exception as exc:
        load_errors.append(
            f"TensorFlow Keras model: {exc}"
        )

    try:
        import tf_keras

        return tf_keras.models.load_model(
            model_path,
            compile=False,
        )
    except Exception as exc:
        load_errors.append(
            f"TF-Keras model: {exc}"
        )

    model, _ = build_classifier(
        architecture,
        num_classes,
        learning_rate,
    )

    base, _ = os.path.splitext(model_path)

    weight_candidates = [
        companion_weights_path(model_path),
        f"{base}_checkpoint.weights.h5",
        model_path,
    ]

    for weights_path in weight_candidates:
        if (
            not os.path.isfile(weights_path)
            or os.path.getsize(weights_path) < 1000
        ):
            continue

        try:
            model.load_weights(weights_path)
            return model
        except Exception as exc:
            load_errors.append(
                f"Weights {weights_path}: {exc}"
            )

    detail = " | ".join(load_errors)

    raise ValueError(
        f"Unable to load model weights for {model_path}. "
        f"Details: {detail}"
    )


def save_classifier_bundle(model, model_path: str) -> None:
    model.save(model_path)
    model.save_weights(companion_weights_path(model_path))


def unfreeze_last_layers(model, architecture: str, num_layers: int = None) -> None:
    architecture = (architecture or "vgg16").lower()
    if num_layers is None:
        num_layers = FINE_TUNE_LAYERS.get(architecture, 4)
    backbone = _find_backbone(model)
    for layer in backbone.layers:
        layer.trainable = False
    for layer in backbone.layers[-num_layers:]:
        layer.trainable = True
