"""Load Teachable Machine TF.js models without full tensorflowjs optional dependencies."""

from __future__ import annotations

import sys
import types
from typing import Any


def _stub_optional_tfjs_modules() -> None:
    if getattr(_stub_optional_tfjs_modules, "_done", False):
        return

    jax_mod = types.ModuleType("jax")
    jax_mod.monitoring = types.SimpleNamespace(record_scalar=lambda *args, **kwargs: None)
    jax_mod.experimental = types.ModuleType("jax.experimental")
    jax_mod.experimental.jax2tf = types.ModuleType("jax.experimental.jax2tf")

    stubs = {
        "tensorflow_decision_forests": types.ModuleType("tensorflow_decision_forests"),
        "tensorflow_hub": types.ModuleType("tensorflow_hub"),
        "jax": jax_mod,
        "jaxlib": types.ModuleType("jaxlib"),
        "flax": types.ModuleType("flax"),
        "importlib_resources": types.ModuleType("importlib_resources"),
        "jax.experimental": jax_mod.experimental,
        "jax.experimental.jax2tf": jax_mod.experimental.jax2tf,
    }
    for name, module in stubs.items():
        sys.modules.setdefault(name, module)

    _stub_optional_tfjs_modules._done = True  # type: ignore[attr-defined]


def load_keras_model(config_json_path: str, **kwargs: Any):
    _stub_optional_tfjs_modules()
    import tensorflow  # noqa: F401
    import tf_keras  # noqa: F401
    from tensorflowjs.converters.keras_tfjs_loader import load_keras_model as _load

    return _load(config_json_path, **kwargs)
