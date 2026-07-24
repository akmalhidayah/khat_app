"""Check TensorFlow / TensorFlow.js dependencies for server-side TM evaluation."""

from __future__ import annotations

from typing import List, Tuple


INSTALL_HINT = (
    "Dependensi evaluasi belum lengkap. Jalankan di folder proyek:\n"
    "  installer\\install_eval_deps.bat\n"
    "atau:\n"
    "  .\\.venv\\Scripts\\pip.exe install tensorflow==2.20.0 tensorflowjs==4.22.0 h5py>=3.11.0"
)


def check_keras_eval_dependencies() -> Tuple[bool, List[str], str]:
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return False, ["tensorflow"], (
            "TensorFlow belum terpasang. Jalankan: installer\\install_eval_deps.bat "
            "atau .\\.venv\\Scripts\\pip.exe install tensorflow==2.20.0 h5py tf-keras"
        )
    return True, [], ""


def check_eval_dependencies() -> Tuple[bool, List[str], str]:
    missing: List[str] = []

    try:
        import tensorflow  # noqa: F401
    except ImportError:
        missing.append("tensorflow")

    if "tensorflow" not in missing:
        try:
            from services.tfjs_import_bootstrap import load_keras_model  # noqa: F401
        except ImportError:
            missing.append("tensorflowjs")

    if missing:
        detail = ", ".join(missing)
        return False, missing, f"{INSTALL_HINT}\n(Paket yang belum terpasang: {detail})"
    return True, [], ""
