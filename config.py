import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv()

_BASE_PATH = Path(__file__).resolve().parent

# Default external asset locations (override via .env)
DEFAULT_EXTERNAL_DATASET_DIR = r"D:\dataset_kaligrafi"
DEFAULT_EXTERNAL_MODEL_DIR = r"D:\model_treaning"

_EXTERNAL_DATASET_DIR = os.getenv("EXTERNAL_DATASET_DIR", DEFAULT_EXTERNAL_DATASET_DIR).strip()
_EXTERNAL_MODEL_DIR = os.getenv("EXTERNAL_MODEL_DIR", DEFAULT_EXTERNAL_MODEL_DIR).strip()
_EXTERNAL_MODEL_FILE = os.getenv("EXTERNAL_MODEL_FILE", "keras_model.h5").strip()
_EXTERNAL_LABELS_FILE = os.getenv("EXTERNAL_LABELS_FILE", "labels.txt").strip()
_USE_EXTERNAL_DATASET = os.getenv(
    "USE_EXTERNAL_DATASET",
    "1" if _EXTERNAL_DATASET_DIR else "0",
).strip().lower() in ("1", "true", "yes")
_USE_EXTERNAL_MODEL = os.getenv(
    "USE_EXTERNAL_MODEL",
    "1" if _EXTERNAL_MODEL_DIR else "0",
).strip().lower() in ("1", "true", "yes")


def _resolve_external_model_path(model_dir: str, preferred_file: str) -> str:
    """Pick the trained model file from an external directory."""
    root = Path(model_dir)
    if not model_dir or not root.is_dir():
        return str(root / preferred_file) if preferred_file else ""

    candidates: list[Path] = []
    if preferred_file:
        candidates.append(root / preferred_file)
    candidates.extend(
        [
            root / "keras_model.h5",
            root / "khat_best.keras",
            root / "khat_latest.keras",
            root / "model.keras",
        ]
    )
    for path in candidates:
        if path.is_file() and path.stat().st_size > 10_000:
            return str(path)
    for pattern in ("*.keras", "*.h5"):
        for path in sorted(root.glob(pattern)):
            if path.is_file() and path.stat().st_size > 10_000:
                return str(path)
    return str(root / preferred_file) if preferred_file else ""


def _dataset_paths_from_base(base: Path) -> dict[str, str]:
    return {
        "DATASET_DIR": str(base),
        "RAW_DATASET_DIR": str(base),
        "NON_KHAT_DIR": str(base / "non_khat"),
        "KHAT_DETECTOR_DATASET_DIR": str(base / "khat_detector"),
        "KHAT_DETECTOR_TRAIN_DIR": str(base / "khat_detector" / "train"),
        "KHAT_DETECTOR_VALIDATION_DIR": str(base / "khat_detector" / "validation"),
        "KHAT_DETECTOR_TEST_DIR": str(base / "khat_detector" / "test"),
        "REJECTED_DATASET_DIR": str(base / "rejected"),
        "DATASET_TMP_DIR": str(base / "tmp"),
        "PROCESSED_DATASET_DIR": str(base / "processed"),
        "MODEL_READY_DIR": str(base / "model_ready"),
        "PROCESSED_BALANCED_DIR": str(base / "processed_balanced"),
        "TRAIN_DIR": str(base / "train"),
        "VALIDATION_DIR": str(base / "validation"),
        "TEST_DIR": str(base / "test"),
        "OPTIMIZED_DIR": str(base / "optimized"),
        "OPTIMIZED_DISPLAY_DIR": str(base / "optimized" / "display"),
        "OPTIMIZED_MODEL_DIR": str(base / "optimized" / "model"),
    }


def _active_dataset_paths() -> dict[str, str]:
    if _USE_EXTERNAL_DATASET and _EXTERNAL_DATASET_DIR:
        return _dataset_paths_from_base(Path(_EXTERNAL_DATASET_DIR))
    paths = _dataset_paths_from_base(_BASE_PATH / "dataset")
    paths["RAW_DATASET_DIR"] = str(_BASE_PATH / "dataset" / "raw")
    paths["NON_KHAT_DIR"] = str(_BASE_PATH / "dataset" / "raw" / "non_khat")
    return paths


_DATASET_PATHS = _active_dataset_paths()
_EXTERNAL_MODEL_PATH = (
    _resolve_external_model_path(_EXTERNAL_MODEL_DIR, _EXTERNAL_MODEL_FILE)
    if _USE_EXTERNAL_MODEL and _EXTERNAL_MODEL_DIR
    else ""
)
_RESOLVED_EXTERNAL_MODEL_FILE = (
    Path(_EXTERNAL_MODEL_PATH).name if _EXTERNAL_MODEL_PATH else _EXTERNAL_MODEL_FILE
)


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "khat-dev-secret-key")
    _DB_USER = os.getenv("DB_USERNAME", os.getenv("DB_USER", "root"))
    _DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD", ""))
    _DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
    _DB_PORT = os.getenv("DB_PORT", "3306")
    _DB_NAME = os.getenv("DB_DATABASE", os.getenv("DB_NAME", "db_khat_classification"))

    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://{_DB_USER}:{_DB_PASSWORD}@{_DB_HOST}:{_DB_PORT}/{_DB_NAME}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "1024")) * 1024 * 1024
    UPLOAD_MAX_SIZE_MB = int(os.getenv("MAX_UPLOAD_MB", "1024"))
    UPLOAD_CHUNK_SIZE_MB = 20
    UPLOAD_CHUNK_SIZE_BYTES = UPLOAD_CHUNK_SIZE_MB * 1024 * 1024

    APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
    APP_PORT = int(os.getenv("APP_PORT", os.getenv("FLASK_PORT", "5002")))
    APP_SIMPLE_MODE = os.getenv("APP_SIMPLE_MODE", "1").strip().lower() in ("1", "true", "yes")

    EXTERNAL_DATASET_DIR = _EXTERNAL_DATASET_DIR
    EXTERNAL_MODEL_DIR = _EXTERNAL_MODEL_DIR
    EXTERNAL_MODEL_FILE = _RESOLVED_EXTERNAL_MODEL_FILE
    EXTERNAL_LABELS_FILE = _EXTERNAL_LABELS_FILE
    USE_EXTERNAL_DATASET = _USE_EXTERNAL_DATASET
    USE_EXTERNAL_MODEL = _USE_EXTERNAL_MODEL
    USE_TEACHABLE_MACHINE = os.getenv(
        "USE_TEACHABLE_MACHINE",
        "0" if _USE_EXTERNAL_MODEL else "1",
    ).strip().lower() in ("1", "true", "yes")
    EXTERNAL_DATASET_SOURCE = _EXTERNAL_DATASET_DIR if _USE_EXTERNAL_DATASET and _EXTERNAL_DATASET_DIR else ""

    BASE_DIR = str(_BASE_PATH)
    LOGS_DIR = str(_BASE_PATH / "logs")
    TEMP_DIR = str(_BASE_PATH / "temp")
    INSTANCE_DIR = str(_BASE_PATH / "instance")
    UPLOAD_FOLDER = str(_BASE_PATH / "static" / "uploads")
    DATASET_PREVIEW_FOLDER = str(_BASE_PATH / "static" / "dataset_preview")
    EVALUATION_FOLDER = str(_BASE_PATH / "static" / "evaluation")
    EVALUATION_PREVIEW_DIR = str(_BASE_PATH / "static" / "evaluation_previews")
    MODEL_DIR = str(_BASE_PATH / "model")
    TEACHABLE_MODEL_DIR = str(_BASE_PATH / "static" / "model")
    TEACHABLE_MODEL_JSON = str(_BASE_PATH / "static" / "model" / "model.json")
    TEACHABLE_MODEL_METADATA = str(_BASE_PATH / "static" / "model" / "metadata.json")
    TEACHABLE_MODEL_WEIGHTS = str(_BASE_PATH / "static" / "model" / "weights.bin")

    EXTERNAL_MODEL_PATH = _EXTERNAL_MODEL_PATH
    EXTERNAL_LABELS_PATH = (
        str(Path(_EXTERNAL_MODEL_DIR) / _EXTERNAL_LABELS_FILE)
        if _USE_EXTERNAL_MODEL and _EXTERNAL_MODEL_DIR
        else ""
    )
    MODEL_PATH = str(_BASE_PATH / "model" / "khat_best.keras")
    LATEST_MODEL_PATH = str(_BASE_PATH / "model" / "khat_latest.keras")
    BEST_MODEL_PATH = str(_BASE_PATH / "model" / "khat_best.keras")
    LEGACY_BEST_MODEL_PATH = str(_BASE_PATH / "model" / "khat_vgg16_best.keras")

    CLASS_INDICES_PATH = str(_BASE_PATH / "model" / "class_indices.json")
    CLASS_WEIGHTS_PATH = str(_BASE_PATH / "model" / "class_weights.json")
    MODEL_METADATA_PATH = str(_BASE_PATH / "model" / "model_metadata.json")
    TRAINING_HISTORY_PATH = str(_BASE_PATH / "model" / "training_history.json")
    TRAINING_CSV_LOG_PATH = str(_BASE_PATH / "model" / "training_log.csv")
    EVALUATION_RESULT_PATH = str(_BASE_PATH / "model" / "evaluation_result.json")
    DATASET_QUALITY_REPORT_PATH = str(_BASE_PATH / "model" / "dataset_quality_report.json")
    IMAGE_QUALITY_REPORT_PATH = str(_BASE_PATH / "model" / "image_quality_report.json")
    DATASET_SPLIT_REPORT_PATH = str(_BASE_PATH / "model" / "dataset_split_report.json")
    DATASET_BALANCE_REPORT_PATH = str(_BASE_PATH / "model" / "dataset_balance_report.json")
    LABEL_AUDIT_REPORT_PATH = str(_BASE_PATH / "model" / "label_audit_report.json")
    LABEL_MISMATCH_REPORT_PATH = str(_BASE_PATH / "model" / "label_mismatch_report.json")
    DUPLICATE_REPORT_PATH = str(_BASE_PATH / "model" / "duplicate_report.json")
    MISCLASSIFICATION_REPORT_PATH = str(_BASE_PATH / "model" / "misclassification_report.json")
    PREDICTION_AUDIT_REPORT_PATH = str(_BASE_PATH / "model" / "prediction_audit_report.json")
    CONFUSION_PAIR_REPORT_PATH = str(_BASE_PATH / "model" / "confusion_pair_report.json")
    BEST_MODEL_METADATA_PATH = str(_BASE_PATH / "model" / "best_model_metadata.json")
    CLASSIFICATION_REPORT_CSV_PATH = str(_BASE_PATH / "static" / "evaluation" / "classification_report.csv")
    KHAT_DETECTOR_PATH = str(_BASE_PATH / "model" / "khat_detector.keras")
    KHAT_DETECTOR_METADATA_PATH = str(_BASE_PATH / "model" / "khat_detector_metadata.json")
    KHAT_DETECTOR_INDICES_PATH = str(_BASE_PATH / "model" / "khat_detector_indices.json")
    NON_KHAT_DETECTION_REPORT_PATH = str(_BASE_PATH / "model" / "non_khat_detection_report.json")
    KHAT_DETECTOR_BALANCE_REPORT_PATH = str(_BASE_PATH / "model" / "khat_detector_balance_report.json")
    KHAT_DETECTOR_EVALUATION_PATH = str(_BASE_PATH / "model" / "khat_detector_evaluation.json")
    KHAT_DETECTOR_THRESHOLD_REPORT_PATH = str(_BASE_PATH / "model" / "khat_detector_threshold_report.json")
    KHAT_DETECTOR_SETTINGS_PATH = str(_BASE_PATH / "model" / "khat_detector_settings.json")
    KHAT_ACCEPT_THRESHOLD = float(os.getenv("KHAT_ACCEPT_THRESHOLD", os.getenv("KHAT_DETECTION_THRESHOLD", "0.70")))
    KHAT_REJECT_THRESHOLD = float(os.getenv("KHAT_REJECT_THRESHOLD", "0.50"))
    KHAT_BORDERLINE_THRESHOLD = float(os.getenv("KHAT_BORDERLINE_THRESHOLD", "0.65"))
    KHAT_HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("KHAT_HIGH_CONFIDENCE_THRESHOLD", "0.85"))
    KHAT_DETECTION_THRESHOLD = KHAT_ACCEPT_THRESHOLD
    KHAT_DETECTION_UNCERTAIN_MIN = KHAT_REJECT_THRESHOLD
    # Stage 2: reject images that do not clearly match one of the 4 trained classes.
    STAGE2_MIN_CONFIDENCE = float(os.getenv("STAGE2_MIN_CONFIDENCE", "0.70"))
    STAGE2_MIN_MARGIN = float(os.getenv("STAGE2_MIN_MARGIN", "0.05"))
    STAGE2_MIN_CONFIDENCE_AFTER_STAGE1 = float(
        os.getenv("STAGE2_MIN_CONFIDENCE_AFTER_STAGE1", "0.32")
    )
    STAGE2_MIN_MARGIN_AFTER_STAGE1 = float(
        os.getenv("STAGE2_MIN_MARGIN_AFTER_STAGE1", "0.10")
    )    # Minimum visual similarity (%) to training images for acceptance.
    STAGE2_PROBABLE_CONFIDENCE = float(os.getenv("STAGE2_PROBABLE_CONFIDENCE", "0.55"))
    STAGE2_PROBABLE_MARGIN = float(os.getenv("STAGE2_PROBABLE_MARGIN", "0.10"))
    STAGE2_MIN_DATASET_SIMILARITY = float(os.getenv("STAGE2_MIN_DATASET_SIMILARITY", "55"))
    STAGE2_MIN_CLASS_SIMILARITY = float(os.getenv("STAGE2_MIN_CLASS_SIMILARITY", "60"))
    # Soften displayed/inference probabilities toward visual similarity across classes.
    PROB_SOFTMAX_TEMPERATURE = float(os.getenv("PROB_SOFTMAX_TEMPERATURE", "2.5"))
    PROB_SIMILARITY_BLEND = float(os.getenv("PROB_SIMILARITY_BLEND", "0.35"))
    PROB_MIN_FLOOR = float(os.getenv("PROB_MIN_FLOOR", "0.03"))
    # After Stage 1 confirms calligraphy, only reject if similarity is near-zero (OOD leak).
    STAGE2_OOD_SIMILARITY_FLOOR = float(os.getenv("STAGE2_OOD_SIMILARITY_FLOOR", "22"))
    DATASET_ALIGN_STRONG_SIMILARITY = float(os.getenv("DATASET_ALIGN_STRONG_SIMILARITY", "75"))
    DATASET_ALIGN_MODERATE_SIMILARITY = float(os.getenv("DATASET_ALIGN_MODERATE_SIMILARITY", "55"))
    DATASET_ALIGN_OVERRIDE_GAP = float(os.getenv("DATASET_ALIGN_OVERRIDE_GAP", "12"))
    USE_MODEL_ENSEMBLE = os.getenv("USE_MODEL_ENSEMBLE", "1").strip().lower() in ("1", "true", "yes")
    ENSEMBLE_LOCAL_WEIGHT = float(os.getenv("ENSEMBLE_LOCAL_WEIGHT", "0.85"))
    ENSEMBLE_EXTERNAL_WEIGHT = float(os.getenv("ENSEMBLE_EXTERNAL_WEIGHT", "0.15"))
    ENSEMBLE_SINGLE_MIN_CONFIDENCE = float(os.getenv("ENSEMBLE_SINGLE_MIN_CONFIDENCE", "0.72"))
    ENSEMBLE_SINGLE_MIN_MARGIN = float(os.getenv("ENSEMBLE_SINGLE_MIN_MARGIN", "0.18"))
    ENSEMBLE_GATE_MIN_CONFIDENCE = float(os.getenv("ENSEMBLE_GATE_MIN_CONFIDENCE", "42"))
    ENSEMBLE_GATE_MIN_MARGIN = float(os.getenv("ENSEMBLE_GATE_MIN_MARGIN", "12"))
    FUSION_MODEL_WEIGHT = float(os.getenv("FUSION_MODEL_WEIGHT", "0.72"))
    FUSION_UNCERTAIN_MODEL_WEIGHT = float(os.getenv("FUSION_UNCERTAIN_MODEL_WEIGHT", "0.45"))
    FUSION_STRONG_MODEL_WEIGHT = float(os.getenv("FUSION_STRONG_MODEL_WEIGHT", "0.88"))
    FUSION_UNCERTAIN_CONFIDENCE = float(os.getenv("FUSION_UNCERTAIN_CONFIDENCE", "0.58"))
    FUSION_UNCERTAIN_MARGIN = float(os.getenv("FUSION_UNCERTAIN_MARGIN", "0.12"))
    # Keep clear CNN top class; ornate borders confuse pHash toward Diwani Jali.
    FUSION_KEEP_MODEL_MIN_CONFIDENCE = float(os.getenv("FUSION_KEEP_MODEL_MIN_CONFIDENCE", "0.38"))
    FUSION_KEEP_MODEL_MIN_MARGIN = float(os.getenv("FUSION_KEEP_MODEL_MIN_MARGIN", "0.08"))
    KHAT_DETECTOR_MIN_SAMPLES = 500
    ACCURACY_TARGET = 0.85

    CANONICAL_CLASS_ORDER = {
        "naskhi": 0,
        "diwani": 1,
        "diwani_jali": 2,
        "tsuluts": 3,
    }
    CLASS_DISPLAY_NAMES = {
        "naskhi": "Khat Naskhi",
        "diwani": "Khat Diwani",
        "diwani_jali": "Khat Diwani Jali",
        "tsuluts": "Khat Tsuluts",
    }

    DATASET_DIR = _DATASET_PATHS["DATASET_DIR"]
    RAW_DATASET_DIR = _DATASET_PATHS["RAW_DATASET_DIR"]
    NON_KHAT_DIR = _DATASET_PATHS["NON_KHAT_DIR"]
    KHAT_DETECTOR_DATASET_DIR = _DATASET_PATHS["KHAT_DETECTOR_DATASET_DIR"]
    KHAT_DETECTOR_TRAIN_DIR = _DATASET_PATHS["KHAT_DETECTOR_TRAIN_DIR"]
    KHAT_DETECTOR_VALIDATION_DIR = _DATASET_PATHS["KHAT_DETECTOR_VALIDATION_DIR"]
    KHAT_DETECTOR_TEST_DIR = _DATASET_PATHS["KHAT_DETECTOR_TEST_DIR"]
    REJECTED_DATASET_DIR = _DATASET_PATHS["REJECTED_DATASET_DIR"]
    DATASET_TMP_DIR = _DATASET_PATHS["DATASET_TMP_DIR"]
    PROCESSED_DATASET_DIR = _DATASET_PATHS["PROCESSED_DATASET_DIR"]
    MODEL_READY_DIR = _DATASET_PATHS["MODEL_READY_DIR"]
    PROCESSED_BALANCED_DIR = _DATASET_PATHS["PROCESSED_BALANCED_DIR"]
    TRAIN_DIR = _DATASET_PATHS["TRAIN_DIR"]
    VALIDATION_DIR = _DATASET_PATHS["VALIDATION_DIR"]
    TEST_DIR = _DATASET_PATHS["TEST_DIR"]
    OPTIMIZED_DIR = _DATASET_PATHS["OPTIMIZED_DIR"]
    OPTIMIZED_DISPLAY_DIR = _DATASET_PATHS["OPTIMIZED_DISPLAY_DIR"]
    OPTIMIZED_MODEL_DIR = _DATASET_PATHS["OPTIMIZED_MODEL_DIR"]
    IMAGE_OPTIMIZATION_REPORT_PATH = str(_BASE_PATH / "model" / "image_optimization_report.json")
    IMAGE_OPTIMIZATION_STATUS_PATH = str(_BASE_PATH / "model" / "image_optimization_status.json")
    DATASET_UPLOAD_SETTINGS_PATH = str(_BASE_PATH / "model" / "dataset_upload_settings.json")
    DATASET_UPLOAD_REPORT_PATH = str(_BASE_PATH / "model" / "dataset_upload_report.json")
    DATASET_HASH_INDEX_PATH = str(_BASE_PATH / "model" / "dataset_hash_index.json")

    CLASS_LABELS = ["naskhi", "diwani", "diwani_jali", "tsuluts"]
    CLASSIFICATION_ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

    # Smaller default batch reduces OOM on Windows/XAMPP hosts during eval.
    EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "32"))
    EVAL_USE_TTA = os.getenv("EVAL_USE_TTA", "1").strip().lower() in ("1", "true", "yes", "on")
    EVAL_USE_ENSEMBLE = os.getenv("EVAL_USE_ENSEMBLE", "1").strip().lower() in ("1", "true", "yes", "on")
    EVAL_DEFER_PREVIEWS = os.getenv("EVAL_DEFER_PREVIEWS", "1").strip().lower() in ("1", "true", "yes", "on")
    EVAL_SKIP_CALIBRATION_ON_RUN = os.getenv("EVAL_SKIP_CALIBRATION_ON_RUN", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )
    EVAL_PLOT_DPI = int(os.getenv("EVAL_PLOT_DPI", "100"))
    EVAL_FAST_MODE = os.getenv("EVAL_FAST_MODE", "1").strip().lower() in ("1", "true", "yes", "on")
    EVAL_USE_INFERENCE_PIPELINE = os.getenv("EVAL_USE_INFERENCE_PIPELINE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # Match live classification ensemble weights (external 0.15 / local 0.85).
    EVAL_ENSEMBLE_EXTERNAL_WEIGHT = float(
        os.getenv("EVAL_ENSEMBLE_EXTERNAL_WEIGHT", os.getenv("ENSEMBLE_EXTERNAL_WEIGHT", "0.15"))
    )
    EVAL_ENSEMBLE_LOCAL_WEIGHT = float(
        os.getenv("EVAL_ENSEMBLE_LOCAL_WEIGHT", os.getenv("ENSEMBLE_LOCAL_WEIGHT", "0.85"))
    )
    PRESERVE_DATASET_IMAGES = os.getenv("PRESERVE_DATASET_IMAGES", "1").strip().lower() in ("1", "true", "yes", "on")
    MODEL_PAGE_CACHE_TTL = int(os.getenv("MODEL_PAGE_CACHE_TTL", "30"))
