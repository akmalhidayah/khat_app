from .database import db
from .user import Pengguna, User
from .khat_class import KelasKhat, KhatClass
from .dataset_image import GambarDataset, Dataset, DatasetImage
from .model_version import VersiModel, ModelVersion
from .classification_history import RiwayatKlasifikasi, ClassificationHistory, ClassificationResult
from .prediction_probability import ProbabilitasPrediksi, PredictionProbability
from .algorithm_calculation import PerhitunganAlgoritma, AlgorithmCalculation
from .evaluasi_model import (
    EvaluasiModel,
    EvaluationResult,
    EvaluationSession,
    HasilEvaluasi,
    ModelEvaluation,
    SesiEvaluasi,
    TIPE_DETAIL,
    TIPE_RINGKASAN,
    buat_ringkasan_evaluasi,
)
from .correction_log import LogKoreksi, CorrectionLog

CORE_TABLES = [
    "pengguna",
    "kelas_khat",
    "gambar_dataset",
    "versi_model",
    "riwayat_klasifikasi",
    "probabilitas_prediksi",
    "perhitungan_algoritma",
    "evaluasi_model",
    "log_koreksi",
]

__all__ = [
    "db",
    "CORE_TABLES",
    "Pengguna",
    "User",
    "KelasKhat",
    "KhatClass",
    "GambarDataset",
    "Dataset",
    "DatasetImage",
    "VersiModel",
    "ModelVersion",
    "RiwayatKlasifikasi",
    "ClassificationHistory",
    "ClassificationResult",
    "ProbabilitasPrediksi",
    "PredictionProbability",
    "PerhitunganAlgoritma",
    "AlgorithmCalculation",
    "EvaluasiModel",
    "ModelEvaluation",
    "EvaluationSession",
    "EvaluationResult",
    "SesiEvaluasi",
    "HasilEvaluasi",
    "TIPE_RINGKASAN",
    "TIPE_DETAIL",
    "buat_ringkasan_evaluasi",
    "LogKoreksi",
    "CorrectionLog",
]
