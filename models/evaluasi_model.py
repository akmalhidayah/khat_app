"""Unified evaluation table — summary + per-image detail via tipe_record."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

from .database import db

TIPE_RINGKASAN = "summary"
TIPE_DETAIL = "detail"


class EvaluasiModel(db.Model):
    __tablename__ = "evaluasi_model"

    id = db.Column(db.Integer, primary_key=True)
    id_model = db.Column(db.Integer, db.ForeignKey("versi_model.id"), nullable=True, index=True)
    id_gambar_dataset = db.Column(db.Integer, db.ForeignKey("gambar_dataset.id"), nullable=True, index=True)
    id_kelas_asli = db.Column(db.Integer, db.ForeignKey("kelas_khat.id"), nullable=True, index=True)
    id_kelas_prediksi = db.Column(db.Integer, db.ForeignKey("kelas_khat.id"), nullable=True, index=True)
    confidence = db.Column(db.Float, nullable=True)
    id_kelas_top2 = db.Column(db.Integer, db.ForeignKey("kelas_khat.id"), nullable=True, index=True)
    skor_top2 = db.Column(db.Float, nullable=True)
    margin_top2 = db.Column(db.Float, nullable=True)
    benar = db.Column(db.Boolean, nullable=True)
    jenis_error = db.Column(db.String(50), nullable=True)
    rekomendasi = db.Column(db.Text, nullable=True)
    url_gambar = db.Column(db.String(512), nullable=True)
    total_gambar = db.Column(db.Integer, nullable=True)
    prediksi_benar = db.Column(db.Integer, nullable=True)
    prediksi_salah = db.Column(db.Integer, nullable=True)
    akurasi = db.Column(db.Float, nullable=True)
    presisi = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    tipe_record = db.Column(db.String(20), nullable=False, default=TIPE_DETAIL, index=True)
    id_induk = db.Column(db.Integer, db.ForeignKey("evaluasi_model.id"), nullable=True, index=True)
    id_legacy = db.Column(db.Integer, nullable=True, index=True)
    nama_file_simpan = db.Column(db.String(255), nullable=True)
    data_ekstra_json = db.Column(db.Text, nullable=True)
    dibuat_pada = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    model = db.relationship("VersiModel", backref=db.backref("evaluasi", lazy="dynamic"))
    gambar = db.relationship("GambarDataset", backref=db.backref("evaluasi", lazy="dynamic"))
    kelas_asli = db.relationship("KelasKhat", foreign_keys=[id_kelas_asli])
    kelas_prediksi = db.relationship("KelasKhat", foreign_keys=[id_kelas_prediksi])
    kelas_top2 = db.relationship("KelasKhat", foreign_keys=[id_kelas_top2])
    induk = db.relationship("EvaluasiModel", remote_side=[id], backref=db.backref("detail_rows", lazy="dynamic"))
    log_koreksi = db.relationship("LogKoreksi", back_populates="evaluasi", lazy="dynamic")

    def _extra(self) -> Dict[str, Any]:
        if not self.data_ekstra_json:
            return {}
        try:
            data = json.loads(self.data_ekstra_json)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def _set_extra(self, key: str, value: Any) -> None:
        data = self._extra()
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
        self.data_ekstra_json = json.dumps(data, ensure_ascii=False) if data else None

    @classmethod
    def ringkasan_query(cls):
        return cls.query.filter_by(tipe_record=TIPE_RINGKASAN)

    @classmethod
    def detail_query(cls):
        return cls.query.filter_by(tipe_record=TIPE_DETAIL)

    @property
    def model_version_id(self):
        return self.id_model

    @model_version_id.setter
    def model_version_id(self, value):
        self.id_model = value

    @property
    def total_images(self):
        return self.total_gambar

    @total_images.setter
    def total_images(self, value):
        self.total_gambar = value

    @property
    def correct_predictions(self):
        return self.prediksi_benar

    @correct_predictions.setter
    def correct_predictions(self, value):
        self.prediksi_benar = value

    @property
    def wrong_predictions(self):
        return self.prediksi_salah

    @wrong_predictions.setter
    def wrong_predictions(self, value):
        self.prediksi_salah = value

    @property
    def accuracy(self):
        return self.akurasi

    @accuracy.setter
    def accuracy(self, value):
        self.akurasi = value

    @property
    def precision_score(self):
        return self.presisi

    @precision_score.setter
    def precision_score(self, value):
        self.presisi = value

    @property
    def recall_score(self):
        return self.recall

    @recall_score.setter
    def recall_score(self, value):
        self.recall = value

    @property
    def dataset_image_id(self):
        return self.id_gambar_dataset

    @dataset_image_id.setter
    def dataset_image_id(self, value):
        self.id_gambar_dataset = value

    @property
    def true_class_id(self):
        return self.id_kelas_asli

    @true_class_id.setter
    def true_class_id(self, value):
        self.id_kelas_asli = value

    @property
    def predicted_class_id(self):
        return self.id_kelas_prediksi

    @predicted_class_id.setter
    def predicted_class_id(self, value):
        self.id_kelas_prediksi = value

    @property
    def confidence_score(self):
        return self.confidence

    @confidence_score.setter
    def confidence_score(self, value):
        self.confidence = value

    @property
    def top2_class_id(self):
        return self.id_kelas_top2

    @top2_class_id.setter
    def top2_class_id(self, value):
        self.id_kelas_top2 = value

    @property
    def top2_score(self):
        return self.skor_top2

    @top2_score.setter
    def top2_score(self, value):
        self.skor_top2 = value

    @property
    def top2_margin(self):
        return self.margin_top2

    @property
    def is_correct(self):
        return self.benar

    @is_correct.setter
    def is_correct(self, value):
        self.benar = value

    @property
    def error_type(self):
        return self.jenis_error

    @error_type.setter
    def error_type(self, value):
        self.jenis_error = value

    @property
    def recommendation(self):
        return self.rekomendasi

    @recommendation.setter
    def recommendation(self, value):
        self.rekomendasi = value

    @property
    def image_url(self):
        return self.url_gambar

    @image_url.setter
    def image_url(self, value):
        self.url_gambar = value

    @property
    def stored_filename(self):
        return self.nama_file_simpan

    @stored_filename.setter
    def stored_filename(self, value):
        self.nama_file_simpan = value

    @property
    def evaluation_session_id(self):
        return self.id_induk

    @evaluation_session_id.setter
    def evaluation_session_id(self, value):
        self.id_induk = value

    @property
    def legacy_evaluation_id(self):
        return self.id_legacy

    @legacy_evaluation_id.setter
    def legacy_evaluation_id(self, value):
        self.id_legacy = value

    @property
    def created_at(self):
        return self.dibuat_pada

    @created_at.setter
    def created_at(self, value):
        self.dibuat_pada = value

    @property
    def selesai_pada(self):
        return self.dibuat_pada

    @property
    def completed_at(self):
        return self.dibuat_pada

    @property
    def model_name(self):
        if self.model:
            return self.model.nama_model
        return self._extra().get("model_name")

    @model_name.setter
    def model_name(self, value):
        self._set_extra("model_name", value)

    @property
    def confusion_matrix(self):
        return self._extra().get("confusion_matrix", "[]")

    @confusion_matrix.setter
    def confusion_matrix(self, value):
        self._set_extra("confusion_matrix", value if isinstance(value, str) else json.dumps(value))

    @property
    def classification_report(self):
        return self._extra().get("classification_report", "{}")

    @classification_report.setter
    def classification_report(self, value):
        self._set_extra("classification_report", value if isinstance(value, str) else json.dumps(value))

    @property
    def training_accuracy(self):
        return self._extra().get("training_accuracy")

    @training_accuracy.setter
    def training_accuracy(self, value):
        self._set_extra("training_accuracy", value)

    @property
    def validation_accuracy(self):
        return self._extra().get("validation_accuracy")

    @validation_accuracy.setter
    def validation_accuracy(self, value):
        self._set_extra("validation_accuracy", value)

    @property
    def training_loss(self):
        return self._extra().get("training_loss")

    @training_loss.setter
    def training_loss(self, value):
        self._set_extra("training_loss", value)

    @property
    def validation_loss(self):
        return self._extra().get("validation_loss")

    @validation_loss.setter
    def validation_loss(self, value):
        self._set_extra("validation_loss", value)


def buat_ringkasan_evaluasi(**kwargs) -> EvaluasiModel:
    row = EvaluasiModel(tipe_record=TIPE_RINGKASAN)
    mapping = {
        "accuracy": "akurasi",
        "precision_score": "presisi",
        "recall_score": "recall",
        "total_images": "total_gambar",
        "correct_predictions": "prediksi_benar",
        "wrong_predictions": "prediksi_salah",
    }
    for key, value in kwargs.items():
        if key in mapping:
            setattr(row, mapping[key], value)
        elif key in ("confusion_matrix", "classification_report", "model_name", "training_accuracy", "validation_accuracy", "training_loss", "validation_loss"):
            setattr(row, key, value)
        elif hasattr(EvaluasiModel, key) and key not in ("tipe_record",):
            setattr(row, key, value)
    return row


ModelEvaluation = EvaluasiModel
EvaluationSession = EvaluasiModel
EvaluationResult = EvaluasiModel
SesiEvaluasi = EvaluasiModel
HasilEvaluasi = EvaluasiModel
