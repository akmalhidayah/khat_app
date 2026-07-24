"""Classification history (riwayat_klasifikasi) — 3NF core + JSON for legacy UI fields."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

from .database import db


class RiwayatKlasifikasi(db.Model):
    __tablename__ = "riwayat_klasifikasi"

    id = db.Column(db.Integer, primary_key=True)
    id_pengguna = db.Column(db.Integer, db.ForeignKey("pengguna.id"), nullable=True, index=True)
    id_model = db.Column(db.Integer, db.ForeignKey("versi_model.id"), nullable=True, index=True)
    id_gambar_dataset = db.Column(db.Integer, db.ForeignKey("gambar_dataset.id"), nullable=True, index=True)
    nama_file_upload = db.Column(db.String(255), nullable=True)
    path_upload = db.Column(db.String(512), nullable=True)
    id_kelas_diharapkan = db.Column(db.Integer, db.ForeignKey("kelas_khat.id"), nullable=True, index=True)
    id_kelas_prediksi = db.Column(db.Integer, db.ForeignKey("kelas_khat.id"), nullable=True, index=True)
    sumber_kelas_diharapkan = db.Column(db.String(50), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    id_kelas_top2 = db.Column(db.Integer, db.ForeignKey("kelas_khat.id"), nullable=True, index=True)
    skor_top2 = db.Column(db.Float, nullable=True)
    margin_top2 = db.Column(db.Float, nullable=True)
    status_validasi = db.Column(db.String(100), nullable=True)
    keputusan_akhir = db.Column(db.String(100), nullable=True)
    sumber_input = db.Column(db.String(50), nullable=True, default="upload")
    id_legacy = db.Column(db.Integer, nullable=True, unique=True, index=True)
    data_ekstra_json = db.Column(db.Text, nullable=True)
    dibuat_pada = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    pengguna = db.relationship("Pengguna", backref=db.backref("riwayat_klasifikasi", lazy="dynamic"))
    model = db.relationship("VersiModel", backref=db.backref("riwayat_klasifikasi", lazy="dynamic"))
    gambar = db.relationship("GambarDataset", back_populates="riwayat_klasifikasi")
    kelas_diharapkan = db.relationship("KelasKhat", foreign_keys=[id_kelas_diharapkan])
    kelas_prediksi = db.relationship(
        "KelasKhat", foreign_keys=[id_kelas_prediksi], back_populates="prediksi_riwayat"
    )
    kelas_top2 = db.relationship("KelasKhat", foreign_keys=[id_kelas_top2])
    probabilitas = db.relationship(
        "ProbabilitasPrediksi",
        back_populates="riwayat",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    perhitungan = db.relationship(
        "PerhitunganAlgoritma",
        back_populates="riwayat",
        uselist=False,
        cascade="all, delete-orphan",
    )
    log_koreksi = db.relationship("LogKoreksi", back_populates="riwayat", lazy="dynamic")

    _COLUMN_ALIASES = {
        "filename": "nama_file_upload",
        "uploaded_filename": "nama_file_upload",
        "image_path": "path_upload",
        "confidence_score": "confidence",
        "validation_status": "status_validasi",
        "final_decision": "keputusan_akhir",
        "input_source": "sumber_input",
        "expected_class_source": "sumber_kelas_diharapkan",
        "top_2_score": "skor_top2",
        "top2_margin": "margin_top2",
        "created_at": "dibuat_pada",
        "legacy_result_id": "id_legacy",
        "user_id": "id_pengguna",
        "model_version_id": "id_model",
        "dataset_image_id": "id_gambar_dataset",
    }

    _EXTRA_FIELDS = {
        "naskhi_score", "diwani_score", "diwani_jali_score", "tsuluts_score",
        "probability_naskhi", "probability_diwani", "probability_diwani_jali", "probability_tsuluts",
        "input_status", "is_khat", "rejection_reason", "reliability_level", "review_status",
        "confidence_label", "explanation_text", "model_name", "model_source", "model_runtime",
        "model_input_size", "similarity_status", "similarity_score", "nearest_dataset_image",
        "nearest_dataset_class", "similarity_risk_level", "similarity_message",
        "similarity_recommendation", "manual_expected_class", "correction_label", "correction_notes",
        "source_type", "preprocessing_mode", "detection_status", "detection_decision",
        "softmax_scores_json", "preprocessing_steps_json", "calculation_notes",
        "khat_probability", "non_khat_probability", "known_class_status",
        "original_width", "original_height", "processed_width", "processed_height",
        "manual_review_required", "top_2_class", "predicted_class", "expected_class",
        "stage1_decision", "confidence_level", "top1_class", "top1_score",
    }

    def __init__(self, **kwargs):
        extra = {}
        class_fields = {}
        for key, value in list(kwargs.items()):
            if key in ("predicted_class", "expected_class", "top_2_class", "manual_expected_class"):
                class_fields[key] = value
                kwargs.pop(key)
            elif key in self._EXTRA_FIELDS:
                extra[key] = value
                kwargs.pop(key)
            elif key in self._COLUMN_ALIASES:
                kwargs[self._COLUMN_ALIASES[key]] = kwargs.pop(key)
        if extra:
            kwargs["data_ekstra_json"] = json.dumps(extra, ensure_ascii=False)
        super().__init__(**kwargs)
        for key, value in class_fields.items():
            setattr(self, key, value)

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

    def _get_extra(self, key: str, default=None):
        return self._extra().get(key, default)

    @property
    def filename(self):
        return self.nama_file_upload or self._get_extra("filename", "")

    @filename.setter
    def filename(self, value):
        self.nama_file_upload = value

    @property
    def uploaded_filename(self):
        return self.nama_file_upload

    @uploaded_filename.setter
    def uploaded_filename(self, value):
        self.nama_file_upload = value

    @property
    def image_path(self):
        return self.path_upload or ""

    @image_path.setter
    def image_path(self, value):
        self.path_upload = value

    @property
    def predicted_class(self):
        if self.kelas_prediksi:
            return self.kelas_prediksi.slug
        return self._get_extra("predicted_class")

    @predicted_class.setter
    def predicted_class(self, value):
        self._set_extra("predicted_class", value)
        if value:
            from services.db_compat import get_class_id_by_slug
            self.id_kelas_prediksi = get_class_id_by_slug(value)

    @property
    def expected_class(self):
        if self.kelas_diharapkan:
            return self.kelas_diharapkan.slug
        return self._get_extra("expected_class")

    @expected_class.setter
    def expected_class(self, value):
        self._set_extra("expected_class", value)
        if value:
            from services.db_compat import get_class_id_by_slug
            self.id_kelas_diharapkan = get_class_id_by_slug(value)

    @property
    def confidence_score(self):
        return self.confidence

    @confidence_score.setter
    def confidence_score(self, value):
        self.confidence = value

    @property
    def validation_status(self):
        return self.status_validasi

    @validation_status.setter
    def validation_status(self, value):
        self.status_validasi = value

    @property
    def final_decision(self):
        return self.keputusan_akhir

    @final_decision.setter
    def final_decision(self, value):
        self.keputusan_akhir = value

    @property
    def input_source(self):
        return self.sumber_input or "upload"

    @input_source.setter
    def input_source(self, value):
        self.sumber_input = value

    @property
    def expected_class_source(self):
        return self.sumber_kelas_diharapkan or self._get_extra("expected_class_source")

    @expected_class_source.setter
    def expected_class_source(self, value):
        self.sumber_kelas_diharapkan = value

    @property
    def top_2_class(self):
        if self.kelas_top2:
            return self.kelas_top2.slug
        return self._get_extra("top_2_class")

    @top_2_class.setter
    def top_2_class(self, value):
        self._set_extra("top_2_class", value)
        if value:
            from services.db_compat import get_class_id_by_slug
            self.id_kelas_top2 = get_class_id_by_slug(value)

    @property
    def top_2_score(self):
        return self.skor_top2

    @top_2_score.setter
    def top_2_score(self, value):
        self.skor_top2 = value

    @property
    def top2_margin(self):
        return self.margin_top2

    @top2_margin.setter
    def top2_margin(self, value):
        self.margin_top2 = value

    @property
    def created_at(self):
        return self.dibuat_pada

    @created_at.setter
    def created_at(self, value):
        self.dibuat_pada = value

    @property
    def legacy_result_id(self):
        return self.id_legacy

    @legacy_result_id.setter
    def legacy_result_id(self, value):
        self.id_legacy = value

    @property
    def user_id(self):
        return self.id_pengguna

    @user_id.setter
    def user_id(self, value):
        self.id_pengguna = value

    @property
    def model_version_id(self):
        return self.id_model

    @model_version_id.setter
    def model_version_id(self, value):
        self.id_model = value

    @property
    def dataset_image_id(self):
        return self.id_gambar_dataset

    @dataset_image_id.setter
    def dataset_image_id(self, value):
        self.id_gambar_dataset = value

    @property
    def expected_class_id(self):
        return self.id_kelas_diharapkan

    @expected_class_id.setter
    def expected_class_id(self, value):
        self.id_kelas_diharapkan = value

    @property
    def predicted_class_id(self):
        return self.id_kelas_prediksi

    @predicted_class_id.setter
    def predicted_class_id(self, value):
        self.id_kelas_prediksi = value

    @property
    def top2_class_id(self):
        return self.id_kelas_top2

    @top2_class_id.setter
    def top2_class_id(self, value):
        self.id_kelas_top2 = value

    def _score_for_slug(self, slug: str) -> float:
        from .khat_class import KelasKhat
        for prob in self.probabilitas.all():
            if prob.kelas and prob.kelas.slug == slug:
                return float(prob.skor_softmax or 0)
        return float(self._get_extra(f"{slug}_score", 0) or 0)

    @property
    def naskhi_score(self):
        return self._score_for_slug("naskhi")

    @naskhi_score.setter
    def naskhi_score(self, value):
        self._set_extra("naskhi_score", value)

    @property
    def diwani_score(self):
        return self._score_for_slug("diwani")

    @diwani_score.setter
    def diwani_score(self, value):
        self._set_extra("diwani_score", value)

    @property
    def diwani_jali_score(self):
        return self._score_for_slug("diwani_jali")

    @diwani_jali_score.setter
    def diwani_jali_score(self, value):
        self._set_extra("diwani_jali_score", value)

    @property
    def tsuluts_score(self):
        return self._score_for_slug("tsuluts")

    @tsuluts_score.setter
    def tsuluts_score(self, value):
        self._set_extra("tsuluts_score", value)

    @property
    def probability_naskhi(self):
        return self.naskhi_score

    @probability_naskhi.setter
    def probability_naskhi(self, value):
        self.naskhi_score = value

    @property
    def probability_diwani(self):
        return self.diwani_score

    @property
    def probability_diwani_jali(self):
        return self.diwani_jali_score

    @property
    def probability_tsuluts(self):
        return self.tsuluts_score

    @property
    def khat_probability(self):
        calc = self.perhitungan
        if calc and calc.probabilitas_khat is not None:
            return calc.probabilitas_khat
        return self._get_extra("khat_probability")

    @khat_probability.setter
    def khat_probability(self, value):
        self._set_extra("khat_probability", value)

    @property
    def non_khat_probability(self):
        calc = self.perhitungan
        if calc and calc.probabilitas_non_khat is not None:
            return calc.probabilitas_non_khat
        return self._get_extra("non_khat_probability")

    @non_khat_probability.setter
    def non_khat_probability(self, value):
        self._set_extra("non_khat_probability", value)

    @property
    def input_status(self):
        return self._get_extra("input_status", "khat")

    @input_status.setter
    def input_status(self, value):
        self._set_extra("input_status", value)

    @property
    def is_khat(self):
        return self._get_extra("is_khat", True)

    @is_khat.setter
    def is_khat(self, value):
        self._set_extra("is_khat", value)

    @property
    def review_status(self):
        return self._get_extra("review_status")

    @review_status.setter
    def review_status(self, value):
        self._set_extra("review_status", value)

    @property
    def confidence_label(self):
        calc = self.perhitungan
        if calc and calc.level_confidence:
            return calc.level_confidence
        return self._get_extra("confidence_label")

    @confidence_label.setter
    def confidence_label(self, value):
        self._set_extra("confidence_label", value)

    @property
    def explanation_text(self):
        return self._get_extra("explanation_text")

    @explanation_text.setter
    def explanation_text(self, value):
        self._set_extra("explanation_text", value)

    @property
    def model_name(self):
        return self.model.nama_model if self.model else self._get_extra("model_name")

    @model_name.setter
    def model_name(self, value):
        self._set_extra("model_name", value)

    @property
    def model_source(self):
        return self.model.sumber_model if self.model else self._get_extra("model_source")

    @model_source.setter
    def model_source(self, value):
        self._set_extra("model_source", value)

    @property
    def model_runtime(self):
        return self.model.runtime if self.model else self._get_extra("model_runtime")

    @model_runtime.setter
    def model_runtime(self, value):
        self._set_extra("model_runtime", value)

    @property
    def model_input_size(self):
        calc = self.perhitungan
        if calc and calc.ukuran_input_model:
            return calc.ukuran_input_model
        return self._get_extra("model_input_size")

    @model_input_size.setter
    def model_input_size(self, value):
        self._set_extra("model_input_size", value)

    @property
    def manual_review_required(self):
        return self._get_extra("manual_review_required", False)

    @manual_review_required.setter
    def manual_review_required(self, value):
        self._set_extra("manual_review_required", value)

    @property
    def similarity_status(self):
        return self._get_extra("similarity_status")

    @similarity_status.setter
    def similarity_status(self, value):
        self._set_extra("similarity_status", value)

    @property
    def similarity_score(self):
        return self._get_extra("similarity_score")

    @similarity_score.setter
    def similarity_score(self, value):
        self._set_extra("similarity_score", value)

    @property
    def nearest_dataset_image(self):
        return self._get_extra("nearest_dataset_image")

    @nearest_dataset_image.setter
    def nearest_dataset_image(self, value):
        self._set_extra("nearest_dataset_image", value)

    @property
    def nearest_dataset_class(self):
        return self._get_extra("nearest_dataset_class")

    @nearest_dataset_class.setter
    def nearest_dataset_class(self, value):
        self._set_extra("nearest_dataset_class", value)

    @property
    def similarity_risk_level(self):
        return self._get_extra("similarity_risk_level")

    @similarity_risk_level.setter
    def similarity_risk_level(self, value):
        self._set_extra("similarity_risk_level", value)

    @property
    def similarity_message(self):
        return self._get_extra("similarity_message")

    @similarity_message.setter
    def similarity_message(self, value):
        self._set_extra("similarity_message", value)

    @property
    def similarity_recommendation(self):
        return self._get_extra("similarity_recommendation")

    @similarity_recommendation.setter
    def similarity_recommendation(self, value):
        self._set_extra("similarity_recommendation", value)

    @property
    def manual_expected_class(self):
        return self._get_extra("manual_expected_class")

    @manual_expected_class.setter
    def manual_expected_class(self, value):
        self._set_extra("manual_expected_class", value)

    @property
    def correction_label(self):
        return self._get_extra("correction_label")

    @correction_label.setter
    def correction_label(self, value):
        self._set_extra("correction_label", value)

    @property
    def correction_notes(self):
        return self._get_extra("correction_notes")

    @correction_notes.setter
    def correction_notes(self, value):
        self._set_extra("correction_notes", value)

    @property
    def source_type(self):
        return self._get_extra("source_type")

    @source_type.setter
    def source_type(self, value):
        self._set_extra("source_type", value)

    @property
    def preprocessing_mode(self):
        return self._get_extra("preprocessing_mode")

    @preprocessing_mode.setter
    def preprocessing_mode(self, value):
        self._set_extra("preprocessing_mode", value)

    @property
    def known_class_status(self):
        calc = self.perhitungan
        if calc and calc.status_kelas_dikenal:
            return calc.status_kelas_dikenal
        return self._get_extra("known_class_status")

    @known_class_status.setter
    def known_class_status(self, value):
        self._set_extra("known_class_status", value)

    @property
    def original_width(self):
        calc = self.perhitungan
        return calc.lebar_asli if calc else self._get_extra("original_width")

    @original_width.setter
    def original_width(self, value):
        self._set_extra("original_width", value)

    @property
    def original_height(self):
        calc = self.perhitungan
        return calc.tinggi_asli if calc else self._get_extra("original_height")

    @original_height.setter
    def original_height(self, value):
        self._set_extra("original_height", value)

    @property
    def processed_width(self):
        calc = self.perhitungan
        return calc.lebar_proses if calc else self._get_extra("processed_width")

    @processed_width.setter
    def processed_width(self, value):
        self._set_extra("processed_width", value)

    @property
    def processed_height(self):
        calc = self.perhitungan
        return calc.tinggi_proses if calc else self._get_extra("processed_height")

    @processed_height.setter
    def processed_height(self, value):
        self._set_extra("processed_height", value)

    @property
    def detection_status(self):
        return self._get_extra("detection_status")

    @detection_status.setter
    def detection_status(self, value):
        self._set_extra("detection_status", value)

    @property
    def detection_decision(self):
        calc = self.perhitungan
        if calc and calc.keputusan_tahap1:
            return calc.keputusan_tahap1
        return self._get_extra("detection_decision")

    @detection_decision.setter
    def detection_decision(self, value):
        self._set_extra("detection_decision", value)

    @property
    def stage1_decision(self):
        return self.detection_decision

    @stage1_decision.setter
    def stage1_decision(self, value):
        self.detection_decision = value

    @property
    def rejection_reason(self):
        return self._get_extra("rejection_reason")

    @rejection_reason.setter
    def rejection_reason(self, value):
        self._set_extra("rejection_reason", value)

    @property
    def reliability_level(self):
        return self._get_extra("reliability_level")

    @reliability_level.setter
    def reliability_level(self, value):
        self._set_extra("reliability_level", value)

    @property
    def top1_class(self):
        return self.predicted_class

    @property
    def top1_score(self):
        return self.confidence

    @property
    def confidence_level(self):
        return self.confidence_label

    @confidence_level.setter
    def confidence_level(self, value):
        self.confidence_label = value

    @property
    def softmax_scores_json(self):
        return self._get_extra("softmax_scores_json")

    @softmax_scores_json.setter
    def softmax_scores_json(self, value):
        self._set_extra("softmax_scores_json", value)

    @property
    def preprocessing_steps_json(self):
        calc = self.perhitungan
        if calc:
            return calc.langkah_preprocessing_json
        return self._get_extra("preprocessing_steps_json")

    @preprocessing_steps_json.setter
    def preprocessing_steps_json(self, value):
        self._set_extra("preprocessing_steps_json", value)

    @property
    def calculation_notes(self):
        calc = self.perhitungan
        if calc:
            return calc.catatan_perhitungan
        return self._get_extra("calculation_notes")

    @calculation_notes.setter
    def calculation_notes(self, value):
        self._set_extra("calculation_notes", value)


ClassificationHistory = RiwayatKlasifikasi
ClassificationResult = RiwayatKlasifikasi
