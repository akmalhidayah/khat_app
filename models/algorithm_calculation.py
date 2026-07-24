"""Algorithm calculation metadata (perhitungan_algoritma)."""

from datetime import datetime

from .database import db


class PerhitunganAlgoritma(db.Model):
    __tablename__ = "perhitungan_algoritma"

    id = db.Column(db.Integer, primary_key=True)
    id_riwayat = db.Column(
        db.Integer,
        db.ForeignKey("riwayat_klasifikasi.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    lebar_asli = db.Column(db.Integer, nullable=True)
    tinggi_asli = db.Column(db.Integer, nullable=True)
    lebar_proses = db.Column(db.Integer, nullable=True)
    tinggi_proses = db.Column(db.Integer, nullable=True)
    ukuran_input_model = db.Column(db.String(20), nullable=True)
    bentuk_tensor = db.Column(db.String(50), nullable=True)
    langkah_preprocessing_json = db.Column(db.Text, nullable=True)
    probabilitas_khat = db.Column(db.Float, nullable=True)
    probabilitas_non_khat = db.Column(db.Float, nullable=True)
    keputusan_tahap1 = db.Column(db.String(100), nullable=True)
    level_confidence = db.Column(db.String(100), nullable=True)
    status_kelas_dikenal = db.Column(db.String(100), nullable=True)
    rumus_json = db.Column(db.Text, nullable=True)
    catatan_perhitungan = db.Column(db.Text, nullable=True)
    dibuat_pada = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    riwayat = db.relationship("RiwayatKlasifikasi", back_populates="perhitungan")

    @property
    def classification_id(self):
        return self.id_riwayat

    @classification_id.setter
    def classification_id(self, value):
        self.id_riwayat = value

    @property
    def original_width(self):
        return self.lebar_asli

    @original_width.setter
    def original_width(self, value):
        self.lebar_asli = value

    @property
    def original_height(self):
        return self.tinggi_asli

    @original_height.setter
    def original_height(self, value):
        self.tinggi_asli = value

    @property
    def processed_width(self):
        return self.lebar_proses

    @processed_width.setter
    def processed_width(self, value):
        self.lebar_proses = value

    @property
    def processed_height(self):
        return self.tinggi_proses

    @processed_height.setter
    def processed_height(self, value):
        self.tinggi_proses = value

    @property
    def model_input_size(self):
        return self.ukuran_input_model

    @model_input_size.setter
    def model_input_size(self, value):
        self.ukuran_input_model = value

    @property
    def tensor_shape(self):
        return self.bentuk_tensor

    @tensor_shape.setter
    def tensor_shape(self, value):
        self.bentuk_tensor = value

    @property
    def preprocessing_steps_json(self):
        return self.langkah_preprocessing_json

    @preprocessing_steps_json.setter
    def preprocessing_steps_json(self, value):
        self.langkah_preprocessing_json = value

    @property
    def khat_probability(self):
        return self.probabilitas_khat

    @khat_probability.setter
    def khat_probability(self, value):
        self.probabilitas_khat = value

    @property
    def non_khat_probability(self):
        return self.probabilitas_non_khat

    @non_khat_probability.setter
    def non_khat_probability(self, value):
        self.probabilitas_non_khat = value

    @property
    def stage1_decision(self):
        return self.keputusan_tahap1

    @stage1_decision.setter
    def stage1_decision(self, value):
        self.keputusan_tahap1 = value

    @property
    def confidence_level(self):
        return self.level_confidence

    @confidence_level.setter
    def confidence_level(self, value):
        self.level_confidence = value

    @property
    def known_class_status(self):
        return self.status_kelas_dikenal

    @known_class_status.setter
    def known_class_status(self, value):
        self.status_kelas_dikenal = value

    @property
    def formula_summary_json(self):
        return self.rumus_json

    @formula_summary_json.setter
    def formula_summary_json(self, value):
        self.rumus_json = value

    @property
    def calculation_notes(self):
        return self.catatan_perhitungan

    @calculation_notes.setter
    def calculation_notes(self, value):
        self.catatan_perhitungan = value

    @property
    def created_at(self):
        return self.dibuat_pada

    @property
    def classification(self):
        return self.riwayat


AlgorithmCalculation = PerhitunganAlgoritma
