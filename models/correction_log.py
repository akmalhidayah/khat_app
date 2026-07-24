"""Label correction audit trail (log_koreksi)."""

from datetime import datetime

from .database import db


class LogKoreksi(db.Model):
    __tablename__ = "log_koreksi"

    id = db.Column(db.Integer, primary_key=True)
    id_riwayat = db.Column(
        db.Integer, db.ForeignKey("riwayat_klasifikasi.id"), nullable=True, index=True
    )
    id_evaluasi = db.Column(
        db.Integer, db.ForeignKey("evaluasi_model.id"), nullable=True, index=True
    )
    id_gambar_dataset = db.Column(db.Integer, db.ForeignKey("gambar_dataset.id"), nullable=True, index=True)
    id_kelas_lama = db.Column(db.Integer, db.ForeignKey("kelas_khat.id"), nullable=True, index=True)
    id_kelas_koreksi = db.Column(db.Integer, db.ForeignKey("kelas_khat.id"), nullable=False, index=True)
    catatan_koreksi = db.Column(db.Text, nullable=True)
    status_koreksi = db.Column(db.String(50), nullable=True, default="pending")
    direview_oleh = db.Column(db.Integer, db.ForeignKey("pengguna.id"), nullable=True, index=True)
    direview_pada = db.Column(db.DateTime, nullable=True)
    dibuat_pada = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    riwayat = db.relationship("RiwayatKlasifikasi", back_populates="log_koreksi")
    evaluasi = db.relationship("EvaluasiModel", back_populates="log_koreksi")
    gambar = db.relationship("GambarDataset")
    kelas_lama = db.relationship("KelasKhat", foreign_keys=[id_kelas_lama])
    kelas_koreksi = db.relationship("KelasKhat", foreign_keys=[id_kelas_koreksi])
    reviewer = db.relationship("Pengguna", foreign_keys=[direview_oleh])

    @property
    def classification_id(self):
        return self.id_riwayat

    @classification_id.setter
    def classification_id(self, value):
        self.id_riwayat = value

    @property
    def evaluation_result_id(self):
        return self.id_evaluasi

    @evaluation_result_id.setter
    def evaluation_result_id(self, value):
        self.id_evaluasi = value

    @property
    def id_hasil_evaluasi(self):
        return self.id_evaluasi

    @id_hasil_evaluasi.setter
    def id_hasil_evaluasi(self, value):
        self.id_evaluasi = value

    @property
    def dataset_image_id(self):
        return self.id_gambar_dataset

    @dataset_image_id.setter
    def dataset_image_id(self, value):
        self.id_gambar_dataset = value

    @property
    def old_class_id(self):
        return self.id_kelas_lama

    @old_class_id.setter
    def old_class_id(self, value):
        self.id_kelas_lama = value

    @property
    def corrected_class_id(self):
        return self.id_kelas_koreksi

    @corrected_class_id.setter
    def corrected_class_id(self, value):
        self.id_kelas_koreksi = value

    @property
    def correction_note(self):
        return self.catatan_koreksi

    @correction_note.setter
    def correction_note(self, value):
        self.catatan_koreksi = value

    @property
    def correction_status(self):
        return self.status_koreksi

    @correction_status.setter
    def correction_status(self, value):
        self.status_koreksi = value

    @property
    def reviewed_by(self):
        return self.direview_oleh

    @reviewed_by.setter
    def reviewed_by(self, value):
        self.direview_oleh = value

    @property
    def reviewed_at(self):
        return self.direview_pada

    @reviewed_at.setter
    def reviewed_at(self, value):
        self.direview_pada = value

    @property
    def created_at(self):
        return self.dibuat_pada

    @property
    def classification(self):
        return self.riwayat

    @property
    def evaluation_result(self):
        return self.evaluasi

    @property
    def dataset_image(self):
        return self.gambar

    @property
    def old_class(self):
        return self.kelas_lama

    @property
    def corrected_class(self):
        return self.kelas_koreksi


CorrectionLog = LogKoreksi
