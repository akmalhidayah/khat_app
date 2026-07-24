"""Master table for Khat style classes (kelas_khat)."""

from datetime import datetime

from .database import db


class KelasKhat(db.Model):
    __tablename__ = "kelas_khat"

    id = db.Column(db.Integer, primary_key=True)
    nama_kelas = db.Column(db.String(100), nullable=False)
    slug = db.Column(db.String(50), unique=True, nullable=False, index=True)
    deskripsi = db.Column(db.Text, nullable=True)
    ciri_utama = db.Column(db.Text, nullable=True)
    dibuat_pada = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    diperbarui_pada = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    gambar_dataset = db.relationship("GambarDataset", back_populates="kelas", lazy="dynamic")
    prediksi_riwayat = db.relationship(
        "RiwayatKlasifikasi",
        foreign_keys="RiwayatKlasifikasi.id_kelas_prediksi",
        back_populates="kelas_prediksi",
        lazy="dynamic",
    )

    @property
    def name(self):
        return self.nama_kelas

    @name.setter
    def name(self, value):
        self.nama_kelas = value

    @property
    def description(self):
        return self.deskripsi

    @description.setter
    def description(self, value):
        self.deskripsi = value

    @property
    def main_characteristics(self):
        return self.ciri_utama

    @main_characteristics.setter
    def main_characteristics(self, value):
        self.ciri_utama = value

    @property
    def created_at(self):
        return self.dibuat_pada

    @property
    def updated_at(self):
        return self.diperbarui_pada

    def __repr__(self) -> str:
        return f"<KelasKhat {self.slug}>"


KhatClass = KelasKhat
