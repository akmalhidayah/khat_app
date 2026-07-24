"""Dataset image registry (gambar_dataset)."""

from datetime import datetime

from .database import db


class GambarDataset(db.Model):
    __tablename__ = "gambar_dataset"

    id = db.Column(db.Integer, primary_key=True)
    id_kelas = db.Column(db.Integer, db.ForeignKey("kelas_khat.id"), nullable=False, index=True)
    nama_file = db.Column(db.String(255), nullable=False)
    nama_file_asli = db.Column(db.String(255), nullable=False)
    path_asli = db.Column(db.String(512), nullable=False)
    path_proses = db.Column(db.String(512), nullable=True)
    path_model_ready = db.Column(db.String(512), nullable=True)
    format_file = db.Column(db.String(20), nullable=True)
    ukuran_file = db.Column(db.Integer, nullable=True)
    lebar = db.Column(db.Integer, nullable=True)
    tinggi = db.Column(db.Integer, nullable=True)
    sumber_data = db.Column(db.String(50), nullable=True, default="raw")
    hash_file = db.Column(db.String(64), nullable=True, index=True)
    status = db.Column(db.String(30), nullable=False, default="active")
    split_data = db.Column(db.String(20), nullable=True, index=True)
    id_legacy = db.Column(db.Integer, nullable=True, unique=True, index=True)
    dibuat_pada = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    diperbarui_pada = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    kelas = db.relationship("KelasKhat", back_populates="gambar_dataset")
    riwayat_klasifikasi = db.relationship("RiwayatKlasifikasi", back_populates="gambar", lazy="dynamic")

    @property
    def filename(self):
        return self.nama_file

    @filename.setter
    def filename(self, value):
        self.nama_file = value

    @property
    def original_filename(self):
        return self.nama_file_asli

    @original_filename.setter
    def original_filename(self, value):
        self.nama_file_asli = value

    @property
    def image_path(self):
        return self.path_asli

    @image_path.setter
    def image_path(self, value):
        self.path_asli = value

    @property
    def class_name(self):
        return self.kelas.slug if self.kelas else None

    @property
    def data_type(self):
        return self.sumber_data or "raw"

    @data_type.setter
    def data_type(self, value):
        self.sumber_data = value

    @property
    def image_format(self):
        return self.format_file

    @image_format.setter
    def image_format(self, value):
        self.format_file = value

    @property
    def image_size(self):
        if self.lebar and self.tinggi:
            return f"{self.lebar}x{self.tinggi}"
        return ""

    @image_size.setter
    def image_size(self, value):
        if value and "x" in str(value).lower():
            parts = str(value).lower().split("x")
            try:
                self.lebar, self.tinggi = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                pass

    @property
    def created_at(self):
        return self.dibuat_pada

    @created_at.setter
    def created_at(self, value):
        self.dibuat_pada = value

    @property
    def legacy_dataset_id(self):
        return self.id_legacy

    @legacy_dataset_id.setter
    def legacy_dataset_id(self, value):
        self.id_legacy = value

    def __repr__(self) -> str:
        return f"<GambarDataset {self.id} {self.nama_file}>"


DatasetImage = GambarDataset
Dataset = GambarDataset
