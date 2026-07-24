from datetime import datetime

from .database import db


class VersiModel(db.Model):
    __tablename__ = "versi_model"

    id = db.Column(db.Integer, primary_key=True)
    versi = db.Column(db.String(50), nullable=False)
    nama_model = db.Column(db.String(255), nullable=False)
    sumber_model = db.Column(db.String(100), nullable=False)
    runtime = db.Column(db.String(100), nullable=True)
    file_model = db.Column(db.String(512), nullable=True)
    file_metadata = db.Column(db.String(512), nullable=True)
    file_weights = db.Column(db.String(512), nullable=True)
    lebar_input = db.Column(db.Integer, nullable=True, default=224)
    tinggi_input = db.Column(db.Integer, nullable=True, default=224)
    label_json = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(30), nullable=True, default="active")
    akurasi = db.Column(db.Float, nullable=True)
    presisi = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    jumlah_kelas = db.Column(db.Integer, nullable=True, default=4)
    jumlah_gambar_dataset = db.Column(db.Integer, nullable=True)
    catatan = db.Column(db.Text, nullable=True)
    aktif = db.Column(db.Boolean, nullable=False, default=False)
    dibuat_pada = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    diaktifkan_pada = db.Column(db.DateTime, nullable=True)

    @property
    def model_name(self):
        return self.nama_model

    @model_name.setter
    def model_name(self, value):
        self.nama_model = value

    @property
    def model_source(self):
        return self.sumber_model

    @model_source.setter
    def model_source(self, value):
        self.sumber_model = value

    @property
    def model_version(self):
        return self.versi

    @model_version.setter
    def model_version(self, value):
        self.versi = value

    @property
    def model_runtime(self):
        return self.runtime

    @model_runtime.setter
    def model_runtime(self, value):
        self.runtime = value

    @property
    def version(self):
        return self.versi

    @version.setter
    def version(self, value):
        self.versi = value

    @property
    def source(self):
        return self.sumber_model

    @source.setter
    def source(self, value):
        self.sumber_model = value

    @property
    def model_file(self):
        return self.file_model

    @model_file.setter
    def model_file(self, value):
        self.file_model = value

    @property
    def metadata_file(self):
        return self.file_metadata

    @metadata_file.setter
    def metadata_file(self, value):
        self.file_metadata = value

    @property
    def weights_file(self):
        return self.file_weights

    @weights_file.setter
    def weights_file(self, value):
        self.file_weights = value

    @property
    def input_width(self):
        return self.lebar_input

    @input_width.setter
    def input_width(self, value):
        self.lebar_input = value

    @property
    def input_height(self):
        return self.tinggi_input

    @input_height.setter
    def input_height(self, value):
        self.tinggi_input = value

    @property
    def labels_json(self):
        return self.label_json

    @labels_json.setter
    def labels_json(self, value):
        self.label_json = value

    @property
    def evaluation_accuracy(self):
        return self.akurasi

    @evaluation_accuracy.setter
    def evaluation_accuracy(self, value):
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
    def num_classes(self):
        return self.jumlah_kelas

    @num_classes.setter
    def num_classes(self, value):
        self.jumlah_kelas = value

    @property
    def dataset_images(self):
        return self.jumlah_gambar_dataset

    @dataset_images.setter
    def dataset_images(self, value):
        self.jumlah_gambar_dataset = value

    @property
    def notes(self):
        return self.catatan

    @notes.setter
    def notes(self, value):
        self.catatan = value

    @property
    def is_active(self):
        return self.aktif

    @is_active.setter
    def is_active(self, value):
        self.aktif = value

    @property
    def created_at(self):
        return self.dibuat_pada

    @created_at.setter
    def created_at(self, value):
        self.dibuat_pada = value

    @property
    def activated_at(self):
        return self.diaktifkan_pada

    @activated_at.setter
    def activated_at(self, value):
        self.diaktifkan_pada = value

    @property
    def training_date(self):
        return self.diaktifkan_pada

    @training_date.setter
    def training_date(self, value):
        self.diaktifkan_pada = value


ModelVersion = VersiModel
