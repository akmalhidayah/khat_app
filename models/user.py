from datetime import datetime

from werkzeug.security import check_password_hash, generate_password_hash

from .database import db


class Pengguna(db.Model):
    __tablename__ = "pengguna"

    id = db.Column(db.Integer, primary_key=True)
    nama_lengkap = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), nullable=True)
    password_hash = db.Column("password_hash", db.String(255), nullable=False)
    peran = db.Column(db.String(20), default="admin", nullable=False)
    status = db.Column(db.String(20), default="aktif", nullable=False)
    dibuat_pada = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    diperbarui_pada = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    @property
    def name(self):
        return self.nama_lengkap

    @name.setter
    def name(self, value):
        self.nama_lengkap = value

    @property
    def password(self):
        return self.password_hash

    @password.setter
    def password(self, value):
        self.password_hash = value

    @property
    def role(self):
        return self.peran

    @role.setter
    def role(self, value):
        self.peran = value

    @property
    def created_at(self):
        return self.dibuat_pada

    @created_at.setter
    def created_at(self, value):
        self.dibuat_pada = value

    def set_password(self, raw_password: str) -> None:
        self.password_hash = generate_password_hash(raw_password, method="pbkdf2:sha256")

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)


User = Pengguna
