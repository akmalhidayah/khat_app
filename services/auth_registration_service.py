"""Self-service account registration for new pengguna accounts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

from models import User, db

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,50}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LEN = 6
MIN_NAME_LEN = 2


@dataclass
class RegistrationResult:
    ok: bool
    message: str
    user: Optional[User] = None
    field_errors: Optional[Dict[str, str]] = None


def _normalize_fields(form_data) -> Dict[str, str]:
    return {
        "name": (form_data.get("name") or "").strip(),
        "username": (form_data.get("username") or "").strip(),
        "email": (form_data.get("email") or "").strip(),
        "password": (form_data.get("password") or "").strip(),
        "password_confirm": (form_data.get("password_confirm") or "").strip(),
    }


def validate_registration_fields(fields: Dict[str, str]) -> Dict[str, str]:
    errors: Dict[str, str] = {}
    name = fields["name"]
    username = fields["username"]
    email = fields["email"]
    password = fields["password"]
    password_confirm = fields["password_confirm"]

    if len(name) < MIN_NAME_LEN:
        errors["name"] = "Nama lengkap minimal 2 karakter."

    if not username:
        errors["username"] = "Username wajib diisi."
    elif not USERNAME_RE.match(username):
        errors["username"] = "Username 3–50 karakter (huruf, angka, underscore)."
    elif User.query.filter_by(username=username).first():
        errors["username"] = "Username sudah digunakan. Pilih yang lain."

    if email and not EMAIL_RE.match(email):
        errors["email"] = "Format email tidak valid."
    elif email and User.query.filter(User.email == email).first():
        errors["email"] = "Email sudah terdaftar."

    if len(password) < MIN_PASSWORD_LEN:
        errors["password"] = f"Password minimal {MIN_PASSWORD_LEN} karakter."

    if not password_confirm:
        errors["password_confirm"] = "Konfirmasi password wajib diisi."
    elif password != password_confirm:
        errors["password_confirm"] = "Konfirmasi password tidak cocok."

    return errors


def register_pengguna(form_data) -> RegistrationResult:
    """Create a new account with role pengguna. Never allows self-register as admin."""
    fields = _normalize_fields(form_data)
    field_errors = validate_registration_fields(fields)
    if field_errors:
        first = next(iter(field_errors.values()))
        return RegistrationResult(ok=False, message=first, field_errors=field_errors)

    user = User(
        name=fields["name"],
        username=fields["username"],
        email=fields["email"] or None,
        role="pengguna",
        status="aktif",
    )
    user.set_password(fields["password"])
    db.session.add(user)
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        return RegistrationResult(
            ok=False,
            message="Gagal menyimpan akun. Silakan coba lagi.",
            field_errors={"username": "Username mungkin sudah digunakan."},
        )

    return RegistrationResult(
        ok=True,
        message="Pendaftaran berhasil. Silakan masuk dengan akun baru Anda.",
        user=user,
    )


def registration_form_defaults(form_data=None) -> Dict[str, str]:
    if not form_data:
        return {"name": "", "username": "", "email": ""}
    fields = _normalize_fields(form_data)
    return {
        "name": fields["name"],
        "username": fields["username"],
        "email": fields["email"],
    }
