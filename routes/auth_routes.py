import os
import mimetypes

from flask import Blueprint, current_app, flash, redirect, render_template, request, send_file, session, url_for

from models import User
from services.auth_registration_service import (
    register_pengguna,
    registration_form_defaults,
)
from services.login_ui_service import resolve_login_decorative_image

auth_bp = Blueprint("auth", __name__)


def _auth_page_context(**extra):
    path, _source = resolve_login_decorative_image(current_app.config)
    if path and os.path.isfile(path):
        version = int(os.path.getmtime(path))
        decorative_image_url = url_for("auth.login_decorative_image", v=version)
    else:
        decorative_image_url = None
    ctx = {"decorative_image_url": decorative_image_url}
    ctx.update(extra)
    return ctx


def _login_context():
    return _auth_page_context()


@auth_bp.route("/login/art")
def login_decorative_image():
    path, _source = resolve_login_decorative_image(current_app.config)
    if not path or not os.path.isfile(path):
        return ("", 404)
    mime, _ = mimetypes.guess_type(path)
    response = send_file(path, mimetype=mime or "image/jpeg", max_age=86400)
    response.headers["Cache-Control"] = "public, max-age=86400"
    return response


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if not username or not password:
            flash("Username dan password wajib diisi.", "danger")
            return render_template("login.html", **_login_context())
        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(password):
            flash("Username atau password salah. Silakan coba lagi.", "danger")
            return render_template("login.html", **_login_context())
        session["user_id"] = user.id
        session["name"] = user.name
        session["role"] = user.role
        flash("Login berhasil.", "success")
        if current_app.config.get("APP_SIMPLE_MODE"):
            if user.role == "admin":
                return redirect(url_for("dashboard.dashboard"))
            return redirect(url_for("classification.classify_image"))
        return redirect(url_for("dashboard.dashboard"))
    return render_template("login.html", **_login_context())


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("dashboard.dashboard"))

    form_values = registration_form_defaults()
    field_errors = {}

    if request.method == "POST":
        form_values = registration_form_defaults(request.form)
        result = register_pengguna(request.form)
        if result.ok:
            flash(result.message, "success")
            return redirect(url_for("auth.login"))
        field_errors = result.field_errors or {}
        flash(result.message, "danger")
        return render_template(
            "register.html",
            form_values=form_values,
            field_errors=field_errors,
            **_auth_page_context(),
        )

    return render_template(
        "register.html",
        form_values=form_values,
        field_errors=field_errors,
        **_auth_page_context(),
    )


@auth_bp.route("/logout")
def logout():
    session.clear()
    flash("Anda telah keluar dari sistem.", "info")
    return redirect(url_for("auth.login"))
