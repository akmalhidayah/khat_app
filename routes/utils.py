from functools import wraps
from flask import flash, redirect, session, url_for


def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login first.", "warning")
            return redirect(url_for("auth.login"))
        return func(*args, **kwargs)
    return wrapper


def admin_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session or session.get("role") != "admin":
            flash("Admin access required.", "danger")
            return redirect(url_for("dashboard.dashboard"))
        return func(*args, **kwargs)
    return wrapper
