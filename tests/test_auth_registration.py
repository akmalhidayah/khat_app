"""Unit tests for registration field validation (no DB)."""

from services.auth_registration_service import USERNAME_RE, EMAIL_RE, MIN_PASSWORD_LEN


def test_username_pattern_accepts_valid():
    assert USERNAME_RE.match("peneliti01")
    assert USERNAME_RE.match("user_abc")
    assert USERNAME_RE.match("ABC123")


def test_username_pattern_rejects_invalid():
    assert not USERNAME_RE.match("ab")
    assert not USERNAME_RE.match("user-name")
    assert not USERNAME_RE.match("user name")
    assert not USERNAME_RE.match("")


def test_email_pattern():
    assert EMAIL_RE.match("a@b.co")
    assert not EMAIL_RE.match("not-an-email")
    assert not EMAIL_RE.match("@missing.local")


def test_min_password_len():
    assert MIN_PASSWORD_LEN == 6
