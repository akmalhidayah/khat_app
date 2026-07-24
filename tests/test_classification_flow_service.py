"""Tests for user classification flow service."""

import os
import tempfile
import unittest
from unittest.mock import patch

from app import create_app
from models import KelasKhat, Pengguna, ProbabilitasPrediksi, PerhitunganAlgoritma, RiwayatKlasifikasi, db
from services.classification_flow_service import allowed_classification_file


class TestClassificationFlow(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.ctx = self.app.app_context()
        self.ctx.push()

    def tearDown(self):
        self.ctx.pop()

    def test_allowed_extensions(self):
        self.assertTrue(allowed_classification_file("sample.jpg"))
        self.assertTrue(allowed_classification_file("sample.JPEG"))
        self.assertTrue(allowed_classification_file("sample.png"))
        self.assertFalse(allowed_classification_file("sample.webp"))
        self.assertFalse(allowed_classification_file("sample.pdf"))

    @patch("services.classification_flow_service.ensure_active_cnn_model_version")
    def test_persist_classification_result(self, mock_model):
        mock_model.return_value = 1
        user = Pengguna.query.filter_by(username="user").first()
        if not user:
            user = Pengguna(nama_lengkap="Test", username="testflow", peran="pengguna")
            user.set_password("test")
            db.session.add(user)
            db.session.commit()

        for slug, name in [
            ("naskhi", "Khat Naskhi"),
            ("diwani", "Khat Diwani"),
            ("diwani_jali", "Khat Diwani Jali"),
            ("tsuluts", "Khat Tsuluts"),
        ]:
            if not KelasKhat.query.filter_by(slug=slug).first():
                db.session.add(KelasKhat(nama_kelas=name, slug=slug))
        db.session.commit()

        from services.classification_flow_service import persist_classification_result

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(b"\x89PNG\r\n\x1a\n")
            path = tmp.name
        try:
            prediction = {
                "predicted_class": "naskhi",
                "confidence": 0.91,
                "scores": {
                    "naskhi": 0.91,
                    "diwani": 0.04,
                    "diwani_jali": 0.03,
                    "tsuluts": 0.02,
                },
            }
            row = persist_classification_result(
                user_id=user.id,
                abs_path=path,
                rel_path="static/uploads/test.png",
                original_filename="test.png",
                prediction=prediction,
            )
            self.assertIsNotNone(row.id)
            probs = ProbabilitasPrediksi.query.filter_by(id_riwayat=row.id).count()
            calc = PerhitunganAlgoritma.query.filter_by(id_riwayat=row.id).first()
            self.assertEqual(probs, 4)
            self.assertIsNotNone(calc)
            self.assertEqual(row.id_pengguna, user.id)
        finally:
            if os.path.exists(path):
                os.remove(path)
            RiwayatKlasifikasi.query.filter_by(nama_file_upload="test.png").delete()
            db.session.commit()


if __name__ == "__main__":
    unittest.main()
