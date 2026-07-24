"""Tests for Khat class characteristics reference data."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.khat_characteristics_service import (
    build_characteristics_context,
    get_khat_characteristics,
)


class KhatCharacteristicsTests(unittest.TestCase):
    def test_diwani_characteristics(self):
        data = get_khat_characteristics("diwani")
        self.assertIsNotNone(data)
        self.assertEqual(data["title"], "Ciri Utama Khat Diwani")
        self.assertEqual(len(data["characteristics"]), 5)

    def test_build_context_for_predicted_only(self):
        ctx = build_characteristics_context("naskhi")
        self.assertFalse(ctx["is_mismatch"])
        self.assertIsNone(ctx["expected"])
        self.assertEqual(ctx["predicted"]["class_display"], "Naskhi")

    def test_build_context_mismatch_shows_comparison(self):
        ctx = build_characteristics_context(
            "diwani_jali",
            expected_class="tsuluts",
            validation_status="Prediction Mismatch",
            filename_mismatch=True,
        )
        self.assertTrue(ctx["is_mismatch"])
        self.assertTrue(ctx["show_comparison"])
        self.assertEqual(ctx["predicted_display"], "Diwani Jali")
        self.assertEqual(ctx["expected_display"], "Tsuluts")
        self.assertIn("Perhatian", ctx["mismatch_warning"])

    def test_unknown_class_returns_none(self):
        self.assertIsNone(get_khat_characteristics("unknown"))


if __name__ == "__main__":
    unittest.main()
