"""Tests for Diwani vs Diwani Jali disambiguation."""

import os
import tempfile
import unittest

from PIL import Image, ImageDraw

from services.diwani_pair_service import (
    apply_contested_style_pairs,
    compute_jali_density_features,
    disambiguate_diwani_pair,
    sanitize_score_dict,
)


class DiwaniPairDisambiguationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="diwani_pair_")

    def tearDown(self):
        for name in os.listdir(self.tmp):
            os.remove(os.path.join(self.tmp, name))
        os.rmdir(self.tmp)

    def _save(self, name: str, image: Image.Image) -> str:
        path = os.path.join(self.tmp, name)
        image.save(path, format="JPEG", quality=90)
        return path

    def test_dense_ornament_and_dots_prefers_jali(self):
        # Diwani Jali-like: dense strokes + many decorative dots/titik.
        img = Image.new("RGB", (400, 400), (235, 230, 220))
        draw = ImageDraw.Draw(img)
        for y in range(10, 390, 8):
            draw.line([(10, y), (390, y + 6)], fill=(5, 5, 5), width=4)
            for x in range(15, 385, 10):
                draw.ellipse([x, y - 2, x + 5, y + 3], fill=(0, 0, 0))
                draw.rectangle([x + 2, y + 2, x + 6, y + 6], fill=(20, 20, 20))
        path = self._save("jali_like.jpg", img)
        feats = compute_jali_density_features(path)
        self.assertIsNotNone(feats)
        self.assertGreater(feats["ornament_index"], 0.33)
        # Dense ornament may merge small dots into ink; holes still mark fill/ornamen.
        self.assertGreater(feats["dot_count"] + feats["hole_count"], 20)
        self.assertGreater(feats["jali_preference"], 0.0)

        scores = {"naskhi": 0.05, "diwani": 0.46, "diwani_jali": 0.44, "tsuluts": 0.05}
        out = sanitize_score_dict(disambiguate_diwani_pair(scores, path))
        self.assertGreaterEqual(out["diwani_jali"], out["diwani"])

    def test_open_flowing_few_ornaments_prefers_diwani(self):
        # Diwani-like: open flowing arcs, sparse ornaments/titik.
        img = Image.new("RGB", (400, 400), (250, 248, 242))
        draw = ImageDraw.Draw(img)
        for y in (90, 180, 270):
            draw.arc([40, y - 40, 360, y + 50], start=200, end=340, fill=(25, 25, 25), width=4)
        path = self._save("diwani_like.jpg", img)
        feats = compute_jali_density_features(path)
        self.assertIsNotNone(feats)
        self.assertLess(feats["ornament_index"], 0.33)
        self.assertLess(feats["jali_preference"], 0.0)

        scores = {"naskhi": 0.05, "diwani": 0.44, "diwani_jali": 0.46, "tsuluts": 0.05}
        out = sanitize_score_dict(disambiguate_diwani_pair(scores, path))
        self.assertGreaterEqual(out["diwani"], out["diwani_jali"])

    def test_non_pair_top_untouched(self):
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        path = self._save("other.jpg", img)
        scores = {"naskhi": 0.70, "diwani": 0.15, "diwani_jali": 0.10, "tsuluts": 0.05}
        out = disambiguate_diwani_pair(scores, path)
        self.assertEqual(out["naskhi"], scores["naskhi"])
        self.assertEqual(out["tsuluts"], scores["tsuluts"])

    def test_confident_pair_with_weak_visual_stays_stable(self):
        # Large model margin + empty image should not aggressively flip labels.
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        path = self._save("blank.jpg", img)
        scores = {"naskhi": 0.04, "diwani": 0.72, "diwani_jali": 0.20, "tsuluts": 0.04}
        out = disambiguate_diwani_pair(scores, path)
        self.assertEqual(max(out, key=out.get), "diwani")

    def test_apply_contested_style_pairs_preserves_naskhi_lead(self):
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        path = self._save("naskhi_lead.jpg", img)
        scores = {"naskhi": 0.62, "diwani": 0.18, "diwani_jali": 0.12, "tsuluts": 0.08}
        out = apply_contested_style_pairs(scores, path, list(scores.keys()))
        self.assertEqual(max(out, key=out.get), "naskhi")


if __name__ == "__main__":
    unittest.main()
