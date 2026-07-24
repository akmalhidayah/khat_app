"""Tests for Diwani Jali vs Tsuluts contested-pair disambiguation."""

import os
import tempfile
import unittest

from PIL import Image, ImageDraw

from services.jali_tsuluts_pair_service import disambiguate_jali_tsuluts_pair
from services.diwani_pair_service import sanitize_score_dict


class JaliTsulutsPairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="jali_tsuluts_")

    def tearDown(self):
        for name in os.listdir(self.tmp):
            os.remove(os.path.join(self.tmp, name))
        os.rmdir(self.tmp)

    def _save(self, name: str, image: Image.Image) -> str:
        path = os.path.join(self.tmp, name)
        image.save(path, format="JPEG", quality=90)
        return path

    def test_dense_ornament_prefers_jali_over_tsuluts(self):
        img = Image.new("RGB", (400, 400), (235, 230, 220))
        draw = ImageDraw.Draw(img)
        for y in range(10, 390, 8):
            draw.line([(10, y), (390, y + 6)], fill=(5, 5, 5), width=4)
            for x in range(15, 385, 10):
                draw.ellipse([x, y - 2, x + 5, y + 3], fill=(0, 0, 0))
        path = self._save("jali_vs_tsuluts.jpg", img)
        scores = {"naskhi": 0.04, "diwani": 0.06, "diwani_jali": 0.44, "tsuluts": 0.46}
        out = sanitize_score_dict(disambiguate_jali_tsuluts_pair(scores, path))
        self.assertGreaterEqual(out["diwani_jali"], out["tsuluts"])

    def test_open_monumental_strokes_prefers_tsuluts(self):
        img = Image.new("RGB", (400, 400), (250, 248, 242))
        draw = ImageDraw.Draw(img)
        # Few large monumental curves, little decorative fill.
        draw.arc([30, 40, 370, 220], start=200, end=340, fill=(15, 15, 15), width=10)
        draw.arc([50, 160, 350, 360], start=20, end=160, fill=(20, 20, 20), width=9)
        path = self._save("tsuluts_like.jpg", img)
        scores = {"naskhi": 0.04, "diwani": 0.06, "diwani_jali": 0.46, "tsuluts": 0.44}
        out = sanitize_score_dict(disambiguate_jali_tsuluts_pair(scores, path))
        self.assertGreaterEqual(out["tsuluts"], out["diwani_jali"])

    def test_non_pair_untouched(self):
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        path = self._save("other.jpg", img)
        scores = {"naskhi": 0.55, "diwani": 0.20, "diwani_jali": 0.15, "tsuluts": 0.10}
        out = disambiguate_jali_tsuluts_pair(scores, path)
        self.assertEqual(out["naskhi"], scores["naskhi"])
        self.assertEqual(out["diwani"], scores["diwani"])


if __name__ == "__main__":
    unittest.main()
