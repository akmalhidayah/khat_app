"""QR/barcode gate must reject codes but not multi-line Arabic calligraphy."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw

from services.validation_pipeline.arabic_script import _detect_qr_or_barcode, detect_arabic_script


class QrBarcodeGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "_tmp_qr_gate")
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        for name in os.listdir(self.tmp_dir):
            path = os.path.join(self.tmp_dir, name)
            if os.path.isfile(path):
                os.remove(path)
        try:
            os.rmdir(self.tmp_dir)
        except OSError:
            pass

    def _save(self, name: str, image: Image.Image) -> str:
        path = os.path.join(self.tmp_dir, name)
        image.save(path, format="JPEG", quality=90)
        return path

    def _make_multiline_calligraphy(self) -> Image.Image:
        """Dense multi-line dark strokes on parchment (Surah practice sheet style)."""
        img = Image.new("RGB", (420, 420), (232, 214, 176))
        draw = ImageDraw.Draw(img)
        ink = (28, 22, 18)
        red = (180, 40, 40)
        for i, y in enumerate((55, 100, 145, 190, 235, 280, 325, 370)):
            draw.arc([30, y - 28, 390, y + 32], start=200, end=340, fill=ink, width=4)
            draw.line([(40, y + 4), (380, y + 10)], fill=ink, width=3)
            draw.ellipse([90, y - 18, 102, y - 6], fill=ink)
            draw.ellipse([210, y - 16, 220, y - 6], fill=ink)
            draw.ellipse([310, y - 20, 320, y - 10], fill=ink)
            # Teacher correction marks — must not look like barcode rejection.
            if i in (1, 2, 5):
                draw.line([(70, y + 2), (250, y + 6)], fill=red, width=2)
        return img

    def _make_barcode_strip(self) -> Image.Image:
        img = Image.new("RGB", (420, 180), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        x = 20
        while x < 400:
            width = 2 if (x // 3) % 2 == 0 else 4
            draw.rectangle([x, 40, x + width, 140], fill=(0, 0, 0))
            x += width + 2
        return img

    def test_rejects_barcode_strip(self):
        path = self._save("barcode.jpg", self._make_barcode_strip())
        self.assertTrue(_detect_qr_or_barcode(path))

    def test_accepts_multiline_calligraphy_with_red_marks(self):
        path = self._save("surah_practice.jpg", self._make_multiline_calligraphy())
        self.assertFalse(_detect_qr_or_barcode(path), "calligraphy must not match barcode stripe heuristic")
        result = detect_arabic_script(path)
        self.assertTrue(result.get("accepted"), result)

    def test_accepts_real_rejected_upload_when_present(self):
        real = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "static",
            "uploads",
            "b9c18b59c9984b8d975f4c5f918571af.jpg",
        )
        if not os.path.isfile(real):
            self.skipTest("real upload not present")
        self.assertFalse(_detect_qr_or_barcode(real), "real Surah calligraphy was false-rejected as barcode")
        result = detect_arabic_script(real)
        self.assertTrue(result.get("accepted"), result)


if __name__ == "__main__":
    unittest.main()
