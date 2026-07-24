"""Tests for Khat vs Non-Khat detection threshold logic."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw

from services.khat_detector_service import assess_calligraphy_content, resolve_detection


class MockConfig(dict):
    pass


CONFIG = MockConfig(
    {
        "KHAT_ACCEPT_THRESHOLD": 0.70,
        "KHAT_REJECT_THRESHOLD": 0.50,
        "KHAT_BORDERLINE_THRESHOLD": 0.65,
        "KHAT_HIGH_CONFIDENCE_THRESHOLD": 0.85,
        "KHAT_DETECTOR_SETTINGS_PATH": "/tmp/nonexistent_khat_settings.json",
    }
)


class CalligraphyContentGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "_tmp_content_gate")
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

    def test_rejects_person_animal_plant_photos(self):
        cases = {
            "person.jpg": self._make_person(),
            "animal.jpg": self._make_animal(),
            "plant.jpg": self._make_plant(),
            "ocean.jpg": self._make_ocean(),
        }
        for name, image in cases.items():
            path = self._save(name, image)
            result = assess_calligraphy_content(path)
            self.assertTrue(result["is_photographic_non_khat"], name)
            self.assertFalse(result["looks_like_calligraphy"], name)
            self.assertTrue(result["is_non_khat_content"], name)
            self.assertIn("bukan merupakan tulisan arab", (result["rejection_reason"] or "").lower())

    def test_rejects_studio_animal_on_white_background(self):
        """Cow/animal photo on white studio backdrop must not be labeled as khat."""
        img = Image.new("RGB", (447, 447), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Brown cow-like body + head silhouette
        draw.ellipse([70, 140, 360, 360], fill=(150, 95, 45))
        draw.ellipse([250, 80, 390, 220], fill=(140, 88, 40))
        draw.ellipse([300, 120, 325, 145], fill=(20, 20, 20))
        draw.polygon([(70, 220), (40, 200), (55, 260)], fill=(130, 80, 35))
        path = self._save("studio_cow.jpg", img)
        result = assess_calligraphy_content(path)
        self.assertTrue(result["is_photographic_non_khat"], result.get("scene_scores"))
        self.assertTrue(result["is_non_khat_content"], result.get("scene_scores"))
        self.assertFalse(result["looks_like_calligraphy"])
        self.assertEqual(result["rejection_reason"], "Gambar bukan merupakan tulisan Arab.")

    def test_rejects_chickens_on_dark_background(self):
        """Animal photos on dark studio/nature grounds must be rejected (not Naskhi)."""
        real = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "static",
            "uploads",
            "e6320233c57842e98e7843833f25026c.png",
        )
        if os.path.isfile(real):
            path = real
        else:
            img = Image.new("RGB", (480, 360), (8, 8, 8))
            draw = ImageDraw.Draw(img)
            # Rooster (warm brown + red comb)
            draw.ellipse([60, 100, 200, 250], fill=(160, 90, 35))
            draw.ellipse([130, 60, 200, 130], fill=(150, 80, 30))
            draw.polygon([(155, 40), (175, 70), (145, 70)], fill=(210, 30, 25))
            # Hen
            draw.ellipse([260, 140, 400, 260], fill=(190, 150, 90))
            draw.ellipse([340, 100, 400, 160], fill=(180, 140, 80))
            draw.polygon([(360, 85), (375, 105), (350, 105)], fill=(200, 40, 30))
            draw.rectangle([40, 250, 440, 280], fill=(90, 70, 45))
            path = self._save("chickens_dark.jpg", img)
        result = assess_calligraphy_content(path)
        self.assertTrue(result["is_photographic_non_khat"], result.get("scene_scores"))
        self.assertTrue(result["is_non_khat_content"], result.get("scene_scores"))
        self.assertFalse(result["looks_like_calligraphy"])
        self.assertEqual(result["rejection_reason"], "Gambar bukan merupakan tulisan Arab.")

    def test_rejects_blue_poster_with_people(self):
        """Color poster / group photo must be rejected even on a dark blue background."""
        img = Image.new("RGB", (400, 500), (12, 45, 140))
        draw = ImageDraw.Draw(img)
        # Three simplified people (heads + torsos)
        for x in (70, 170, 270):
            draw.ellipse([x, 120, x + 70, 190], fill=(230, 190, 160))
            draw.rectangle([x + 10, 190, x + 60, 320], fill=(30, 80, 170))
            draw.rectangle([x + 5, 190, x + 65, 250], fill=(15, 15, 15))
        draw.rectangle([40, 40, 360, 90], fill=(250, 250, 250))
        path = self._save("blue_poster_people.jpg", img)
        result = assess_calligraphy_content(path)
        self.assertTrue(result["is_photographic_non_khat"], result.get("scene_scores"))
        self.assertTrue(result["is_non_khat_content"], result.get("scene_scores"))
        self.assertFalse(result["looks_like_calligraphy"])
        self.assertEqual(result["rejection_reason"], "Gambar bukan merupakan tulisan Arab.")

    def test_rejects_building_outdoor_scene_with_sky(self):
        """Mosque/building photo with sky and vegetation must be rejected."""
        img = Image.new("RGB", (400, 400), (70, 150, 220))  # sky
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 250, 400, 400], fill=(40, 130, 55))  # grass/trees
        draw.rectangle([80, 120, 320, 300], fill=(240, 240, 235))  # building
        draw.ellipse([120, 60, 280, 160], fill=(235, 235, 230))  # dome
        draw.polygon([(190, 40), (210, 40), (200, 20)], fill=(200, 170, 60))
        path = self._save("mosque_outdoor.jpg", img)
        result = assess_calligraphy_content(path)
        self.assertTrue(result["is_photographic_non_khat"], result.get("scene_scores"))
        self.assertTrue(result["is_non_khat_content"], result.get("scene_scores"))
        self.assertFalse(result["looks_like_calligraphy"])
        self.assertEqual(result["rejection_reason"], "Gambar bukan merupakan tulisan Arab.")

    def test_accepts_dark_ink_calligraphy_on_parchment(self):
        """Black calligraphy on beige parchment must not be rejected as a human photo."""
        img = Image.new("RGB", (420, 420), (214, 188, 150))
        draw = ImageDraw.Draw(img)
        # Connected cursive-like strokes (not a face)
        for y in (90, 150, 210, 270):
            draw.arc([40, y - 35, 380, y + 45], start=200, end=340, fill=(18, 18, 18), width=5)
            draw.line([(55, y + 8), (365, y + 18)], fill=(12, 12, 12), width=4)
            draw.ellipse([300, y - 10, 320, y + 10], outline=(10, 10, 10), width=3)
        path = self._save("parchment_khat.jpg", img)
        result = assess_calligraphy_content(path)
        scene = result.get("scene_scores") or {}
        self.assertFalse(result["is_photographic_non_khat"], scene)
        self.assertFalse(scene.get("is_human_photo"), scene)
        self.assertFalse(result["is_non_khat_content"], scene)
        self.assertTrue(result["looks_like_calligraphy"], scene)

    def test_rejects_ocean_and_plant_scenes(self):
        for name, maker in (("ocean2.jpg", self._make_ocean), ("plant2.jpg", self._make_plant)):
            path = self._save(name, maker())
            result = assess_calligraphy_content(path)
            self.assertTrue(result["is_photographic_non_khat"], result.get("scene_scores"))
            self.assertTrue(result["is_non_khat_content"])

    def test_rejects_latin_alphabet_letters(self):
        path = self._save("latin_alphabet.jpg", self._make_latin_alphabet())
        result = assess_calligraphy_content(path)
        self.assertTrue(result["is_latin_script_non_khat"], result.get("latin_gate"))
        self.assertTrue(result["is_non_khat_content"])
        self.assertFalse(result["looks_like_calligraphy"])
        self.assertIn("bukan merupakan tulisan arab", (result["rejection_reason"] or "").lower())

    def test_rejects_decorative_latin_sentence(self):
        path = self._save("latin_sentence.jpg", self._make_decorative_latin_sentence())
        result = assess_calligraphy_content(path)
        self.assertTrue(result["is_latin_script_non_khat"], result.get("latin_gate"))
        self.assertTrue(result["is_non_khat_content"])
        self.assertFalse(result["looks_like_calligraphy"])

    def test_rejects_connected_latin_lettering_wordmark(self):
        """Cursive Latin wordmark like 'Lettering' must be rejected, not labeled Diwani."""
        # Prefer the real failing upload when present in the workspace.
        real = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "static",
            "uploads",
            "caaae43ce7c943059fa344ae1487f443.jpeg",
        )
        if os.path.isfile(real):
            path = real
        else:
            img = Image.new("RGB", (480, 360), (248, 248, 248))
            draw = ImageDraw.Draw(img)
            try:
                from PIL import ImageFont
                font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 90)
            except Exception:
                font = ImageFont.load_default()
            draw.text((40, 120), "Lettering", fill=(15, 15, 15), font=font)
            path = self._save("latin_lettering_word.jpg", img)
        result = assess_calligraphy_content(path)
        self.assertTrue(result["is_latin_script_non_khat"], result.get("latin_gate"))
        self.assertTrue(result["is_non_khat_content"], result.get("latin_gate"))
        self.assertFalse(result["looks_like_calligraphy"])
        self.assertEqual(result["rejection_reason"], "Input bukan merupakan tulisan Arab.")

    def test_accepts_simple_calligraphy_like_image(self):
        img = Image.new("RGB", (400, 400), (248, 245, 238))
        draw = ImageDraw.Draw(img)
        for y in (80, 140, 200, 260):
            draw.arc([40, y - 30, 360, y + 40], start=200, end=340, fill=(20, 20, 20), width=4)
            draw.line([(60, y), (340, y + 8)], fill=(15, 15, 15), width=3)
        path = self._save("khat_like.jpg", img)
        result = assess_calligraphy_content(path)
        self.assertFalse(result["is_photographic_non_khat"])
        self.assertFalse(result["is_latin_script_non_khat"], result.get("latin_gate"))
        self.assertFalse(result["is_non_khat_content"])
        self.assertTrue(result["looks_like_calligraphy"])

    def test_accepts_gold_calligraphy_on_parchment(self):
        """Gold/brown ink on aged paper must not be mistaken for human/animal skin."""
        path = self._save("gold_khat.jpg", self._make_gold_calligraphy_on_parchment())
        result = assess_calligraphy_content(path)
        self.assertFalse(result["is_photographic_non_khat"], result.get("scene_scores"))
        self.assertFalse(result["is_latin_script_non_khat"], result.get("latin_gate"))
        self.assertFalse(result["is_non_khat_content"], result.get("scene_scores"))
        self.assertTrue(result["looks_like_calligraphy"])
        self.assertTrue((result.get("scene_scores") or {}).get("looks_like_ink_on_paper"))

    def test_accepts_light_calligraphy_on_dark_panel(self):
        """White/gold script on dark brown ornate panel must pass Stage 1 content gate."""
        path = self._save("dark_panel_khat.jpg", self._make_dark_panel_calligraphy())
        result = assess_calligraphy_content(path)
        self.assertFalse(result["is_photographic_non_khat"], result.get("scene_scores"))
        self.assertFalse(result["is_non_khat_content"], result.get("scene_scores"))
        self.assertTrue(result["looks_like_calligraphy"])
        self.assertTrue((result.get("scene_scores") or {}).get("looks_like_light_ink_on_dark"))

    def test_accepts_arabic_cursive_like_strokes(self):
        """Arabic-like connected horizontal strokes must not be rejected as Latin."""
        path = self._save("arabic_cursive.jpg", self._make_arabic_cursive_like())
        result = assess_calligraphy_content(path)
        self.assertFalse(result["is_latin_script_non_khat"], result.get("latin_gate"))
        self.assertFalse(result["is_non_khat_content"], result.get("latin_gate"))
        self.assertTrue(result["looks_like_calligraphy"])

    def test_accepts_bismillah_like_calligraphy(self):
        """Basmala-style Arabic calligraphy must not be rejected as Latin abjad."""
        path = self._save("bismillah_like.jpg", self._make_bismillah_like())
        result = assess_calligraphy_content(path)
        self.assertFalse(result["is_latin_script_non_khat"], result.get("latin_gate"))
        self.assertFalse(result["is_photographic_non_khat"], result.get("scene_scores"))
        self.assertFalse(result["is_non_khat_content"], result.get("latin_gate"))
        self.assertTrue(result["looks_like_calligraphy"])

    def test_accepts_rendered_arabic_text_when_font_available(self):
        img = self._make_arabic_text_if_possible()
        if img is None:
            self.skipTest("Arabic font not available on this machine")
        path = self._save("arabic_text.jpg", img)
        result = assess_calligraphy_content(path)
        self.assertFalse(result["is_latin_script_non_khat"], result.get("latin_gate"))
        self.assertFalse(result["is_non_khat_content"], result.get("latin_gate"))

    def test_rejects_single_latin_letter(self):
        path = self._save("latin_letter_a.jpg", self._make_single_latin_letter("A"))
        result = assess_calligraphy_content(path)
        self.assertTrue(result["is_latin_script_non_khat"], result.get("latin_gate"))
        self.assertTrue(result["is_non_khat_content"])
        self.assertFalse(result["looks_like_calligraphy"])

    def test_rejects_latin_word_hello(self):
        path = self._save("latin_hello.jpg", self._make_latin_word("HELLO"))
        result = assess_calligraphy_content(path)
        self.assertTrue(result["is_latin_script_non_khat"], result.get("latin_gate"))
        self.assertTrue(result["is_non_khat_content"])

    def _make_gold_calligraphy_on_parchment(self) -> Image.Image:
        # Mimic Basmala-style gold strokes on cream parchment (user false-reject case).
        img = Image.new("RGB", (520, 280), (236, 222, 190))
        draw = ImageDraw.Draw(img)
        gold = (150, 105, 35)
        dark_gold = (110, 70, 20)
        for y in (90, 140, 190):
            draw.arc([40, y - 40, 480, y + 50], start=200, end=340, fill=gold, width=6)
            draw.line([(50, y), (470, y + 8)], fill=dark_gold, width=4)
            draw.arc([80, y - 25, 220, y + 20], start=160, end=20, fill=gold, width=3)
            draw.ellipse([150, y - 32, 162, y - 20], fill=dark_gold)
            draw.ellipse([300, y - 28, 312, y - 16], fill=dark_gold)
        return img

    def _make_dark_panel_calligraphy(self) -> Image.Image:
        # White/gold script on dark brown ground with ornate border (user false-reject case).
        img = Image.new("RGB", (520, 360), (72, 42, 22))
        draw = ImageDraw.Draw(img)
        # Dense ornamental border (warm brown/gold, not green plant photo).
        for i in range(0, 520, 18):
            draw.ellipse([i, 8, i + 22, 30], outline=(160, 110, 40), width=2)
            draw.ellipse([i, 330, i + 22, 352], outline=(160, 110, 40), width=2)
        for j in range(0, 360, 18):
            draw.ellipse([8, j, 30, j + 22], outline=(140, 95, 35), width=2)
            draw.ellipse([490, j, 512, j + 22], outline=(140, 95, 35), width=2)
        # Central light calligraphy strokes.
        for y in (110, 160, 210, 255):
            draw.arc([70, y - 28, 450, y + 34], start=200, end=340, fill=(235, 225, 200), width=5)
            draw.line([(90, y), (430, y + 6)], fill=(250, 240, 210), width=3)
            draw.ellipse([180, y - 22, 192, y - 10], fill=(240, 220, 160))
            draw.ellipse([320, y - 18, 332, y - 6], fill=(240, 220, 160))
        return img

    def _make_arabic_cursive_like(self) -> Image.Image:
        img = Image.new("RGB", (520, 280), (250, 247, 240))
        draw = ImageDraw.Draw(img)
        for y in (90, 150, 210):
            draw.arc([30, y - 35, 490, y + 45], start=200, end=340, fill=(18, 18, 18), width=5)
            draw.line([(40, y), (480, y + 6)], fill=(12, 12, 12), width=4)
            draw.line([(70, y - 18), (200, y - 8)], fill=(20, 20, 20), width=3)
            draw.ellipse([120, y - 28, 132, y - 16], fill=(15, 15, 15))
            draw.ellipse([260, y - 30, 272, y - 18], fill=(15, 15, 15))
            draw.ellipse([400, y + 10, 412, y + 22], fill=(15, 15, 15))
        return img

    def _make_bismillah_like(self) -> Image.Image:
        # Continuous Arabic baseline + flourishes (user false-reject as Latin).
        img = Image.new("RGB", (640, 280), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        ink = (12, 12, 12)
        y = 150
        draw.line([(40, y), (600, y)], fill=ink, width=6)
        draw.arc([40, y - 55, 220, y + 20], start=200, end=340, fill=ink, width=5)
        draw.arc([180, y - 70, 380, y + 10], start=190, end=350, fill=ink, width=6)
        draw.arc([340, y - 50, 520, y + 25], start=200, end=330, fill=ink, width=5)
        draw.line([(480, y - 40), (600, y - 10)], fill=ink, width=4)
        draw.line([(120, y - 35), (120, y - 8)], fill=ink, width=3)
        draw.line([(260, y - 48), (260, y - 12)], fill=ink, width=3)
        draw.line([(400, y - 30), (400, y - 5)], fill=ink, width=3)
        draw.ellipse([150, y - 42, 162, y - 30], fill=ink)
        draw.ellipse([300, y - 55, 312, y - 43], fill=ink)
        draw.ellipse([450, y + 8, 462, y + 20], fill=ink)
        draw.arc([520, y - 90, 620, y - 20], start=40, end=200, fill=ink, width=4)
        return img

    def _make_arabic_text_if_possible(self):
        try:
            from PIL import ImageFont
        except ImportError:
            return None
        font = None
        # Prefer real Arabic fonts only — Latin fallbacks render Arabic poorly and
        # can look like discrete Latin-like glyphs in tests.
        for path in (
            r"C:\Windows\Fonts\arabic.ttf",
            r"C:\Windows\Fonts\arabtype.ttf",
            r"C:\Windows\Fonts\Candarab.ttf",
            r"C:\Windows\Fonts\GARABD.TTF",
            r"C:\Windows\Fonts\tradbdo.ttf",
            r"C:\Windows\Fonts\trado.ttf",
        ):
            if not os.path.isfile(path):
                continue
            try:
                font = ImageFont.truetype(path, 54)
                probe = Image.new("RGB", (80, 80), (255, 255, 255))
                ImageDraw.Draw(probe).text((8, 8), "ب", fill=(0, 0, 0), font=font)
                break
            except OSError:
                font = None
                continue
        if font is None:
            return None
        img = Image.new("RGB", (640, 240), (252, 250, 245))
        draw = ImageDraw.Draw(img)
        try:
            draw.text((40, 80), "بسم الله الرحمن", fill=(10, 10, 10), font=font)
        except Exception:
            return None
        return img

    def _make_single_latin_letter(self, letter: str = "A") -> Image.Image:
        img = Image.new("RGB", (400, 400), (250, 248, 242))
        draw = ImageDraw.Draw(img)
        try:
            from PIL import ImageFont

            font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 220)
        except OSError:
            try:
                from PIL import ImageFont

                font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 200)
            except OSError:
                font = None
        draw.text((110, 70), letter, fill=(10, 10, 10), font=font)
        return img

    def _make_latin_word(self, word: str = "HELLO") -> Image.Image:
        img = Image.new("RGB", (560, 220), (252, 250, 245))
        draw = ImageDraw.Draw(img)
        try:
            from PIL import ImageFont

            font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 72)
        except OSError:
            font = None
        draw.text((40, 70), word, fill=(10, 10, 10), font=font)
        return img

    def _make_latin_alphabet(self) -> Image.Image:
        img = Image.new("RGB", (520, 220), (250, 248, 242))
        draw = ImageDraw.Draw(img)
        try:
            from PIL import ImageFont

            font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 42)
        except OSError:
            font = None
        draw.text((24, 70), "ABCDEFGHIJKLM", fill=(15, 15, 15), font=font)
        draw.text((24, 130), "NOPQRSTUVWXYZ", fill=(15, 15, 15), font=font)
        return img

    def _make_decorative_latin_sentence(self) -> Image.Image:
        # Mixed-size English typography poster (user false-accept as Naskhi).
        img = Image.new("RGB", (720, 420), (248, 246, 240))
        draw = ImageDraw.Draw(img)
        try:
            from PIL import ImageFont

            small = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 28)
            medium = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 44)
            large = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 64)
        except OSError:
            small = medium = large = None
        draw.text((80, 40), "the true", fill=(30, 30, 30), font=small)
        draw.text((40, 90), "DIFFERENCE", fill=(15, 15, 15), font=large)
        draw.text((70, 180), "BETWEEN", fill=(20, 20, 20), font=medium)
        draw.text((50, 250), "LETTERING", fill=(15, 15, 15), font=large)
        draw.text((60, 330), "TYPOGRAPHY & Calligraphy", fill=(25, 25, 25), font=medium)
        return img

    def _make_person(self) -> Image.Image:
        img = Image.new("RGB", (400, 400), (210, 170, 140))
        draw = ImageDraw.Draw(img)
        draw.ellipse([120, 60, 280, 240], fill=(240, 200, 170))
        draw.rectangle([140, 240, 260, 400], fill=(40, 60, 120))
        return img

    def _make_animal(self) -> Image.Image:
        img = Image.new("RGB", (400, 400), (120, 160, 80))
        draw = ImageDraw.Draw(img)
        draw.ellipse([80, 120, 320, 320], fill=(160, 110, 60))
        draw.ellipse([150, 150, 200, 200], fill=(20, 20, 20))
        draw.ellipse([220, 150, 270, 200], fill=(20, 20, 20))
        return img

    def _make_plant(self) -> Image.Image:
        img = Image.new("RGB", (400, 400), (90, 160, 50))
        draw = ImageDraw.Draw(img)
        for idx in range(40):
            x = (idx * 37) % 360
            y = (idx * 53) % 360
            draw.ellipse([x, y, x + 40, y + 20], fill=(40, 150, 60))
        return img

    def _make_ocean(self) -> Image.Image:
        img = Image.new("RGB", (400, 400), (30, 110, 190))
        draw = ImageDraw.Draw(img)
        for y in range(0, 400, 18):
            shade = 140 + (y % 40)
            draw.rectangle([0, y, 400, y + 12], fill=(20, shade, 210))
        draw.ellipse([40, 40, 120, 90], fill=(250, 250, 220))
        return img


class KhatDetectionThresholdTests(unittest.TestCase):
    def test_confirmed_khat_at_high_confidence(self):
        result = resolve_detection(0.90, CONFIG)
        self.assertEqual(result["input_status"], "khat")
        self.assertEqual(result["detection_status"], "confirmed_khat")
        self.assertTrue(result["stage2_allowed"])
        self.assertFalse(result["manual_review_required"])

    def test_moderate_khat_at_74_percent(self):
        """74.49% should continue classification, not reject."""
        result = resolve_detection(0.7449, CONFIG)
        self.assertEqual(result["input_status"], "khat")
        self.assertEqual(result["detection_status"], "moderate_khat")
        self.assertTrue(result["stage2_allowed"])
        self.assertTrue(result["manual_review_required"])
        self.assertIn("Moderate Confidence", result["title"])

    def test_moderate_khat_at_threshold_boundary(self):
        result = resolve_detection(0.70, CONFIG)
        self.assertEqual(result["input_status"], "khat")
        self.assertTrue(result["stage2_allowed"])

    def test_borderline_khat_is_rejected(self):
        """Below accept (e.g. 64–69%) must not receive a class label."""
        result = resolve_detection(0.699, CONFIG)
        self.assertEqual(result["input_status"], "non_khat")
        self.assertEqual(result["detection_status"], "borderline_khat")
        self.assertFalse(result["stage2_allowed"])
        self.assertEqual(result["detection_decision"], "rejected")
        self.assertIn("Rejected", result["title"])

    def test_uncertain_khat_is_rejected(self):
        result = resolve_detection(0.55, CONFIG)
        self.assertEqual(result["input_status"], "non_khat")
        self.assertEqual(result["detection_status"], "uncertain_khat")
        self.assertFalse(result["stage2_allowed"])
        self.assertEqual(result["detection_decision"], "rejected")

    def test_uncertain_force_classify(self):
        result = resolve_detection(0.55, CONFIG, force_classify=True)
        self.assertEqual(result["input_status"], "uncertain")
        self.assertTrue(result["stage2_allowed"])
        self.assertTrue(result["manual_review_required"])
        self.assertEqual(result["detection_decision"], "continue_caution")

    def test_non_khat_below_reject_threshold(self):
        result = resolve_detection(0.49, CONFIG)
        self.assertEqual(result["input_status"], "non_khat")
        self.assertFalse(result["stage2_allowed"])
        self.assertEqual(result["detection_decision"], "rejected")

    def test_people_photo_band_rejected(self):
        """Regression: 64.39% people photo must be rejected, not labeled Diwani Jali."""
        result = resolve_detection(0.6439, CONFIG)
        self.assertEqual(result["input_status"], "non_khat")
        self.assertFalse(result["stage2_allowed"])
        self.assertFalse(result["is_khat"])

    def test_old_strict_threshold_would_have_rejected_74_percent(self):
        """Document regression: 0.80 accept would wrongly block 74% Khat."""
        strict = resolve_detection(0.7449, {**CONFIG, "KHAT_ACCEPT_THRESHOLD": 0.80})
        self.assertNotEqual(strict["input_status"], "khat")


if __name__ == "__main__":
    unittest.main()
