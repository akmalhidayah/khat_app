"""Tests for dataset fusion and characteristic shape prior."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw

from services.dataset_alignment_service import (
    align_prediction_with_dataset,
    apply_characteristic_shape_prior,
    fuse_model_and_dataset_scores,
)


class _Cfg(dict):
    pass


class DatasetAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="align_test_")
        self.cfg = _Cfg(
            {
                "FUSION_MODEL_WEIGHT": 0.72,
                "FUSION_UNCERTAIN_MODEL_WEIGHT": 0.45,
                "FUSION_STRONG_MODEL_WEIGHT": 0.88,
                "FUSION_UNCERTAIN_CONFIDENCE": 0.58,
                "FUSION_UNCERTAIN_MARGIN": 0.12,
                "FUSION_KEEP_MODEL_MIN_CONFIDENCE": 0.38,
                "FUSION_KEEP_MODEL_MIN_MARGIN": 0.08,
                "STAGE2_MIN_CONFIDENCE": 0.70,
                "STAGE2_MIN_MARGIN": 0.15,
                "DATASET_ALIGN_STRONG_SIMILARITY": 75,
                "DATASET_ALIGN_OVERRIDE_GAP": 12,
                "DATASET_ROOT": self.tmp,
                "DATASET_SIMILARITY_CACHE_PATH": os.path.join(self.tmp, "sim_cache.json"),
            }
        )

    def tearDown(self):
        for name in os.listdir(self.tmp):
            path = os.path.join(self.tmp, name)
            if os.path.isfile(path):
                os.remove(path)
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

    def _blank_image(self, name: str = "blank.jpg") -> str:
        path = os.path.join(self.tmp, name)
        Image.new("RGB", (120, 120), (245, 240, 230)).save(path, quality=90)
        return path

    def test_align_forwards_kept_model_decision(self):
        scores = {
            "naskhi": 0.55,
            "diwani_jali": 0.20,
            "diwani": 0.15,
            "tsuluts": 0.10,
        }
        labels = list(scores.keys())
        path = self._blank_image()
        # Force clear-model path by stubbing similarity empty via missing index —
        # clear_model returns kept_model_decision before needing real sims when
        # conf/margin meet thresholds. With blank image sim may be empty.
        result = fuse_model_and_dataset_scores(scores, labels, path, self.cfg)
        self.assertTrue(result.get("kept_model_decision"), result)
        wrapped = align_prediction_with_dataset(scores, labels, path, self.cfg)
        self.assertTrue(wrapped.get("kept_model_decision"), wrapped)
        self.assertEqual(wrapped.get("predicted_class"), "naskhi")

    def test_characteristic_prior_leaves_strong_winner(self):
        scores = {
            "naskhi": 0.70,
            "diwani_jali": 0.12,
            "diwani": 0.10,
            "tsuluts": 0.08,
        }
        path = self._blank_image("strong.jpg")
        out = apply_characteristic_shape_prior(scores, path, list(scores.keys()))
        self.assertEqual(max(out, key=out.get), "naskhi")
        self.assertAlmostEqual(out["naskhi"], scores["naskhi"], places=2)

    def test_characteristic_prior_helps_ambiguous_low_ornament_toward_naskhi(self):
        # Sparse horizontal strokes on paper → Naskhi-like (low ornament).
        img = Image.new("RGB", (256, 256), (248, 244, 236))
        draw = ImageDraw.Draw(img)
        for y in (70, 120, 170, 220):
            draw.arc([30, y - 20, 226, y + 28], start=200, end=340, fill=(20, 20, 20), width=3)
            draw.line([(40, y), (210, y + 4)], fill=(15, 15, 15), width=2)
        path = os.path.join(self.tmp, "naskhi_like.jpg")
        img.save(path, quality=92)

        scores = {
            "naskhi": 0.34,
            "diwani_jali": 0.30,
            "diwani": 0.20,
            "tsuluts": 0.16,
        }
        out = apply_characteristic_shape_prior(scores, path, list(scores.keys()), max_boost=0.18)
        self.assertEqual(max(out, key=out.get), "naskhi", out)
        self.assertGreaterEqual(out["naskhi"], scores["naskhi"])


if __name__ == "__main__":
    unittest.main()
