"""Smoke tests for evaluation pipeline helpers."""

import unittest

from services.evaluation_pipeline_service import (
    _local_tta_variant_names,
    _remap_probability_row,
)
import numpy as np


class EvaluationPipelineHelperTests(unittest.TestCase):
    def test_tta_variants_match_live_family_without_hflip(self):
        names = _local_tta_variant_names(True)
        self.assertIn("orig", names)
        self.assertIn("bright", names)
        self.assertIn("contrast", names)
        self.assertIn("rot3", names)
        self.assertNotIn("hflip", names)

    def test_remap_probability_row_preserves_mass(self):
        row = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
        source = ["diwani_jali", "tsuluts", "diwani", "naskhi"]
        target = ["naskhi", "diwani", "diwani_jali", "tsuluts"]
        out = _remap_probability_row(row, source, target)
        self.assertAlmostEqual(float(out.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(out[2]), 0.4, places=6)  # diwani_jali
        self.assertAlmostEqual(float(out[1]), 0.2, places=6)  # diwani


if __name__ == "__main__":
    unittest.main()
