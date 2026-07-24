"""Tests for calibrator class-order alignment."""

import unittest

import numpy as np

from services.calibration_service import apply_calibrator, _reorder_probability_columns


class CalibratorAlignmentTests(unittest.TestCase):
    def test_reorder_probability_columns(self):
        probs = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64)
        source = ["diwani_jali", "tsuluts", "diwani", "naskhi"]
        target = ["naskhi", "diwani", "diwani_jali", "tsuluts"]
        out = _reorder_probability_columns(probs, source, target)
        np.testing.assert_allclose(out[0], [0.4, 0.3, 0.1, 0.2], atol=1e-6)

    def test_apply_calibrator_remaps_bias_to_model_order(self):
        # Model order (external): diwani_jali, tsuluts, diwani, naskhi
        # Calibrator order (canonical): naskhi, diwani, diwani_jali, tsuluts
        probs = np.array([[0.40, 0.25, 0.20, 0.15]], dtype=np.float32)
        calibrator = {
            "class_order": ["naskhi", "diwani", "diwani_jali", "tsuluts"],
            "temperature": 1.0,
            "matrix": None,
            "bias": np.array([0.0, 0.0, 2.0, -2.0], dtype=np.float32),  # boost jali, cut tsuluts
            "borderline_margin": 0.0,
        }
        model_order = ["diwani_jali", "tsuluts", "diwani", "naskhi"]
        out = apply_calibrator(probs, calibrator, model_class_order=model_order)[0]
        # After remapped bias, diwani_jali (index 0) should rise vs tsuluts (index 1).
        self.assertGreater(out[0], out[1])
        self.assertAlmostEqual(float(out.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
