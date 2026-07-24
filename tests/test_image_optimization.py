"""Tests for image optimization helpers."""

import os
import tempfile

from PIL import Image

from services.image_optimization_service import optimize_image


def test_optimize_image_resizes_and_compresses():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "sample.png")
        output_path = os.path.join(temp_dir, "optimized")

        Image.new("RGB", (1200, 800), color=(180, 90, 40)).save(input_path, format="PNG")

        result = optimize_image(input_path, output_path, max_size=512, quality=80, output_format="JPEG")

        assert result["status"] == "optimized"
        assert result["width"] <= 512
        assert result["height"] <= 512
        assert result["optimized_size"] > 0
        assert result["optimized_size"] <= result["original_size"]
        assert os.path.isfile(result["output_path"])
