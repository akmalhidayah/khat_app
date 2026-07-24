"""Pre-CNN input validation pipeline for Arabic Khat classification.

Modules:
- validation.py — file / resolution checks
- arabic_script.py — EasyOCR Arabic detection
- object_detector.py — YOLOv8 forbidden-object gate
- feature_extractor.py — penultimate CNN embeddings
- similarity.py — cosine similarity vs training gallery
- ood_detector.py — Maximum Softmax Probability OOD
- pipeline.py — orchestration
"""

from services.validation_pipeline.pipeline import (
    pipeline_enabled,
    to_flask_rejection,
    validate_before_cnn,
    validate_distribution,
)

__all__ = [
    "pipeline_enabled",
    "validate_before_cnn",
    "validate_distribution",
    "to_flask_rejection",
]
