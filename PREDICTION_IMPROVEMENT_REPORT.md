# Prediction Pipeline V2 — Improvement Report

## Root causes confirmed

1. An unavailable binary detector returned `is_khat=True` and probability `1.0`.
2. The Stage 2 gate lowered its post-Stage-1 defaults to roughly 0.32 confidence
   and 0.02 margin, then allowed confidence alone to short-circuit acceptance.
3. Raw, fused, and best-single confidence/margin were combined with `max()`.
4. The object named `raw_scores` was mutated by pair rules before being returned.
5. Stage 2 called the content gate again instead of reusing the Stage 1 trace.
6. Model output cardinality was not checked immediately after cached model load.
7. Inference preprocessing existed in several variants and lacked numerical
   diagnostics.

## Changes

- Added an immutable production class contract and strict mapping validator.
- Added model output-shape validation at load time.
- Added one aspect-preserving local classifier preprocessing function with EXIF
  transpose, RGB conversion, float32/finite validation, and diagnostics.
- Changed missing/failed binary detector output to uncertain with null
  probabilities and mandatory manual review.
- Added structured Stage 1 decision/evidence/feature fields while preserving all
  legacy response keys.
- Made raw classifier evidence authoritative for Stage 2; refinement remains
  diagnostic/advisory.
- Required both confidence and top-two margin; removed soft similarity salvage.
- Preserved immutable `raw_model_scores` and exposed `refined_scores`.
- Reused the Stage 1 trace in Stage 2 instead of recalculating the content gate.
- Added explicit `confirmed_class`, `probable_class`, `uncertain_class`, and
  `abstained` response diagnostics.
- Added manifest audit and read-only exact/dHash leakage-report scripts.
- Added tests for mapping, preprocessing, unavailable detector, hard rejection,
  low-margin abstention, and non-max-merged scores.

## Evaluation status

No production-accuracy increase is claimed. The checkout has no local dataset
split and the active model binary is disabled/missing; the host also lacks Flask,
Pillow, NumPy, TensorFlow and pytest. Consequently:

- Python syntax validation can be run and is reported separately.
- Model/corpus metrics cannot be produced honestly on this host.
- The leakage report has status `not_run_no_dataset_files`.
- The real-image integration manifest test skips explicitly when fixtures are
  unavailable.

The previous stored test accuracy (75.36%) is historical metadata, not a V2 result.

## Compatibility and risks

Existing response keys remain. New abstention fields are additive. Records that
previously received a low-margin class can now have no final class; callers must
treat this as an intentional safety decision. Deploy first in shadow/audit mode
if business workflow currently assumes every khat-like input receives a label.

Strict mapping checks can expose old model bundles whose metadata is incomplete
or whose output order differs. Such a failure is intentional and safer than
silently assigning the wrong style.

## Required validation on the model host

```bash
python -m pytest -q tests/test_khat_detector.py tests/test_prediction_pipeline_v2.py
python -m pytest -q
python scripts/audit_prediction_pipeline.py \
  --manifest tests/fixtures/prediction_manifest.json \
  --output model/prediction_audit
python scripts/find_dataset_duplicates.py \
  --dataset dataset \
  --output model/dataset_leakage_report
```

Calibrate Stage 2 thresholds from the validation manifest, grouped by
`source_group`; do not tune against the held-out test split.

