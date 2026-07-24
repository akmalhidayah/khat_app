# Prediction Pipeline Audit

## Scope and baseline

This audit describes the code on branch `server-stable-before-cursor` before the
prediction-pipeline-v2 changes. The production contract is four local classifier
outputs in this exact order:

`naskhi`, `diwani`, `diwani_jali`, `tsuluts`.

Syntax compilation passed before this work. The Python 3.14 host did not have the
project dependencies installed, so the baseline test discovery stopped at import
time (19 import errors; 4 dependency-free modules passed).

## Actual request path

1. `routes/classification_routes.py` validates the uploaded file, stores it, and
   calls `prediction_service.predict_image`.
2. The optional validation package checks file integrity/resolution, OCR/QR and
   forbidden objects. Missing OCR/YOLO generally fails open to the heuristic gate.
3. `detect_khat` calls `assess_calligraphy_content`.
4. The content gate derives paper, saturation, edge, ink, scene, Haar-face/skin,
   connected-component, baseline and Latin-template features.
5. A hard Latin/photo decision blocks Stage 2, including when force-classify is
   requested.
6. If present, the binary khat/non-khat model receives a padded 224x224 image.
   Before this change, an unavailable detector returned khat probability 1.0.
7. `_classify_khat_type` selects Teachable Machine, one Keras classifier, or the
   local/external ensemble.
8. The four-class model emits probabilities. The local path can then apply
   calibration, contested-pair rules, dataset pHash fusion, and shape priors.
9. `refine_class_probabilities` temperature-softens and optionally blends dataset
   similarity for display.
10. The distribution/OOD validator may inspect an embedding and MSP.
11. `_class_recognition_gate` decides whether to label or abstain.
12. The route persists the returned decision, probabilities and diagnostics.

## Stage audit

| Stage | Input/output | Threshold/fallback | Principal risks |
|---|---|---|---|
| Upload | FileStorage → stored RGB image | extension, image verification, upload limit | decompression bombs and invalid content need bounded decoding |
| Validation package | image → accepted/rejected trace | OCR/YOLO env thresholds; unavailable tools fail open | duplicate heuristic calls; false acceptance when optional tools are missing |
| Content features | image → feature snapshot | numerous literal edge/paper/color/layout cutoffs | scattered thresholds; synthetic photo rules can overfit |
| Script gate | features → Latin/Arabic evidence | template hits, cursive/baseline/layout evidence | decorative Latin and fragmented Arabic overlap |
| Photo gate | features → photo evidence | Haar+skin, blob, scene/color evidence | grayscale people conflict with monochrome calligraphy |
| Stage 1 combine | script+photo evidence → explicit status | hard Latin/photo; otherwise confirmed/likely/uncertain | old boolean response hid conflicting evidence |
| Binary detector | 224x224 batch → khat probability | accept .70, borderline .65, reject .50 | old unavailable fallback asserted 1.0 certainty |
| Mapping | output index → class slug | four unique contiguous indexes | external/local mappings could be silently mixed |
| Preprocessing | image → `(1,224,224,3)` float32 | EXIF, RGB, trim/pad, architecture transform | external direct resize distorted aspect; architecture metadata can be wrong |
| Four-class model | batch → raw scores | model softmax/calibration | confidence is not correctness; wrong mapping is catastrophic |
| Ensemble | component scores → fused raw scores | configurable local/external weights | “best single” confidence was later max-merged |
| Alignment | raw scores + pHash → fused scores | weak-model and similarity thresholds | pHash can flip style due to border/layout, not calligraphic identity |
| Refinement | fused scores → display scores | temperature 2.5, similarity blend .35 | display distribution can conceal raw uncertainty |
| Stage 2 | raw/refined evidence → class/abstain | formerly ~.35 confidence and ~.02 margin after Stage 1 | low-margin labels; `max()` across sources inflated acceptance |
| Persistence | result → relational rows/audit | broad exception fallbacks | diagnostics may be lost while prediction still succeeds |

## Conflicts found

- `STAGE2_MIN_CONFIDENCE_AFTER_STAGE1=0.32` and margin `0.02` conflict with
  the global .70/.05 gate and allow near-ties.
- `_class_recognition_gate` used the maximum of fused, raw and best-single
  confidence/margin instead of evaluating one declared source.
- Stage 2 called `assess_calligraphy_content` again although Stage 1 had already
  produced the result.
- pair correction and shape/similarity fusion mutated the variable called
  `raw_scores`; the true model output was no longer preserved.
- local, external and Teachable Machine mappings were resolved through shared
  metadata paths. A model-specific mapping must be validated before inference.
- EfficientNetB0 in modern Keras includes rescaling internally; its
  `preprocess_input` is a pass-through. External TM-style H5 expects `[-1,1]`.
  Architecture selection therefore has to be explicit and diagnosed.

## Decision policy

- Hard evidence for Latin/photo/non-script produces `rejected_input`.
- Missing binary detector is `uncertain`, never probability 1.0.
- Raw classifier scores remain immutable and primary.
- Refined/similarity scores are advisory and cannot force a conflicting class.
- Confidence and top-two margin are tested separately.
- Insufficient evidence yields `uncertain_class`, `abstained=true`, and no final
  class. Existing response keys remain present for template/database compatibility.
- Threshold defaults are conservative configuration values and must be calibrated
  on a source-grouped validation manifest before being described as optimal.

