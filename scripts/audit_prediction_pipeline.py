#!/usr/bin/env python3
"""Manifest-driven, abstention-aware prediction audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLASS_NAMES = ["naskhi", "diwani", "diwani_jali", "tsuluts"]
REQUIRED_FIELDS = {
    "path", "expected_content", "expected_class", "stage1_decision", "stage1_reason",
    "stage1_features", "binary_detector", "raw_model_scores", "refined_scores",
    "top1", "top2", "margin", "dataset_similarity", "final_status", "final_class",
    "processing_time_ms", "exception",
}


def _safe(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value


def _metrics(records):
    khat = [r for r in records if r["expected_content"] == "khat"]
    non = [r for r in records if r["expected_content"] != "khat"]
    accepted = [r for r in records if r["final_class"]]
    correct = [r for r in accepted if r["final_class"] == r["expected_class"]]
    false_accept = [r for r in non if r["final_class"]]
    false_reject = [r for r in khat if r["final_status"] == "rejected_input"]
    per_class = {}
    f1s, recalls = [], []
    for label in CLASS_NAMES:
        tp = sum(r["expected_class"] == label and r["final_class"] == label for r in records)
        fp = sum(r["expected_class"] != label and r["final_class"] == label for r in records)
        fn = sum(r["expected_class"] == label and r["final_class"] != label for r in records)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
        f1s.append(f1)
        recalls.append(recall)
    return {
        "samples": len(records),
        "coverage": len(accepted) / len(records) if records else 0.0,
        "selective_accuracy": len(correct) / len(accepted) if accepted else None,
        "uncertain": sum(r["final_status"] == "uncertain_class" for r in records),
        "false_acceptance_rate": len(false_accept) / len(non) if non else None,
        "false_rejection_rate": len(false_reject) / len(khat) if khat else None,
        "macro_f1": sum(f1s) / len(f1s),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "per_class": per_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="model/prediction_audit")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    from app import create_app
    from services.prediction_service import predict_image

    app = create_app()
    records = []
    with app.app_context():
        for item in manifest:
            started = time.perf_counter()
            record = {
                "path": item.get("path"), "expected_content": item.get("expected_content"),
                "expected_class": item.get("expected_class"), "source_group": item.get("source_group"),
                "exception": None,
            }
            try:
                result = predict_image(str((manifest_path.parent / item["path"]).resolve()))
                stage1 = result.get("stage1_trace") or {}
                raw = result.get("raw_model_scores") or {}
                refined = result.get("refined_scores") or result.get("scores") or {}
                ordered = sorted(raw.items(), key=lambda pair: pair[1], reverse=True)
                record.update({
                    "stage1_decision": stage1.get("decision_code") or result.get("detection_status"),
                    "stage1_reason": stage1.get("decision_reason") or result.get("stage1_message"),
                    "stage1_features": stage1.get("feature_snapshot") or {},
                    "binary_detector": {
                        "available": result.get("detector_available"),
                        "khat_probability": result.get("khat_probability"),
                        "non_khat_probability": result.get("non_khat_probability"),
                    },
                    "raw_model_scores": raw, "refined_scores": refined,
                    "top1": ordered[0] if ordered else None,
                    "top2": ordered[1] if len(ordered) > 1 else None,
                    "margin": result.get("top2_margin_pct"),
                    "dataset_similarity": result.get("similarity_scores"),
                    "final_status": result.get("final_status") or (
                        "rejected_input" if result.get("input_status") == "non_khat" else "uncertain_class"
                    ),
                    "final_class": result.get("final_class") or result.get("predicted_class"),
                })
            except Exception as exc:
                record.update({key: None for key in REQUIRED_FIELDS - record.keys()})
                record["final_status"] = "error"
                record["exception"] = f"{type(exc).__name__}: {exc}"
            record["processing_time_ms"] = round((time.perf_counter() - started) * 1000, 2)
            records.append(record)

    metrics = _metrics(records)
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    fields = sorted(set().union(*(r.keys() for r in records)))
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _safe(row.get(key)) for key in fields} for row in records)
    (output / "per_class_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "calibration_candidates.json").write_text(
        json.dumps({"status": "requires validation corpus", "records": len(records)}, indent=2),
        encoding="utf-8",
    )
    for name, predicate in {
        "errors_false_accept.csv": lambda r: r["expected_content"] != "khat" and r["final_class"],
        "errors_false_reject.csv": lambda r: r["expected_content"] == "khat" and r["final_status"] == "rejected_input",
        "errors_wrong_class.csv": lambda r: r["final_class"] and r["final_class"] != r["expected_class"],
    }.items():
        subset = [r for r in records if predicate(r)]
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: _safe(row.get(key)) for key in fields} for row in subset)
    # Portable matrix tables without adding sklearn.
    for filename, expected_key, predicted_key, labels in (
        ("stage1_confusion_matrix.csv", "expected_content", "stage1_decision",
         sorted({str(r["expected_content"]) for r in records} | {str(r["stage1_decision"]) for r in records})),
        ("stage2_confusion_matrix.csv", "expected_class", "final_class", CLASS_NAMES + ["None"]),
    ):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle); writer.writerow(["expected/predicted", *labels])
            for expected in labels:
                writer.writerow([expected, *[
                    sum(str(r.get(expected_key)) == expected and str(r.get(predicted_key)) == predicted for r in records)
                    for predicted in labels
                ]])
    (output / "summary.md").write_text(
        "# Prediction Audit\n\n"
        f"- Samples: {metrics['samples']}\n- Coverage: {metrics['coverage']:.4f}\n"
        f"- Selective accuracy: {metrics['selective_accuracy']}\n"
        f"- Macro F1: {metrics['macro_f1']:.4f}\n"
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.4f}\n"
        f"- Uncertain: {metrics['uncertain']}\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
