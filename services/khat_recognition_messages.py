"""Pesan konsisten untuk validasi input dan klasifikasi khat."""

from __future__ import annotations

from typing import Dict, List, Optional


STATUS_LABEL = "Khat Tidak Dikenali"
STATUS_LABEL_SHORT = "Tidak Dikenali"

# --- Pesan validasi input (aturan bisnis) ---
MSG_NOT_ARABIC_IMAGE = "Gambar bukan merupakan tulisan Arab."
MSG_NOT_ARABIC_SCRIPT = "Input bukan merupakan tulisan Arab."
MSG_OUTSIDE_TRAINED_CLASSES = (
    "Tulisan Arab terdeteksi, tetapi jenis khat tidak termasuk dalam dataset pelatihan."
)
MSG_LOW_DATASET_SIMILARITY = (
    "Tulisan Arab tidak memiliki kemiripan yang cukup dengan data pelatihan "
    "sehingga tidak dapat diklasifikasikan."
)

REJECTION_NOT_ARABIC_IMAGE = "not_arabic_image"
REJECTION_NOT_ARABIC_SCRIPT = "not_arabic_script"
REJECTION_OUTSIDE_CLASSES = "outside_trained_classes"
REJECTION_LOW_SIMILARITY = "low_dataset_similarity"


def user_facing_class_list(display_list: str) -> str:
    return display_list or "Naskhi, Diwani, Diwani Jali, Tsuluts"


def primary_message(display_list: str = "") -> str:
    _ = display_list
    return MSG_OUTSIDE_TRAINED_CLASSES


def detail_message(display_list: str) -> str:
    classes = user_facing_class_list(display_list)
    return (
        f"Sistem hanya mengklasifikasikan Khat Naskhi, Khat Tsuluts, Khat Diwani, "
        f"dan Khat Diwani Jali. Kelas yang didukung: {classes}."
    )


def flash_message(display_list: str = "") -> str:
    _ = display_list
    return MSG_OUTSIDE_TRAINED_CLASSES


def flash_for_prediction(prediction: Optional[Dict] = None, display_list: str = "") -> str:
    """Pilih flash message sesuai jenis penolakan prediksi."""
    prediction = prediction or {}
    code = prediction.get("rejection_code") or ""
    message = (prediction.get("message") or prediction.get("rejection_reason") or "").strip()
    if code == REJECTION_NOT_ARABIC_IMAGE or message == MSG_NOT_ARABIC_IMAGE:
        return MSG_NOT_ARABIC_IMAGE
    if code == REJECTION_NOT_ARABIC_SCRIPT or message == MSG_NOT_ARABIC_SCRIPT:
        return MSG_NOT_ARABIC_SCRIPT
    if code == REJECTION_LOW_SIMILARITY or message == MSG_LOW_DATASET_SIMILARITY:
        return MSG_LOW_DATASET_SIMILARITY
    if prediction.get("input_status") == "unrecognized":
        return MSG_OUTSIDE_TRAINED_CLASSES
    if message in {
        MSG_NOT_ARABIC_IMAGE,
        MSG_NOT_ARABIC_SCRIPT,
        MSG_OUTSIDE_TRAINED_CLASSES,
        MSG_LOW_DATASET_SIMILARITY,
    }:
        return message
    if prediction.get("input_status") in ("non_khat", "uncertain"):
        return message or MSG_NOT_ARABIC_IMAGE
    return flash_message(display_list)


def rejection_summary(reasons: List[str]) -> str:
    if not reasons:
        return MSG_OUTSIDE_TRAINED_CLASSES
    for reason in reasons:
        if reason in {
            MSG_NOT_ARABIC_IMAGE,
            MSG_NOT_ARABIC_SCRIPT,
            MSG_OUTSIDE_TRAINED_CLASSES,
            MSG_LOW_DATASET_SIMILARITY,
        }:
            return reason
    return " ".join(reasons)


def build_unrecognized_payload(
    *,
    display_list: str,
    reasons: Optional[List[str]] = None,
    rejection_code: str = REJECTION_OUTSIDE_CLASSES,
    message: Optional[str] = None,
) -> Dict[str, str]:
    reasons = reasons or []
    summary = rejection_summary(reasons)
    final_message = message or (
        MSG_LOW_DATASET_SIMILARITY
        if rejection_code == REJECTION_LOW_SIMILARITY
        else MSG_OUTSIDE_TRAINED_CLASSES
    )
    return {
        "input_status": "unrecognized",
        "status_label": STATUS_LABEL,
        "message": final_message,
        "detail_message": detail_message(display_list),
        "rejection_reason": summary if summary in {
            MSG_NOT_ARABIC_IMAGE,
            MSG_NOT_ARABIC_SCRIPT,
            MSG_OUTSIDE_TRAINED_CLASSES,
            MSG_LOW_DATASET_SIMILARITY,
        } else final_message,
        "rejection_code": rejection_code,
        "stage2_message": (
            f"Sistem hanya mengenali khat dari kelas dataset training: "
            f"{user_facing_class_list(display_list)}. "
            "Input di luar kelas tersebut atau di bawah ambang kemiripan/keyakinan ditolak."
        ),
        "flash_message": final_message,
    }


def reason_invalid_class(display_list: str = "") -> str:
    _ = display_list
    return MSG_OUTSIDE_TRAINED_CLASSES


def reason_low_confidence(conf_pct: float = 0.0, min_conf_pct: float = 85.0) -> str:
    _ = conf_pct
    _ = min_conf_pct
    return MSG_LOW_DATASET_SIMILARITY


def reason_low_margin(margin_pct: float = 0.0, min_margin_pct: float = 0.0) -> str:
    _ = margin_pct
    _ = min_margin_pct
    return MSG_LOW_DATASET_SIMILARITY


def reason_low_similarity() -> str:
    return MSG_LOW_DATASET_SIMILARITY


def unrecognized_from_system_error(
    display_list: str = "",
    exc: Optional[BaseException] = None,
) -> Dict[str, str]:
    """User-facing unrecognized payload for internal/path errors during classification."""
    reasons = [MSG_OUTSIDE_TRAINED_CLASSES]
    payload = build_unrecognized_payload(display_list=display_list, reasons=reasons)
    if exc is not None and is_path_mount_error(exc):
        payload["rejection_reason"] = MSG_OUTSIDE_TRAINED_CLASSES
    return payload


def is_path_mount_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "mount" in text and "start on mount" in text


def should_treat_as_unrecognized(exc: BaseException) -> bool:
    if is_path_mount_error(exc):
        return True
    text = str(exc).lower()
    return "relpath" in text or "different drives" in text
