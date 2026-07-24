"""Visual characteristics reference for Arabic Khat classes (informational only)."""

from typing import Any, Dict, List, Optional

CLASS_KEYS = ("diwani", "diwani_jali", "naskhi", "tsuluts")

CLASS_DISPLAY = {
    "diwani": "Diwani",
    "diwani_jali": "Diwani Jali",
    "naskhi": "Naskhi",
    "tsuluts": "Tsuluts",
}

KHAT_CHARACTERISTICS: Dict[str, Dict[str, Any]] = {
    "diwani": {
        "title": "Ciri Utama Khat Diwani",
        "description": (
            "Khat Diwani memiliki karakter tulisan yang lentur, elegan, dan mengalir. "
            "Bentuk hurufnya banyak menggunakan lengkungan halus, susunan huruf cenderung dekoratif, "
            "serta memiliki irama visual yang lembut."
        ),
        "characteristics": [
            "Bentuk huruf melengkung dan lentur.",
            "Komposisi tulisan mengalir dan elegan.",
            "Banyak lekukan halus.",
            "Dekoratif, tetapi tidak terlalu padat.",
            "Keterbacaan sedang karena bentuknya artistik.",
        ],
    },
    "diwani_jali": {
        "title": "Ciri Utama Khat Diwani Jali",
        "description": (
            "Khat Diwani Jali merupakan pengembangan dari Diwani yang lebih padat, rumit, dan penuh ornamen. "
            "Gaya ini memiliki susunan huruf yang kompleks, banyak titik hias, serta ruang kosong yang lebih sedikit."
        ),
        "characteristics": [
            "Tulisan sangat dekoratif dan kompleks.",
            "Komposisi lebih padat dibanding Diwani biasa.",
            "Banyak titik, ornamen, dan hiasan visual.",
            "Ruang kosong lebih sedikit.",
            "Kesan visual mewah, resmi, dan artistik.",
        ],
    },
    "naskhi": {
        "title": "Ciri Utama Khat Naskhi",
        "description": (
            "Khat Naskhi memiliki bentuk huruf yang rapi, jelas, dan mudah dibaca. "
            "Gaya ini sering digunakan dalam penulisan mushaf, buku, dan teks Arab panjang "
            "karena susunan hurufnya lebih teratur."
        ),
        "characteristics": [
            "Huruf rapi, sederhana, dan mudah dibaca.",
            "Bentuk tulisan cenderung horizontal dan teratur.",
            "Minim ornamen.",
            "Spasi antarhuruf dan antarbaris lebih jelas.",
            "Cocok untuk teks panjang dan penulisan formal.",
        ],
    },
    "tsuluts": {
        "title": "Ciri Utama Khat Tsuluts",
        "description": (
            "Khat Tsuluts memiliki karakter megah, tegas, dan monumental. "
            "Hurufnya cenderung besar, memiliki tarikan garis panjang, lengkungan luas, "
            "serta sering digunakan pada dekorasi masjid, judul, dan karya kaligrafi besar."
        ),
        "characteristics": [
            "Huruf besar, tinggi, dan dominan.",
            "Tarikan garis panjang dan tegas.",
            "Lengkungan besar dan artistik.",
            "Komposisi terlihat megah dan monumental.",
            "Dekoratif, tetapi tidak sepadat Diwani Jali.",
        ],
    },
}

DISCLAIMER_NOTE = (
    "Catatan: Ciri utama ini digunakan sebagai informasi pendukung. "
    "Keputusan akhir tetap mengacu pada hasil prediksi model, confidence score, "
    "validation status, dan manual review jika diperlukan."
)

MISMATCH_WARNING = (
    "Perhatian: Ciri utama yang ditampilkan mengikuti kelas hasil prediksi model. "
    "Karena hasil prediksi tidak sesuai dengan expected class, pengguna disarankan "
    "membandingkan ciri ini dengan label gambar secara manual."
)


def normalize_class_key(class_key: Optional[str]) -> Optional[str]:
    if not class_key:
        return None
    key = str(class_key).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "diwanijali": "diwani_jali",
        "diwani_jali": "diwani_jali",
        "thuluth": "tsuluts",
        "thuluts": "tsuluts",
        "suluth": "tsuluts",
    }
    return aliases.get(key, key)


def get_khat_characteristics(class_key: Optional[str]) -> Optional[Dict[str, Any]]:
    normalized = normalize_class_key(class_key)
    if not normalized or normalized not in KHAT_CHARACTERISTICS:
        return None
    data = KHAT_CHARACTERISTICS[normalized]
    return {
        "class_key": normalized,
        "class_display": CLASS_DISPLAY.get(normalized, normalized.replace("_", " ").title()),
        "title": data["title"],
        "description": data["description"],
        "characteristics": list(data["characteristics"]),
    }


def get_characteristics_catalog() -> Dict[str, Dict[str, Any]]:
    """Full catalog keyed by class slug (for JSON/JS export)."""
    return {
        key: get_khat_characteristics(key)
        for key in CLASS_KEYS
    }


def build_characteristics_context(
    predicted_class: Optional[str],
    expected_class: Optional[str] = None,
    validation_status: Optional[str] = None,
    filename_mismatch: bool = False,
) -> Optional[Dict[str, Any]]:
    predicted = get_khat_characteristics(predicted_class)
    if not predicted:
        return None

    is_mismatch = bool(
        filename_mismatch
        or validation_status == "Prediction Mismatch"
        or (
            expected_class
            and normalize_class_key(expected_class) != normalize_class_key(predicted_class)
        )
    )
    expected = get_khat_characteristics(expected_class) if is_mismatch and expected_class else None

    def _comparison_view(chars: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not chars:
            return None
        return {
            **chars,
            "short_description": chars["description"],
            "short_characteristics": chars["characteristics"][:3],
        }

    return {
        "card_title": "Ciri Utama Khat",
        "card_subtitle": "Informasi karakter visual berdasarkan hasil klasifikasi.",
        "predicted": predicted,
        "expected": expected,
        "predicted_comparison": _comparison_view(predicted),
        "expected_comparison": _comparison_view(expected),
        "is_mismatch": is_mismatch,
        "validation_status": validation_status,
        "predicted_display": predicted["class_display"],
        "expected_display": expected["class_display"] if expected else None,
        "disclaimer_note": DISCLAIMER_NOTE,
        "mismatch_warning": MISMATCH_WARNING if is_mismatch else None,
        "show_comparison": bool(is_mismatch and expected),
    }
