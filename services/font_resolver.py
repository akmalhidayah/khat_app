"""Cached cross-platform TrueType font resolution for Latin diagnostics."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

REGULAR_LATIN_FONT_CANDIDATES: Tuple[str, ...] = (
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSans-Regular.ttf",
)

BOLD_LATIN_FONT_CANDIDATES: Tuple[str, ...] = (
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSans-Bold.ttf",
)


@lru_cache(maxsize=2)
def resolve_latin_font_paths(bold: bool = False) -> Tuple[Path, ...]:
    """Return existing known font paths without recursive filesystem searches."""
    candidates = BOLD_LATIN_FONT_CANDIDATES if bold else REGULAR_LATIN_FONT_CANDIDATES
    return tuple(Path(candidate) for candidate in candidates if Path(candidate).is_file())


@lru_cache(maxsize=64)
def load_latin_font(size: int, bold: bool = False):
    """Load the first usable FreeType font, or return ``None`` explicitly."""
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in resolve_latin_font_paths(bold):
        try:
            font = ImageFont.truetype(str(path), int(size))
        except OSError:
            continue
        if isinstance(font, ImageFont.FreeTypeFont):
            return font
    return None


def latin_font_diagnostics() -> dict:
    regular = resolve_latin_font_paths(False)
    bold = resolve_latin_font_paths(True)
    return {
        "regular": [str(path) for path in regular],
        "bold": [str(path) for path in bold],
        "font_count": len(set(regular + bold)),
    }

