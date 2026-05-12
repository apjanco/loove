"""
Unicode Character Database (UCD) loader.

Downloads and caches two UCD files:
  - UnicodeData.txt  — character names and general categories
  - Scripts.txt      — script assignment for every code point

Both files use range-based notation so we build efficient range-lookup
structures instead of materialising a mapping for all 1.1 M code points.

Usage:
    scripts = load_scripts()
    script  = scripts.get(0x0915)   # → "Devanagari"

    names   = load_char_names()
    name    = names.get(0x0041)     # → "LATIN CAPITAL LETTER A"
"""
from __future__ import annotations

import bisect
import re
from pathlib import Path

import requests

_ROOT = Path(__file__).parents[2]
_CACHE_DIR = _ROOT / "data" / "ucd"

_UCD_URL     = "https://unicode.org/Public/UCD/latest/ucd/UnicodeData.txt"
_SCRIPTS_URL = "https://unicode.org/Public/UCD/latest/ucd/Scripts.txt"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _download(url: str, cache_path: Path) -> str:
    """Download a text file if not already cached, then return its contents."""
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    cache_path.write_text(resp.text, encoding="utf-8")
    return resp.text


# ---------------------------------------------------------------------------
# Script lookup
# ---------------------------------------------------------------------------

class ScriptRanges:
    """
    Efficient O(log n) script lookup backed by sorted ranges from Scripts.txt.

    Scripts.txt records look like:
        0000..001F    ; Common
        0041..005A    ; Latin
        0915          ; Devanagari
    """

    def __init__(self, ranges: list[tuple[int, int, str]]):
        # ranges is sorted by start code point
        self._starts = [r[0] for r in ranges]
        self._ranges = ranges

    def get(self, cp: int, default: str = "Unknown") -> str:
        idx = bisect.bisect_right(self._starts, cp) - 1
        if idx >= 0:
            start, end, script = self._ranges[idx]
            if start <= cp <= end:
                return script
        return default

    def __getitem__(self, cp: int) -> str:
        return self.get(cp)

    def __contains__(self, cp: int) -> bool:
        return self.get(cp) != "Unknown"


def load_scripts() -> ScriptRanges:
    """Parse Scripts.txt and return an efficient range-based lookup."""
    text = _download(_SCRIPTS_URL, _CACHE_DIR / "Scripts.txt")

    ranges: list[tuple[int, int, str]] = []
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        m = re.match(r"([0-9A-F]+)(?:\.\.([0-9A-F]+))?\s*;\s*(\S+)", line)
        if not m:
            continue
        start = int(m.group(1), 16)
        end   = int(m.group(2), 16) if m.group(2) else start
        script = m.group(3)
        ranges.append((start, end, script))

    ranges.sort(key=lambda r: r[0])
    return ScriptRanges(ranges)


# ---------------------------------------------------------------------------
# Character name lookup
# ---------------------------------------------------------------------------

def load_char_names() -> dict[int, str]:
    """
    Parse UnicodeData.txt and return a sparse dict of code point → name.

    Range markers (e.g. <CJK Ideograph, First>) are skipped — those code
    points do not have individual names in UnicodeData.txt.
    """
    text = _download(_UCD_URL, _CACHE_DIR / "UnicodeData.txt")

    names: dict[int, str] = {}
    for line in text.splitlines():
        if not line:
            continue
        parts = line.split(";")
        if len(parts) < 2:
            continue
        cp_str, name = parts[0], parts[1]
        if name.startswith("<"):
            continue  # Range marker
        try:
            names[int(cp_str, 16)] = name
        except ValueError:
            continue

    return names
