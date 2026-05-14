"""
CLDR loader: download and parse CLDR locale XML to extract exemplar character sets.

Source: https://github.com/unicode-org/cldr  (release-47 tag)

Each locale XML file (e.g. common/main/hi.xml) contains an <exemplarCharacters>
element inside <characters>.  The content is in Unicode Set Notation:

    [क ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व श ष स ह]
    [\u0621-\u063A \u0641-\u064A]
    [a-z {ch} {dz}]

We parse this into a flat set of Unicode code points.

Multi-character sequences like {ch} are expanded to individual code points
(U+0063, U+0068). This is conservative: a model that can encode 'c' and 'h'
individually is counted as "capable" of representing the Welsh digraph 'ch'.
Grapheme-cluster–level scoring is a future enhancement.

Output: data/cldr/languages.json
{
  "hi": {
    "locale_id": "hi",
    "name":      "Hindi",
    "script":    "Deva",
    "exemplar_main":      [2325, 2326, ...],  // sorted list of ints
    "exemplar_auxiliary": [...]
  },
  ...
}
"""
from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from babel import Locale, UnknownLocaleError
try:
    from babel.core import locale_identifiers
except ImportError:
    from babel.localedata import locale_identifiers

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parents[2]
_CACHE_DIR = _ROOT / "data" / "cldr" / "raw"
_OUT_PATH = _ROOT / "data" / "cldr" / "languages.json"

_CLDR_BASE = (
    "https://raw.githubusercontent.com/unicode-org/cldr/release-47/common/main"
)
_REQUEST_DELAY = 0.05   # seconds between GitHub raw requests (be polite)


# ---------------------------------------------------------------------------
# Unicode Set Notation parser
# ---------------------------------------------------------------------------

def parse_unicode_set(s: str) -> set[int]:
    """
    Parse a CLDR Unicode Set Notation string into a set of Unicode code points.

    Handles:
      - Single characters separated by spaces: [a b c]
      - Ranges:  [a-z]  or  [\u0621-\u063A]
      - Multi-char sequences: [{ch} {dz}]  → expands to individual code points
      - \\uXXXX and \\UXXXXXXXX escapes
      - Negated sets [^ ...] → returns empty set (not relevant for exemplars)
    """
    s = s.strip()
    if not s.startswith("["):
        return set()

    # Strip outer brackets
    inner = s[1:]
    if inner.endswith("]"):
        inner = inner[:-1]
    inner = inner.strip()

    if inner.startswith("^"):
        return set()  # Negated sets not used in exemplar strings; skip.

    codepoints: set[int] = set()

    # 1. Extract multi-char sequences {xx} and expand to individual code points.
    for seq in re.findall(r"\{([^}]+)\}", inner):
        for ch in seq:
            codepoints.add(ord(ch))
    inner = re.sub(r"\{[^}]+\}", " ", inner)

    # 2. Replace \uXXXX / \UXXXXXXXX escape sequences with the actual characters.
    def _replace_escape(m: re.Match) -> str:
        return chr(int(m.group(1), 16))

    inner = re.sub(r"\\[uU]([0-9A-Fa-f]{4,8})", _replace_escape, inner)

    # 3. Split by whitespace; each token is either a range (x-y) or a single char.
    for token in inner.split():
        if not token:
            continue

        # Detect range: exactly "X-Y" where X and Y are single Unicode chars.
        # We look for a '-' that is neither the first nor last character.
        if len(token) >= 3 and "-" in token[1:-1]:
            dash = token.index("-", 1)
            before = token[:dash]
            after = token[dash + 1:]
            if len(before) == 1 and len(after) == 1:
                start, end = ord(before), ord(after)
                if start <= end:
                    for cp in range(start, end + 1):
                        codepoints.add(cp)
                continue
            # Fall through: treat as multi-char token.

        for ch in token:
            codepoints.add(ord(ch))

    return codepoints


# ---------------------------------------------------------------------------
# CLDR XML fetch + parse
# ---------------------------------------------------------------------------

def _fetch_locale_xml(locale_id: str) -> str | None:
    """Download (and cache on disk) the CLDR XML for a locale."""
    cache_file = _CACHE_DIR / f"{locale_id}.xml"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    url = f"{_CLDR_BASE}/{locale_id}.xml"
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(resp.text, encoding="utf-8")
        time.sleep(_REQUEST_DELAY)
        return resp.text
    except requests.RequestException as exc:
        print(f"  [cldr] Warning: could not fetch {locale_id}: {exc}")
        return None


def _extract_exemplars(xml_text: str) -> dict[str, set[int]]:
    """
    Parse exemplarCharacters elements from a CLDR locale XML string.
    Returns a dict keyed by tier name: "main", "auxiliary", "index", "punctuation".
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {}

    characters_elem = root.find(".//characters")
    if characters_elem is None:
        return {}

    result: dict[str, set[int]] = {}
    for elem in characters_elem.findall("exemplarCharacters"):
        tier = elem.get("type", "main")
        if elem.text:
            cps = parse_unicode_set(elem.text)
            if cps:
                result[tier] = cps

    return result


def _locale_display_name(locale_id: str) -> tuple[str, str]:
    """Return (english_name, script_code) for a locale via babel."""
    try:
        loc = Locale.parse(locale_id)
        # babel stores script as a script code like "Latn", "Deva", "Arab"
        script = str(loc.script) if loc.script else ""
        # get_display_name('en') returns the name in English
        name = str(loc.get_display_name("en") or locale_id)
        return name, script
    except (UnknownLocaleError, ValueError):
        return locale_id, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_language_database(
    locales: list[str] | None = None,
    force_refresh: bool = False,
) -> dict[str, dict]:
    """
    Build (or load from cache) the CLDR language database.

    Args:
        locales: Specific locale IDs to process.  If None, all base-language
                 locales known to babel are processed (no region/script suffixes).
        force_refresh: Re-download CLDR XML even if already cached.

    Returns:
        A dict  locale_id → language_entry  suitable for JSON serialisation.
    """
    if _OUT_PATH.exists() and not force_refresh and locales is None:
        return json.loads(_OUT_PATH.read_text(encoding="utf-8"))

    if locales is None:
        # Base-language locales only (e.g. "hi", not "hi_IN" or "hi_Latn")
        locales = sorted(
            lid for lid in locale_identifiers()
            if "_" not in lid and lid != "root"
        )

    db: dict[str, dict] = {}

    for locale_id in locales:
        xml_text = _fetch_locale_xml(locale_id)
        if xml_text is None:
            continue

        exemplars = _extract_exemplars(xml_text)
        if not exemplars.get("main"):
            # No usable exemplar data — skip this locale.
            continue

        name, script = _locale_display_name(locale_id)

        db[locale_id] = {
            "locale_id": locale_id,
            "name": name,
            "script": script,
            "exemplar_main": sorted(exemplars.get("main", set())),
            "exemplar_auxiliary": sorted(exemplars.get("auxiliary", set())),
        }

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cldr] Saved {len(db)} languages → {_OUT_PATH}")
    return db


def load_language_database() -> dict[str, dict]:
    """Load the previously built CLDR language database from disk."""
    if not _OUT_PATH.exists():
        raise FileNotFoundError(
            f"CLDR language database not found at {_OUT_PATH}.\n"
            "Run:  python scripts/fetch_cldr.py"
        )
    return json.loads(_OUT_PATH.read_text(encoding="utf-8"))
