"""
Fertility analysis using the UDHR (Universal Declaration of Human Rights) corpus.

Fertility = tokens / characters (non-whitespace).
A lower fertility means the model tokenises the language efficiently — it has
learned multi-character subwords for it.  A high fertility (e.g. 4+ tokens per
character) means the model is falling back to individual bytes and has no
real language-level understanding of the script.

UDHR is freely available in ~500 languages and is the standard benchmark corpus
for tokenizer fertility analysis.  Files live at:
    https://unicode.org/udhr/d/udhr_{code}.txt

The index XML (maps language codes → file codes) is at the unicode-org/udhr
GitHub repo:
    https://raw.githubusercontent.com/unicode-org/udhr/main/index/index.xml

Index XML element format:
    <udhr f="hin" iso639-3="hin" iso15924="Deva" xml:lang="hi" ...>

We key our locale→file mapping on the xml:lang prefix (e.g. "hi") which aligns
with CLDR locale IDs.  If a locale has multiple UDHR translations we use the
first entry in document order (which is usually the most complete one).
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

import requests

_ROOT = Path(__file__).parents[2]
_CACHE_DIR = _ROOT / "data" / "udhr"
_INDEX_URL = (
    "https://raw.githubusercontent.com/unicode-org/udhr/main/index/index.xml"
)
_UDHR_TEXT_BASE = "https://unicode.org/udhr/d"

# Lazy-loaded index: locale_prefix (e.g. "hi") → UDHR file code (e.g. "hin")
_LOCALE_TO_FILE: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# UDHR index
# ---------------------------------------------------------------------------

def _load_index() -> dict[str, str]:
    """
    Return a mapping from CLDR locale prefix to UDHR file code.
    Cached in memory; the XML is cached on disk.
    """
    global _LOCALE_TO_FILE
    if _LOCALE_TO_FILE is not None:
        return _LOCALE_TO_FILE

    cache_path = _CACHE_DIR / "index.xml"
    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
    else:
        resp = requests.get(_INDEX_URL, timeout=30)
        resp.raise_for_status()
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(resp.text, encoding="utf-8")
        text = resp.text

    mapping: dict[str, str] = {}
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        _LOCALE_TO_FILE = mapping
        return mapping

    # xml:lang uses the XML namespace URI
    _XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

    for elem in root.iter("udhr"):
        file_code = elem.get("f") or elem.get("id", "")
        xml_lang  = elem.get(_XML_LANG, "")
        # xml:lang may be "hi", "zh-Hant", "sr-Latn" — use the primary subtag
        locale_prefix = xml_lang.split("-")[0] if xml_lang else ""
        if file_code and locale_prefix and locale_prefix not in mapping:
            mapping[locale_prefix] = file_code

    _LOCALE_TO_FILE = mapping
    return mapping


def get_available_locales() -> list[str]:
    """Return all locale prefixes for which a UDHR translation exists."""
    return sorted(_load_index().keys())


# ---------------------------------------------------------------------------
# UDHR text fetch
# ---------------------------------------------------------------------------

def _fetch_udhr_text(file_code: str) -> str | None:
    """Download and cache a UDHR plain-text file."""
    cache_path = _CACHE_DIR / f"{file_code}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")

    url = f"{_UDHR_TEXT_BASE}/udhr_{file_code}.txt"
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(resp.content)
        time.sleep(0.05)
        return resp.content.decode("utf-8", errors="replace")
    except requests.RequestException:
        return None


def _extract_body(text: str, max_chars: int = 4000) -> str:
    """
    Strip the UDHR file header and return up to `max_chars` characters of body.

    UDHR plain-text files begin with a header section (copyright, language info)
    before the actual Declaration text.  The body typically starts at "Preamble"
    or "Article 1" / a numbered article heading.
    """
    lines = text.splitlines()
    body_lines: list[str] = []
    in_header = True

    for line in lines:
        stripped = line.strip()
        if in_header:
            # Heuristic: body starts when we see a line that looks like an article
            # heading or the Preamble.  Header lines are often copyright notices,
            # blank, or contain "UDHR".
            if re.match(r"^(Preamble|Article|ARTICLE|\d+\.)\b", stripped):
                in_header = False
            elif stripped and not any(
                stripped.startswith(pfx)
                for pfx in ("©", "#", "UDHR", "Universal Declaration", "United Nations")
            ):
                in_header = False

        if not in_header and stripped:
            body_lines.append(stripped)

    body = " ".join(body_lines)
    return body[:max_chars]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_fertility(
    locale_id: str,
    tokenize_fn: Callable[[str], list[int]],
) -> dict | None:
    """
    Compute fertility metrics for a language using its UDHR translation.

    Args:
        locale_id:   CLDR locale prefix, e.g. "hi", "ar", "zh".
        tokenize_fn: A callable that accepts a text string and returns a list
                     of token IDs.  Should be the model's own tokenizer.

    Returns:
        A dict with fertility metrics, or None if no UDHR data is available.

        {
            "tokens_per_char":  float,  # lower is better
            "tokens_per_word":  float,  # lower is better
            "sample_chars":     int,    # non-whitespace chars used
            "sample_tokens":    int,
        }
    """
    index = _load_index()
    file_code = index.get(locale_id)
    if not file_code:
        return None

    raw_text = _fetch_udhr_text(file_code)
    if not raw_text:
        return None

    body = _extract_body(raw_text)
    if len(body) < 50:
        return None

    try:
        token_ids = tokenize_fn(body)
    except Exception:
        return None

    n_tokens = len(token_ids)
    n_chars  = sum(1 for ch in body if not ch.isspace())
    n_words  = len(body.split())

    if n_chars == 0 or n_words == 0:
        return None

    return {
        "tokens_per_char": round(n_tokens / n_chars, 4),
        "tokens_per_word": round(n_tokens / n_words, 4),
        "sample_chars":    n_chars,
        "sample_tokens":   n_tokens,
    }


def prefetch_all(locale_ids: list[str]) -> None:
    """
    Pre-download UDHR texts for the given locales.
    Safe to call multiple times — cached files are skipped.
    """
    index = _load_index()
    for locale_id in locale_ids:
        file_code = index.get(locale_id)
        if file_code:
            _fetch_udhr_text(file_code)
