"""
Fertility analysis using the UDHR (Universal Declaration of Human Rights) corpus.

Fertility = tokens / characters (non-whitespace).
A lower fertility means the model tokenises the language efficiently — it has
learned multi-character subwords for it.  A high fertility (e.g. 4+ tokens per
character) means the model is falling back to individual bytes and has no
real language-level understanding of the script.

UDHR is freely available in ~500 languages and is the standard benchmark corpus
for tokenizer fertility analysis.  Files are hosted on GitHub as XML:
    https://raw.githubusercontent.com/eric-muller/udhr/main/data/udhr/udhr_{code}.xml

The index XML (maps language codes → file codes) lives at:
    https://raw.githubusercontent.com/eric-muller/udhr/main/data/udhr/index.xml

Index XML element format:
    <udhr f="hin" iso639-3="hin" iso15924="Deva" xml:lang="hi" ...>

We key our locale→file mapping on the xml:lang prefix (e.g. "hi") which aligns
with CLDR locale IDs.  If a locale has multiple UDHR translations we use the
first entry in document order (which is usually the most complete one).
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

import requests

_ROOT = Path(__file__).parents[2]
_CACHE_DIR = _ROOT / "data" / "udhr"
_INDEX_URL = (
    "https://raw.githubusercontent.com/eric-muller/udhr/main/data/udhr/index.xml"
)
_UDHR_XML_BASE = (
    "https://raw.githubusercontent.com/eric-muller/udhr/main/data/udhr"
)

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

    for elem in root.iter("udhr"):
        file_code = elem.get("f") or elem.get("id", "")
        # New index format uses bcp47 attribute (e.g. "hi", "zh-Hant", "sr-Latn")
        bcp47 = elem.get("bcp47", "")
        # Use the primary language subtag to align with CLDR locale keys
        locale_prefix = bcp47.split("-")[0] if bcp47 else ""
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

def _fetch_udhr_xml(file_code: str) -> str | None:
    """Download and cache a UDHR XML file."""
    cache_path = _CACHE_DIR / f"{file_code}.xml"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")

    url = f"{_UDHR_XML_BASE}/udhr_{file_code}.xml"
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


def _extract_body(xml_text: str, max_chars: int = 4000) -> str:
    """
    Extract article/paragraph text from a UDHR XML file.

    The UDHR XML schema has <preamble>, <article>, <para>, and <title>
    elements.  We collect all <para> and <title> text nodes, skipping the
    top-level <udhr> header attributes which carry metadata, not body text.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ""

    parts: list[str] = []
    # Tags are namespace-qualified: {http://efele.net/udhr}para etc.
    # Match by local name only so we don't hard-code the namespace URL.
    for elem in root.iter():
        local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if local in ("title", "para") and elem.text:
            stripped = elem.text.strip()
            if stripped:
                parts.append(stripped)

    body = " ".join(parts)
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

    raw_text = _fetch_udhr_xml(file_code)
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
            _fetch_udhr_xml(file_code)
