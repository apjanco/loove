"""
Wikidata writing-system loader — fill script gaps GlotScript leaves open.

GlotScript maps ~7,000 ISO 639-3 codes to scripts, but leaves many historical
and sparsely-attested languages with no script at all (122 of loove's historical
languages, for instance).  Wikidata models the same relation on its language
items via property P282 ("writing system"), and each writing-system item carries
its ISO 15924 code in P506, so a single SPARQL join yields
``iso639-3 -> ISO 15924 script`` for every language Wikidata knows.

This module is a thin, cached client for that join.  It is deliberately scoped
the same way as ``glotscript_loader``: it answers "what script(s) is this
language written in", nothing more.  It never invents exemplar characters, so a
language gains a script here but only becomes *scoreable* if that script is one
``historic_scripts`` can build a Unicode-block exemplar for.

The cache accumulates: fetching a small set (e.g. the 122 script-less historical
languages) and later fetching more merges into the same file, so the loader can
grow toward "all languages with a Wikidata script" without refetching.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_CACHE_DIR = _ROOT / "data" / "wikidata"
_CACHE = _CACHE_DIR / "language_scripts.json"

_ENDPOINT = "https://query.wikidata.org/sparql"
# Wikimedia requires a descriptive User-Agent; requests without one are blocked.
_USER_AGENT = (
    "loove-language-coverage/0.1 "
    "(LLM language-coverage research; https://github.com/loove)"
)

# P220 = ISO 639-3 code (language), P282 = writing system, P506 = ISO 15924 code.
_QUERY = """SELECT ?iso ?code WHERE {{
  {values}
  ?lang wdt:P220 ?iso .
  ?lang wdt:P282 ?ws .
  ?ws wdt:P506 ?code .
}}"""


def _run_query(values_clause: str, timeout: int, retries: int = 3) -> list[tuple[str, str]]:
    """Run one SPARQL query and return (iso, script_code) pairs."""
    query = _QUERY.format(values=values_clause)
    url = _ENDPOINT + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/sparql-results+json",
        },
    )
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as exc:  # network hiccup / 429 — back off and retry
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"Wikidata query failed after {retries} tries: {last_exc}")

    out: list[tuple[str, str]] = []
    for b in data["results"]["bindings"]:
        iso = b.get("iso", {}).get("value")
        code = b.get("code", {}).get("value")
        if iso and code:
            out.append((iso, code))
    return out


def _values_clause(iso_codes: list[str]) -> str:
    quoted = " ".join(f'"{c}"' for c in iso_codes)
    return f"VALUES ?iso {{ {quoted} }}"


def fetch_scripts(
    iso_codes: list[str] | None = None,
    *,
    chunk: int = 300,
    timeout: int = 90,
    pause: float = 1.0,
) -> dict[str, list[str]]:
    """
    Fetch ``iso639-3 -> [ISO 15924]`` from Wikidata and merge into the cache.

    Args:
        iso_codes: Restrict to these ISO 639-3 codes (queried in chunks).
                   None fetches every language Wikidata has an ISO 639-3 + script
                   for — the full expansion.
        chunk:     ISO codes per SPARQL request when a subset is given.

    Returns the freshly-fetched mapping (not the whole merged cache).
    """
    pairs: list[tuple[str, str]] = []
    if iso_codes:
        codes = sorted(set(iso_codes))
        for i in range(0, len(codes), chunk):
            batch = codes[i:i + chunk]
            pairs.extend(_run_query(_values_clause(batch), timeout))
            time.sleep(pause)
    else:
        pairs = _run_query("", timeout)

    fetched: dict[str, list[str]] = {}
    for iso, code in pairs:
        fetched.setdefault(iso, [])
        if code not in fetched[iso]:
            fetched[iso].append(code)
    for iso in fetched:
        fetched[iso].sort()

    _merge_into_cache(fetched)
    return fetched


def _merge_into_cache(fetched: dict[str, list[str]]) -> None:
    """Merge freshly-fetched scripts into the on-disk cache."""
    existing = _load_raw()
    scripts = existing.get("scripts", {})
    for iso, codes in fetched.items():
        merged = sorted(set(scripts.get(iso, [])) | set(codes))
        scripts[iso] = merged
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE.write_text(
        json.dumps(
            {
                "source": "wikidata P220/P282/P506",
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "count": len(scripts),
                "scripts": dict(sorted(scripts.items())),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_raw() -> dict:
    if not _CACHE.exists():
        return {}
    try:
        return json.loads(_CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load_wikidata_scripts() -> dict[str, list[str]]:
    """Return the cached ``iso639-3 -> [ISO 15924]`` mapping (empty if none)."""
    return _load_raw().get("scripts", {})
