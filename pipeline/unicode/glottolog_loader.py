"""
Glottolog loader: download and parse the Glottolog languoid CSV to build
a comprehensive language registry that acts as the master list of languages
for the dashboard.

Source: https://glottolog.org/meta/downloads
File:   glottolog_languoid.csv.zip  (v5.3, ~576 KB)
License: CC BY 4.0

What Glottolog adds over CLDR
------------------------------
CLDR covers ~250 locales with exemplar character sets — the languages major
tech companies actively support.  Glottolog catalogs ~8,000+ distinct
languages, giving us a denominator that reflects the world's true linguistic
diversity.

Key fields from the CSV we use
-------------------------------
  id              — Glottocode (e.g. "stan1295"), unique stable identifier
  name            — English name
  level           — "language" | "dialect" | "family"  (we keep only "language")
  bookkeeping     — bool; bookkeeping rows are legacy/admin entries, exclude them
  iso639P3code    — ISO 639-3 code (3-letter); bridge to CLDR locale IDs
  macroarea       — geographic region: Africa | Eurasia | Papunesia |
                    Australia | South America | North America
  family_id       — Glottocode of the top-level family (isolates: same as id)
  latitude        — float or empty
  longitude       — float or empty

Cross-referencing with CLDR
----------------------------
We use the `langcodes` library to map ISO 639-3 → BCP 47 tag, then look that
up in the CLDR language database.  A match means we have exemplar character
data; no match means the language is tracked in the registry but cannot yet
be scored for character coverage.

Output: data/glottolog/languoids.json
{
  "hin1269": {                     // glottocode as key
    "glottocode":   "hin1269",
    "name":         "Hindi",
    "iso639_3":     "hin",
    "cldr_locale":  "hi",          // null if no CLDR match
    "macroarea":    "Eurasia",
    "family_id":    "indo1319",
    "family_name":  "Indo-European",
    "latitude":     25.0,
    "longitude":    77.0,
    "has_cldr":     true
  },
  ...
}

Output: data/glottolog/families.json
{
  "indo1319": { "glottocode": "indo1319", "name": "Indo-European" },
  ...
}
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import requests

try:
    import langcodes
    _HAS_LANGCODES = True
except ImportError:
    _HAS_LANGCODES = False

_ROOT = Path(__file__).parents[2]
_CACHE_DIR = _ROOT / "data" / "glottolog"
_LANGUOIDS_OUT = _CACHE_DIR / "languoids.json"
_FAMILIES_OUT  = _CACHE_DIR / "families.json"

# v5.3 (latest as of 2026-05)
_CSV_ZIP_URL = (
    "https://cdstar.eva.mpg.de/bitstreams/EAEA0-608B-9919-A962-0/"
    "glottolog_languoid.csv.zip"
)
_CSV_CACHE = _CACHE_DIR / "glottolog_languoid.csv"

_MACROAREAS = {
    "africa", "eurasia", "papunesia",
    "australia", "south america", "north america",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_csv(force_refresh: bool = False) -> str:
    """Download and unzip the Glottolog languoid CSV, caching the raw CSV."""
    if _CSV_CACHE.exists() and not force_refresh:
        return _CSV_CACHE.read_text(encoding="utf-8")

    print(f"[glottolog] Downloading languoid CSV from {_CSV_ZIP_URL} …")
    resp = requests.get(_CSV_ZIP_URL, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # The zip contains a single CSV; its name may vary by version.
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            raise RuntimeError("No CSV file found in the Glottolog zip.")
        raw = zf.read(names[0]).decode("utf-8")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CSV_CACHE.write_text(raw, encoding="utf-8")
    print(f"[glottolog] Cached to {_CSV_CACHE}")
    return raw


# ---------------------------------------------------------------------------
# ISO 639-3 → CLDR locale bridge
# ---------------------------------------------------------------------------

def _iso3_to_cldr(iso3: str) -> str | None:
    """
    Map an ISO 639-3 code to the most likely CLDR / BCP 47 locale ID.

    Returns a bare language subtag (e.g. "hi", "ar", "nld") suitable for
    looking up in the CLDR language database.  Returns None on failure.
    """
    if not iso3 or not _HAS_LANGCODES:
        return None
    try:
        tag = langcodes.standardize_tag(iso3, macro=False)
        # Strip region/script subtags — we want the base language only
        return tag.split("-")[0] if tag else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CSV parser
# ---------------------------------------------------------------------------

def _parse_csv(raw_csv: str) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Parse the languoid CSV into (languoids, families) dicts.

    languoids: glottocode → entry dict (level == "language" only)
    families:  glottocode → {glottocode, name}  (level == "family" only)
    """
    reader = csv.DictReader(io.StringIO(raw_csv))

    all_rows: list[dict] = list(reader)
    available_cols = set(all_rows[0].keys()) if all_rows else set()

    # Detect column name variants between Glottolog versions
    def _col(preferred: str, *alts: str) -> str:
        for c in (preferred, *alts):
            if c in available_cols:
                return c
        return preferred  # will fail gracefully later

    col_id        = _col("id", "glottocode")
    col_name      = _col("name")
    col_level     = _col("level")
    col_bookkeep  = _col("bookkeeping")
    col_iso3      = _col("iso639P3code", "ISO 639-3")
    col_macro     = _col("macroarea")
    col_family    = _col("family_id")
    col_lat       = _col("latitude")
    col_lon       = _col("longitude")

    languoids: dict[str, dict] = {}
    families:  dict[str, dict] = {}

    for row in all_rows:
        glottocode = (row.get(col_id) or "").strip()
        level      = (row.get(col_level) or "").strip().lower()
        bookkeep   = (row.get(col_bookkeep) or "False").strip().lower()

        if not glottocode:
            continue

        # Collect family-level entries for the families index
        if level == "family":
            families[glottocode] = {
                "glottocode": glottocode,
                "name": (row.get(col_name) or "").strip(),
            }
            continue

        # Only keep language-level, non-bookkeeping rows
        if level != "language" or bookkeep in ("true", "1", "yes"):
            continue

        iso3       = (row.get(col_iso3) or "").strip()
        macroarea  = (row.get(col_macro) or "").strip()
        family_id  = (row.get(col_family) or "").strip()

        # Latitude / longitude — may be empty
        try:
            lat: float | None = float(row[col_lat]) if row.get(col_lat, "").strip() else None
        except ValueError:
            lat = None
        try:
            lon: float | None = float(row[col_lon]) if row.get(col_lon, "").strip() else None
        except ValueError:
            lon = None

        cldr_locale = _iso3_to_cldr(iso3) if iso3 else None

        languoids[glottocode] = {
            "glottocode":  glottocode,
            "name":        (row.get(col_name) or "").strip(),
            "iso639_3":    iso3 or None,
            "cldr_locale": cldr_locale,
            "macroarea":   macroarea or None,
            "family_id":   family_id or None,
            "family_name": None,         # filled in below
            "latitude":    lat,
            "longitude":   lon,
            "has_cldr":    cldr_locale is not None,
        }

    # Back-fill family names
    for entry in languoids.values():
        fid = entry["family_id"]
        if fid and fid in families:
            entry["family_name"] = families[fid]["name"]
        elif fid == entry["glottocode"]:
            # Language isolate: the language itself is its own family
            entry["family_name"] = entry["name"] + " (isolate)"

    return languoids, families


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_languoid_database(force_refresh: bool = False) -> dict[str, dict]:
    """
    Download, parse, and cache the Glottolog languoid database.

    Returns:
        languoids dict  (glottocode → entry)
    """
    if _LANGUOIDS_OUT.exists() and not force_refresh:
        data = json.loads(_LANGUOIDS_OUT.read_text(encoding="utf-8"))
        return data["languoids"]

    raw_csv = _download_csv(force_refresh=force_refresh)
    languoids, families = _parse_csv(raw_csv)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _LANGUOIDS_OUT.write_text(
        json.dumps({"languoids": languoids}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _FAMILIES_OUT.write_text(
        json.dumps(families, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with_cldr    = sum(1 for e in languoids.values() if e["has_cldr"])
    without_cldr = len(languoids) - with_cldr
    print(
        f"[glottolog] {len(languoids):,} languages parsed.\n"
        f"            {with_cldr:,} have a CLDR locale match.\n"
        f"            {without_cldr:,} have no CLDR character data."
    )

    return languoids


def load_languoid_database() -> dict[str, dict]:
    """Load the previously built Glottolog languoid database from disk."""
    if not _LANGUOIDS_OUT.exists():
        raise FileNotFoundError(
            f"Glottolog database not found at {_LANGUOIDS_OUT}.\n"
            "Run:  python scripts/fetch_glottolog.py"
        )
    data = json.loads(_LANGUOIDS_OUT.read_text(encoding="utf-8"))
    return data["languoids"]


def load_families() -> dict[str, dict]:
    """Load the Glottolog language family index from disk."""
    if not _FAMILIES_OUT.exists():
        raise FileNotFoundError(
            f"Glottolog families file not found at {_FAMILIES_OUT}.\n"
            "Run:  python scripts/fetch_glottolog.py"
        )
    return json.loads(_FAMILIES_OUT.read_text(encoding="utf-8"))


def cldr_to_glottolog_index(languoids: dict[str, dict]) -> dict[str, list[str]]:
    """
    Build a reverse index: CLDR locale ID → list of glottocodes.

    One CLDR locale may correspond to multiple Glottolog language entries
    (e.g. Serbo-Croatian dialects all map to "sr").
    """
    index: dict[str, list[str]] = {}
    for glottocode, entry in languoids.items():
        locale = entry.get("cldr_locale")
        if locale:
            index.setdefault(locale, []).append(glottocode)
    return index
