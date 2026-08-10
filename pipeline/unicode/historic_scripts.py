"""
Historic-script exemplars: make dead-script languages scoreable.

Why this module exists
----------------------
loove scores a language by classifying its characters into coverage tiers, and
that requires knowing the language's characters.  CLDR supplies those for 310
modern locales, so historical languages written in an extinct script — Gothic,
Cuneiform, Egyptian Hieroglyphs — have no exemplar set and cannot be scored at
all.  They vanish into the Glottolog "stub" tier, indistinguishable from a
living language we simply haven't reached.

For an extinct script there is no living community and therefore no CLDR
exemplar to wait for, but there *is* a definitive character inventory: the
Unicode block(s) that encode the script.  This module synthesises an exemplar
set from those blocks so the ordinary tier classifier can score the language.

Scope and honesty
-----------------
This only helps languages whose script is used by no living language (a genuine
tokenizer gap) *and* is encoded in Unicode.  A historical language written in a
still-living script (Ancient Greek in Greek, Old English in Latin) is already
reachable at the code-point level, so a synthetic exemplar would add nothing —
those are left out.  The synthesised exemplar is the whole script inventory, not
a per-language subset, so a high score means "the tokenizer covers the script",
which for these dead scripts is the honest question.  Provenance is recorded as
``exemplar_source = "unicode_block"`` so the number is never mistaken for a
CLDR-grounded one.

Scripts Unicode does not encode at all (Mayan Hieroglyphs, Jurchen) cannot be
given an exemplar; ``scripts_all_unencoded`` lets the coverage layer flag those
languages as having no representable script rather than guessing.
"""
from __future__ import annotations

import unicodedata
from functools import lru_cache

from pipeline.unicode.glottolog_loader import load_languoid_database
from pipeline.unicode.glotscript_loader import orthographic_main
from pipeline.unicode.wikidata_scripts import load_wikidata_scripts

# ISO 15924 variants that are stylistic forms of a living script, not scripts of
# their own; normalising them prevents e.g. Latin-Fraktur languages from being
# mistaken for an extinct-script gap.
_SCRIPT_ALIASES: dict[str, str] = {
    "Latf": "Latn",   # Latin (Fraktur variant)
    "Latg": "Latn",   # Latin (Gaelic variant)
    "Aran": "Arab",   # Arabic (Nastaliq variant)
    "Syre": "Syrc",   # Syriac (Estrangelo variant)
    "Syrj": "Syrc",   # Syriac (Western variant)
    "Syrn": "Syrc",   # Syriac (Eastern variant)
}

# ISO 15924 → Unicode block range(s), inclusive, for extinct scripts Unicode
# encodes.  Ranges are filtered to assigned code points at build time, so a
# range that spans reserved slots is harmless.
_SCRIPT_BLOCKS: dict[str, list[tuple[int, int]]] = {
    "Samr": [(0x0800, 0x083F)],                         # Samaritan
    "Ogam": [(0x1680, 0x169F)],                         # Ogham
    "Runr": [(0x16A0, 0x16FF)],                         # Runic
    "Copt": [(0x2C80, 0x2CFF), (0x102E0, 0x102FF)],     # Coptic (+ Epact numbers)
    "Linb": [(0x10000, 0x1007F), (0x10080, 0x100FF),
             (0x10100, 0x1013F)],                       # Linear B (+ Aegean numbers)
    "Cprt": [(0x10800, 0x1083F)],                       # Cypriot
    "Ital": [(0x10300, 0x1032F)],                       # Old Italic
    "Goth": [(0x10330, 0x1034F)],                       # Gothic
    "Perm": [(0x10350, 0x1037F)],                       # Old Permic
    "Ugar": [(0x10380, 0x1039F)],                       # Ugaritic
    "Xpeo": [(0x103A0, 0x103DF)],                       # Old Persian
    "Aghb": [(0x10530, 0x1056F)],                       # Caucasian Albanian
    "Lina": [(0x10600, 0x1077F)],                       # Linear A
    "Phnx": [(0x10900, 0x1091F)],                       # Phoenician
    "Lydi": [(0x10920, 0x1093F)],                       # Lydian
    "Lyci": [(0x10280, 0x1029F)],                       # Lycian
    "Cari": [(0x102A0, 0x102DF)],                       # Carian
    "Mero": [(0x10980, 0x1099F)],                       # Meroitic Hieroglyphs
    "Merc": [(0x109A0, 0x109FF)],                       # Meroitic Cursive
    "Khar": [(0x10A00, 0x10A5F)],                       # Kharoshthi
    "Sarb": [(0x10A60, 0x10A7F)],                       # Old South Arabian
    "Narb": [(0x10A80, 0x10A9F)],                       # Old North Arabian
    "Armi": [(0x10840, 0x1085F)],                       # Imperial Aramaic
    "Palm": [(0x10860, 0x1087F)],                       # Palmyrene
    "Nbat": [(0x10880, 0x108AF)],                       # Nabataean
    "Hatr": [(0x108E0, 0x108FF)],                       # Hatran
    "Avst": [(0x10B00, 0x10B3F)],                       # Avestan
    "Prti": [(0x10B40, 0x10B5F)],                       # Inscriptional Parthian
    "Phli": [(0x10B60, 0x10B7F)],                       # Inscriptional Pahlavi
    "Phlp": [(0x10B80, 0x10BAF)],                       # Psalter Pahlavi
    "Mani": [(0x10AC0, 0x10AFF)],                       # Manichaean
    "Sogo": [(0x10F00, 0x10F2F)],                       # Old Sogdian
    "Sogd": [(0x10F30, 0x10F6F)],                       # Sogdian
    "Chrs": [(0x10FB0, 0x10FDF)],                       # Chorasmian
    "Elym": [(0x10FE0, 0x10FFF)],                       # Elymaic
    "Brah": [(0x11000, 0x1107F)],                       # Brahmi
    "Ahom": [(0x11700, 0x1174F)],                       # Ahom
    "Kawi": [(0x11F00, 0x11F5F)],                       # Kawi
    "Sidd": [(0x11580, 0x115FF)],                       # Siddham
    "Phag": [(0xA840, 0xA87F)],                         # Phags-pa
    "Marc": [(0x11C70, 0x11CBF)],                       # Marchen
    "Medf": [(0x16E40, 0x16E9F)],                       # Medefaidrin
    "Xsux": [(0x12000, 0x123FF), (0x12400, 0x1247F),
             (0x12480, 0x1254F)],                       # Cuneiform
    "Egyp": [(0x13000, 0x1342F), (0x13430, 0x1343F)],   # Egyptian Hieroglyphs
    "Hluw": [(0x14400, 0x1467F)],                       # Anatolian Hieroglyphs
    "Tang": [(0x17000, 0x187FF), (0x18800, 0x18AFF),
             (0x18D00, 0x18D8F)],                       # Tangut
    "Kits": [(0x18B00, 0x18CFF)],                       # Khitan Small Script
}

# Extinct scripts with no Unicode encoding: a language written only in one of
# these has no representable code points in any tokenizer, ever.
_UNENCODED_HISTORIC: frozenset[str] = frozenset({
    "Maya",   # Mayan Hieroglyphs (proposed, not encoded)
    "Jurc",   # Jurchen
    "Kitl",   # Khitan Large Script
    "Pelm",   # Proto-Elamite
    "Roro",   # Rongorongo
})


def _alias(script: str) -> str:
    return _SCRIPT_ALIASES.get(script, script)


def is_encoded(script: str) -> bool:
    """True when we can synthesise an exemplar for this script from Unicode."""
    return _alias(script) in _SCRIPT_BLOCKS


def scripts_all_unencoded(scripts: list[str]) -> bool:
    """True when every script given is a known extinct, Unicode-unencoded one."""
    norm = [_alias(s) for s in scripts if s]
    return bool(norm) and all(s in _UNENCODED_HISTORIC for s in norm)


@lru_cache(maxsize=None)
def assigned_codepoints(script: str) -> tuple[int, ...]:
    """The assigned code points in a script's Unicode block(s)."""
    ranges = _SCRIPT_BLOCKS.get(_alias(script))
    if not ranges:
        return ()
    out: list[int] = []
    for start, end in ranges:
        for cp in range(start, end + 1):
            try:
                unicodedata.name(chr(cp))
            except ValueError:
                continue  # unassigned code point
            out.append(cp)
    return tuple(out)


def build_historic_language_db(locales: list[str] | None = None) -> dict[str, dict]:
    """
    Build synthetic, exemplar-bearing language entries for historical languages
    written in an extinct, Unicode-encoded script.

    The result is keyed by glottocode and shaped like a CLDR ``language_db``
    entry, so it can be merged straight into the map passed to
    ``compute_coverage`` and scored by the ordinary tier classifier.

    Only languages whose script is used by no living language are included: a
    historical language sharing a modern script is already reachable and needs
    no synthetic exemplar.
    """
    try:
        languoids = load_languoid_database()
    except FileNotFoundError:
        return {}

    # Wikidata fills script gaps GlotScript leaves open (cached; empty if never
    # fetched).  Used only as a fallback when GlotScript has nothing.
    wikidata = load_wikidata_scripts()

    # Scripts still used by a living language — the exemplar for these already
    # exists (or is moot), so they are not the gap this module fills.
    living: set[str] = set()
    for e in languoids.values():
        if e.get("is_historical") is False and e.get("iso639_3"):
            living.update(_alias(s) for s in orthographic_main(e["iso639_3"]))

    out: dict[str, dict] = {}
    unclassified: set[str] = set()

    for glottocode, e in languoids.items():
        if not (e.get("is_historical") and not e.get("has_cldr") and e.get("iso639_3")):
            continue
        scripts = [_alias(s) for s in orthographic_main(e["iso639_3"])]
        if not scripts:
            scripts = [_alias(s) for s in wikidata.get(e["iso639_3"], [])]
        historic_only = [s for s in scripts if s not in living]
        if not historic_only:
            continue  # written in a living script — already reachable
        encoded = [s for s in historic_only if s in _SCRIPT_BLOCKS]
        if not encoded:
            unclassified.update(
                s for s in historic_only if s not in _UNENCODED_HISTORIC
            )
            continue
        if locales is not None and glottocode not in locales:
            continue
        script = encoded[0]
        cps = assigned_codepoints(script)
        if not cps:
            continue
        out[glottocode] = {
            "locale_id":          glottocode,
            "name":               e.get("name", glottocode),
            "script":             script,
            "exemplar_main":      sorted(cps),
            "exemplar_auxiliary": [],
            "exemplar_source":    "unicode_block",
            "has_cldr":           False,
            "is_historical":      True,
            "language_type":      e.get("language_type"),
            "glottocode":         e.get("glottocode"),
            "iso639_3":           e.get("iso639_3"),
            "macroarea":          e.get("macroarea"),
            "family_id":          e.get("family_id"),
            "family_name":        e.get("family_name"),
            "latitude":           e.get("latitude"),
            "longitude":          e.get("longitude"),
        }

    if unclassified:
        print(
            f"[historic] {len(unclassified)} historic-only script(s) neither "
            f"encoded nor listed as unencoded (skipped): "
            f"{', '.join(sorted(unclassified))}"
        )

    return out
