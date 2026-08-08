"""
GlotScript-R loader — language → writing-system metadata.

GlotScript (Kargaran, Imani, Yvon, Schuetze; LREC-COLING 2024) ships two
things.  The PyPI package ``glotscript`` is a *text* script detector: give it
a string, it tells you which Unicode script the characters belong to.  That is
not what we need.

What we need is GlotScript-R, the *resource*: a table mapping ~7,000 ISO 639-3
language codes to the script(s) that language is written in.  It lives in the
project's GitHub repository under ``metadata/GlotScript.tsv`` and is not
distributed on PyPI, so we vendor a copy under ``data/glotscript/``.

Why this matters here
---------------------
loove scores a tokenizer by asking, for each language, what fraction of that
language's characters survive tokenization.  That requires knowing the
language's characters, which we get from CLDR exemplar sets — and CLDR covers
310 locales.  Everything else in Glottolog is a metadata stub we cannot score.

GlotScript does not give us exemplar characters, so it cannot by itself make a
language scoreable.  What it gives us is the language's *script*, and script is
enough to answer a coarser but still useful question: is this language written
in a system the tokenizer represents at all?  A tokenizer with no Ethiopic
codepoints in its vocabulary cannot serve Amharic, Tigrinya, or the other ~20
Ethiopic-script languages, and we can say that without exemplar data for any
of them.

So the contract is deliberately narrow.  This module answers "what script is
this language written in", and callers may use that for script-level reachability
claims.  It must not be used to synthesise a coverage score; a language with a
script but no exemplars stays unscored, and says so.

Data shape
----------
The TSV has one row per ISO 639-3 code::

    ISO639-3  ISO15924-Main  Wiki-aux  SIL-aux  Lrec2800-aux  SIL2-aux
    amh       Ethi
    aze       Latn           Arab,Cyrl

``ISO15924-Main`` holds *every* script the language is attested in, as an
unordered comma-separated set — 603 of the 8,035 rows carry more than one.  The
four ``-aux`` columns are additional scripts attested by individual sources; we
keep them but treat them as secondary, because a script being attested by one
source is weaker evidence than it being in the main set.

Order is NOT priority
---------------------
It is tempting to read the first element as the primary script.  It is not.
English is ``"Brai, Shaw, Latn, Dsrt, Runr"`` and Hindi is
``"Brai, Deva, Latn, Mahj"`` — taking element zero yields Braille for both.
Some rows do happen to lead with the primary script (Russian ``"Cyrl, Brai"``,
Japanese ``"Jpan, Latn, …"``) but many do not, and there is no field that marks
which is which.

So this module never collapses the set to a single "the" script.  Callers get
the full set and must phrase their question in terms that a set can answer —
"is any script of this language represented" rather than "is its script
represented".  A single-script API would be quietly wrong for at least the 603
multi-script languages and visibly wrong for English.

Transcription systems
---------------------
Braille is a tactile transcription of some other script rather than a language's
own orthography, and its presence in a tokenizer says nothing about whether that
tokenizer serves the language.  It is tracked separately from the orthographic
scripts so reachability judgements are not made on it.

Special values
--------------
``Zxxx`` is ISO 15924 for "unwritten".  It is not a script and languages
carrying it are excluded from reachability claims rather than counted as
unreachable — a spoken-only language is not evidence against a tokenizer.
``Zyyy`` (common) and ``Zinh`` (inherited) are similarly not real scripts.
"""

from __future__ import annotations

import csv
from pathlib import Path

# ISO 15924 codes that denote the absence of a writing system rather than a
# writing system.  Languages carrying these are excluded from script-level
# analysis rather than counted as failures.
NON_SCRIPTS: frozenset[str] = frozenset({"Zxxx", "Zyyy", "Zinh", "Zzzz"})

# Scripts that transcribe another script rather than being a language's own
# orthography.  A tokenizer containing Braille codepoints is not thereby able
# to serve Hindi, so these are excluded from reachability judgements while
# still being reported.
TRANSCRIPTION_SCRIPTS: frozenset[str] = frozenset({"Brai"})

_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "glotscript" / "GlotScript.tsv"

_CACHE: dict[str, dict] | None = None


def _split_scripts(cell: str) -> list[str]:
    """
    Parse a script cell into a list of ISO 15924 codes.

    Cells may hold a single code ("Latn"), a comma-separated list with
    inconsistent spacing ("Arab, Latn"), or be empty.  Non-script sentinels
    are filtered out here so callers never see them.
    """
    if not cell:
        return []
    out = []
    for part in cell.replace(";", ",").split(","):
        code = part.strip()
        if code and code not in NON_SCRIPTS and code not in out:
            out.append(code)
    return out


def load_glotscript(path: Path | str | None = None) -> dict[str, dict]:
    """
    Load GlotScript-R as ``iso639_3 -> entry``.

    Each entry is::

        {
          "iso639_3":    "aze",
          "main":        ["Latn"],        # primary script(s), sentinels removed
          "aux":         ["Arab", "Cyrl"] # attested elsewhere, deduped
          "unwritten":   False,           # True if the row said Zxxx and nothing else
        }

    Returns an empty dict if the data file is absent, so that callers degrade
    to "no script information" rather than crashing.  Result is cached.
    """
    global _CACHE
    if _CACHE is not None and path is None:
        return _CACHE

    p = Path(path) if path else _DATA_PATH
    if not p.exists():
        if path is None:
            _CACHE = {}
        return {}

    table: dict[str, dict] = {}
    with p.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            iso = (row.get("ISO639-3") or "").strip()
            if not iso:
                continue
            raw_main = (row.get("ISO15924-Main") or "").strip()
            main = _split_scripts(raw_main)

            aux: list[str] = []
            for col in ("Wiki-aux", "SIL-aux", "Lrec2800-aux", "SIL2-aux"):
                for code in _split_scripts(row.get(col) or ""):
                    if code not in main and code not in aux:
                        aux.append(code)

            table[iso] = {
                "iso639_3":  iso,
                "main":      main,
                "aux":       aux,
                # Only call it unwritten if the source said so *and* offered
                # no alternative — an empty cell means unknown, not unwritten.
                "unwritten": raw_main in NON_SCRIPTS and not aux,
            }

    if path is None:
        _CACHE = table
    return table


def scripts_for(iso639_3: str | None, include_transcription: bool = False) -> list[str]:
    """
    Return every script the language is attested in, main set first.

    Deliberately returns a list, not a single code: the source data is an
    unordered set and picking one element yields Braille for English and Hindi
    (see module docstring).  Returns ``[]`` when the language is unknown,
    unwritten, or has no ISO 639-3 code.

    By default transcription systems (Braille) are excluded, because their
    presence in a tokenizer is not evidence it can serve the language.
    """
    if not iso639_3:
        return []
    entry = load_glotscript().get(iso639_3)
    if not entry:
        return []
    out = list(entry["main"]) + [a for a in entry["aux"] if a not in entry["main"]]
    if not include_transcription:
        out = [s for s in out if s not in TRANSCRIPTION_SCRIPTS]
    return out


def orthographic_main(iso639_3: str | None) -> list[str]:
    """
    The main-set scripts, excluding transcription systems.

    This is the set to use for reachability: "does the tokenizer represent any
    script this language is actually written in".
    """
    if not iso639_3:
        return []
    entry = load_glotscript().get(iso639_3)
    if not entry:
        return []
    return [s for s in entry["main"] if s not in TRANSCRIPTION_SCRIPTS]


def clear_cache() -> None:
    """Drop the module-level cache (used by tests)."""
    global _CACHE
    _CACHE = None
