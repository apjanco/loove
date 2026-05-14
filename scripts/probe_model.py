#!/usr/bin/env python3
"""
Probe a black-box LLM for language coverage using behavioural tests.

Unlike ingest_model.py (which requires access to the tokenizer vocabulary),
probe_model.py uses only API access. It identifies languages that are commonly
out-of-vocabulary (OOV) across the models we have pre-computed coverage data
for, then runs three tests against the target model:

  1. Echo fidelity   — can the model reproduce the script character-for-character?
  2. Script ID       — does it recognise the writing system?
  3. Translation     — can it comprehend the text?

OOV languages are auto-selected as those whose median weighted_score across
all known models is below --threshold (default 0.5) and that appear in at
least --min-models coverage files. Override with --locales.

Results are saved to data/probes/<safe_model_name>__<YYYY-MM-DD>.json.

Examples
--------
  # Probe Gemini 2.0 Flash for all commonly-OOV languages
  python scripts/probe_model.py \\
      --model gemini-2.0-flash --api-type gemini \\
      --api-key $GEMINI_API_KEY

  # Probe a custom OpenAI-compatible endpoint
  python scripts/probe_model.py \\
      --model my-model --api-type openai \\
      --base-url http://localhost:11434/v1 \\
      --api-key ollama

  # Probe only specific languages
  python scripts/probe_model.py \\
      --model gpt-4o --api-type openai \\
      --api-key $OPENAI_API_KEY \\
      --locales am bo zgh lo km

  # Lower threshold to catch partially-covered languages too
  python scripts/probe_model.py \\
      --model claude-3-5-sonnet-20241022 --api-type openai \\
      --base-url https://api.anthropic.com/v1 \\
      --api-key $ANTHROPIC_API_KEY \\
      --threshold 0.7 --min-models 3
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from statistics import median
from typing import Iterator

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

COVERAGE_DIR = ROOT / "data" / "coverage"
UDHR_DIR     = ROOT / "data" / "udhr"
PROBES_DIR   = ROOT / "data" / "probes"

# ---------------------------------------------------------------------------
# CLDR locale → UDHR filename stem mapping
# ---------------------------------------------------------------------------
_LOCALE_TO_UDHR: dict[str, str] = {
    "aa":  "aar",        "ab":  "abk",        "af":  "afr",
    "ak":  "aka_akuapem","am":  "amh",        "ar":  "arb",
    "az":  "azj_cyrl",  "be":  "bel",        "bg":  "bul",
    "bh":  "bho",       "bn":  "ben",        "bo":  "bod",
    "br":  "bre",       "bs":  "bos_cyrl",   "ca":  "cat",
    "cs":  "ces",       "cy":  "cym",        "da":  "dan",
    "de":  "deu_1901",  "dv":  "div",        "dz":  "dzo",
    "el":  "ell_monotonic", "en": "eng",     "eo":  "epo",
    "es":  "spa",       "et":  "est",        "eu":  "eus",
    "fa":  "pes_1",     "fi":  "fin",        "fr":  "fra",
    "ga":  "gle",       "gl":  "glg",        "gu":  "guj",
    "ha":  "hau_NE",    "haw": "haw",        "he":  "heb",
    "hi":  "hin",       "hr":  "hrv",        "hu":  "hun",
    "hy":  "hye",       "id":  "ind",        "ig":  "ibo",
    "is":  "isl",       "it":  "ita",        "iu":  "ike",
    "ja":  "jpn",       "ka":  "kat",        "kk":  "kaz",
    "km":  "khm",       "kn":  "kan",        "ko":  "kor",
    "ky":  "kir",       "la":  "lat",        "lb":  "ltz",
    "ln":  "lin",       "lo":  "lao",        "lt":  "lit",
    "mi":  "mri",       "mk":  "mkd",        "ml":  "mal",
    "mn":  "khk",       "mr":  "mar",        "mt":  "mlt",
    "my":  "mya",       "ne":  "nep",        "nl":  "nld",
    "os":  "oss",       "pa":  "pan",        "pl":  "pol",
    "ps":  "pbu",       "pt":  "por_BR",     "ro":  "ron_1953",
    "ru":  "rus",       "rw":  "kin",        "sa":  "san",
    "sc":  "src",       "shn": "shn",        "si":  "sin",
    "sk":  "slk",       "sl":  "slv",        "sn":  "sna",
    "so":  "som",       "sq":  "sqi",        "sr":  "srp_cyrl",
    "ss":  "ssw",       "st":  "sot",        "su":  "sun",
    "sv":  "swe",       "sw":  "swa",        "ta":  "tam",
    "te":  "tel",       "tg":  "tgk",        "th":  "tha",
    "ti":  "tir",       "tk":  "tuk_cyrl",   "tn":  "tsn",
    "tr":  "tur",       "tt":  "tat",        "ug":  "uig_arab",
    "uk":  "ukr",       "ur":  "urd",        "uz":  "uzn_cyrl",
    "vai": "vai",       "vi":  "vie",        "wo":  "wol",
    "xh":  "xho",       "yi":  "ydd",        "yo":  "yor",
    "yue": "yue",       "zh":  "cmn_hans",   "zgh": "zgh",
    "zu":  "zul",
}

# ---------------------------------------------------------------------------
# Expected script keywords per locale (for Test 2)
# ---------------------------------------------------------------------------
_LOCALE_SCRIPT: dict[str, str] = {
    "am":  "ethiopic",   "ti":  "ethiopic",
    "ar":  "arabic",     "fa":  "arabic",     "ur":  "arabic",
    "ug":  "arabic",     "ps":  "arabic",
    "bo":  "tibetan",
    "zh":  "han",        "yue": "han",
    "ja":  "kanji",
    "ko":  "hangul",
    "lo":  "lao",
    "km":  "khmer",
    "my":  "myanmar",    "shn": "myanmar",
    "hi":  "devanagari", "mr":  "devanagari", "ne":  "devanagari",
    "sa":  "devanagari",
    "bn":  "bengali",
    "pa":  "gurmukhi",
    "gu":  "gujarati",
    "ta":  "tamil",
    "te":  "telugu",
    "kn":  "kannada",
    "ml":  "malayalam",
    "si":  "sinhala",
    "th":  "thai",
    "ka":  "georgian",
    "hy":  "armenian",
    "mn":  "cyrillic",   "ru":  "cyrillic",   "uk":  "cyrillic",
    "be":  "cyrillic",   "bg":  "cyrillic",   "sr":  "cyrillic",
    "el":  "greek",
    "he":  "hebrew",
    "yi":  "hebrew",
    "zgh": "tifinagh",
    "vai": "vai",
}

# ---------------------------------------------------------------------------
# Fallback sentences (used when no UDHR file exists)
# Keyed by locale then by script keyword
# ---------------------------------------------------------------------------
_LOCALE_FALLBACK: dict[str, str] = {
    "am":  "ፈጣኑ ቡናማ ቀበሮ ሰነፉ ውሻ ላይ ዘለለ።",
    "ti":  "ናይ ሰብ መሰላት ዓለም-ለኸ ምልክዓት።",
    "bo":  "བོད་ཀྱི་སྐད་ཡིག་ནི་རྒྱ་ཆེན་པོ་ཡིན།",
    "lo":  "ໝາປ່າສີນ້ຳຕານໄວ ໂດດຂ້າມໝາຂີ້ຄ້ານ.",
    "km":  "កញ្ជ្រោងពណ៌ត្នោតរហ័សលោតឆ្លងផ្ទៃឆ្កែខ្ជិល។",
    "my":  "လျင်မြန်သောအညိုရောင်မြေခွေးသည် ပျင်းရိသောခွေးကိုကျော်ခုန်သွားသည်။",
    "shn": "ၸုမ်းမိူင်းႁူမ်ႈသုမ်ႈ တႃႇပၢႆးမၢၵ်ႈမႄးၶိုၼ်ႈမႃးၶႄႈ",
    "zgh": "ⴰⵎⴰⵣⵉⵖ ⵉⵛⵛⴰⵔ ⵜⴰⵎⴰⵣⵉⵖⵜ.",
    "vai": "ꕉꕞꕱ ꔓꘋ ꖸ ꕃꔤ ꗡꘋ ꔻꔤ ꕪꕞ",
    "dz":  "གནས་སྟངས་ཀྱི་ལམ་ལུགས་ཤིག",
}

_SCRIPT_FALLBACK: dict[str, str] = {
    "ethiopic":   "ፈጣኑ ቡናማ ቀበሮ ሰነፉ ውሻ ላይ ዘለለ።",
    "tibetan":    "བོད་ཀྱི་སྐད་ཡིག་ནི་རྒྱ་ཆེན་པོ་ཡིན།",
    "lao":        "ໝາປ່າສີນ້ຳຕານໄວ ໂດດຂ້າມໝາຂີ້ຄ້ານ.",
    "khmer":      "កញ្ជ្រោងពណ៌ត្នោតរហ័សលោតឆ្លងផ្ទៃឆ្កែខ្ជិល។",
    "myanmar":    "လျင်မြန်သောအညိုရောင်မြေခွေးသည် ပျင်းရိသောခွေးကိုကျော်ခုန်သွားသည်།",
    "tifinagh":   "ⴰⵎⴰⵣⵉⵖ ⵉⵛⵛⴰⵔ ⵜⴰⵎⴰⵣⵉⵖⵜ.",
    "han":        "那只敏捷的棕色狐狸跳过了那只懒惰的狗。",
    "hangul":     "빠른 갈색 여우가 게으른 개를 뛰어넘었다.",
    "kanji":      "素早い茶色のキツネは怠け者の犬を飛び越えた。",
    "arabic":     "الثعلب البني السريع يقفز فوق الكلب الكسول.",
    "devanagari": "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदी।",
    "bengali":    "দ্রুত বাদামী শিয়াল অলস কুকুরের উপর দিয়ে লাফিয়ে গেল।",
    "tamil":      "வேகமான பழுப்பு நிற நரி சோம்பேறி நாய் மீது தாண்டியது.",
    "telugu":     "వేగవంతమైన గోధుమ రంగు నక్క సోమరి కుక్క మీదుగా దూకింది.",
    "kannada":    "ವೇಗದ ಕಂದು ನರಿ ಸೋಮಾರಿ ನಾಯಿಯ ಮೇಲೆ ಹಾರಿತು.",
    "malayalam":  "വേഗമേറിയ തവിട്ടുനിറമുള്ള കുറുക്കൻ മടിയനായ നായ്ക്കു മേൽ ചാടി.",
    "sinhala":    "ශීඝ්‍ර දුඹුරු හිවල් කම්මැලි බල්ලා හා පනිනවා.",
    "gurmukhi":   "ਤੇਜ਼ ਭੂਰੀ ਲੂੰਬੜੀ ਆਲਸੀ ਕੁੱਤੇ ਉੱਤੇ ਛਾਲ ਮਾਰ ਗਈ।",
    "gujarati":   "ઝડપી ભૂરી શિયાળ આળસુ કૂતરા પર કૂદી.",
    "georgian":   "სწრაფი ყავისფერი მელა ზარმაც ძაღლს გადაეხტა.",
    "armenian":   "Արագ շագանակագույն աղվեսը ծույլ շանն ընդ վրայ թռավ.",
    "thai":       "สุนัขจิ้งจอกสีน้ำตาลที่ว่องไวกระโดดข้ามสุนัขขี้เกียจ",
    "hebrew":     "השועל החום המהיר קפץ מעל הכלב העצלן.",
    "greek":      "Η γρήγορη καφέ αλεπού πήδηξε πάνω από τον τεμπέλη σκύλο.",
    "cyrillic":   "Быстрая коричневая лисица перепрыгнула через ленивую собаку.",
    "vai":        "ꕉꕞꕱ ꔓꘋ ꖸ ꕃꔤ ꗡꘋ ꔻꔤ ꕪꕞ",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _udhr_sentence(locale: str) -> str | None:
    """Extract a short representative sentence from the UDHR corpus."""
    stem = _LOCALE_TO_UDHR.get(locale)
    if not stem:
        return None
    path = UDHR_DIR / f"{stem}.xml"
    if not path.exists():
        return None
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        texts: list[str] = []
        for el in root.iter():
            t = (el.text or "").strip()
            if len(t) >= 40 and not t.isdigit():
                texts.append(t)
        # Prefer a sentence from 50–200 chars
        for t in texts:
            if 50 <= len(t) <= 200:
                return t
        # Fallback: first long-enough text node
        return texts[0] if texts else None
    except Exception:
        return None


def _probe_sentence(locale: str) -> str:
    """Return the best available probe sentence for a locale."""
    # 1. Locale-specific fallback (curated)
    if locale in _LOCALE_FALLBACK:
        return _LOCALE_FALLBACK[locale]
    # 2. UDHR corpus
    udhr = _udhr_sentence(locale)
    if udhr:
        return udhr
    # 3. Script-level fallback
    script = _LOCALE_SCRIPT.get(locale)
    if script and script in _SCRIPT_FALLBACK:
        return _SCRIPT_FALLBACK[script]
    # 4. Last resort: use the locale code itself as a marker
    return locale


def _char_fidelity(original: str, response: str) -> float:
    """Ratio of original non-whitespace characters reproduced in response."""
    orig_clean = "".join(c for c in original if not c.isspace())
    if not orig_clean:
        return 1.0
    resp_clean = "".join(c for c in response if not c.isspace())
    return difflib.SequenceMatcher(
        None, orig_clean, resp_clean, autojunk=False
    ).ratio()


def _has_byte_artifacts(text: str) -> bool:
    """Detect raw-byte leakage patterns like <0xe1>, \\xe1, or U+FFFD."""
    return bool(re.search(
        r'(\\x[0-9a-fA-F]{2}|<0x[0-9a-fA-F]{2}>|0x[0-9a-fA-F]{2}|\ufffd)',
        text,
    ))


# Matches genuine refusal to translate/understand — NOT incidental occurrences
# of "cannot" inside a valid translation (e.g. "I cannot go there" → English).
_REFUSAL_RE = re.compile(
    r"(cannot|can.?t|unable|not able)\s+(to\s+)?(translate|understand|read|process|identify)"
    r"|don.?t\s+(understand|recognize|know)\s+(this|the)\s+(text|script|language|writing|characters?)"
    r"|this\s+(text|script|language|writing)\s+is\s+(unknown|unfamiliar|unrecognized|not supported)",
    re.IGNORECASE,
)


def _safe_model_name(model: str) -> str:
    """Convert a model identifier to a filesystem-safe string."""
    return re.sub(r'[^A-Za-z0-9._-]', '_', model)


# ---------------------------------------------------------------------------
# OOV language detection
# ---------------------------------------------------------------------------

def find_oov_locales(
    threshold: float,
    min_models: int,
) -> list[tuple[str, float, int]]:
    """
    Scan all coverage files and return locales that are commonly OOV.

    Returns a list of (locale, median_score, model_count) tuples sorted by
    median_score ascending (worst languages first).

    A locale qualifies if:
    - It appears in at least `min_models` coverage files.
    - Its median weighted_score across those files is < `threshold`.
    """
    coverage_files = sorted(COVERAGE_DIR.glob("*.json"))
    if not coverage_files:
        print(f"[!] No coverage files found in {COVERAGE_DIR}")
        return []

    # locale → list of scores across all models
    locale_scores: dict[str, list[float]] = {}

    for path in coverage_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"    [warn] Could not read {path.name}: {exc}")
            continue
        langs = data.get("languages", {})
        for locale, info in langs.items():
            main = info.get("main") or {}
            score = main.get("weighted_score")
            if score is not None:
                locale_scores.setdefault(locale, []).append(float(score))

    results: list[tuple[str, float, int]] = []
    for locale, scores in locale_scores.items():
        if len(scores) < min_models:
            continue
        med = median(scores)
        if med < threshold:
            results.append((locale, med, len(scores)))

    results.sort(key=lambda t: t[1])  # worst first
    return results


# ---------------------------------------------------------------------------
# API callers
# ---------------------------------------------------------------------------

def _call_openai(
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict],
    timeout: int = 60,
) -> str:
    """Call an OpenAI-compatible chat completion endpoint."""
    import urllib.request
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": 400,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return body["choices"][0]["message"]["content"].strip()


def _call_gemini(
    api_key: str,
    model: str,
    prompt: str,
    timeout: int = 60,
) -> str:
    """Call the Google Gemini generateContent endpoint."""
    import urllib.request
    url = (
        f"https://generativelanguage.googleapis.com/v1beta"
        f"/models/{model}:generateContent?key={api_key}"
    )
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 400},
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return body["candidates"][0]["content"]["parts"][0]["text"].strip()


# ---------------------------------------------------------------------------
# Single-locale probe
# ---------------------------------------------------------------------------

def probe_locale(
    locale: str,
    lang_name: str,
    api_type: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    pause: float,
) -> dict:
    """Run the 3-test battery for one locale. Returns a result dict."""

    probe_text = _probe_sentence(locale)
    expected_script = _LOCALE_SCRIPT.get(locale, "")

    def call(prompt: str) -> tuple[str, str | None]:
        time.sleep(pause)
        try:
            if api_type == "gemini":
                return _call_gemini(api_key, model, prompt, timeout), None
            else:
                msgs = [{"role": "user", "content": prompt}]
                return _call_openai(api_key, base_url, model, msgs, timeout), None
        except Exception as exc:
            return "", str(exc)

    # ── Test 1: Echo ──────────────────────────────────────────────────
    echo_resp, echo_err = call(
        f"Please copy the following text exactly as written, "
        f"preserving every character without translation or modification:"
        f"\n\n{probe_text}"
    )
    if echo_err:
        fidelity = 0.0
        byte_artifacts = False
    else:
        fidelity = _char_fidelity(probe_text, echo_resp)
        byte_artifacts = _has_byte_artifacts(echo_resp)

    # ── Test 2: Script identification ─────────────────────────────────
    script_resp, script_err = call(
        f"What writing system or script is used in this text? "
        f"Answer with just the script name (one to three words):\n\n{probe_text}"
    )
    if script_err or not expected_script:
        script_recognized = None  # unknown
    else:
        script_recognized = expected_script.lower() in script_resp.lower()

    # ── Test 3: Translation ───────────────────────────────────────────
    trans_resp, trans_err = call(
        f"Translate the following {lang_name} text into English. "
        f"If you cannot understand the text, say so explicitly:\n\n{probe_text}"
    )
    if trans_err:
        translation_refused = True
    else:
        translation_refused = bool(_REFUSAL_RE.search(trans_resp))

    # ── Verdict ───────────────────────────────────────────────────────
    if fidelity >= 0.95 and script_recognized and not translation_refused:
        verdict = "Strong"
    elif fidelity >= 0.75 or script_recognized:
        verdict = "Partial"
    else:
        verdict = "Poor"

    return {
        "locale":              locale,
        "lang_name":           lang_name,
        "probe_text":          probe_text,
        "echo_response":       echo_resp,
        "echo_error":          echo_err,
        "echo_fidelity":       round(fidelity, 4),
        "byte_artifacts":      byte_artifacts,
        "expected_script":     expected_script,
        "script_response":     script_resp,
        "script_error":        script_err,
        "script_recognized":   script_recognized,
        "translation":         trans_resp,
        "translation_error":   trans_err,
        "translation_refused": translation_refused,
        "verdict":             verdict,
    }


# ---------------------------------------------------------------------------
# Language name lookup (best-effort via CLDR data)
# ---------------------------------------------------------------------------

def _build_name_map() -> dict[str, str]:
    """Return a locale → English name mapping from the CLDR languages file."""
    cldr_path = ROOT / "data" / "cldr" / "languages.json"
    if not cldr_path.exists():
        return {}
    try:
        data = json.loads(cldr_path.read_text(encoding="utf-8"))
        # Support both flat {locale: name} and nested structures
        if isinstance(data, dict):
            flat: dict[str, str] = {}
            for k, v in data.items():
                if isinstance(v, str):
                    flat[k] = v
                elif isinstance(v, dict) and "name" in v:
                    flat[k] = v["name"]
            return flat
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe a black-box LLM for language coverage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", required=True,
        help="Name of the model to probe (e.g. gemini-2.0-flash, gpt-4o).",
    )
    parser.add_argument(
        "--api-type", required=True, choices=["openai", "gemini"],
        help="API backend: 'openai' for OpenAI-compatible endpoints, "
             "'gemini' for Google Gemini.",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key. Defaults to OPENAI_API_KEY or GEMINI_API_KEY env var.",
    )
    parser.add_argument(
        "--base-url", default="https://api.openai.com/v1",
        help="Base URL for OpenAI-compatible endpoints "
             "(default: https://api.openai.com/v1).",
    )
    parser.add_argument(
        "--locales", nargs="+", metavar="LOCALE",
        help="Specific locale codes to probe (e.g. am bo zgh). "
             "Default: auto-detect commonly-OOV languages.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Coverage score threshold for 'commonly OOV' (default: 0.5).",
    )
    parser.add_argument(
        "--min-models", type=int, default=5,
        help="Minimum number of known models a locale must appear in to qualify "
             "(default: 5).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROBES_DIR,
        help="Directory to save probe results JSON (default: data/probes/).",
    )
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="API request timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--pause", type=float, default=1.0,
        help="Seconds to wait between API calls to avoid rate-limiting "
             "(default: 1.0).",
    )
    parser.add_argument(
        "--controls", nargs="*",
        metavar="LOCALE",
        default=["en", "es", "fr", "de", "zh", "ja", "ar", "ru", "hi", "ko"],
        help="Control locales (well-covered languages) prepended to the probe "
             "list as a sanity check. Pass --controls with no arguments to "
             "disable. Default: en es fr de zh ja ar ru hi ko",
    )
    args = parser.parse_args()

    # ── API key resolution ─────────────────────────────────────────────
    api_key = args.api_key
    if not api_key:
        env_var = "GEMINI_API_KEY" if args.api_type == "gemini" else "OPENAI_API_KEY"
        api_key = os.environ.get(env_var, "")
    if not api_key:
        print(
            f"[!] No API key provided. Set --api-key or the "
            f"{'GEMINI_API_KEY' if args.api_type == 'gemini' else 'OPENAI_API_KEY'} "
            f"environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Locale selection ───────────────────────────────────────────────
    name_map = _build_name_map()

    if args.locales:
        target_locales = [(loc, 0.0, 0) for loc in args.locales]
        print(f"[probe] Probing {len(target_locales)} user-specified locale(s).")
    else:
        print(
            f"[1/3] Scanning coverage data for OOV languages "
            f"(threshold={args.threshold}, min_models={args.min_models})…"
        )
        target_locales = find_oov_locales(args.threshold, args.min_models)
        print(f"      Found {len(target_locales)} commonly-OOV locale(s).")

    if not target_locales:
        print("[!] No locales to probe. Use --locales to specify them manually.")
        sys.exit(0)

    # ── Control locales ────────────────────────────────────────────────
    control_locales_set: set[str] = set()
    if args.controls:
        control_locales_set = set(args.controls)
        existing_locs = {loc for loc, *_ in target_locales}
        controls_to_add = [
            (loc, 1.0, -1)
            for loc in args.controls
            if loc not in existing_locs
        ]
        target_locales = controls_to_add + target_locales
        print(
            f"[+] Prepending {len(controls_to_add)} control locale(s) "
            f"({', '.join(args.controls)}) as sanity checks."
        )

    # ── Run probes ─────────────────────────────────────────────────────
    print(
        f"\n[2/3] Probing model '{args.model}' ({args.api_type}) "
        f"across {len(target_locales)} language(s)…\n"
    )

    results: dict[str, dict] = {}
    col_w = max(len(loc) for loc, *_ in target_locales) + 2

    header = (
        f"  {'Locale':<{col_w}} {'Language':<30} {'Fidelity':>9} "
        f"{'Script':>8} {'Trans':>7}  {'Verdict':<8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, (locale, med_score, _model_count) in enumerate(target_locales, 1):
        lang_name = name_map.get(locale, locale)
        prefix = f"  [{i}/{len(target_locales)}] {locale:<{col_w}} {lang_name:<30}"
        print(prefix, end="", flush=True)

        try:
            result = probe_locale(
                locale=locale,
                lang_name=lang_name,
                api_type=args.api_type,
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                timeout=args.timeout,
                pause=args.pause,
            )
            result["median_known_score"] = round(med_score, 4)
            result["is_control"] = locale in control_locales_set
            results[locale] = result

            fid = f"{result['echo_fidelity']:.0%}"
            scr = ("yes" if result["script_recognized"] else
                   "no" if result["script_recognized"] is False else "?")
            tr  = "no" if result["translation_refused"] else "yes"
            v   = result["verdict"]
            art = " [byte-leak]" if result["byte_artifacts"] else ""
            ctrl = " [control]" if result["is_control"] else ""
            print(f" {fid:>9} {scr:>8} {tr:>7}  {v:<8}{art}{ctrl}")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            results[locale] = {
                "locale": locale,
                "error": str(exc),
                "is_control": locale in control_locales_set,
            }

    # ── Save results ───────────────────────────────────────────────────
    print(f"\n[3/3] Saving results…")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_model_name(args.model)
    today = date.today().isoformat()
    out_path = args.output_dir / f"{safe_name}__{today}.json"

    payload = {
        "probe_model":  args.model,
        "api_type":     args.api_type,
        "base_url":     args.base_url if args.api_type == "openai" else None,
        "probed_at":    today,
        "threshold":    args.threshold,
        "min_models":   args.min_models,
        "total_locales": len(results),
        "results":      results,
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"      → {out_path}")

    # ── Summary ────────────────────────────────────────────────────────
    verdicts = [r.get("verdict", "?") for r in results.values()]
    strong  = verdicts.count("Strong")
    partial = verdicts.count("Partial")
    poor    = verdicts.count("Poor")
    errors  = sum(1 for r in results.values() if "error" in r)
    print(
        f"\n  Summary: {strong} Strong / {partial} Partial / {poor} Poor"
        + (f" / {errors} Errors" if errors else "")
    )


if __name__ == "__main__":
    main()
