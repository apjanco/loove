---
title: LLM Vocabulary Coverage Dashboard
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.14.0"
app_file: app.py
pinned: false
---

<img height="200" src="loove-image.jpg" />

# Loove — LLM Language Coverage Dashboard

A data pipeline and dashboard for measuring how well large language models (LLMs) support the world's languages — not just whether a script can be encoded, but *how well* it is supported.

## What It Measures

Coverage is not binary. A model may technically encode any Unicode character via byte fallback, but that tells us almost nothing about whether it has real language competence. Loove assigns each character in a language's alphabet to one of four tiers:

| Tier | Meaning | Weight |
|------|---------|--------|
| **0 — Native token** | A dedicated single-character token exists. The model almost certainly saw this character in training. | 1.0 |
| **1 — Embedded** | The character appears inside multi-character tokens but has no standalone token. The model knows it exists, but cannot produce it atomically. | 0.7 |
| **2 — Byte fallback** | No token surface contains this character; it is encoded as raw UTF-8 bytes. The model has no language-level signal for it. | 0.2 |
| **3 — Unreachable** | Cannot be encoded at all (applies to non-byte-fallback tokenizers like BERT for unseen scripts). | 0.0 |

The **weighted coverage score** for a language is the tier-weighted average across all characters in its CLDR exemplar set. The dashboard surfaces both the aggregate score and a per-tier breakdown for drill-down.

In addition to vocabulary-based tier scoring, Loove measures **fertility** — tokens per character on real text (UDHR corpus) — which captures the practical cost of using a language with a given model.

---

## Data Sources

| Source | What it provides | License |
|--------|-----------------|---------|
| [Unicode CLDR](https://github.com/unicode-org/cldr) | Exemplar character sets for ~250 locales — the definitive list of characters each language uses | Unicode License |
| [Glottolog 5.3](https://glottolog.org/meta/downloads) | Registry of ~8,000 languages with ISO 639-3 codes, language families, macroareas, and coordinates | CC BY 4.0 |
| [Unicode UCD](https://unicode.org/Public/UCD/latest/ucd/) | Character names and script assignments | Unicode License |
| [UDHR corpus](https://unicode.org/udhr/) | Universal Declaration of Human Rights in ~500 languages — used for fertility scoring | Public domain |
| HuggingFace Hub | Tokenizer files for open-source models | Varies per model |
| tiktoken | OpenAI model tokenizers | MIT |

---

## Project Layout

```
loove/
├── requirements.txt
├── pipeline/
│   ├── tokenizers/
│   │   ├── base.py                 # ModelVocabData — the only type stored to disk
│   │   ├── hf_tokenizer.py         # HuggingFace AutoTokenizer extraction
│   │   └── tiktoken_tokenizer.py   # tiktoken (OpenAI) extraction
│   ├── unicode/
│   │   ├── cldr_loader.py          # CLDR exemplar character parser
│   │   ├── ucd_loader.py           # UnicodeData.txt + Scripts.txt loader
│   │   └── glottolog_loader.py     # Glottolog languoid CSV parser
│   └── analysis/
│       ├── tier_classifier.py      # Tier 0–3 classification + weighted score
│       ├── fertility.py            # UDHR download + tokens/char metric
│       └── coverage.py             # Aggregates everything → JSON
├── scripts/
│   ├── fetch_cldr.py               # One-off: build data/cldr/languages.json
│   ├── fetch_glottolog.py          # One-off: build data/glottolog/languoids.json
│   ├── fetch_udhr.py               # One-off: pre-cache UDHR corpus texts
│   └── ingest_model.py             # Run full pipeline for one model
└── data/                           # Generated — not committed to git
    ├── cldr/
    │   ├── raw/                    # Cached per-locale CLDR XML files
    │   └── languages.json          # Parsed exemplar data for all locales
    ├── glottolog/
    │   ├── glottolog_languoid.csv  # Raw CSV (cached)
    │   ├── languoids.json          # Parsed language registry
    │   └── families.json           # Language family index
    ├── ucd/
    │   ├── UnicodeData.txt
    │   └── Scripts.txt
    ├── udhr/
    │   ├── index.xml
    │   └── *.txt                   # Per-language UDHR texts
    ├── models/
    │   └── {model_id}.json         # Extracted code-point sets (no raw vocab)
    └── coverage/
        └── {model_id}.json         # Full per-language coverage results
```

> **Privacy note**: Raw tokenizer vocabulary files are never written to disk. Only derived data (Unicode code-point sets) is stored, in compliance with commercial API terms of service that prohibit redistribution of raw token lists.

---

## Setup

```bash
# Python 3.10+ required
pip install -r requirements.txt
```

---

## Running the Pipeline

### Step 1 — Fetch reference data (once)

```bash
# Download and parse CLDR exemplar character sets (~250 locales, ~5 min)
python scripts/fetch_cldr.py

# Download Glottolog language registry (~8,000 languages)
python scripts/fetch_glottolog.py --stats

# Pre-cache UDHR corpus for fertility scoring (optional but recommended)
python scripts/fetch_udhr.py
```

### Step 2 — Ingest a model

```bash
# OpenAI model via tiktoken (no token needed)
python scripts/ingest_model.py --model gpt-4o --source tiktoken

# With fertility scoring
python scripts/ingest_model.py --model gpt-4o --source tiktoken --fertility

# HuggingFace open-source model
python scripts/ingest_model.py --model mistralai/Mistral-7B-v0.1 --source hf

# Gated HuggingFace model (accept the license on HF first)
HF_TOKEN=hf_xxx python scripts/ingest_model.py \
    --model meta-llama/Llama-2-13b-chat-hf --source hf --fertility

# Local tokenizer directory
python scripts/ingest_model.py \
    --model my-model --source hf --local /path/to/tokenizer/dir

# Analyse only a subset of languages
python scripts/ingest_model.py --model gpt-4o --source tiktoken \
    --locales hi ar zh ja ko sw yo
```

### Output

Each ingested model produces two files:

- `data/models/{model_id}.json` — extracted code-point sets (fast to recompute coverage from)
- `data/coverage/{model_id}.json` — full per-language results including tier breakdown, optional fertility, and Glottolog metadata

**Sample coverage entry:**

```json
"hi": {
  "name":        "Hindi",
  "script":      "Deva",
  "glottocode":  "hin1269",
  "iso639_3":    "hin",
  "macroarea":   "Eurasia",
  "family_id":   "indo1319",
  "family_name": "Indo-European",
  "latitude":    25.0,
  "longitude":   77.0,
  "has_cldr":    true,
  "main": {
    "total":          68,
    "weighted_score": 0.9559,
    "tier0_count":    62,
    "tier1_count":    4,
    "tier2_count":    2,
    "tier3_count":    0,
    "tier1": [2366, 2367],
    "tier2": [2385],
    "tier3": []
  },
  "fertility": {
    "tokens_per_char": 1.42,
    "tokens_per_word": 5.31,
    "sample_chars":    2847,
    "sample_tokens":   4041
  }
}
```

Languages tracked in Glottolog but lacking CLDR exemplar data appear as stub entries with `"has_cldr": false` — they carry family/macroarea metadata but no tier analysis.

---

## Supported Tokenizer Formats

| Format | Models | Notes |
|--------|--------|-------|
| **HuggingFace AutoTokenizer** | Llama, Mistral, Falcon, Gemma, Qwen, Phi, … | Handles ByteLevelBPE, SentencePiece, and WordPiece variants automatically |
| **tiktoken** | GPT-4, GPT-4o, GPT-3.5-turbo, o1, o3, … | Pass model alias or encoding name (`cl100k_base`, `o200k_base`) |

HuggingFace gated models (Llama 2/3, Gemma) require accepting the license on [huggingface.co](https://huggingface.co) and providing an access token via `--hf-token` or the `HF_TOKEN` environment variable.

---

## Architecture (planned)

```
data/coverage/*.json
        │
        ▼
  FastAPI backend          GET /api/coverage
  analysis service    →    GET /api/models
                           GET /api/languages
                           GET /api/leaderboard
        │
        ▼
  React + Recharts frontend
  ├─ Leaderboard table (sort by language, script, family, macroarea)
  ├─ Per-model drill-down (tier breakdown + fertility chart)
  ├─ Language detail (all models × one language)
  └─ Export (CSV / JSON)
```

---

## License

Source code: MIT.  
Data produced by this pipeline is derived from sources with their own licenses — see the Data Sources table above. Do not publicly redistribute raw tokenizer vocabulary data obtained from commercial API providers.
