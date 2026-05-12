Below is a practical guide to **collecting model‑vocabulary data** for the dashboard.  
It covers the most common sources, the technical steps to extract the token list, and the legal / operational considerations you’ll need to keep in mind.

---

## 1. Where to Find the Vocabulary

| Source | Typical Access Method | What You Get | Notes |
|--------|-----------------------|--------------|-------|
| **Open‑source LLMs (Hugging Face, GitHub, model repos)** | `git clone` or `pip install transformers` + `AutoTokenizer.from_pretrained()` | Tokenizer files (`vocab.txt`, `merges.txt`, `tokenizer.json`, `sentencepiece.model`, etc.) | Most are released under permissive licenses (MIT, Apache‑2.0). Verify the repo’s LICENSE. |
| **Commercial APIs (OpenAI, Anthropic, Google Gemini, Cohere, etc.)** | Provider‑specific API (e.g., `GET /v1/models/{model}/tokenizer`) *or* a **private endpoint** that returns the token list (some providers expose a “tokenizer” endpoint for debugging). | Usually a JSON list of tokens or a BPE merge table. | Requires API key, rate limits, and often a **non‑redistribution** clause. Use only for analysis, not for publishing the raw list. |
| **Model‑hosting platforms (Hugging Face Hub, ModelScope, TensorFlow Hub)** | `hf_hub_download`, `modelscope` CLI, or direct HTTP download of the `tokenizer` folder. | Same as open‑source; sometimes split into multiple files (`vocab.json`, `tokenizer_config.json`). | Platform may enforce usage restrictions (e.g., “research‑only”). |
| **Academic papers / supplemental material** | PDF / supplementary zip, sometimes a CSV of the top‑N tokens. | Partial vocab (often only the most frequent tokens). | Useful for sanity‑checking but not sufficient for full coverage. |
| **Community‑curated lists** (e.g., `tiktoken` for OpenAI models) | `pip install tiktoken` → `tiktoken.encoding_for_model("gpt‑4")` | Tokenizer object that can enumerate all token IDs → token strings. | Great for OpenAI models; still subject to the provider’s TOS. |
| **Self‑trained tokenizers** (if you want to compare a custom tokenizer) | `tokenizers` library → `ByteLevelBPETokenizer`, `SentencePieceProcessor`. | You control the vocab entirely. | Useful for “what‑if” experiments, not for existing commercial models. |

---

## 2. Technical Extraction Steps

Below are snippets for the three most common tokenizer formats. All code assumes a Python 3.10+ environment.

### 2.1. Hugging Face (BPE / WordPiece)

```python
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-2-13b-chat-hf"   # any HF model name
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# 1) Get the full token list (as strings)
tokens = tokenizer.get_vocab()          # dict: token_str -> token_id
token_list = [tok for tok, _ in sorted(tokens.items(), key=lambda x: x[1])]

# 2) Persist to disk (one token per line)
with open("vocab.txt", "w", encoding="utf-8") as f:
    for tok in token_list:
        f.write(tok + "\n")
```

*If the tokenizer is a `SentencePieceProcessor`:*

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("spm.model")                     # file from HF repo
tokens = [sp.id_to_piece(i) for i in range(sp.Get_piece_size())]
```

### 2.2. OpenAI / tiktoken (GPT‑4, GPT‑3.5)

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")   # or "gpt-3.5-turbo"
token_list = [enc.decode([i]) for i in range(enc.max_token_value + 1)]

with open("gpt4_vocab.txt", "w", encoding="utf-8") as f:
    for tok in token_list:
        f.write(tok + "\n")
```

### 2.3. Anthropic (Claude) – via public endpoint (if available)

```python
import requests, json

api_key = "YOUR_ANTHROPIC_API_KEY"
model = "claude-2.1"

headers = {"x-api-key": api_key, "content-type": "application/json"}
payload = {"model": model, "type": "tokenizer"}   # hypothetical endpoint
resp = requests.post("https://api.anthropic.com/v1/tokenizer", headers=headers, json=payload)

tokens = resp.json()["tokens"]   # list of strings
with open("claude_vocab.txt", "w", encoding="utf-8") as f:
    for tok in tokens:
        f.write(tok + "\n")
```

> **Tip:** If the provider does not expose a direct endpoint, you can infer the vocab by iterating over all possible token IDs (e.g., `range(0, 2**16)`) and calling the API’s `encode`/`decode` functions until you hit an error. This is slower but works for many BPE‑style APIs.

---

## 3. Normalising Tokens to Unicode Code Points

The dashboard works on **Unicode code points**, not on raw token strings. After you have the token list:

```python
def token_to_codepoints(tok: str) -> set[int]:
    # Decode any escape sequences (e.g., "\u00E9") and split into characters
    return {ord(ch) for ch in tok}

all_codepoints = set()
for tok in token_list:
    all_codepoints.update(token_to_codepoints(tok))
```

*Special handling:*

| Situation | Action |
|-----------|--------|
| **Byte‑level tokenizers** (e.g., LLaMA’s `ByteLevelBPETokenizer`) | Tokens may contain raw bytes (`\x00`). Decode them with `bytes(tok, "utf‑8", "replace")` before `ord`. |
| **Control characters / whitespace** | Keep them; they are part of Unicode and affect language coverage (e.g., `U+000A` line feed). |
| **Unicode Normalisation** | Apply `unicodedata.normalize("NFKC", tok)` to collapse equivalent forms before extracting code points. |

Store the resulting set of code points per model in the database (e.g., as a PostgreSQL `int[]` column or a compressed bitmap). This set is what you’ll intersect with the language‑character maps.

---

## 4. Legal / Licensing Checklist

| Item | What to Verify | Typical Outcome |
|------|----------------|-----------------|
| **Model License** | Look at the `LICENSE` file in the repo or the model card on HF. | Most open‑source LLMs are Apache‑2.0 / MIT → free to redistribute the vocab. |
| **Tokenizer License** | Tokenizer files often share the model’s license, but double‑check (`tokenizer.json`, `vocab.txt`). | Same as model license. |
| **Commercial API TOS** | Review the “Data Usage” and “Redistribution” sections. | Usually **you may not publish the raw token list**, but you can compute and display derived statistics (coverage percentages, missing‑character counts). |
| **Privacy / PII** | Token lists are public, but if you ever log user‑submitted text, scrub PII. | Follow your organization’s data‑privacy policy. |
| **Attribution** | Provide a citation to the model and tokenizer source. | Add a “Data Sources” page in the UI. |

*If a source is **non‑redistributable**, store the vocab only in a **private** location and expose only the **coverage metrics** (which are derived data).*

---

## 5. Operational Workflow for the Dashboard

1. **Upload / Register Model**  
   * Admin provides: model name, source (HF, API, local file), optional license text.  
   * Backend triggers a **background worker** to download the tokenizer files and compute the code‑point set.

2. **Cache the Code‑Point Set**  
   * Store as a **compressed bitmap** (`roaringbitmap` or `bitarray`) for fast set operations.  
   * Also keep the **raw token list** (optional, for debugging).

3. **Re‑run on Updates**  
   * Schedule a weekly job to check for newer tokenizer releases (e.g., a new HF version).  
   * If the hash of the tokenizer files changes, recompute the coverage and invalidate the cache.

4. **Expose API**  
   * `GET /api/models/{id}/vocab` → returns **metadata only** (size, hash).  
   * `GET /api/models/{id}/coverage?lang=eng` → returns coverage % and list of missing Unicode points (no raw tokens).

---

## 6. Quick “Starter” Script (End‑to‑End)

```python
#!/usr/bin/env python3
"""
One‑off script to fetch a tokenizer from HuggingFace,
convert it to a set of Unicode code points,
and write a JSON summary for the dashboard.
"""

import json, sys, unicodedata
from pathlib import Path
from transformers import AutoTokenizer

def token_to_cps(tok: str) -> set[int]:
    # Normalise then split into characters
    norm = unicodedata.normalize("NFKC", tok)
    return {ord(ch) for ch in norm}

def main(model_id: str, out_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    vocab = tokenizer.get_vocab()
    cps = set()
    for tok in sorted(vocab, key=vocab.get):
        cps.update(token_to_cps(tok))

    summary = {
        "model_id": model_id,
        "vocab_size": len(vocab),
        "codepoint_count": len(cps),
        "codepoints": sorted(cps),          # optional – can be omitted for size
    }

    Path(out_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"✅ {model_id} → {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: fetch_vocab.py <hf_model_id> <output_json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
```

Run it locally:

```bash
python fetch_vocab.py meta-llama/Llama-2-13b-chat-hf llama2_13b.json
```

The resulting JSON can be loaded directly into the dashboard’s ingestion pipeline.

---

## 7. TL;DR Checklist for Your Team

| ✅ Action | 📄 Details |
|----------|-----------|
| **Identify target models** | List HF repo names, API endpoints, and any private models. |
| **Write a downloader per source** | Use `AutoTokenizer`, `tiktoken`, or HTTP calls. |
| **Normalize tokens → Unicode code points** | Apply NFKC, handle byte‑level tokens. |
| **Store as compressed bitmap** | Fast set intersection later. |
| **Validate licensing** | Ensure you can keep the data private or only expose derived stats. |
| **Automate with a background worker** | Celery / FastAPI background tasks. |
| **Expose coverage API** | No raw vocab leakage. |
| **Document data sources** | Add a “Data Sources” page in the UI for transparency. |

With this pipeline in place, you’ll be able to populate the leaderboard with accurate, up‑to‑date coverage numbers for any LLM you support. Happy building!