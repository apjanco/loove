#!/usr/bin/env python3
"""
Ingest a model's tokenizer and compute language coverage.

Steps:
  1. Load tokenizer (HuggingFace Hub or tiktoken) and extract the Unicode
     code-point sets (codepoints_single, codepoints_any).
  2. Save derived vocab data to data/models/{model_id}.json.
     Raw token strings and vocabulary files are NOT stored.
  3. Load (or build) the CLDR language database.
  4. Classify every CLDR exemplar character into Tiers 0-3.
  5. Optionally compute fertility from the UDHR corpus.
  6. Save full coverage result to data/coverage/{model_id}.json.

Examples:
  # OpenAI model via tiktoken
  python scripts/ingest_model.py --model gpt-4 --source tiktoken

  # OpenAI model with fertility
  python scripts/ingest_model.py --model gpt-4o --source tiktoken --fertility

  # HuggingFace open-source model
  python scripts/ingest_model.py --model mistralai/Mistral-7B-v0.1 --source hf

  # Gated HuggingFace model (token from env or flag)
  HF_TOKEN=hf_xxx python scripts/ingest_model.py \\
      --model meta-llama/Llama-2-13b-chat-hf --source hf

  # Local tokenizer directory (model_id used only as the stored identifier)
  python scripts/ingest_model.py \\
      --model my-custom-model --source hf --local /path/to/tokenizer/dir

  # Only compute coverage for a subset of languages
  python scripts/ingest_model.py --model gpt-4 --source tiktoken \\
      --locales hi ar zh ja ko
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parents[1]))

from pipeline.tokenizers import hf_tokenizer, tiktoken_tokenizer
from pipeline.tokenizers.base import ModelVocabData
from pipeline.unicode.cldr_loader import build_language_database
from pipeline.analysis.coverage import compute_coverage, save_coverage, save_vocab


# ---------------------------------------------------------------------------
# Tokenize-function factories (for fertility scoring)
# ---------------------------------------------------------------------------

def _make_hf_tokenize_fn(model_id: str, hf_token: str | None) -> Callable:
    from transformers import AutoTokenizer
    kwargs = {"trust_remote_code": True}
    if hf_token:
        kwargs["token"] = hf_token
    tok = AutoTokenizer.from_pretrained(model_id, **kwargs)
    return lambda text: tok.encode(text, add_special_tokens=False)


def _make_tiktoken_fn(model_name: str) -> Callable:
    import tiktoken
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding(model_name)
    return lambda text: enc.encode(text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a model tokenizer and compute language coverage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace repo ID, tiktoken model name, or local path.",
    )
    parser.add_argument(
        "--source", required=True, choices=["hf", "tiktoken"],
        help="Tokenizer backend: 'hf' for HuggingFace, 'tiktoken' for OpenAI.",
    )
    parser.add_argument(
        "--hf-token", default=os.environ.get("HF_TOKEN"),
        help="HuggingFace access token (for gated models). "
             "Defaults to the HF_TOKEN env var.",
    )
    parser.add_argument(
        "--local", metavar="PATH",
        help="Load tokenizer from this local directory instead of the Hub. "
             "--model is still used as the stored model identifier.",
    )
    parser.add_argument(
        "--fertility", action="store_true",
        help="Compute fertility scores using UDHR corpus. "
             "Requires UDHR files (run fetch_udhr.py first, or they are "
             "downloaded on demand).",
    )
    parser.add_argument(
        "--locales", nargs="+", metavar="LOCALE",
        help="Only compute coverage for these locale IDs (e.g. hi ar zh). "
             "Default: all locales in the CLDR database.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Extract vocabulary
    # ------------------------------------------------------------------
    print(f"[1/4] Extracting tokenizer: {args.model} ({args.source})")
    if args.source == "hf":
        vocab: ModelVocabData = hf_tokenizer.extract(
            model_id=args.model,
            hf_token=args.hf_token,
            local_path=args.local,
        )
    else:
        vocab = tiktoken_tokenizer.extract(args.model)

    print(
        f"      Vocab size:          {vocab.vocab_size:>10,} tokens\n"
        f"      Single-char tokens:  {len(vocab.codepoints_single):>10,} unique code points\n"
        f"      Multi-token chars:   {len(vocab.codepoints_any) - len(vocab.codepoints_single):>10,} unique code points\n"
        f"      Byte fallback:       {vocab.has_byte_fallback}"
    )

    # ------------------------------------------------------------------
    # 2. Save derived vocab data (no raw tokens stored)
    # ------------------------------------------------------------------
    print("[2/4] Saving vocab data…")
    vocab_path = save_vocab(vocab)
    print(f"      → {vocab_path}")

    # ------------------------------------------------------------------
    # 3. Load CLDR language database
    # ------------------------------------------------------------------
    print("[3/4] Loading CLDR language database…")
    language_db = build_language_database(locales=args.locales)
    print(f"      {len(language_db)} languages loaded")

    # ------------------------------------------------------------------
    # 4. Compute coverage (with optional fertility)
    # ------------------------------------------------------------------
    print("[4/4] Computing coverage…")

    tokenize_fn = None
    if args.fertility:
        print("      Fertility enabled — building tokenize function…")
        if args.source == "hf":
            tokenize_fn = _make_hf_tokenize_fn(
                args.local or args.model, args.hf_token
            )
        else:
            tokenize_fn = _make_tiktoken_fn(args.model)

    result = compute_coverage(
        vocab=vocab,
        language_db=language_db,
        tokenize_fn=tokenize_fn,
    )

    # Summary statistics
    lang_results = result["languages"]
    scored    = {lid: v for lid, v in lang_results.items() if v.get("has_cldr", True) and "main" in v}
    stubs     = {lid: v for lid, v in lang_results.items() if not v.get("has_cldr", True)}
    scores    = [v["main"]["weighted_score"] for v in scored.values()]

    if scores:
        avg     = sum(scores) / len(scores)
        perfect = sum(1 for s in scores if s >= 1.0)
        worst   = sorted(
            ((lid, v["main"]["weighted_score"]) for lid, v in scored.items()),
            key=lambda x: x[1],
        )[:5]
        print(f"\n      CLDR languages analysed:        {len(scores)}")
        print(f"      Glottolog stub languages:        {len(stubs)}")
        print(f"      Average coverage score:          {avg:.1%}")
        print(f"      Languages at full coverage:      {perfect}")
        print("      Lowest coverage:")
        for lid, score in worst:
            name = scored[lid].get("name", lid)
            print(f"        {lid:<10} {name:<25} {score:.1%}")

    out_path = save_coverage(result)
    print(f"\n      Coverage saved → {out_path}")


if __name__ == "__main__":
    main()
