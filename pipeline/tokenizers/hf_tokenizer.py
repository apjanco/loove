"""
HuggingFace tokenizer extractor.

Handles the three main internal token formats used by HF tokenizers:

  1. ByteLevelBPE (GPT-2, RoBERTa, LLaMA-3, Mistral …)
     Token strings use the GPT-2 byte→unicode mapping, e.g. "Ġhello" → " hello".

  2. SentencePiece with byte fallback (LLaMA-2, T5, mBART …)
     Normal tokens use ▁ as a space prefix; byte tokens look like "<0x41>".

  3. WordPiece (BERT, DistilBERT …)
     Continuation tokens use "##". No byte fallback; true OOV characters are
     replaced with [UNK] → they are Tier 3.

In all cases we decode each token to its actual Unicode string via
`tokenizer.convert_tokens_to_string([token])`, then extract code points.
For ByteLevelBPE we use the canonical byte→unicode mapping directly so we
never lose information to a multi-token decode context.
"""
from __future__ import annotations

import os
import re

from .base import ModelVocabData


# ---------------------------------------------------------------------------
# GPT-2 byte ↔ unicode mapping (canonical implementation)
# ---------------------------------------------------------------------------

def _gpt2_bytes_to_unicode() -> dict[int, str]:
    """Return the byte→unicode char mapping used by GPT-2 ByteLevelBPE."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_B2U: dict[int, str] = _gpt2_bytes_to_unicode()
_U2B: dict[str, int] = {v: k for k, v in _B2U.items()}


def _decode_gpt2_token(token_str: str) -> bytes | None:
    """
    Decode a GPT-2 byte-level BPE token string to raw bytes.
    Returns None if the string contains chars not in the GPT-2 mapping
    (e.g. special tokens like <|endoftext|>).
    """
    try:
        return bytes([_U2B[ch] for ch in token_str])
    except KeyError:
        return None


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract(
    model_id: str,
    hf_token: str | None = None,
    local_path: str | None = None,
) -> ModelVocabData:
    """
    Load a HuggingFace tokenizer and extract its Unicode code-point coverage.

    Args:
        model_id:   HuggingFace repo ID, e.g. "meta-llama/Llama-2-13b-chat-hf".
        hf_token:   Optional HF access token for gated models. Falls back to the
                    HF_TOKEN environment variable if not provided.
        local_path: If given, load from this local directory instead of the Hub.
                    The model_id is still used as the canonical identifier in the
                    stored data.
    """
    from transformers import AutoTokenizer

    token = hf_token or os.environ.get("HF_TOKEN")
    source_path = local_path or model_id
    load_kwargs: dict = {"trust_remote_code": True}
    if token:
        load_kwargs["token"] = token

    try:
        tokenizer = AutoTokenizer.from_pretrained(source_path, **load_kwargs)
    except OSError as exc:
        msg = str(exc).lower()
        if "gated" in msg or "access" in msg or "401" in msg or "403" in msg:
            raise PermissionError(
                f"Model '{model_id}' is gated.\n"
                "  1. Accept the license at https://huggingface.co/{model_id}\n"
                "  2. Pass your token via --hf-token or HF_TOKEN env var."
            ) from exc
        raise

    vocab: dict[str, int] = tokenizer.get_vocab()
    vocab_size = len(vocab)

    # ------------------------------------------------------------------
    # Detect tokenizer flavour
    # ------------------------------------------------------------------
    is_bpe_byte_level = hasattr(tokenizer, "byte_encoder")

    # SentencePiece byte-fallback: look for <0xXX> tokens in the vocab
    _sp_byte_re = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
    has_sp_byte_tokens = any(
        _sp_byte_re.match(k) for k in list(vocab)[:500]
    )

    has_byte_fallback = is_bpe_byte_level or has_sp_byte_tokens

    special_tokens: set[str] = set(tokenizer.all_special_tokens or [])

    codepoints_single: set[int] = set()
    codepoints_any: set[int] = set()

    for token_str in vocab:
        if token_str in special_tokens:
            continue

        # Skip SentencePiece raw-byte tokens — they are the fallback mechanism,
        # not evidence that the model represents the character meaningfully.
        if has_sp_byte_tokens and _sp_byte_re.match(token_str):
            continue

        decoded: str | None = None

        if is_bpe_byte_level:
            # Decode via the GPT-2 byte mapping (lossless, no context needed)
            raw = _decode_gpt2_token(token_str)
            if raw is not None:
                decoded = raw.decode("utf-8", errors="replace")
            # If _decode_gpt2_token returns None it's a special token with
            # characters outside the mapping — skip it.
        else:
            # WordPiece / SentencePiece / other: use the tokenizer's own decoder.
            # convert_tokens_to_string works at the single-token level and strips
            # internal markers (## for WordPiece, ▁ for SP).
            try:
                decoded = tokenizer.convert_tokens_to_string([token_str])
            except Exception:
                # Last resort: treat the raw token string as the surface form.
                decoded = token_str

        if not decoded:
            continue

        # Extract code points, ignoring the Unicode replacement character
        # (U+FFFD) which signals a decode error rather than real coverage.
        cps = [ord(ch) for ch in decoded if ord(ch) != 0xFFFD]
        if not cps:
            continue

        codepoints_any.update(cps)

        # Single code-point token → this character has Tier-0 support.
        if len(cps) == 1:
            codepoints_single.add(cps[0])

    return ModelVocabData(
        model_id=model_id,
        source="huggingface",
        vocab_size=vocab_size,
        has_byte_fallback=has_byte_fallback,
        codepoints_single=codepoints_single,
        codepoints_any=codepoints_any,
    )
