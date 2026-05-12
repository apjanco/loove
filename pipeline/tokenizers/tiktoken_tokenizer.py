"""
tiktoken extractor for OpenAI models.

tiktoken is always byte-level: every possible UTF-8 byte sequence can be
represented, so has_byte_fallback is always True.

The key distinction we draw:
  - A token whose bytes decode cleanly to a single Unicode character → Tier 0.
  - A token whose bytes decode to multiple characters → each char goes to Tier 1
    (unless another token already gives it Tier 0).
  - A token whose bytes are NOT valid UTF-8 on their own is a partial byte-sequence
    token (raw byte fallback). It does NOT contribute to code-point coverage —
    that is intentional, because representing U+4E2D (中) as three separate byte
    tokens [0xE4, 0xB8, 0xAD] is degraded (Tier 2) support, not Tier 0.
"""
from __future__ import annotations

from .base import ModelVocabData

# Mapping from model name aliases to tiktoken encoding names, kept here so we
# can surface a clear error when an unknown model is requested.
_KNOWN_ENCODINGS: dict[str, str] = {
    "gpt-4":                  "cl100k_base",
    "gpt-4-turbo":            "cl100k_base",
    "gpt-4o":                 "o200k_base",
    "gpt-4o-mini":            "o200k_base",
    "gpt-3.5-turbo":          "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    "o1":                     "o200k_base",
    "o1-mini":                "o200k_base",
    "o3":                     "o200k_base",
    "o3-mini":                "o200k_base",
    # Pass encoding names directly as well
    "cl100k_base":            "cl100k_base",
    "o200k_base":             "o200k_base",
    "p50k_base":              "p50k_base",
    "r50k_base":              "r50k_base",
}


def extract(model_name: str) -> ModelVocabData:
    """
    Extract ModelVocabData from a tiktoken encoding.

    Args:
        model_name: A model alias ("gpt-4", "gpt-4o") or a direct tiktoken
                    encoding name ("cl100k_base", "o200k_base").
    """
    import tiktoken

    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Try as a direct encoding name
        try:
            enc = tiktoken.get_encoding(model_name)
        except Exception:
            known = ", ".join(sorted(_KNOWN_ENCODINGS))
            raise ValueError(
                f"Unknown tiktoken model or encoding: '{model_name}'.\n"
                f"Known values: {known}"
            )

    vocab_size = enc.n_vocab
    codepoints_single: set[int] = set()
    codepoints_any: set[int] = set()

    for token_id in range(vocab_size):
        try:
            raw: bytes = enc.decode_single_token_bytes(token_id)
        except Exception:
            continue

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            # This is a partial-byte or raw-byte fallback token.
            # It provides Tier-2 coverage but does NOT count as a real
            # code-point token — skip it here.
            continue

        cps = [ord(ch) for ch in text if ord(ch) != 0xFFFD]
        if not cps:
            continue

        codepoints_any.update(cps)
        if len(cps) == 1:
            codepoints_single.add(cps[0])

    return ModelVocabData(
        model_id=model_name,
        source="tiktoken",
        vocab_size=vocab_size,
        has_byte_fallback=True,   # tiktoken is always byte-level
        codepoints_single=codepoints_single,
        codepoints_any=codepoints_any,
    )
