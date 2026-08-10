"""
Regression tests for byte-fallback detection and extraction determinism.

Background
----------
`has_byte_fallback` was originally derived from two heuristics:

    is_bpe_byte_level = hasattr(tokenizer, "byte_encoder")
    has_sp_byte_tokens = any(_sp_byte_re.match(k) for k in list(vocab)[:500])

Both fail on current `transformers`. `byte_encoder` exists only on slow
(Python) tokenizers, while fast tokenizers are the default since 4.x; and the
`<0xXX>` scan inspects the first 500 keys of `get_vocab()`, a dict whose
iteration order is not token-id order, so the byte tokens frequently fall
outside the window.

Consequence: characters that byte fallback places in Tier 2 (weight 0.2) were
scored as Tier 3 (weight 0.0), understating every affected model's coverage —
by up to 6.3 percentage points of mean score. The same flag also gated whether
`<0xXX>` tokens were skipped during extraction, so two runs over the same
tokenizer could produce different code-point sets: Gemma-2 2b/9b and 27b share
one tokenizer but their stored sets differ by exactly U+0000 and U+0017.

The per-model expectations below were established by round-tripping characters
from scripts absent from each vocabulary through the live tokenizers.

The parametrized tests hit the HuggingFace Hub; the rest are offline.
Run with:  pytest tests/test_byte_fallback.py
"""
from __future__ import annotations

import pytest

from pipeline.tokenizers import hf_tokenizer
from pipeline.tokenizers.hf_tokenizer import _PROBE_CANDIDATES, _probe_byte_fallback

# Confirmed by live round-trip: byte fallback works, though the old heuristics
# reported False for every one of these.
HAS_FALLBACK = [
    "Qwen/Qwen2.5-0.5B",
    "tiiuae/falcon-7b",
    "allenai/OLMo-2-1124-7B",
    "microsoft/phi-4",
]

# Genuine unknown-token substitution: False is the correct answer. These guard
# against a "fix" that simply returns True everywhere.
NO_FALLBACK = [
    "google-bert/bert-base-multilingual-cased",
    "FacebookAI/xlm-roberta-base",
    "facebook/xglm-564M",
]


# ---------------------------------------------------------------------------
# Offline tests: the probe's decision logic
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """
    Minimal stand-in exposing the three methods the probe uses.

    lossless=False simulates a tokenizer that drops unrepresentable input;
    unk_token set and returned simulates explicit unknown-token substitution.
    """

    def __init__(self, *, lossless: bool = True, unk_token: str | None = None,
                 emit_unk: bool = False) -> None:
        self.lossless = lossless
        self.unk_token = unk_token
        self.emit_unk = emit_unk
        self.encoded: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:  # noqa: ARG002
        self.encoded.append(text)
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:  # noqa: ARG002
        return self.encoded[-1] if self.lossless else ""

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:  # noqa: ARG002
        return [self.unk_token] if self.emit_unk else ["a", "b", "c"]


def test_lossless_roundtrip_is_fallback() -> None:
    assert _probe_byte_fallback(_FakeTokenizer(), set()) is True


def test_lossy_roundtrip_is_not_fallback() -> None:
    assert _probe_byte_fallback(_FakeTokenizer(lossless=False), set()) is False


def test_unk_substitution_is_not_fallback() -> None:
    tok = _FakeTokenizer(unk_token="[UNK]", emit_unk=True)
    assert _probe_byte_fallback(tok, set()) is False


def test_no_usable_probe_returns_false() -> None:
    """
    When the vocabulary already covers every probe character there is nothing
    to test, and a lossless round-trip would prove nothing. The helper must
    report False rather than a false positive.
    """
    covered = {ord(c) for c in _PROBE_CANDIDATES}
    tok = _FakeTokenizer()
    assert _probe_byte_fallback(tok, covered) is False
    assert tok.encoded == [], "covered characters must not be probed"


def test_probe_skips_covered_characters() -> None:
    """Only characters absent from the vocabulary are valid probes."""
    covered = {ord(_PROBE_CANDIDATES[0])}
    tok = _FakeTokenizer()
    _probe_byte_fallback(tok, covered)
    assert _PROBE_CANDIDATES[0] not in tok.encoded
    assert _PROBE_CANDIDATES[1] in tok.encoded


def test_byte_token_skip_is_unconditional() -> None:
    """
    `<0xXX>` tokens must be excluded from the code-point sets regardless of any
    flag. This is the determinism fix: the skip used to be gated on a flag
    whose value depended on where the byte tokens fell in dict order, so the
    control characters U+0000 and U+0017 were admitted on some runs only.
    """
    assert hf_tokenizer._SP_BYTE_RE.match("<0x00>")
    assert hf_tokenizer._SP_BYTE_RE.match("<0x17>")
    assert hf_tokenizer._SP_BYTE_RE.match("<0xFF>")
    assert not hf_tokenizer._SP_BYTE_RE.match("<0xGG>")
    assert not hf_tokenizer._SP_BYTE_RE.match("hello")

    source = hf_tokenizer.extract.__code__.co_consts
    assert not any(
        isinstance(c, str) and "has_sp_byte_tokens" in c for c in source
    ), "extraction must no longer branch on the old vocabulary-shape flag"


# ---------------------------------------------------------------------------
# Live tests: real tokenizers from the Hub
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_id", HAS_FALLBACK)
def test_live_byte_fallback_detected(model_id: str) -> None:
    data = hf_tokenizer.extract(model_id)
    assert data.has_byte_fallback is True, (
        f"{model_id} round-trips out-of-vocabulary characters losslessly, "
        "so byte fallback must be detected"
    )


@pytest.mark.parametrize("model_id", NO_FALLBACK)
def test_live_unk_substitution_detected(model_id: str) -> None:
    data = hf_tokenizer.extract(model_id)
    assert data.has_byte_fallback is False, (
        f"{model_id} replaces out-of-vocabulary characters with its unknown "
        "token, which is not byte fallback"
    )


def test_live_extraction_is_repeatable() -> None:
    """Two extractions of the same tokenizer must agree exactly."""
    first = hf_tokenizer.extract("Qwen/Qwen2.5-0.5B")
    second = hf_tokenizer.extract("Qwen/Qwen2.5-0.5B")
    assert first.codepoints_single == second.codepoints_single
    assert first.codepoints_any == second.codepoints_any
    assert first.has_byte_fallback == second.has_byte_fallback
