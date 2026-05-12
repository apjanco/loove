"""
Tier classifier: assign a coverage tier to each Unicode code point in a
language's exemplar set, relative to a model's extracted vocabulary data.

Tier definitions
----------------
  Tier 0 — Native single token
    At least one token in the vocabulary decodes to exactly this character.
    The model almost certainly encountered this character during training and
    can predict/generate it as a single unit.
    Weight: 1.0

  Tier 1 — Embedded in multi-character tokens
    The character appears within longer tokens but has no dedicated single-
    character token.  The model has seen this character in context but cannot
    produce it atomically.
    Weight: 0.7

  Tier 2 — Byte-fallback only
    The character is absent from all token surfaces, but the tokenizer can
    still encode it by splitting its UTF-8 byte sequence into individual byte
    tokens.  The model has essentially no language-specific training signal
    for this character — it sees raw bytes.
    Weight: 0.2

  Tier 3 — Unreachable
    The character cannot be encoded at all (no byte fallback, character absent
    from vocabulary).  Applies to WordPiece tokenizers (e.g. BERT) for scripts
    not in their training vocabulary.
    Weight: 0.0

The weighted score per language is:
    score = Σ(tier_weight[tier] for cp in exemplar) / len(exemplar)
"""
from __future__ import annotations

from dataclasses import dataclass

from pipeline.tokenizers.base import ModelVocabData

TIER_WEIGHTS: dict[int, float] = {
    0: 1.0,
    1: 0.7,
    2: 0.2,
    3: 0.0,
}


@dataclass
class TierResult:
    """Tier breakdown for a set of code points against one model's vocabulary."""

    tier0: list[int]   # Sorted lists of code points at each tier
    tier1: list[int]
    tier2: list[int]
    tier3: list[int]

    @property
    def total(self) -> int:
        return len(self.tier0) + len(self.tier1) + len(self.tier2) + len(self.tier3)

    @property
    def weighted_score(self) -> float:
        """Coverage score in [0.0, 1.0]. Higher is better."""
        if self.total == 0:
            return 0.0
        raw = (
            len(self.tier0) * TIER_WEIGHTS[0]
            + len(self.tier1) * TIER_WEIGHTS[1]
            + len(self.tier2) * TIER_WEIGHTS[2]
            + len(self.tier3) * TIER_WEIGHTS[3]
        )
        return raw / self.total

    def to_dict(self) -> dict:
        return {
            "total":          self.total,
            "weighted_score": round(self.weighted_score, 4),
            "tier0_count":    len(self.tier0),
            "tier1_count":    len(self.tier1),
            "tier2_count":    len(self.tier2),
            "tier3_count":    len(self.tier3),
            # Store missing / degraded code points for drill-down UI
            # (omit tier0 from JSON — it's the happy path)
            "tier1": self.tier1,
            "tier2": self.tier2,
            "tier3": self.tier3,
        }


def classify(codepoints: set[int], vocab: ModelVocabData) -> TierResult:
    """
    Classify each code point in `codepoints` into the four coverage tiers
    relative to `vocab`.

    The best tier for a code point is determined by what the vocabulary provides:
      - codepoints_single: Tier 0
      - codepoints_any (but not single): Tier 1
      - not in any token, but has_byte_fallback: Tier 2
      - otherwise: Tier 3
    """
    tier0: list[int] = []
    tier1: list[int] = []
    tier2: list[int] = []
    tier3: list[int] = []

    for cp in sorted(codepoints):
        if cp in vocab.codepoints_single:
            tier0.append(cp)
        elif cp in vocab.codepoints_any:
            tier1.append(cp)
        elif vocab.has_byte_fallback:
            tier2.append(cp)
        else:
            tier3.append(cp)

    return TierResult(tier0=tier0, tier1=tier1, tier2=tier2, tier3=tier3)
