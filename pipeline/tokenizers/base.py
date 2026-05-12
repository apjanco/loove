"""
Base types for tokenizer vocabulary extraction.

We store only derived data (code-point sets), never raw vocabulary files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ModelVocabData:
    """
    Derived vocabulary data for a single model.

    Raw token strings are intentionally NOT stored — only the Unicode code points
    that each token contributes to, separated into two sets that drive tier scoring:

      codepoints_single  — code points where at least one token decodes to exactly
                           that single character (Tier 0 candidates).
      codepoints_any     — all code points appearing anywhere in any decoded token,
                           including inside longer subword tokens (Tier 0 + 1 candidates).

    A code point present in codepoints_any but NOT in codepoints_single is Tier 1.
    A code point absent from codepoints_any but encodable via byte fallback is Tier 2.
    A code point unreachable even by byte fallback is Tier 3.
    """

    model_id: str
    source: str           # "huggingface" | "tiktoken"
    vocab_size: int
    has_byte_fallback: bool
    codepoints_single: set[int] = field(default_factory=set)
    codepoints_any: set[int] = field(default_factory=set)
    computed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "source": self.source,
            "vocab_size": self.vocab_size,
            "has_byte_fallback": self.has_byte_fallback,
            "codepoints_single": sorted(self.codepoints_single),
            "codepoints_any": sorted(self.codepoints_any),
            "computed_at": self.computed_at,
        }

    @classmethod
    def from_json_dict(cls, d: dict) -> "ModelVocabData":
        return cls(
            model_id=d["model_id"],
            source=d["source"],
            vocab_size=d["vocab_size"],
            has_byte_fallback=d["has_byte_fallback"],
            codepoints_single=set(d["codepoints_single"]),
            codepoints_any=set(d["codepoints_any"]),
            computed_at=d.get("computed_at", ""),
        )
