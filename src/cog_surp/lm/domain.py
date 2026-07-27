"""Immutable language-model scoring records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScoringRequest:
    """One context and observed target region to score."""

    request_id: str
    context: str
    target: str

    @property
    def text(self) -> str:
        """The exact teacher-forced text passed to the tokenizer."""
        return self.context + self.target

    @property
    def target_span(self) -> tuple[int, int]:
        """Target character offsets in ``text``."""
        return len(self.context), len(self.text)


@dataclass(frozen=True, slots=True)
class TokenObservation:
    """Observed next-token probability and tokenizer alignment."""

    position: int
    token_id: int
    token: str
    start: int
    end: int
    log_probability: float | None


@dataclass(frozen=True, slots=True)
class ScoringResult:
    """Token-level observations plus an explicit region aggregate."""

    request_id: str
    text: str
    target_start: int
    target_end: int
    tokens: tuple[TokenObservation, ...]
    probability_strategy: str
    target_token_positions: tuple[int, ...]
    target_log_probability: float
    target_surprisal_nats: float
    target_surprisal_bits: float
    model_id: str
    model_revision: str
    tokenizer_revision: str
    dtype: str
    quantization: str
