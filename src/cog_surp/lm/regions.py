"""Explicit token-to-linguistic-region probability strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from cog_surp.lm.domain import TokenObservation


@dataclass(frozen=True, slots=True)
class RegionProbability:
    """Aggregate probability for one observed character region."""

    strategy: str
    token_positions: tuple[int, ...]
    log_probability: float

    @property
    def surprisal_nats(self) -> float:
        return -self.log_probability

    @property
    def surprisal_bits(self) -> float:
        return -self.log_probability / math.log(2)


class RegionProbabilityStrategy(Protocol):
    """Port for auditable token-to-region probability allocation."""

    name: str

    def aggregate(
        self,
        *,
        text: str,
        region_start: int,
        region_end: int,
        tokens: tuple[TokenObservation, ...],
    ) -> RegionProbability:
        """Aggregate observed token probabilities for a character region."""


class SubtokenSumStrategy:
    """Conventional baseline: sum every token overlapping the region."""

    name = "subtoken-sum"

    def aggregate(
        self,
        *,
        text: str,
        region_start: int,
        region_end: int,
        tokens: tuple[TokenObservation, ...],
    ) -> RegionProbability:
        del text
        selected = tuple(
            token
            for token in tokens
            if token.end > region_start and token.start < region_end
        )
        return _aggregate_selected(self.name, selected)


class BoundaryAwareStrategy:
    """Include boundary-space tokens but reject non-whitespace leakage."""

    name = "boundary-aware"

    def aggregate(
        self,
        *,
        text: str,
        region_start: int,
        region_end: int,
        tokens: tuple[TokenObservation, ...],
    ) -> RegionProbability:
        selected = tuple(
            token
            for token in tokens
            if token.end > region_start and token.start < region_end
        )
        if not selected:
            raise ValueError("no tokenizer tokens overlap the target region")
        for token in selected:
            left_leak = text[token.start : min(token.end, region_start)]
            right_leak = text[max(token.start, region_end) : token.end]
            if left_leak and not left_leak.isspace():
                raise ValueError(
                    f"token {token.position} crosses a non-whitespace left boundary"
                )
            if right_leak and not right_leak.isspace():
                raise ValueError(
                    f"token {token.position} crosses a non-whitespace right boundary"
                )
        return _aggregate_selected(self.name, selected)


def _aggregate_selected(
    strategy: str, selected: tuple[TokenObservation, ...]
) -> RegionProbability:
    if not selected:
        raise ValueError("no tokenizer tokens overlap the target region")
    missing = [token.position for token in selected if token.log_probability is None]
    if missing:
        raise ValueError(f"target tokens lack preceding context at positions {missing}")
    return RegionProbability(
        strategy=strategy,
        token_positions=tuple(token.position for token in selected),
        log_probability=sum(
            token.log_probability
            for token in selected
            if token.log_probability is not None
        ),
    )
