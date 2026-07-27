"""Language-model probability engines and region strategies."""

from cog_surp.lm.backends import MockBackend, SurprisalBackend, TransformersBackend
from cog_surp.lm.comparison import compare_probability_strategies
from cog_surp.lm.domain import ScoringRequest, ScoringResult, TokenObservation
from cog_surp.lm.regions import BoundaryAwareStrategy, SubtokenSumStrategy

__all__ = [
    "BoundaryAwareStrategy",
    "MockBackend",
    "ScoringRequest",
    "ScoringResult",
    "SubtokenSumStrategy",
    "SurprisalBackend",
    "TokenObservation",
    "TransformersBackend",
    "compare_probability_strategies",
]
