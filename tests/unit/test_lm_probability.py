from __future__ import annotations

import math

import pytest

from cog_surp.lm.backends import MockBackend, observed_token_log_probabilities
from cog_surp.lm.domain import ScoringRequest, TokenObservation
from cog_surp.lm.regions import BoundaryAwareStrategy, SubtokenSumStrategy

torch = pytest.importorskip("torch")


def test_hand_calculated_causal_shift_and_first_token_exclusion() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, math.log(3.0)],
                [math.log(4.0), 0.0],
                [0.0, 0.0],
            ]
        ]
    )
    input_ids = torch.tensor([[0, 1, 0]])
    mask = torch.tensor([[1, 1, 1]])

    observed = observed_token_log_probabilities(logits, input_ids, mask)

    assert torch.isnan(observed[0, 0])
    assert observed[0, 1].item() == pytest.approx(math.log(3 / 4))
    assert observed[0, 2].item() == pytest.approx(math.log(4 / 5))


def test_padding_mask_excludes_padded_observations() -> None:
    logits = torch.zeros((1, 3, 2))
    ids = torch.tensor([[0, 1, 0]])
    mask = torch.tensor([[1, 1, 0]])

    observed = observed_token_log_probabilities(logits, ids, mask)

    assert observed[0, 1].item() == pytest.approx(-math.log(2))
    assert torch.isnan(observed[0, 2])


def test_boundary_aware_accepts_leading_space_token() -> None:
    token = TokenObservation(1, 5, "Ġcat", 3, 7, -2.0)

    region = BoundaryAwareStrategy().aggregate(
        text="The cat",
        region_start=4,
        region_end=7,
        tokens=(token,),
    )

    assert region.token_positions == (1,)
    assert region.surprisal_nats == 2.0


def test_boundary_aware_rejects_non_whitespace_leakage() -> None:
    token = TokenObservation(1, 5, "ecat", 2, 6, -2.0)

    with pytest.raises(ValueError, match="non-whitespace left boundary"):
        BoundaryAwareStrategy().aggregate(
            text="thecat",
            region_start=3,
            region_end=6,
            tokens=(token,),
        )


def test_mock_backend_is_deterministic_and_reports_strategy() -> None:
    request = ScoringRequest("item-1", "prime ", "target")
    backend = MockBackend(SubtokenSumStrategy())

    first = backend.score([request])
    second = backend.score([request])

    assert first == second
    assert first[0].probability_strategy == "subtoken-sum"
    assert first[0].target_surprisal_nats == len("target")


@pytest.mark.parametrize(
    ("text", "start", "end", "tokens"),
    [
        (
            "Say hello!",
            4,
            9,
            (TokenObservation(1, 1, "Ġhello", 3, 9, -2.0),),
        ),
        (
            "I saw café.",
            6,
            10,
            (TokenObservation(1, 1, "Ġcafé", 5, 10, -2.0),),
        ),
        (
            "It isn't.",
            3,
            8,
            (
                TokenObservation(1, 1, "Ġis", 2, 5, -1.0),
                TokenObservation(2, 2, "n't", 5, 8, -1.5),
            ),
        ),
        (
            "A well-known fact",
            2,
            12,
            (
                TokenObservation(1, 1, "Ġwell", 1, 6, -1.0),
                TokenObservation(2, 2, "-", 6, 7, -0.5),
                TokenObservation(3, 3, "known", 7, 12, -1.5),
            ),
        ),
        (
            "Wait   target",
            7,
            13,
            (TokenObservation(1, 1, "ĠĠĠtarget", 4, 13, -3.0),),
        ),
        (
            "word, next",
            0,
            5,
            (
                TokenObservation(1, 1, "word", 0, 4, -1.0),
                TokenObservation(2, 2, ",", 4, 5, -0.2),
            ),
        ),
    ],
)
def test_boundary_alignment_fixtures(
    text: str,
    start: int,
    end: int,
    tokens: tuple[TokenObservation, ...],
) -> None:
    region = BoundaryAwareStrategy().aggregate(
        text=text,
        region_start=start,
        region_end=end,
        tokens=tokens,
    )

    assert region.token_positions == tuple(token.position for token in tokens)


def test_sentence_initial_target_without_preceding_context_is_rejected() -> None:
    token = TokenObservation(0, 1, "Hello", 0, 5, None)

    with pytest.raises(ValueError, match="lack preceding context"):
        BoundaryAwareStrategy().aggregate(
            text="Hello",
            region_start=0,
            region_end=5,
            tokens=(token,),
        )


def test_observed_log_probabilities_are_stable_across_float_precision() -> None:
    logits32 = torch.tensor(
        [[[0.2, -0.3], [0.7, -0.1], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    ids = torch.tensor([[0, 1, 0]])
    mask = torch.ones_like(ids)

    score32 = observed_token_log_probabilities(logits32, ids, mask)
    score64 = observed_token_log_probabilities(logits32.double(), ids, mask)

    assert torch.isnan(score32[0, 0])
    assert torch.isnan(score64[0, 0])
    torch.testing.assert_close(
        score32[:, 1:].double(),
        score64[:, 1:],
        atol=1e-7,
        rtol=1e-7,
    )
