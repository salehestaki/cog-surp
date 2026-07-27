"""Exact and deterministic surprisal backend implementations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol, cast

from cog_surp.lm.domain import ScoringRequest, ScoringResult, TokenObservation
from cog_surp.lm.regions import RegionProbabilityStrategy


class SurprisalBackend(Protocol):
    """Backend port shared by exact, high-throughput, and mock scorers."""

    def score(self, batch: Sequence[ScoringRequest]) -> Sequence[ScoringResult]:
        """Score observed targets without sampling or generation."""


def observed_token_log_probabilities(
    logits: Any,
    input_ids: Any,
    attention_mask: Any,
) -> Any:
    """Gather teacher-forced observed-token log probabilities with causal shift."""
    import torch

    if logits.ndim != 3 or input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("expected logits [B,L,V] and ids/mask [B,L]")
    if logits.shape[:2] != input_ids.shape or input_ids.shape != attention_mask.shape:
        raise ValueError(
            "logits, input IDs, and attention mask shapes are inconsistent"
        )
    shifted = torch.log_softmax(logits[:, :-1, :], dim=-1)
    labels = input_ids[:, 1:]
    gathered = shifted.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    valid = attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
    output = torch.full(
        input_ids.shape,
        torch.nan,
        dtype=logits.dtype,
        device=logits.device,
    )
    output[:, 1:] = torch.where(valid, gathered, torch.nan)
    return output


class TransformersBackend:
    """Reference backend using an exact teacher-forced Transformers forward pass."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        strategy: RegionProbabilityStrategy,
        dtype: str = "float32",
        device: str = "cpu",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"unsupported dtype: {dtype}")
        torch_dtype = getattr(torch, dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            use_fast=True,
        )
        if not self._tokenizer.is_fast:
            raise ValueError("reference scoring requires a fast tokenizer")
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "right"
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch_dtype,
        ).to(device)
        self._model.eval()
        self._model_id = model_id
        self._revision = revision
        self._strategy = strategy
        self._dtype = dtype
        self._device = device

    def score(self, batch: Sequence[ScoringRequest]) -> Sequence[ScoringResult]:
        """Score a batch via one forward pass; never call ``generate``."""
        import torch

        if not batch:
            return []
        encoded = self._tokenizer(
            [request.text for request in batch],
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
            padding=True,
        )
        offsets = encoded.pop("offset_mapping")
        model_inputs = {
            key: value.to(self._device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask"}
        }
        with torch.inference_mode():
            output = self._model(**model_inputs)
            observed = observed_token_log_probabilities(
                output.logits,
                model_inputs["input_ids"],
                model_inputs["attention_mask"],
            )
        input_ids = model_inputs["input_ids"].detach().cpu()
        attention = model_inputs["attention_mask"].detach().cpu()
        observed = observed.detach().cpu()
        results: list[ScoringResult] = []
        for row_index, request in enumerate(batch):
            length = int(attention[row_index].sum().item())
            row_ids = input_ids[row_index, :length].tolist()
            row_offsets = offsets[row_index, :length].tolist()
            token_strings = self._tokenizer.convert_ids_to_tokens(row_ids)
            tokens = tuple(
                TokenObservation(
                    position=position,
                    token_id=int(token_id),
                    token=str(token_string),
                    start=int(offset[0]),
                    end=int(offset[1]),
                    log_probability=(
                        None
                        if math.isnan(float(observed[row_index, position]))
                        else float(observed[row_index, position])
                    ),
                )
                for position, (token_id, token_string, offset) in enumerate(
                    zip(row_ids, token_strings, row_offsets, strict=True)
                )
            )
            target_start, target_end = request.target_span
            region = self._strategy.aggregate(
                text=request.text,
                region_start=target_start,
                region_end=target_end,
                tokens=tokens,
            )
            results.append(
                ScoringResult(
                    request_id=request.request_id,
                    text=request.text,
                    target_start=target_start,
                    target_end=target_end,
                    tokens=tokens,
                    probability_strategy=region.strategy,
                    target_token_positions=region.token_positions,
                    target_log_probability=region.log_probability,
                    target_surprisal_nats=region.surprisal_nats,
                    target_surprisal_bits=region.surprisal_bits,
                    model_id=self._model_id,
                    model_revision=self._revision,
                    tokenizer_revision=self._revision,
                    dtype=self._dtype,
                    quantization="none",
                )
            )
        return results


class MockBackend:
    """Deterministic, explicitly non-scientific backend for tests and CI."""

    def __init__(self, strategy: RegionProbabilityStrategy) -> None:
        self._strategy = strategy

    def score(self, batch: Sequence[ScoringRequest]) -> Sequence[ScoringResult]:
        results: list[ScoringResult] = []
        for request in batch:
            target_start, target_end = request.target_span
            token = TokenObservation(
                position=1,
                token_id=1,
                token=request.target,
                start=target_start,
                end=target_end,
                log_probability=-float(len(request.target)),
            )
            region = self._strategy.aggregate(
                text=request.text,
                region_start=target_start,
                region_end=target_end,
                tokens=(token,),
            )
            results.append(
                ScoringResult(
                    request_id=request.request_id,
                    text=request.text,
                    target_start=target_start,
                    target_end=target_end,
                    tokens=(token,),
                    probability_strategy=region.strategy,
                    target_token_positions=region.token_positions,
                    target_log_probability=region.log_probability,
                    target_surprisal_nats=region.surprisal_nats,
                    target_surprisal_bits=region.surprisal_bits,
                    model_id="deterministic-mock",
                    model_revision="fixture-v1",
                    tokenizer_revision="fixture-v1",
                    dtype="float64",
                    quantization="none",
                )
            )
        return cast(Sequence[ScoringResult], results)
