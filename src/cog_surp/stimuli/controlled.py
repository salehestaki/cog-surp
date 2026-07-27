"""Deterministic controlled contrasts and LLM-candidate audit schemas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    import pandas as pd

AnomalyFamily = Literal[
    "selectional-restriction",
    "unrelated-prime-target",
    "world-knowledge",
    "thematic-role-reversal",
    "discourse-inconsistency",
    "negation-sensitive",
    "syntactic-control",
    "plausible-unexpected",
    "implausible-unexpected",
]


class ControlledItem(BaseModel):
    """One hand-authored matched contrast template."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    item_id: str = Field(min_length=1)
    anomaly_family: AnomalyFamily
    context_text: str = Field(min_length=1)
    control_target: str = Field(min_length=1)
    manipulated_target: str = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_targets(self) -> ControlledItem:
        if self.control_target.casefold() == self.manipulated_target.casefold():
            raise ValueError("control and manipulated targets must differ")
        return self


class ControlledGeneratorConfig(BaseModel):
    """Versioned deterministic controlled-stimulus configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: int
    seed: int
    items: tuple[ControlledItem, ...]

    @model_validator(mode="after")
    def unique_items(self) -> ControlledGeneratorConfig:
        identifiers = [item.item_id for item in self.items]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("controlled item IDs must be unique")
        if not self.items:
            raise ValueError("at least one controlled item is required")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> ControlledGeneratorConfig:
        """Load a fully specified generator configuration."""
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def generate_controlled_stimuli(config: ControlledGeneratorConfig) -> pd.DataFrame:
    """Generate paired contrasts deterministically without model sampling."""
    import pandas as pd

    records: list[dict[str, Any]] = []
    for item in config.items:
        for condition, target in (
            ("control", item.control_target),
            ("manipulated", item.manipulated_target),
        ):
            normalized_context = item.context_text.rstrip() + " "
            records.append(
                {
                    "dataset_id": "controlled-generator",
                    "item": item.item_id,
                    "condition": condition,
                    "anomaly_family": item.anomaly_family,
                    "context_text": normalized_context,
                    "target_text": target,
                    "target_word": target,
                    "target_start": len(normalized_context),
                    "target_end": len(normalized_context) + len(target),
                    "target_length_characters": len(target),
                    "generator_seed": config.seed,
                    "generation_method": "deterministic-template",
                    "validation_status": "engineered-model-side-stress-test",
                    "scientific_use": "model-side-stress-test-only",
                    "data_status": "synthetic-stimulus",
                }
            )
    frame = pd.DataFrame.from_records(records)
    diagnostics = frame.pivot(
        index="item",
        columns="condition",
        values="target_length_characters",
    ).assign(
        target_length_difference=lambda value: (
            value["manipulated"] - value["control"]
        ).abs()
    )["target_length_difference"]
    frame = frame.merge(diagnostics, on="item", validate="many_to_one")
    return frame


class LLMAssistedCandidate(BaseModel):
    """Fully auditable LLM-proposed stimulus candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    candidate_id: str = Field(min_length=1)
    generation_model_id: str = Field(min_length=1)
    generation_model_revision: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    decoding_parameters: dict[str, float | int | str | bool]
    context_text: str = Field(min_length=1)
    target_text: str = Field(min_length=1)
    target_start: int = Field(ge=0)
    target_end: int = Field(gt=0)
    anomaly_family: AnomalyFamily
    grammar_pass: bool
    duplicate_pass: bool
    lexical_matching_diagnostics: dict[str, float | int | str | bool]
    semantic_similarity: float = Field(ge=-1, le=1)
    manual_review_status: Literal["pending", "rejected", "approved"]

    @model_validator(mode="after")
    def target_span_matches(self) -> LLMAssistedCandidate:
        full_text = self.context_text + self.target_text
        if self.target_end > len(full_text):
            raise ValueError("candidate target span exceeds full text")
        if full_text[self.target_start : self.target_end] != self.target_text:
            raise ValueError("candidate target span does not match target text")
        return self


def validate_llm_candidates(path: Path) -> pd.DataFrame:
    """Validate JSONL candidates and label them as model-side stress tests."""
    import pandas as pd

    candidates = [
        LLMAssistedCandidate.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not candidates:
        raise ValueError("candidate file is empty")
    identifiers = [candidate.candidate_id for candidate in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate IDs must be unique")
    texts = [
        (candidate.context_text + candidate.target_text).casefold()
        for candidate in candidates
    ]
    if len(set(texts)) != len(texts):
        raise ValueError("candidate file contains duplicate full texts")
    records = []
    for candidate in candidates:
        record = candidate.model_dump(mode="json")
        record.update(
            {
                "input_sha256": hashlib.sha256(
                    (candidate.context_text + candidate.target_text).encode()
                ).hexdigest(),
                "validation_status": "llm-assisted-stress-test",
                "scientific_use": "model-side-stress-test-only",
                "data_status": "synthetic-stimulus",
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)
