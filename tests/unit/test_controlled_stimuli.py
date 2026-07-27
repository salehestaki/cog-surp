from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cog_surp.stimuli import (
    ControlledGeneratorConfig,
    generate_controlled_stimuli,
    validate_llm_candidates,
)


def test_controlled_generator_is_paired_and_deterministic() -> None:
    config = ControlledGeneratorConfig.model_validate(
        {
            "schema_version": 1,
            "seed": 7,
            "items": [
                {
                    "item_id": "i1",
                    "anomaly_family": "world-knowledge",
                    "context_text": "The sky is",
                    "control_target": "blue",
                    "manipulated_target": "square",
                }
            ],
        }
    )

    first = generate_controlled_stimuli(config)
    second = generate_controlled_stimuli(config)

    assert first.equals(second)
    assert set(first["condition"]) == {"control", "manipulated"}
    assert set(first["scientific_use"]) == {"model-side-stress-test-only"}


def test_llm_candidates_require_auditable_metadata(tmp_path: Path) -> None:
    context = "The sky is "
    target = "square"
    record = {
        "candidate_id": "candidate-1",
        "generation_model_id": "fixture/model",
        "generation_model_revision": "abc",
        "prompt_sha256": hashlib.sha256(b"prompt").hexdigest(),
        "seed": 1,
        "decoding_parameters": {"temperature": 0.7},
        "context_text": context,
        "target_text": target,
        "target_start": len(context),
        "target_end": len(context) + len(target),
        "anomaly_family": "world-knowledge",
        "grammar_pass": True,
        "duplicate_pass": True,
        "lexical_matching_diagnostics": {"length_difference": 1},
        "semantic_similarity": 0.1,
        "manual_review_status": "pending",
    }
    path = tmp_path / "candidates.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    frame = validate_llm_candidates(path)

    assert frame.loc[0, "validation_status"] == "llm-assisted-stress-test"
    assert frame.loc[0, "scientific_use"] == "model-side-stress-test-only"

    record.pop("prompt_sha256")
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_llm_candidates(path)
