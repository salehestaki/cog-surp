"""Validated public and controlled stimulus handling."""

from cog_surp.stimuli.controlled import (
    ControlledGeneratorConfig,
    LLMAssistedCandidate,
    generate_controlled_stimuli,
    validate_llm_candidates,
)
from cog_surp.stimuli.derco import load_derco_stimuli
from cog_surp.stimuli.erp_core import load_erp_core_stimuli

__all__ = [
    "ControlledGeneratorConfig",
    "LLMAssistedCandidate",
    "generate_controlled_stimuli",
    "load_derco_stimuli",
    "load_erp_core_stimuli",
    "validate_llm_candidates",
]
