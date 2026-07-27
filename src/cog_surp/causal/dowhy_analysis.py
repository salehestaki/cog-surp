"""DoWhy identification, estimation, and bounded robustness refuters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cog_surp.causal.graph import CausalGraphSpec, minimal_backdoor_adjustment_set


@dataclass(frozen=True, slots=True)
class DoWhyAnalysisResult:
    """Machine-readable causal estimate with explicit interpretation limits."""

    treatment: str
    outcome: str
    identifier_method: str
    adjustment_set: tuple[str, ...]
    estimate: float
    refuters: dict[str, float | None]
    interpretation: str


def estimate_condition_effect(
    *,
    data: Any,
    spec: CausalGraphSpec,
    outcome: str,
    run_refuters: bool = False,
    random_seed: int = 20260727,
) -> DoWhyAnalysisResult:
    """Identify and estimate an experimental condition effect with DoWhy."""
    from dowhy import CausalModel

    spec.validate()
    required = {spec.treatment, outcome}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"causal data missing columns: {sorted(missing)}")
    model = CausalModel(
        data=data,
        treatment=spec.treatment,
        outcome=outcome,
        graph=spec.to_gml_text(),
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=False)
    estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression",
    )
    refuters: dict[str, float | None] = {}
    if run_refuters:
        configurations: dict[str, tuple[str, dict[str, Any]]] = {
            "placebo_treatment": (
                "placebo_treatment_refuter",
                {"placebo_type": "permute", "num_simulations": 20},
            ),
            "random_common_cause": ("random_common_cause", {}),
            "data_subset": (
                "data_subset_refuter",
                {"subset_fraction": 0.8, "num_simulations": 20},
            ),
            "bootstrap": (
                "bootstrap_refuter",
                {"num_simulations": 20},
            ),
            "simulated_unobserved_common_cause": (
                "add_unobserved_common_cause",
                {
                    "simulation_method": "direct-simulation",
                    "confounders_effect_on_treatment": "binary_flip",
                    "confounders_effect_on_outcome": "linear",
                    "effect_strength_on_treatment": 0.05,
                    "effect_strength_on_outcome": 1.0,
                },
            ),
        }
        for label, (method_name, kwargs) in configurations.items():
            refutation = model.refute_estimate(
                estimand,
                estimate,
                method_name=method_name,
                random_seed=random_seed,
                **kwargs,
            )
            refuters[label] = _refuter_value(refutation)
    return DoWhyAnalysisResult(
        treatment=spec.treatment,
        outcome=outcome,
        identifier_method=str(estimand.identifier_method),
        adjustment_set=tuple(
            sorted(minimal_backdoor_adjustment_set(spec, outcome=outcome))
        ),
        estimate=float(estimate.value),
        refuters=refuters,
        interpretation=(
            "Experimental condition effect under the declared randomized-design "
            "and measurement assumptions. Refuters probe specific perturbations "
            "and do not prove the graph true."
        ),
    )


def _refuter_value(refutation: Any) -> float | None:
    value = getattr(refutation, "new_effect", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
