"""Versioned causal graph for experimental effects and alignment boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

CONDITION = "experimental_condition"
COVARIATES = "observed_item_covariates"
PARTICIPANT = "participant_factors"
ITEM = "item_identity"
MODEL = "model_identity"
TOKENIZER = "tokenizer_probability_strategy"
MODEL_MEASURE = "model_prediction_measure"
HUMAN_N400 = "human_n400"
PREPROCESSING = "preprocessing_measurement"
UNSUPPORTED_EDGE = (MODEL_MEASURE, HUMAN_N400)


@dataclass(frozen=True, slots=True)
class CausalGraphSpec:
    """A DAG plus its declared treatment, outcomes, and design regime."""

    graph: nx.DiGraph[str]
    treatment: str
    human_outcome: str
    model_outcome: str
    randomized_condition: bool

    def validate(self) -> None:
        """Reject cycles and unsupported model-to-human causal claims."""
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("causal graph must be acyclic")
        if self.graph.has_edge(*UNSUPPORTED_EDGE):
            raise ValueError(
                "unsupported causal edge model_prediction_measure -> human_n400"
            )
        required = {
            self.treatment,
            self.human_outcome,
            self.model_outcome,
            COVARIATES,
            PARTICIPANT,
            ITEM,
            MODEL,
            TOKENIZER,
            PREPROCESSING,
        }
        missing = required - set(self.graph.nodes)
        if missing:
            raise ValueError(f"causal graph missing nodes: {sorted(missing)}")

    def to_gml_text(self) -> str:
        """Return a DoWhy-compatible directed graph string."""
        self.validate()
        return "\n".join(nx.generate_gml(self.graph))


def build_default_graph(*, randomized_condition: bool = True) -> CausalGraphSpec:
    """Build the prespecified graph without a surprisal-to-EEG edge."""
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_nodes_from(
        [
            CONDITION,
            COVARIATES,
            PARTICIPANT,
            ITEM,
            MODEL,
            TOKENIZER,
            MODEL_MEASURE,
            HUMAN_N400,
            PREPROCESSING,
        ]
    )
    graph.add_edges_from(
        [
            (CONDITION, HUMAN_N400),
            (CONDITION, MODEL_MEASURE),
            (COVARIATES, HUMAN_N400),
            (COVARIATES, MODEL_MEASURE),
            (PARTICIPANT, HUMAN_N400),
            (ITEM, HUMAN_N400),
            (ITEM, MODEL_MEASURE),
            (MODEL, MODEL_MEASURE),
            (TOKENIZER, MODEL_MEASURE),
            (PREPROCESSING, HUMAN_N400),
        ]
    )
    if not randomized_condition:
        graph.add_edge(COVARIATES, CONDITION)
    spec = CausalGraphSpec(
        graph=graph,
        treatment=CONDITION,
        human_outcome=HUMAN_N400,
        model_outcome=MODEL_MEASURE,
        randomized_condition=randomized_condition,
    )
    spec.validate()
    return spec


def minimal_backdoor_adjustment_set(
    spec: CausalGraphSpec, *, outcome: str
) -> frozenset[str]:
    """Return direct treatment causes that also causally precede the outcome."""
    spec.validate()
    if outcome not in {spec.human_outcome, spec.model_outcome}:
        raise ValueError(f"undeclared outcome: {outcome}")
    treatment_parents = set(spec.graph.predecessors(spec.treatment))
    outcome_ancestors = nx.ancestors(spec.graph, outcome)
    return frozenset(treatment_parents & outcome_ancestors)
