from __future__ import annotations

import networkx as nx
import pytest

from cog_surp.causal.graph import (
    COVARIATES,
    HUMAN_N400,
    MODEL_MEASURE,
    UNSUPPORTED_EDGE,
    build_default_graph,
    minimal_backdoor_adjustment_set,
)


def test_default_graph_is_acyclic_and_has_no_surprisal_to_eeg_edge() -> None:
    spec = build_default_graph(randomized_condition=True)

    assert nx.is_directed_acyclic_graph(spec.graph)
    assert not spec.graph.has_edge(MODEL_MEASURE, HUMAN_N400)
    assert minimal_backdoor_adjustment_set(spec, outcome=HUMAN_N400) == frozenset()


def test_nonrandom_design_requires_observed_covariate_adjustment() -> None:
    spec = build_default_graph(randomized_condition=False)

    assert minimal_backdoor_adjustment_set(spec, outcome=HUMAN_N400) == frozenset(
        {COVARIATES}
    )


def test_unsupported_surprisal_to_eeg_claim_is_rejected() -> None:
    spec = build_default_graph()
    spec.graph.add_edge(*UNSUPPORTED_EDGE)

    with pytest.raises(ValueError, match="unsupported causal edge"):
        spec.validate()


def test_gml_is_dowhy_compatible_directed_graph() -> None:
    gml = build_default_graph().to_gml_text()

    assert "directed 1" in gml
    assert 'label "experimental_condition"' in gml
