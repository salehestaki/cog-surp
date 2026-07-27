"""Real-data condition-effect identification audit for ERP CORE H1 and H2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cog_surp.causal.dowhy_analysis import estimate_condition_effect
from cog_surp.causal.graph import build_default_graph

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class CausalAuditArtifacts:
    """Materialized real-data causal audit artifacts."""

    graph: Path
    graph_figure: Path
    report: Path


def _encoded_condition(frame: pd.DataFrame) -> pd.Series:
    values = set(frame["condition"].dropna().astype(str))
    if values != {"related", "unrelated"}:
        raise ValueError(
            f"expected related/unrelated conditions, found {sorted(values)}"
        )
    return frame["condition"].map({"related": 0, "unrelated": 1}).astype(int)


def analyze_real_condition_effects(
    *,
    h1_trials_path: Path,
    h2_surprisal_paths: list[Path],
    output_dir: Path,
    run_id: str,
    random_seed: int = 20260727,
) -> CausalAuditArtifacts:
    """Identify A->Y and A->S under the declared randomized-condition graph."""
    import matplotlib.pyplot as plt
    import networkx as nx
    import pandas as pd

    if len(h2_surprisal_paths) < 2:
        raise ValueError("real causal audit requires at least two H2 model artifacts")
    spec = build_default_graph(randomized_condition=True)
    h1 = pd.read_parquet(h1_trials_path)
    required_h1 = {
        "condition",
        "n400_mean_voltage_uv",
        "participant_included",
        "rejection_status",
    }
    missing_h1 = sorted(required_h1 - set(h1.columns))
    if missing_h1:
        raise ValueError(f"H1 trials lack required columns: {missing_h1}")
    h1 = h1.loc[
        h1["participant_included"].astype(bool) & (h1["rejection_status"] == "accepted")
    ].copy()
    h1["experimental_condition"] = _encoded_condition(h1)
    h1["human_n400"] = h1["n400_mean_voltage_uv"].astype(float)
    h1_result = estimate_condition_effect(
        data=h1[["experimental_condition", "human_n400"]],
        spec=spec,
        outcome="human_n400",
        run_refuters=True,
        random_seed=random_seed,
    )

    h2_results: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    for path in h2_surprisal_paths:
        scores = pd.read_parquet(path)
        required_h2 = {"condition", "target_surprisal_nats", "model_id"}
        missing_h2 = sorted(required_h2 - set(scores.columns))
        if missing_h2:
            raise ValueError(f"H2 scores lack required columns: {missing_h2}")
        models = scores["model_id"].drop_duplicates()
        if len(models) != 1:
            raise ValueError("each H2 score artifact must contain one model")
        model_id = str(models.iloc[0])
        if model_id in seen_models:
            raise ValueError(f"duplicate H2 model in causal audit: {model_id}")
        seen_models.add(model_id)
        scores = scores.copy()
        scores["experimental_condition"] = _encoded_condition(scores)
        scores["model_prediction_measure"] = scores["target_surprisal_nats"].astype(
            float
        )
        result = estimate_condition_effect(
            data=scores[["experimental_condition", "model_prediction_measure"]],
            spec=spec,
            outcome="model_prediction_measure",
            run_refuters=True,
            random_seed=random_seed,
        )
        h2_results.append({"model_id": model_id, **asdict(result)})

    output_dir.mkdir(parents=True, exist_ok=True)
    graph = output_dir / "causal-graph.gml"
    graph.write_text(spec.to_gml_text() + "\n", encoding="utf-8")
    graph_figure = output_dir / "causal-graph.svg"
    positions = nx.spring_layout(spec.graph, seed=random_seed)
    figure, axis = plt.subplots(figsize=(10, 6))
    nx.draw_networkx(
        spec.graph,
        pos=positions,
        ax=axis,
        node_size=1800,
        node_color="#DDEBF7",
        edge_color="#555555",
        font_size=7,
        arrowsize=16,
    )
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(graph_figure)
    plt.close(figure)
    report = output_dir / "causal-condition-effects.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "data_status": "real",
                "design": "ERP CORE randomized/counterbalanced condition",
                "h1_condition_to_human_n400": asdict(h1_result),
                "h2_condition_to_model_measure": h2_results,
                "graph_falsification": {
                    "executed": False,
                    "reason": (
                        "The conceptual DAG intentionally includes latent and "
                        "design nodes not jointly observed in the released "
                        "trial table, so conditional-independence graph "
                        "falsification is not identified from these artifacts."
                    ),
                },
                "primary_estimate_note": (
                    "The equally participant-weighted paired H1 and matched-target "
                    "paired H2 estimates remain primary. These unadjusted DoWhy "
                    "estimates document identification and perturbation sensitivity."
                ),
                "claim_boundary": (
                    "A->Y and A->S are separate condition effects. No S->Y edge "
                    "or model-brain mechanistic claim is identified."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return CausalAuditArtifacts(
        graph=graph,
        graph_figure=graph_figure,
        report=report,
    )
