"""Artifact-only Streamlit dashboard for completed Cog-Surp runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


def _latest(pattern: str) -> Path | None:
    candidates = list(Path("artifacts/runs").glob(pattern))
    return (
        max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
    )


def _path_input(label: str, default: Path | None) -> Path:
    value = st.sidebar.text_input(label, str(default or ""))
    path = Path(value)
    if not value or not path.exists():
        st.error(f"Artifact not found: {value}")
        st.stop()
    return path


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} is not a JSON object")
    return loaded


st.set_page_config(page_title="Cog-Surp", layout="wide")
st.title("Cog-Surp")
st.caption(
    "Reproducible benchmarking of model prediction measures and human N400 responses"
)
st.success("REAL EEG · ERP CORE controlled H1 and DERCo article-0 alignment")
st.warning(
    "Predictive/explanatory alignment only—no mechanistic or neurobiological "
    "homology claim."
)

features_path = _path_input(
    "Features",
    _latest("features-*/features.parquet"),
)
predictive_path = _path_input(
    "Held-out summary",
    _latest("predictive-*/held-out-summary.parquet"),
)
posterior_path = _path_input(
    "Posterior summary",
    _latest("analysis-*/posterior-summary.parquet"),
)
diagnostics_path = _path_input(
    "Diagnostics",
    _latest("analysis-*/diagnostics.json"),
)
robustness_path = _path_input(
    "Cross-model robustness",
    _latest("robustness-*/model-comparison.parquet"),
)
h1_path = _path_input(
    "ERP CORE H1 estimate",
    _latest("eeg-cohort-*/eeg/cohort/h1-condition-effect.json"),
)
h2_path = _path_input(
    "ERP CORE H2 estimate",
    _latest("model-effect-*/h2-model-effect.json"),
)
causal_path = _path_input(
    "Real-data causal audit",
    _latest("causal-*/causal-condition-effects.json"),
)
cluster_path = _path_input(
    "Exploratory sensor-time metadata",
    _latest("cluster-*/cluster-metadata.json"),
)

features = pd.read_parquet(features_path)
predictive = pd.read_parquet(predictive_path)
posterior = pd.read_parquet(posterior_path)
diagnostics = _load_json(diagnostics_path)
robustness = pd.read_parquet(robustness_path)
h1 = _load_json(h1_path)
h2 = _load_json(h2_path)
causal_audit = _load_json(causal_path)
cluster_metadata = _load_json(cluster_path)
cluster_summary = pd.read_parquet(cluster_path.parent / "cluster-summary.parquet")
erp_root = h1_path.parent
participant_qc = pd.read_parquet(erp_root / "participant-qc.parquet")

overview, stimuli, eeg, alignment, causal, provenance = st.tabs(
    [
        "Overview",
        "Stimuli & model",
        "Human EEG",
        "Alignment",
        "Causal assumptions",
        "Provenance",
    ]
)

with overview:
    columns = st.columns(5)
    columns[0].metric("Participants", features["participant"].nunique())
    columns[1].metric("Items", features["item"].nunique())
    columns[2].metric("Observations", f"{len(features):,}")
    columns[3].metric("Max R-hat", f"{diagnostics['max_rhat']:.4f}")
    columns[4].metric(
        "ERP CORE H1",
        f"{h1['primary_rule_based_cohort']['estimate_uv']:.2f} µV",
        help="Unrelated minus related; negative means a larger N400.",
    )
    st.markdown(
        """
        The dashboard reads completed Parquet, JSON, and NetCDF-derived summaries.
        It never launches EEG preprocessing or language-model inference during a
        page rerun. More-negative voltage means a larger N400.
        """
    )

with stimuli:
    st.subheader("ERP CORE matched-target model condition effects")
    h2_columns = st.columns(len(h2["models"]))
    for column, model in zip(h2_columns, h2["models"], strict=True):
        column.metric(
            str(model["model_id"]),
            f"{model['estimate_nats']:.2f} nats",
            (f"95% CI {model['ci95_low_nats']:.2f} to {model['ci95_high_nats']:.2f}"),
            help="Unrelated minus related target surprisal.",
        )
    st.caption(
        "H2 is model behavior on matched publisher stimuli. It is not evidence "
        "that model surprisal causes the human N400."
    )
    st.dataframe(
        pd.DataFrame.from_records(h2["models"]),
        width="stretch",
        hide_index=True,
    )

    st.subheader("DERCo naturalistic word measures")
    item_frame = features.drop_duplicates("item")
    st.plotly_chart(
        px.scatter(
            item_frame,
            x="human_cloze_surprisal_nats",
            y="target_surprisal_nats",
            hover_data=["item", "target_word", "target_token_count"],
            labels={
                "human_cloze_surprisal_nats": "Human cloze surprisal (nats)",
                "target_surprisal_nats": "LM surprisal (nats)",
            },
        ),
        width="stretch",
    )
    st.dataframe(
        item_frame[
            [
                "item",
                "target_word",
                "human_cloze_probability",
                "human_response_entropy_nats",
                "target_surprisal_nats",
                "target_token_count",
                "model_id",
                "probability_strategy",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

with eeg:
    st.subheader("ERP CORE controlled condition effect")
    effect = h1["primary_rule_based_cohort"]
    counts = h1["participant_counts"]
    columns = st.columns(4)
    columns[0].metric("Public participants", counts["public_available"])
    columns[1].metric("Primary included", counts["primary_included"])
    columns[2].metric("Primary excluded", counts["primary_excluded"])
    columns[3].metric(
        "Unrelated - related",
        f"{effect['estimate_uv']:.2f} µV",
        f"95% CI {effect['ci95_low_uv']:.2f} to {effect['ci95_high_uv']:.2f}",
    )
    st.caption(effect["sign_convention"])
    figure_columns = st.columns(2)
    figure_columns[0].image(
        str(erp_root / "grand-average-condition-erp.svg"),
        caption="CPz condition ERPs; negative voltage plotted upward",
        width="stretch",
    )
    figure_columns[1].image(
        str(erp_root / "grand-average-difference-wave.svg"),
        caption="Unrelated-minus-related difference wave",
        width="stretch",
    )
    _, topomap_column, _ = st.columns([1, 2, 1])
    topomap_column.image(
        str(erp_root / "n400-difference-topomap.svg"),
        caption="Canonical MNE scalp map, 300-500 ms",
        width="stretch",
    )
    st.dataframe(participant_qc, width="stretch", hide_index=True)
    st.info(
        "The all-public-participant sensitivity estimate is "
        f"{h1['all_public_participants_sensitivity']['estimate_uv']:.2f} µV. "
        "The primary estimate applies the versioned automated exclusion rules."
    )
    st.subheader("Exploratory sensor-time analysis")
    st.image(
        str(cluster_path.parent / "sensor-time-t-statistic.svg"),
        caption="Sensor-time t statistics; cluster inference is exploratory",
    )
    st.caption(cluster_metadata["interpretation_boundary"])
    st.dataframe(cluster_summary, width="stretch", hide_index=True)

    st.subheader("DERCo naturalistic N400-window outcomes")
    display = features.copy()
    display["cloze_band"] = pd.qcut(
        display["human_cloze_probability"],
        q=3,
        labels=["low", "medium", "high"],
        duplicates="drop",
    )
    st.plotly_chart(
        px.box(
            display,
            x="cloze_band",
            y="n400_mean_voltage_uv",
            points=False,
            labels={
                "cloze_band": "Human predictability tertile",
                "n400_mean_voltage_uv": "N400-window mean voltage (µV)",
            },
        ),
        width="stretch",
    )
    st.caption(
        "These are prespecified ROI/window outcomes, not full ERP waveforms or "
        "source-localized activity."
    )

with alignment:
    st.subheader("Held-out performance")
    st.dataframe(predictive, width="stretch", hide_index=True)
    st.subheader("Cross-model robustness")
    st.dataframe(robustness, width="stretch", hide_index=True)
    fixed_names = {
        "target_surprisal_nats_z",
        "human_cloze_surprisal_nats_z",
        "human_response_entropy_nats_z",
        "word_frequency_z",
        "number_of_letters_z",
        "word_position_z",
    }
    fixed = posterior.loc[posterior["parameter"].isin(fixed_names)].copy()
    lower = next(
        column
        for column in fixed
        if column.startswith("hdi") and column.endswith("_lb")
    )
    upper = next(
        column
        for column in fixed
        if column.startswith("hdi") and column.endswith("_ub")
    )
    fixed["error_minus"] = fixed["mean"] - fixed[lower]
    fixed["error_plus"] = fixed[upper] - fixed["mean"]
    figure = px.scatter(fixed, x="mean", y="parameter")
    figure.update_traces(
        error_x={
            "type": "data",
            "array": fixed["error_plus"],
            "arrayminus": fixed["error_minus"],
        }
    )
    figure.add_vline(x=0, line_dash="dash")
    st.plotly_chart(figure, width="stretch")

with causal:
    st.image(
        str(causal_path.parent / "causal-graph.svg"),
        caption="Versioned conceptual DAG; no model-surprisal-to-EEG edge",
    )
    st.subheader("Identified condition effects and refuters")
    st.json(causal_audit)
    st.markdown(
        """
        - DERCo predictability is observational at the word level.
        - Item identity and lexical/contextual variables affect both model scores
          and EEG outcomes.
        - Participant factors and preprocessing choices affect EEG outcomes.
        - Model identity and tokenization strategy affect model scores.
        - There is deliberately no default causal arrow from LM surprisal to EEG.

        The posterior surprisal coefficient is a conditional association. The
        held-out analysis tests predictive generalization; neither identifies a
        physical causal effect of an LM on a participant's brain.
        """
    )

with provenance:
    st.json(
        {
            "features": str(features_path.resolve()),
            "predictive_summary": str(predictive_path.resolve()),
            "posterior_summary": str(posterior_path.resolve()),
            "diagnostics": str(diagnostics_path.resolve()),
            "robustness": str(robustness_path.resolve()),
            "erp_core_h1": str(h1_path.resolve()),
            "erp_core_h2": str(h2_path.resolve()),
            "causal_audit": str(causal_path.resolve()),
            "exploratory_cluster": str(cluster_path.resolve()),
            "model_id": str(features["model_id"].iloc[0]),
            "model_revision": str(features["model_revision"].iloc[0]),
            "probability_strategy": str(features["probability_strategy"].iloc[0]),
            "data_status": "real",
        }
    )
