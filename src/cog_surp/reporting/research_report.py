"""Generate a bounded Markdown report from completed analysis artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def build_research_report(
    *,
    features_path: Path,
    predictive_summary_path: Path,
    posterior_summary_path: Path,
    diagnostics_path: Path,
    output_path: Path,
    robustness_path: Path | None = None,
    h1_path: Path | None = None,
    h2_path: Path | None = None,
    causal_path: Path | None = None,
    cluster_path: Path | None = None,
) -> None:
    """Write a traceable report without rerunning any scientific computation."""
    import pandas as pd

    features = pd.read_parquet(features_path)
    predictive = pd.read_parquet(predictive_summary_path)
    posterior = pd.read_parquet(posterior_summary_path)
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    lm_row = posterior.loc[posterior["parameter"] == "target_surprisal_nats_z"].iloc[0]
    cloze_row = posterior.loc[
        posterior["parameter"] == "human_cloze_surprisal_nats_z"
    ].iloc[0]
    interval_columns = [
        column
        for column in posterior.columns
        if column.startswith("hdi") and column.endswith(("_lb", "_ub"))
    ]
    if len(interval_columns) != 2:
        raise ValueError("posterior summary must contain one HDI lower/upper pair")
    lower_column = next(column for column in interval_columns if column.endswith("_lb"))
    upper_column = next(column for column in interval_columns if column.endswith("_ub"))

    def held_out(split: str, model: str, metric: str) -> float:
        row = predictive.loc[
            (predictive["split"] == split) & (predictive["model"] == model)
        ]
        if len(row) != 1:
            raise ValueError(f"missing predictive summary row: {split}/{model}")
        return float(row.iloc[0][metric])

    item_gain = held_out(
        "leave-items-out", "lexical-controls", "mean_rmse_uv"
    ) - held_out("leave-items-out", "combined", "mean_rmse_uv")
    participant_gain = held_out(
        "leave-participants-out", "lexical-controls", "mean_rmse_uv"
    ) - held_out("leave-participants-out", "combined", "mean_rmse_uv")
    cloze_item_gain = held_out(
        "leave-items-out", "lexical-controls", "mean_rmse_uv"
    ) - held_out("leave-items-out", "human-cloze", "mean_rmse_uv")
    entropy_item_gain = held_out(
        "leave-items-out", "lexical-controls", "mean_rmse_uv"
    ) - held_out("leave-items-out", "response-entropy", "mean_rmse_uv")
    robustness_section = ""
    if robustness_path is not None:
        robustness = pd.read_parquet(robustness_path)
        if len(robustness) != 2:
            raise ValueError("robustness summary must contain exactly two models")
        rows = robustness.set_index("model_id")
        coefficients = "; ".join(
            f"{model}: {row['surprisal_coefficient_uv_per_sd']:.3f} "
            f"[{row['coefficient_hdi95_lb']:.3f}, "
            f"{row['coefficient_hdi95_ub']:.3f}]"
            for model, row in rows.iterrows()
        )
        robustness_section = f"""
## H5: cross-model robustness

The two model families shared {int(robustness["shared_item_count"].iloc[0])}
scored words. Their word surprisals correlated at Pearson
{robustness["cross_model_pearson"].iloc[0]:.3f} and Spearman
{robustness["cross_model_spearman"].iloc[0]:.3f}. Standardized conditional
coefficients (95% HDI) were: {coefficients}. Both models retained near-zero
held-out R2, so directional replication does not imply strong practical
prediction or shared mechanism.
"""
    h1_section = ""
    if h1_path is not None:
        h1 = json.loads(h1_path.read_text(encoding="utf-8"))
        primary = h1["primary_rule_based_cohort"]
        sensitivity = h1["all_public_participants_sensitivity"]
        counts = h1["participant_counts"]
        h1_section = f"""
## H1: controlled human anomaly effect

The ERP CORE controlled analysis used prespecified CPz mean voltage from
300-500 ms. Of {counts["public_available"]} publicly available participants,
{counts["primary_included"]} passed the versioned automated trial, behavior, and
participant QC rules. The equal-participant-weight unrelated-minus-related
contrast was {primary["estimate_uv"]:.3f} µV (95% t interval
{primary["ci95_low_uv"]:.3f} to {primary["ci95_high_uv"]:.3f}). Negative values
mean a larger N400 for unrelated targets.

The transparent all-{counts["public_available"]}-participant sensitivity
estimate was {sensitivity["estimate_uv"]:.3f} µV (95% interval
{sensitivity["ci95_low_uv"]:.3f} to {sensitivity["ci95_high_uv"]:.3f}). This is
an effect of randomized experimental condition on human voltage; it is not a
causal effect of model surprisal.
"""
    h2_section = ""
    if h2_path is not None:
        h2 = json.loads(h2_path.read_text(encoding="utf-8"))
        model_lines = "\n".join(
            f"- {model['model_id']}: {model['estimate_nats']:.3f} nats "
            f"(95% interval {model['ci95_low_nats']:.3f} to "
            f"{model['ci95_high_nats']:.3f}; "
            f"n={model['n_matched_targets']} matched targets)."
            for model in h2["models"]
        )
        h2_section = f"""
## H2: model response to the controlled manipulation

Exact boundary-aware teacher-forced scoring produced the following
unrelated-minus-related target-surprisal effects:

{model_lines}

Positive values mean unrelated primes increased model target surprisal. The
item-level effects correlated at Pearson
{h2["cross_model_effect_pearson"]:.3f} across these two model families. H2 is
model behavior on matched text; its similarity in direction to H1 does not
identify shared computation, mediation, or model-to-brain causation.
"""
    causal_section = ""
    if causal_path is not None:
        causal = json.loads(causal_path.read_text(encoding="utf-8"))
        h1_causal = causal["h1_condition_to_human_n400"]
        causal_section = f"""
## Causal identification audit

Under the declared randomized-condition DAG, DoWhy identifies an empty
backdoor adjustment set. Its trial-weighted A-to-Y estimate is
{h1_causal["estimate"]:.3f} µV. Placebo, random-common-cause, data-subset,
bootstrap, and simulated-unobserved-common-cause refuters were executed for H1
and both H2 model effects. These perturbations probe named vulnerabilities;
they neither prove the graph nor create an S-to-Y causal estimand.

Graph-implied conditional-independence falsification was not executed because
the released trial artifact does not jointly observe the conceptual DAG's
latent participant/item and measurement nodes. The paired, equal-participant
H1 estimate remains primary.
"""
    cluster_section = ""
    if cluster_path is not None:
        cluster = json.loads(cluster_path.read_text(encoding="utf-8"))
        cluster_section = f"""
## Exploratory sensor-time analysis

A two-sided within-participant spatiotemporal cluster permutation analysis used
{cluster["participants"]} participants and {cluster["n_permutations"]}
permutations over {cluster["time_window_s"][0]:.1f}-
{cluster["time_window_s"][1]:.1f} s. Of {cluster["clusters"]} candidate
clusters, {cluster["clusters_passing_alpha"]} passed the configured cluster
alpha. This analysis is exploratory: cluster significance does not license
claims about exact onset, peak latency, neural source, or anatomical location.
"""
    report = f"""# Cog-Surp real-data research report

Status: **Real EEG; ERP CORE controlled H1/H2 and DERCo article-0 alignment**

{h1_section}

{h2_section}

{causal_section}

{cluster_section}

## Scope

This analysis uses {features["participant"].nunique()} eligible DERCo participants,
{features["item"].nunique()} word items, and {len(features):,} retained
participant-word observations. QPF42 and USQ95 are excluded according to the
publisher-reported excessive-eye-movement criterion. EEG outcomes are mean
voltage from 300-500 ms over Cz, CP1, CP2, and Pz after the publisher's
preprocessing and a -200-0 ms baseline.

## H3: incremental EEG explanation

In the crossed participant/item model, a one-SD increase in SmolLM2-135M
word-region surprisal was associated with {lm_row["mean"]:.3f} µV change in
N400-window voltage (95% HDI {lm_row[lower_column]:.3f} to
{lm_row[upper_column]:.3f}) after human cloze surprisal, human response entropy,
word frequency, word length, and word position were included. More-negative
voltage means a larger N400.

Human cloze surprisal had a conditional association of
{cloze_row["mean"]:.3f} µV per SD (95% HDI
{cloze_row[lower_column]:.3f} to {cloze_row[upper_column]:.3f}).

The combined model improved mean RMSE over lexical controls by only
{item_gain:.3f} µV on held-out items and {participant_gain:.3f} µV on held-out
participants. Mean held-out R² values remained near zero. The coefficient and
the weak held-out improvement must therefore be read together: this slice
shows a modest conditional association but little practical out-of-sample
explanatory gain.

## H4: alternatives to raw LM surprisal

Human cloze surprisal and human response entropy were evaluated separately,
in addition to their combined human-predictability model. On held-out items,
their RMSE improvements over lexical controls were {cloze_item_gain:.3f} µV
and {entropy_item_gain:.3f} µV, respectively. These alternatives are compared
on the same leakage-resistant folds as raw LM surprisal and the combined model.

{robustness_section}

## Diagnostics

- Four NUTS chains; 1,000 tuning and 1,000 retained draws per chain.
- Divergences: {diagnostics["divergences"]}.
- Maximum R-hat: {diagnostics["max_rhat"]:.4f}.
- Minimum bulk ESS: {diagnostics["min_bulk_ess"]:.0f}.
- Posterior-predictive RMSE: {diagnostics["posterior_predictive_rmse_uv"]:.3f} µV.
- Convergence gate passed: {diagnostics["convergence_pass"]}.

## Claim boundary

This result is predictive/explanatory alignment. It does not show that model
surprisal physically causes the human N400, that the model and brain implement
the same computation, or that either system is mechanistically homologous to
the other. The analysis covers one DERCo article and two small model families;
broader model-scale and cross-article robustness remain necessary.

## Artifact lineage

- Features: `{features_path.resolve()}`
- Held-out summary: `{predictive_summary_path.resolve()}`
- Posterior summary: `{posterior_summary_path.resolve()}`
- Diagnostics: `{diagnostics_path.resolve()}`
{f"- Robustness: `{robustness_path.resolve()}`" if robustness_path else ""}
{f"- ERP CORE H1: `{h1_path.resolve()}`" if h1_path else ""}
{f"- ERP CORE H2: `{h2_path.resolve()}`" if h2_path else ""}
{f"- Causal audit: `{causal_path.resolve()}`" if causal_path else ""}
{f"- Exploratory cluster metadata: `{cluster_path.resolve()}`" if cluster_path else ""}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
