# Cog-Surp

Cog-Surp is a reproducible benchmarking workbench for testing whether
language-model prediction measures explain human N400 responses across
controlled and naturalistic language paradigms.

It separately evaluates:

1. the causal effect of a controlled linguistic manipulation on human N400;
2. the effect of that text manipulation on model-derived measures; and
3. held-out predictive alignment between those measures and EEG.

The third result is not evidence that a language model and the human brain use
the same mechanism.

## Development status

Cog-Surp has complete real-data vertical slices: ERP CORE supplies the
controlled cohort N400 effect and DERCo supplies authoritative naturalistic
word-to-EEG alignment for H3. Synthetic data are restricted to tests, demos,
recovery, and power analysis and are always labelled.

## Quick start

Install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync --locked --extra all
uv run cog-surp doctor
uv run pytest
```

The real DERCo article-0 pipeline is driven by immutable CLI inputs:

```bash
uv run cog-surp datasets fetch derco --subject LRK01 --run article_0
uv run cog-surp stimuli validate --config configs/datasets/derco.yaml
uv run cog-surp eeg preprocess --config configs/eeg/derco.yaml \
  --subject LRK01 --article article_0
uv run cog-surp lm score --config configs/models/model_matrix.yaml \
  --stimuli-path artifacts/runs/<stimulus-run>/stimuli.parquet
uv run cog-surp features build \
  --stimuli-path artifacts/runs/<stimulus-run>/stimuli.parquet \
  --surprisal-path artifacts/runs/<lm-run>/surprisal.parquet
uv run cog-surp analyze predictive \
  --features-path artifacts/runs/<feature-run>/features.parquet
uv run cog-surp analyze fit --config configs/analyses/derco_primary.yaml \
  --features-path artifacts/runs/<feature-run>/features.parquet
uv run cog-surp app run
```

The controlled ERP CORE cohort is driven separately so missing trial-word
metadata is never fabricated:

```bash
uv run cog-surp eeg preprocess --config configs/eeg/erp_core_primary.yaml \
  --subject 001 --dataset-manifest artifacts/runs/<dataset-run>/dataset-manifest.json
uv run cog-surp eeg summarize-cohort \
  --config configs/eeg/erp_core_primary.yaml \
  --dataset-manifest artifacts/runs/<dataset-run>/dataset-manifest.json
uv run cog-surp analyze model-effect \
  --surprisal-path artifacts/runs/<erp-smollm-run>/surprisal.parquet \
  --surprisal-path artifacts/runs/<erp-gpt2-run>/surprisal.parquet
uv run cog-surp analyze causal-condition \
  --h1-trials-path artifacts/runs/<cohort-run>/eeg/cohort/accepted-single-trial-n400.parquet \
  --h2-surprisal-path artifacts/runs/<erp-smollm-run>/surprisal.parquet \
  --h2-surprisal-path artifacts/runs/<erp-gpt2-run>/surprisal.parquet
uv run cog-surp eeg cluster-exploratory \
  --config configs/eeg/erp_core_cluster.yaml \
  --cohort-run artifacts/runs/<cohort-run>
uv run cog-surp stimuli generate-controlled \
  --config configs/stimuli/controlled.yaml
```

Data and model weights are not committed. See
[`docs/scientific_scope.md`](docs/scientific_scope.md) for the claim boundary
[`docs/data_dictionary.md`](docs/data_dictionary.md) for variable roles,
[`docs/research_landscape.md`](docs/research_landscape.md) for adjacent tools,
and [`docs/adr/`](docs/adr/) for decisions.

## Data license

Project source code is Apache-2.0. ERP CORE resources are conservatively
treated as CC BY-SA 4.0. DERCo's OSF dataset has no declared dataset license
and is marked `NOASSERTION`; analyze it locally and do not redistribute it.
Downloaded data always retain their original terms.
