# Reproducibility

## Environment

Use Python 3.12 and the committed lock:

```bash
uv sync --locked --extra all
uv run cog-surp doctor
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

## Run identity

CLI run IDs are derived from canonical JSON containing resolved configuration,
input hashes, and hashes of the stage's relevant implementation modules.
Reusing an explicit run ID with different inputs fails.
Scientific outputs are Parquet, JSON, SVG, Markdown, or NetCDF and carry parent
and output SHA-256 values.

`cog-surp provenance snapshot` records:

- the hash of every non-ignored code/config/documentation file;
- Git revision and dirty-state hash (or a null revision before the first
  commit);
- `uv.lock` SHA-256;
- Python, operating system, CPU, and memory;
- the complete installed package inventory; and
- explicitly selected parent artifacts.

Raw EEG, model weights, posterior NetCDF files, and generated artifacts are
ignored by Git. DERCo data must not be redistributed because its dataset
license is undeclared.

## Expensive work and cache behavior

Language-model inference, EEG extraction, and Bayesian sampling occur only
through the CLI, never in Streamlit reruns. Dataset, stimulus, EEG, LM,
feature, predictive, Bayesian, causal, robustness, cluster, and reporting
stages reuse an existing run only when resolved inputs and implementation
hashes match and every manifested output passes SHA-256 verification.
Otherwise the stage is recomputed with an explicit event or fails on an
immutable run-ID conflict.

## Interpretation

Every output records real or synthetic status. Synthetic outputs are test-only.
Condition effects, model behavior, and held-out EEG/model alignment are
separate estimands. A model-surprisal coefficient is not a physical causal
effect on EEG.

## CPU container

The root `Dockerfile` installs Python 3.12.13, uv 0.11.32, and the committed
lock, then runs as an unprivileged user. It includes only repository source,
configuration, tests, documentation metadata, and the synthetic demo—never
raw EEG, model caches, credentials, or local scientific artifacts.

Build and run the offline smoke sequence:

```bash
docker build -t cog-surp:0.1.0 .
docker run --rm cog-surp:0.1.0
docker run --rm cog-surp:0.1.0 --help
docker run --rm --entrypoint pytest cog-surp:0.1.0 \
  tests/unit/test_release_manifest.py -q
docker run --rm cog-surp:0.1.0 report validate-manifest \
  --manifest demo/bundle/release-manifest.json
docker run --rm -p 8501:8501 cog-surp:0.1.0 app run \
  --manifest demo/bundle/release-manifest.json
```

Open `http://localhost:8501` for the final command. Mount `data` and
`artifacts` explicitly for real workflows. DERCo remains non-redistributable.
A CUDA image is not claimed because this host has no GPU and the CUDA
inference path has not been validated.

This sequence was executed successfully on 2026-07-28 with Docker Desktop
4.80.0 and Linux Engine 29.6.1. The resulting `cog-surp:0.1.0` image:

- has image ID
  `sha256:958aea9a7d230e1e88e9daaa0574d6ccb2b5350a909bc6d50564024e77b6873a`;
- is 559,227,888 bytes;
- runs as UID/GID 10001 (`cogsurp`);
- passed the doctor and root-help commands;
- passed all eight release-manifest unit tests inside the container;
- validated the committed 18-artifact synthetic manifest; and
- served the dashboard successfully at `/_stcore/health`.

The default image is CPU/demo-focused: only the locked `data` and `dashboard`
extras are included. Full EEG preprocessing and language-model scoring remain
host workflows installed with `uv sync --locked --extra all`.
