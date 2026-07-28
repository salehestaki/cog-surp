# Changelog

All notable changes to Cog-Surp are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-28

### Added

- Strict Pydantic schema for self-contained release manifests, including
  artifact checksums, run lineage, datasets, pinned model/tokenizer revisions,
  configuration hashes, provenance, citations, limitations, and per-artifact
  evidence status.
- Immutable `cog-surp report manifest` builder and
  `report validate-manifest` integrity command.
- Deterministic 18-artifact synthetic demo and `cog-surp demo build`.
- Committed Streamlit AppTests for all six sections, real/synthetic/mixed
  labels, provenance, missing files, corruption, unsafe paths, and incompatible
  lineage.
- Release metadata, software citation tests, full Apache License 2.0 terms,
  `NOTICE`, release checklist, and release notes.
- CLI `--version` and release/dashboard command module.

### Changed

- Dashboard loading now accepts exactly one validated manifest and never
  discovers artifacts by modification time.
- Every scientific page receives persistent manifest-derived real, synthetic,
  or mixed status wording.
- Dashboard provenance now exposes release ID, source revision, project
  version, datasets, models, and limitations.
- CPU Dockerfile now includes the synthetic demo/configuration/test resources,
  uses the locked environment, excludes private data through `.dockerignore`,
  and runs as an unprivileged user.
- Project and citation metadata now identify Saleh Estaki Organi and the
  Cog-Surp software itself.

### Scientific integrity

- Preserved all verified H1–H5 estimates and their original stage artifacts.
- Kept condition effects, model behavior, and predictive alignment as separate
  estimands.
- Kept the no-model-to-EEG causal-edge boundary and weak held-out prediction
  interpretation.
- No synthetic value is presented as human evidence.

### Validation

- 83 automated tests passed with no skips, xfails, or failures.
- Ruff check/format and strict mypy passed.
- Wheel and source distribution built and installed successfully in isolated
  Python 3.12 environments.
- Synthetic and local empirical release manifests validated and rendered with
  zero Streamlit AppTest exceptions.
- Docker daemon access was unavailable locally; exact external smoke commands
  are documented and Docker validation is not claimed.

[0.1.0]: https://github.com/salehestaki/cog-surp/releases/tag/v0.1.0
