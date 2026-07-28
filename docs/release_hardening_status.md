# Release hardening status

Audit date: 2026-07-28
Branch: `main`
Initial revision: `4ffcc0db80718c3a1b8acdeaef5368ea2f3b062f`
Initial worktree: clean

## Preserved and already validated

- Real ERP CORE H1, matched-stimulus three-model H2, and DERCo H3/H4/H5
  artifacts and reported estimates.
- Exact teacher-forced LM scoring, real/synthetic scientific boundaries,
  checksummed stage manifests, causal refuters, exploratory cluster analysis,
  and artifact-only scientific computation.
- Locked Python 3.12 environment, 57 passing tests, strict mypy, Ruff,
  successful wheel/sdist builds, public GitHub repository, and green CI.
- Existing real-result screenshots, because they were captured from valid
  local artifacts and retain the dashboard's real-data label.

## Release blockers discovered

- The dashboard independently selects the newest file for each artifact type;
  it can therefore combine incompatible runs.
- There is no unified validated release/report manifest or self-contained
  public synthetic demo bundle.
- Streamlit AppTests are not committed.
- `CITATION.cff` incorrectly uses ERP CORE as Cog-Surp's preferred citation;
  author and project URL metadata are incomplete.
- `LICENSE` is an abbreviated Apache notice rather than the complete Apache
  License 2.0 text.
- The Docker image omits configurations, demo files, tests, and a non-root
  runtime user; current Docker execution status requires fresh validation.
- The 2,161-line CLI module needs compatibility-preserving decomposition.
- Changelog, release notes, and an evidence-based release checklist are
  missing.

## Completed in this hardening task

- Added strict unified manifest construction/loading, content-derived release
  IDs, safe paths, checksums, dataset/model/config provenance, and lineage
  rejection tests.
- Removed independent dashboard artifact discovery and made all six sections
  consume one manifest with global and panel-level evidence labels.
- Added committed Streamlit AppTests plus corruption, missing-file, unsafe-path,
  incompatible-run, and scientific-labeling regressions.
- Corrected package/author/project metadata and `CITATION.cff`; added full
  Apache-2.0 terms and `NOTICE`.
- Added a deterministic 18-artifact synthetic demo, sample report, release
  configuration for the preserved empirical artifacts, non-root Docker
  packaging, CLI release module, changelog, release notes, and checklist.
- Built and validated local synthetic and empirical bundles. The empirical
  bundle remains ignored because DERCo redistribution status is
  `NOASSERTION`.

## Dependency-ordered implementation

1. Define, build, and validate one immutable release manifest.
2. Make the dashboard consume only that manifest and enforce data-status
   labels.
3. Add manifest corruption/lineage and Streamlit AppTests.
4. Correct author, package, citation, license, and version metadata.
5. Generate the deterministic synthetic demo/report and improve Docker.
6. Split CLI registration and reusable command support without changing the
   command tree.
7. Run full build, clean-install, demo, documentation, security, Docker, and
   CI gates; then finalize release documentation.

## Current classification

- Unified manifest and manifest-only dashboard: **implemented and validated**
- Committed dashboard tests: **validated**
- Metadata/citation/license: **complete and validated**
- Docker: **built and validated as a non-root CPU/demo image**
- Synthetic public demo: **implemented and validated**
- CLI behavior: **compatible; release/dashboard commands decomposed**
- Full Python gates: **83 tests, Ruff, mypy, wheel/sdist and clean installs pass**
- Scientific results: **preserved; no numerical result changed**
