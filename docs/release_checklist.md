# Cog-Surp v0.1.0 release checklist

Verified on 2026-07-28.

- [x] Repository clean after the final release commit (generated scientific
  runs remain ignored by design).
- [x] Metadata and real author/contact/ORCID information complete.
- [x] No release placeholders remain.
- [x] Full unmodified Apache License 2.0 terms and project `NOTICE` present.
- [x] `CITATION.cff` parses and cites Cog-Surp as software.
- [x] Wheel and source distribution build.
- [x] Clean wheel installation, import, and base doctor smoke pass.
- [x] Clean source-distribution installation and import pass.
- [x] Full test suite passes: 83 passed; 0 skipped, xfailed, or failed.
- [x] Strict typing passes over 50 source files.
- [x] Ruff lint and format checks pass.
- [x] Streamlit AppTests pass for all six sections.
- [x] Missing, corrupt, unsafe, and lineage-mismatched manifests are rejected.
- [x] Non-root CPU Docker image builds; doctor, help, manifest tests, demo
  validation, and live dashboard health check pass.
- [x] Synthetic demo builds, validates, and renders without model/data download.
- [x] Real/synthetic/mixed status contracts reviewed and regression-tested.
- [x] README commands and one-minute project narrative reviewed.
- [x] Local Markdown links checked.
- [x] Secret, private-path, oversized-file, and redistribution scan completed.
- [x] Scientific claims and separation of estimands reviewed.
- [x] Known limitations and dataset/model licensing boundaries reviewed.
- [x] Local empirical release bundle generated and validated without committing
  restricted artifacts.

The optional CUDA/GPU image and registry publication remain out of scope.
