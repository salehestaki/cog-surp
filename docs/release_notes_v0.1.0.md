# Cog-Surp v0.1.0 release notes

Release date: 2026-07-28

Cog-Surp v0.1.0 is a research-grade public release for reproducible comparison
of language-model prediction measures and human N400 responses. It preserves
the verified empirical results while making their artifact selection,
provenance, and scientific status explicit and testable.

## Highlights

- One validated release manifest is now the sole dashboard/report source of
  truth. It rejects missing, corrupt, unsafe, unknown, or lineage-incompatible
  artifacts.
- Real, synthetic, and mixed evidence status is metadata-derived and enforced
  globally and at panel level.
- The public CPU-only demo includes 18 deterministic artifacts and all six
  dashboard sections. It is labeled synthetic throughout and makes no human
  evidence claim.
- Streamlit AppTests and manifest regressions are committed and run in CI.
- Package metadata, author/contact information, ORCID, `CITATION.cff`, full
  Apache-2.0 terms, and project URLs are internally consistent.
- Release, dashboard, and demo commands moved into a cohesive CLI module while
  preserving the documented command tree.

## Scientific results preserved

No empirical result was recalculated for presentation. README values retain
their stage-artifact lineage:

- ERP CORE H1: -3.669 µV unrelated-minus-related, 95% interval
  [-4.319, -3.019], primary n=34.
- Matched-stimulus H2: +3.057 nats (SmolLM2), +1.628 (GPT-2), and +3.517
  (Qwen2.5-0.5B).
- DERCo H3: -0.215 µV/SD, 95% HDI [-0.369, -0.055], interpreted as a
  conditional association with weak practical held-out prediction.

The local empirical release configuration validates the current H1, H2,
H3/H4/H5, causal, cluster, and integrated-report artifact family. Generated
real bundles are not committed because DERCo redistribution terms are
`NOASSERTION`.

## Migration

Dashboard launches now require one manifest:

```bash
uv run cog-surp app run \
  --manifest demo/bundle/release-manifest.json
```

Build a local empirical bundle with:

```bash
uv run cog-surp report manifest \
  --config configs/reports/release.yaml
```

The former dashboard behavior that independently selected newest artifacts is
removed with no fallback.

## Verification

- `83 passed`; no skipped, xfailed, or failed tests.
- Ruff check and format check passed.
- Strict mypy passed over 50 source files.
- Wheel and sdist built and passed isolated-install imports; the base wheel
  also passed `cog-surp doctor --json`.
- Synthetic and real manifest dashboard runs rendered all six tabs with zero
  exceptions.
- Dockerfile static review passed, but local Windows Docker daemon access was
  denied. The release does not claim a successful image build.

## Known limitations

- DERCo has no declared dataset license; its data and derivatives are not
  redistributed.
- Public EEG and model weights require separate downloads and retain their
  upstream terms.
- Held-out predictive gains are weak; association does not imply mechanistic
  or neurobiological homology.
- Sensor-time results are exploratory, not confirmatory timing or source
  localization.
- Docker build/run verification remains external.
