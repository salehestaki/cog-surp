# Cog-Surp public demo

> **SYNTHETIC TEST/DEMO DATA — NOT HUMAN EVIDENCE**

This small, redistributable bundle exercises every dashboard section without
EEG downloads, model weights, a GPU, or network access. Its values are
deterministic simulations for software inspection only. They are not empirical
measurements and support no conclusion about people, cognition, or
neuroscience.

Build and validate the bundle:

```bash
uv run cog-surp demo build --output demo/bundle
uv run cog-surp report validate-manifest \
  --manifest demo/bundle/release-manifest.json
```

Launch the dashboard:

```bash
uv run cog-surp app run \
  --manifest demo/bundle/release-manifest.json
```

The committed bundle is self-contained. `release-manifest.json` records its
content-derived release ID, source revision, deterministic fixture identity,
run lineage, scientific status, and SHA-256 for every file. Re-running the
builder validates and reuses an identical bundle; it never silently overwrites
a partial or inconsistent directory.

Real EEG and model weights are deliberately excluded from this directory.
See the repository README for the full empirical reproduction path.
