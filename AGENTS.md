# Cog-Surp repository instructions

- Use `uv sync --locked --extra all`; run commands with `uv run`.
- Before handoff, run `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy`, and the relevant `uv run pytest` scope.
- Keep scientific computation in `src/cog_surp`, not Streamlit pages or
  notebooks. Dashboard code reads completed artifacts only.
- Never commit raw/restricted EEG, model weights, secrets, or large generated
  artifacts. Every scientific artifact needs a checksum and parent lineage.
- Preserve immutable run configuration. Reuse a valid cached run; never
  silently overwrite a run ID with changed inputs or configuration.
- Real EEG is the empirical foundation. Synthetic outputs are non-evidential
  and must always display `Synthetic data`.
- Keep condition effects on human EEG, condition effects on model measures,
  and predictive EEG/model alignment separate. Never add a default causal edge
  from model surprisal to human N400 or claim model-brain homology.
- Scientific surprisal uses teacher-forced causal logits with correct shifting,
  observed-token `log_softmax`, nats as canonical units, and an explicit
  token-to-region strategy. Never use generation for probability scoring.
- More-negative voltage means a larger N400. Report coefficient coding and
  sign explicitly.
- Do not use destructive Git commands or discard unrelated working-tree state.
- Completion requires current artifacts and tests to demonstrate the full
  acceptance criteria; plans or partial interfaces are not evidence.
