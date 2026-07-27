# Current state

Audit date: 2026-07-27  
Repository: `D:\Saleh`, branch `main`

## Completed scientific slices

- **H1, controlled human EEG:** ERP CORE run
  `eeg-cohort-f1a3d5d14adf` aggregates all 39 public participants with
  versioned, automated preprocessing and retains 34 under the primary QC
  rules. The equal-participant unrelated-minus-related CPz, 300-500 ms
  contrast is -3.669 microvolts, 95% interval [-4.319, -3.019]. The all-39
  sensitivity estimate is -5.323 microvolts, [-8.556, -2.090]. Negative
  values mean a larger N400 for unrelated targets.
- **H2, matched model response:** `model-effect-553f5f5b92e8` evaluates the
  same 100 ERP CORE targets with exact teacher-forced probabilities. The
  unrelated-minus-related effect is 3.057 nats [2.355, 3.759] for pinned
  SmolLM2-135M and 1.628 [1.127, 2.129] for pinned GPT-2; item effects
  correlate at 0.606. These are model-behavior results, not brain effects.
- **H3, naturalistic alignment:** DERCo article 0 supplies authoritative
  publisher `WordID` linkage for 9,029 observations, 20 participants, and 559
  scored items. The converged crossed Bayesian SmolLM2 run
  `analysis-9a5dad267c7d` estimates a conditional coefficient of -0.215
  microvolts/SD, 95% HDI [-0.369, -0.055], with zero divergences and maximum
  R-hat 1.0056. Held-out improvement is only about 0.015 microvolts RMSE and
  mean R2 remains near zero, so practical predictive gain is weak.
- **H4, alternatives:** `predictive-de8598f91332` separately evaluates human
  cloze probability and human response entropy in leakage-free item and
  participant folds, alongside lexical, LM, and combined specifications.
  Item-held-out RMSE gains over lexical controls are approximately 0.013
  microvolts for cloze and 0.002 for entropy.
- **H5, robustness:** pinned SmolLM2 and GPT-2 article-0 surprisals correlate
  at 0.932 and their conditional coefficients are -0.215 and -0.217
  microvolts/SD with overlapping intervals. Probability-strategy run
  `strategy-bcec8b5dc577` finds exact equality between boundary-aware and
  subtoken-sum scoring on this fixture only. A pinned Qwen2.5-0.5B CPU run is
  being executed as the additional family/scale check.

## Completed engineering and audit layers

- Python 3.12 package, immutable typed configurations, `uv.lock`, CLI,
  pre-commit, GitHub Actions CI, CPU Dockerfile, and six-tab artifact-only
  Streamlit dashboard.
- Checksummed dataset, EEG, stimulus, LM, feature, predictive, Bayesian,
  causal, robustness, cluster, report, and provenance manifests with parent
  lineage. Cache reuse validates configuration identity and output checksums.
- Exact causal-LM scoring covers the causal shift, padding, first-token
  exclusion, offsets, region aggregation, whitespace, punctuation, Unicode,
  contraction, hyphenation, batching, and float32/float64 sensitivity.
- Real-data DoWhy audit `causal-2df1788ad3dd` separately estimates condition
  to human N400 and condition to model surprisal. Bootstrap, subset, placebo,
  random-common-cause, and simulated-unobserved-confounder refuters execute.
  It intentionally contains no model-surprisal-to-human-EEG causal edge.
- Exploratory real-data sensor-time permutation run
  `cluster-f12050fd8fb7` uses 34 participants and 512 permutations, returns 13
  clusters (two at uncorrected alpha 0.05), and is explicitly barred from
  exact onset, peak, anatomical-source, or confirmatory interpretation.
- Deterministic controlled generator `controlled-877dcfe7a41a` emits nine
  separate anomaly families and 18 model-side stress-test rows. The
  LLM-assisted candidate schema records generation provenance and diagnostics
  and never upgrades unreviewed material to validated human stimuli.
- A source distribution and wheel build successfully. The wheel was installed
  into an isolated Python 3.12 environment with only base dependencies, where
  `cog-surp doctor --json` returned `ok: true`.
- Delta research, scientific scope, causal assumptions, preprocessing,
  reproducibility, variable roles, architecture decisions, limitations, and
  data/model licensing are documented.

## External constraints and deliberate boundaries

- Docker CLI access to the local engine named pipe is denied, so the CPU image
  cannot be built in this session; the wheel-based clean-install gate passes.
- No NVIDIA GPU is available. Larger checkpoints are therefore bounded CPU
  robustness jobs rather than a broad scale sweep.
- DERCo has no declared dataset license and is recorded as `NOASSERTION`; its
  local data and all model weights are excluded from Git.
- ERP CORE does not expose an authoritative randomized trial-to-word key.
  Cog-Surp therefore uses it for H1 and a separate matched-stimulus H2, never
  fabricates an item-linked H3, and uses DERCo for word-aligned H3.
- The sensor-time analysis is exploratory, and DoWhy refuters probe specified
  perturbations rather than proving the causal graph true.

## Final release gates

The remaining gates are to finish or explicitly bound the active Qwen CPU
robustness run, regenerate the integrated report and provenance snapshot from
the latest artifacts, run the complete test/dashboard suite, and create the
first local source commit. Remote CI and image-registry publication require an
external remote and are outside this workspace.
