# Implementation progress

## 2026-07-27

- Inspected the initial local workspace: it was not a Git repository and
  contained no project files or local `AGENTS.md`.
- Recorded Windows 10 (64-bit), 15.74 GB RAM (0.93 GB free at inspection),
  313.26 GB free on drive D, no detected NVIDIA tooling, no installed Python,
  and no `uv`.
- Verified ERP CORE N400 access and CC BY-SA 4.0 terms from author and paper
  sources. Direct anonymous checks succeeded for metadata, license, and one
  BIDS EEGLAB recording.
- Accepted ADRs for a modular monolith, real EEG as primary evidence with
  synthetic fixtures only, and exclusion of a default surprisal-to-EEG causal
  edge.
- Installed `uv` 0.11.32 without modifying the shell profile; `uv` provisioned
  CPython 3.12.13.
- Added package metadata, dependency extras, CI, pre-commit, initial
  configuration, and the typed `cog-surp doctor` command.
- The dependency resolver exposed DoWhy's SciPy upper bound; pinned the
  compatible `scipy>=1.15,<1.16` range and generated `uv.lock`.
- Initial unit tests pass. Static checks identified small CLI/package-marker
  issues, which were corrected before the next full verification.
- Added a typed EEG dataset port and ERP CORE adapter with OSF pagination,
  subject filtering, streaming atomic downloads, publisher checksum
  verification, metadata-only mode, and immutable run manifests.
- Live metadata run `dataset-5f0ce3910456` fetched 15 subject-001 metadata
  artifacts (24,617 bytes); all publisher SHA-256 checks passed.
- Found conflicting license declarations (BIDS metadata says CC0; bundled
  license and author site say CC BY-SA 4.0). ADR 0002 records both and applies
  the conservative CC BY-SA terms.
- Live signal run `dataset-4e25cb1c1ed6` fetched and verified 17 artifacts,
  including subject 001's `.set` and `.fdt` files.
- Added versioned preprocessing configuration and `cog-surp eeg preprocess`.
  Configuration validation prevents a no-artifact-correction smoke pipeline
  from being labelled primary.
- Real-data run `eeg-159eab43f235` read 33 channels × 585,728 samples, found
  30 trials for each of target codes 211/212/221/222, and wrote checksummed
  Parquet/JSON/SVG artifacts. Of 120 trials, 104 passed the prespecified 200 µV
  peak-to-peak threshold and 16 were explicitly marked rejected.
- The one-subject smoke summary is non-confirmatory: accepted means were
  1.5296 µV (related, n=53) and 0.0518 µV (unrelated, n=51). It is not an
  empirical project conclusion.
- Twelve tests, Ruff, strict mypy, and the machine-readable doctor check pass.
- Next: parse and validate publisher stimuli, implement exact teacher-forced
  scoring and region strategies, then establish item-level EEG/LM linkage.
- Selected DERCo for word-aligned H3 after validating publisher `WordID`
  metadata against human prediction records. The OSF dataset license is not
  declared, so manifests use `NOASSERTION` and prohibit redistribution.
- Identified the two publisher exclusions as QPF42 and USQ95 and fetched the
  bounded article-0 slice for the 20 remaining participants. All 21 source
  artifacts in run `dataset-bdb6db89668e` passed SHA-256 validation.
- A cohort extraction exposed empty `WordID` prefix epochs in some publisher
  FIF files. The extractor now excludes only rows explicitly outside the
  prediction task and refuses duplicate analytical participant/item keys.
- Produced 9,029 authoritative participant-word joins over 559 scored items in
  `features-53afaa6d8b49`.
- Exact CPU scoring of all 561 usable article words took about 12 minutes at
  batch size 8. Batch 16 exceeded available memory; no partial output was
  promoted. Completed run: `lm-cpu-smoke-1ec20045d244`.
- Grouped five-fold item and participant holdouts completed in
  `predictive-fdbaced97cfd`. The combined model improved RMSE over lexical
  controls by only about 0.015 microvolts and mean R2 remained near zero.
- The first 2-chain Bayesian fit had max R-hat 1.02 and was rejected. The
  strengthened 4-chain, 1,000-draw run `analysis-9a5dad267c7d` passed with no
  divergences, max R-hat 1.0056, and minimum bulk ESS 1,015.
- The standardized conditional SmolLM2 surprisal coefficient was -0.215
  microvolts per SD (95% HDI -0.369 to -0.055). It is reported as conditional
  association only and interpreted alongside the weak held-out gain.
- Added an artifact-only six-tab Streamlit dashboard and report run
  `report-6a36b4a1904f`. Streamlit's application test rendered with no
  exceptions.
- Verification after the milestone: Ruff passed, strict mypy passed, and all
  33 tests passed.
- Completed a full GPT-2 legacy-anchor article run
  `lm-legacy-anchor-57ac8db2a3b6`, its 9,029-row feature and held-out analyses,
  and converged Bayesian run `analysis-28134e9eeca3`.
- Materialized cross-family run `robustness-0d9d41368e91`. SmolLM2 and GPT-2
  surprisals correlate at Pearson 0.932; their standardized conditional
  coefficients are -0.215 and -0.217 microvolts/SD, while both retain near-zero
  held-out R2.
- Regenerated the research report as `report-0947d6d6b14a` with the
  cross-family results and revalidated all dashboard tabs.
- Final verification for this continuation milestone: Ruff formatting/checks,
  strict mypy, 34 tests, and the Streamlit application test all pass.
- Added full DoWhy refuter execution coverage. Placebo, random-common-cause,
  subset, and bootstrap perturbations all return estimates; the deterministic
  placebo result is near zero.
- Added a pinned Python/uv CPU Dockerfile. Local validation is blocked because
  the Docker CLI cannot access `npipe:////./pipe/docker_engine`.
- Scored all 561 eligible DERCo words with SmolLM2 using conventional
  subtoken-sum aggregation and compared them to the boundary-aware run.
  `strategy-d32ca4203d32` found a maximum absolute difference of 0.0 nats for
  this exact fixture; the report explicitly avoids generalizing equivalence.
- Fetched and checksum-verified the full public ERP CORE inventory in
  `dataset-c1e901f7fc6a`: 39 participant folders (001-040 excluding absent
  027) and 290 artifacts.
- Added the versioned primary ERP CORE pipeline: standard montage, robust bad-
  channel detection/interpolation, deterministic 20-component extended-
  infomax ICA, bipolar EOG checks, explicit trial/participant exclusions,
  accepted-condition FIF evokeds, and per-subject ERP figures.
- Completed preprocessing for all 39 public participants. Thirty-four passed
  the prespecified automated QC rules; participants 004, 029, 030, 039, and
  040 exceeded rejection/count rules and remain visible in participant QC.
- Added checksummed cohort run `eeg-cohort-b23c28fbb335`. The primary
  unrelated-minus-related CPz 300-500 ms contrast is -3.669 microvolts (95%
  interval -4.319 to -3.019; n=34). The all-public-participant sensitivity is
  -5.323 microvolts (-8.556 to -2.090; n=39).
- Added equal-participant cohort tests, exact subject-run/config/manifest
  discovery, subject artifact checksum verification, canonical MNE
  topography, condition/difference waves, dashboard ERP views, and integrated
  report `report-474eaff27f96`.
- Added implementation-source hashes to EEG run identity and checksum-gated
  reuse for subject and cohort runs. Repeating the cohort command returned
  `eeg_cohort_reused` without recomputation.
- Extended checksum-gated reuse and implementation hashing uniformly across
  dataset, stimulus, LM, feature, predictive, Bayesian, causal, robustness,
  cluster, and reporting commands, with cache-corruption regression tests.
- Added the controlled H2 estimator over 100 matched ERP CORE targets.
  Implementation-current run `model-effect-553f5f5b92e8` estimates
  unrelated-minus-related surprisal at 3.057 nats for SmolLM2-135M and 1.628
  nats for GPT-2; the item-effect correlation is 0.606.
- Added separate H4 held-out evaluations for exact human cloze probability and
  response entropy. They provide small item-held-out RMSE improvements over
  lexical controls of approximately 0.013 and 0.002 microvolts.
- Added a real-data DoWhy condition-effect audit with bootstrap, subset,
  placebo, random-common-cause, and simulated-unobserved-common-cause
  refuters. Refreshed run: `causal-2df1788ad3dd`.
- Added an explicitly exploratory MNE sensor-time cluster permutation command
  and dashboard/report views. Refreshed run `cluster-f12050fd8fb7` includes 34
  participants, 512 permutations, 13 clusters, and two clusters at alpha
  0.05, without confirmatory timing or source claims.
- Added deterministic controlled stress-test generation for nine distinct
  linguistic families plus a provenance-complete schema for validating
  LLM-assisted candidates. Refreshed run: `controlled-877dcfe7a41a`.
- Expanded probability regressions to whitespace, punctuation, Unicode,
  contractions, hyphenation, multiple spaces, first-token rejection, and
  floating-point sensitivity.
- Added a variable-role data dictionary and a dated delta-research landscape
  covering Brain-Score, minicons, MNE, DoWhy, ERP CORE, and DERCo.
- Rebuilt the package after making scientific extras lazy. A wheel installed
  into a minimal isolated Python 3.12 environment and base
  `cog-surp doctor --json` returned `ok: true`.
- Refreshed implementation-hash-dependent H1 as
  `eeg-cohort-f1a3d5d14adf`; its scientific estimates are identical to the
  prior run.
- Bounded the third-family scale check after a full-story Qwen CPU attempt
  proved impractical at batch size one and produced no promotable artifact.
  Pinned Qwen2.5-0.5B scored all 200 short ERP CORE condition sequences in 33
  seconds as `lm-cross-family-59b0da516c20`.
- Three-model H2 run `model-effect-67ae1006277c` estimates the Qwen
  unrelated-minus-related effect at 3.517 nats (95% interval 2.606 to 4.428).
  Pairwise item-effect correlations are 0.606 (SmolLM2/GPT-2), 0.697
  (SmolLM2/Qwen), and 0.536 (GPT-2/Qwen).
- Corrected the H2 summary schema to report every model pair instead of
  silently repeating the first pair's correlation when more than two models
  are supplied.
- Generated integrated real-data report `report-8562aac501a7` from the current
  H1, three-model H2, H3/H4, causal, cluster, and robustness artifacts.
- Final local release gates pass: Ruff format/check, strict mypy over 45 source
  files, 54 pytest tests, six Streamlit tabs with zero exceptions, source and
  wheel builds, and a base-only isolated wheel doctor check with `ok: true`.

## 2026-07-28

- Rebuilt the public README as an English research landing page with CI and
  scope badges, exact H1-H5 result summaries, a Mermaid architecture map,
  collapsible pipelines, and three real screenshots captured from completed
  dashboard artifacts.
- Improved the dashboard's EEG figure layout so condition and difference
  waveforms remain legible at repository-preview widths.
- Diagnosed the first remote CI failure: all checks passed until `doctor`
  treated the GitHub runner's 8.22 GiB free disk as a hard failure. Doctor now
  requires 2 GiB for installation/CPU fixtures, warns below the documented
  10 GiB real-data recommendation, and has parameterized regression coverage.
- Final local verification after the presentation/CI update: 57 tests, Ruff,
  strict mypy, six dashboard tabs, zero Streamlit exceptions, and all local
  README links pass.
- Added an authoritative release-manifest schema and immutable bundle builder;
  the dashboard now reads one checksummed artifact family and has no
  modification-time discovery fallback.
- Added a deterministic 18-artifact synthetic public demo, manifest-only
  six-section dashboard, and committed AppTests for evidence labels,
  provenance, corruption, missing files, unsafe paths, and incompatible run
  lineage.
- Built and validated a local empirical bundle from the preserved H1, H2,
  H3/H4/H5, causal, cluster, and integrated-report artifacts. It remains
  ignored because DERCo redistribution terms are `NOASSERTION`.
- Corrected author/project/CFF metadata, installed the full Apache-2.0 license
  plus `NOTICE`, improved locked non-root Docker packaging, and added release
  notes/checklist/changelog.
- v0.1.0 Python release gates pass: 83 tests with no skips or failures, Ruff,
  strict mypy over 50 source files, wheel/sdist builds, isolated installs, and
  base-wheel doctor.
- Re-tested Docker through an elevated Docker Desktop Linux Engine session.
  The first full-extra build exposed inappropriate CUDA downloads for a CPU
  image, so the default image was narrowed to locked data/dashboard extras.
  Image `cog-surp:0.1.0` then built successfully as non-root UID/GID 10001 and
  passed doctor, help, eight manifest tests, demo validation, and a live
  Streamlit health endpoint.
