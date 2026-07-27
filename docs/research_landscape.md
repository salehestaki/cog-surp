# Research and technical landscape

Last delta review: 2026-07-27

## Adjacent platforms and libraries

### Brain-Score Language

[Brain-Score](https://www.brain-score.org/) is an active, open benchmarking
platform spanning neural and behavioral measurements, and its
[language package](https://brain-score-language.readthedocs.io/en/latest/)
provides a unified model/benchmark scoring interface. It is the closest broad
model-to-brain benchmarking platform. Cog-Surp does not duplicate its
leaderboard mission: the differentiator here is a narrow, auditable
surprisal–N400 workflow combining exact word-region probabilities, EEG
preprocessing, crossed statistical models, causal-assumption auditing, and
local artifact lineage.

Decision: remain interoperable in concept, but do not adopt Brain-Score as the
local pipeline runtime.

### minicons

[minicons](https://github.com/kanishkamisra/minicons) is an actively maintained
Apache-licensed utility for behavioral and representational LM analysis,
including incremental and conditional sequence scoring. It is a strong
general-purpose alternative for model-side experiments.

Decision: retain Cog-Surp's small reference scorer because its required
artifact schema exposes every observed token, offset, region allocation,
immutable model revision, and EEG join. Use minicons as an external
cross-check candidate rather than hiding the scientific reference path behind
a broader API.

### MNE-Python and MNE-BIDS

[MNE-Python](https://mne.tools/stable/) is the maintained core EEG/MEG
dependency. Its documented spatiotemporal cluster API expects observations ×
time × space and supports adjacency-aware permutation inference. MNE-BIDS is
retained for standards-compatible expansion, while ERP CORE's released
EEGLAB/BIDS-compatible files are currently loaded directly to preserve the
bounded adapter.

Decision: adopt MNE for signal processing and canonical sensor plots; label
cluster inference exploratory and prohibit exact-onset/source claims.

### DoWhy

[DoWhy](https://www.pywhy.org/dowhy/) implements explicit modeling,
identification, estimation, and refutation. Its maintained refuter suite
includes simulated unobserved common causes.

Decision: use DoWhy for the defensible A→Y and A→S condition effects and
assumption audits. Do not use it to manufacture an unsupported S→Y causal
edge. Refuters probe specified perturbations and do not validate the DAG.

## Data resources

- [ERP CORE](https://erpinfo.org/erp-core) remains the bounded controlled H1
  dataset. Cog-Surp conservatively applies CC BY-SA 4.0 because the bundled
  license and author site are more restrictive than one BIDS declaration.
- [DERCo](https://doi.org/10.1038/s41597-024-03915-8) supplies authoritative
  word-level EEG and prediction linkage for naturalistic H3. Its OSF deposit
  declares no dataset license, so local analysis is allowed without
  redistribution.

## Model selection

The runnable matrix uses immutable revisions of SmolLM2-135M and GPT-2 as two
CPU-feasible base-model families. Qwen2.5-0.5B is configured as a larger
cross-family extension but is not promoted to a completed principal result on
this no-GPU host. Model selection is based on transparent model cards,
tokenizer support, license, revision pinning, and feasible exact inference,
not generic leaderboard rank.

## Positioning

Cog-Surp is best positioned as open research software, a reproducibility
platform, and a methodological benchmark—not a clinical product, brain
decoder, or claim of human-like model mechanisms. Existing tools cover
individual layers well; the contribution is the scientifically bounded,
real-EEG, tokenization-aware, participant/item-aware, causal-audited,
artifact-traceable integration.

