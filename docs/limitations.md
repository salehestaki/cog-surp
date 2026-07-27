# Limitations

## Scientific interpretation

Cog-Surp tests experimental effects, model behavior, and predictive alignment
as separate questions. Correlation, similar anomaly sensitivity, or parallel
condition effects do not establish a shared mechanism or a causal
surprisal-to-EEG pathway.

## ERP CORE item linkage

ERP CORE is well suited to the primary controlled human N400 condition effect.
Its public BIDS event files contain onset, sample, and condition code but not
the prime or target word presented on each randomized trial. The EEGLAB event
records likewise expose only type, latency, and urevent. The publisher supplies
two counterbalanced stimulus lists (200 condition rows, 100 targets), but the
currently inspected public files do not provide a trial-to-word key.

Consequently, Cog-Surp must not pretend that ERP CORE single trials have known
target words. The initial single-trial artifact uses `target_word=null` and a
target-event identifier. ERP CORE can support H1, and its public stimuli can
support a separate H2 model-side analysis. Item-level H3 alignment requires
either an authoritative trial key from the publisher or a second real language
EEG dataset with explicit word/event alignment.

## ERP CORE preprocessing sensitivity

The primary reprocessing applies fixed automated ICA, bad-channel, trial, and
participant rules to all 39 publicly available participant folders. It retains
34 participants, fewer than the publisher-reported final N400 sample of 39.
Differences can arise from automated component selection and rejection rather
than publisher-equivalent manual judgments. Cog-Surp therefore reports the
rule-based cohort as primary and all 39 public participants as a named
sensitivity analysis. Both show a more-negative unrelated-minus-related
contrast, but this agreement does not validate every preprocessing choice.

## Licensing

ERP CORE's BIDS description declares CC0, while its bundled license and
author-maintained project page declare CC BY-SA 4.0. Cog-Surp records both and
conservatively applies CC BY-SA 4.0 pending clarification.

DERCo's public OSF project does not declare a dataset license. The associated
article's CC BY license is not assumed to license the deposited EEG data.
Cog-Surp records `NOASSERTION`, permits local analysis, and does not
redistribute the data.

## DERCo article-0 scope

The current H3 result uses one story and publisher-preprocessed epochs. It now
includes two small causal-LM families, but is not cross-article or broad-scale
robustness. Although conditional standardized surprisal coefficients are
negative for both models, improvement over lexical controls is about 0.015
microvolts RMSE and held-out R2 is near zero. Practical predictive gain is
therefore weak.

The model coefficient is an association conditional on measured covariates and
crossed participant/item intercepts. It does not identify a causal effect of
model surprisal on EEG, and it does not establish shared computation or
mechanistic homology.

## Controlled model effects and causal auditing

The ERP CORE H2 analysis scores publisher stimuli but cannot link those words
to an individual participant's randomized trials. H1 and H2 are therefore
separate estimands over the same manipulation, not a trial-level mediation
analysis. Their parallel direction does not identify a surprisal-to-N400 path.

DoWhy identifies and perturbs the randomized condition effects under the
encoded graph. Refuters are stress tests of particular assumptions and data
perturbations; passing them does not prove the graph or eliminate unmeasured
bias. Graph falsification is not reported as executed because the conceptual
graph includes latent and design variables that are not jointly observed.

## Exploratory sensor-time inference

The cluster permutation analysis is an exploratory multiplicity-controlled
localization aid. Its clusters depend on the chosen threshold, adjacency,
window, preprocessing, and participant QC. Cog-Surp does not use it to claim
an exact N400 onset or peak, infer a neural source, or replace the
prespecified CPz 300-500 ms H1 test.

## Generated stimuli

Rule-based and LLM-assisted stimuli are computational stress tests only.
They have not undergone human plausibility, expectancy, grammar, or norming
validation and cannot support claims about human N400 effects.

## Runtime and packaging

The base wheel passes an isolated Python 3.12 installation and doctor check.
The local Docker daemon denies access to its Windows named pipe, so the
provided pinned CPU image has not been built in this environment. No NVIDIA
GPU is available, limiting scale robustness to bounded CPU checkpoints.
