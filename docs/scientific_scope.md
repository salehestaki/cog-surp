# Scientific scope

## Public description

Cog-Surp is a reproducible benchmarking workbench for testing whether
language-model prediction measures explain human N400 responses across
controlled and naturalistic language paradigms.

## Questions kept separate

Cog-Surp treats the following as distinct estimands:

1. **Experimental causality:** Does an assigned semantic manipulation change
   human N400-window voltage?
2. **Model behavior:** Does the same text manipulation change a model-derived
   prediction measure?
3. **Neurocomputational alignment:** Does that measure improve prediction of
   held-out human EEG beyond prespecified experimental, lexical, semantic,
   participant, and item controls?

The third question concerns predictive or explanatory alignment. Similar
outputs, correlations, anomaly sensitivity, or parallel experimental effects
do not establish mechanistic, computational, or neurobiological homology
between a language model and a human brain.

## Primary vertical slice

The primary dataset is the ERP CORE N400 word-pair judgment paradigm. Its
related versus unrelated prime-target manipulation is the primary contrast.
The human outcome is single-trial mean voltage from 300–500 ms at the
publisher-aligned CPz electrode, with a −200–0 ms baseline. Exact preprocessing
parameters, exclusions, event codes, and sensitivity variants are versioned
before inspecting the key condition result. More-negative voltage means a
larger N400; every coefficient report must state its coding and sign.

The initial LM analysis scores the same target-word regions with teacher-forced
causal inference. Natural-log surprisal (nats) is canonical. Every aggregate
identifies its token-to-region probability strategy.

## Evidential boundary

Real human EEG is mandatory for empirical conclusions. Synthetic EEG is
permitted only for tests, demonstrations, estimator recovery, sensitivity,
power analysis, and CI. Every synthetic artifact and dashboard view must be
persistently labelled **Synthetic data** and must be excluded from empirical
claims about human cognition.

No default causal graph contains an edge from model surprisal to human EEG.
Cog-Surp may estimate the randomized condition effect on EEG and the textual
condition effect on model measures. Any stronger causal interpretation needs a
separate intervention, identification argument, and preregistered analysis.

## Confirmatory boundary

The primary dataset, contrast, ROI, N400 window, model family, probability
strategy, covariates, and hierarchical model are fixed in versioned
configuration. Alternative sensors, windows, models, tokenizers, probability
strategies, predictors, and preprocessing settings are labelled exploratory or
robustness analyses and receive appropriate multiplicity treatment.

## Non-goals

The first release does not perform real-time EEG acquisition, train a language
model, decode text from EEG, conduct causal discovery, localize cortical
sources without anatomical data, run a new human-subject study, or claim that
LLMs process language like the human brain.
