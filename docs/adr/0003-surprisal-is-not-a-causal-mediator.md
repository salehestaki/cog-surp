# ADR 0003: Surprisal is not a default causal mediator

- Status: Accepted
- Date: 2026-07-27

## Context

Experimental condition can change both a participant's EEG and a score
computed later from the stimulus by a language model. Correlation or parallel
condition sensitivity does not make that computed score a physical cause of
the recorded EEG.

## Decision

The default DAG includes `condition -> human_n400` and
`condition -> model_measure`, alongside justified item, participant,
measurement, model, tokenizer, and covariate relations. It does not include
`model_measure -> human_n400`.

DoWhy analyses identify and audit the experimental condition effect on human
N400 and the textual condition effect on each model measure where assumptions
permit. Models that add surprisal to EEG controls estimate incremental
predictive or explanatory alignment. The software rejects unsupported causal
mediation labels unless an alternative graph and identification rationale are
explicitly supplied.

## Consequences

Reports distinguish causal condition effects, model behavior, and held-out
alignment. Refuters probe particular assumptions and perturbations; they do not
prove a graph true or establish model-brain homology.

