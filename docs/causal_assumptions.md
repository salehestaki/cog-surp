# Causal assumptions

Cog-Surp keeps three estimands separate:

1. the randomized condition effect on human N400;
2. the same textual condition effect on a model-derived measure; and
3. incremental held-out EEG prediction by that measure.

The default DAG contains:

- `experimental_condition -> human_n400`
- `experimental_condition -> model_prediction_measure`
- item covariates and identity into both relevant outcomes
- participant factors into human N400
- model identity and tokenizer/probability strategy into the model measure
- preprocessing/measurement choices into the measured N400

For randomized ERP CORE assignment, the default graph has no backdoor parent
of condition and therefore no covariate adjustment is required to identify the
condition effect. Precision variables and crossed participant/item structure
still belong in the outcome model. If assignment is nonrandom or matching is
imperfect, observed item covariates may cause condition and require adjustment.

The default graph explicitly forbids
`model_prediction_measure -> human_n400`. Adding model surprisal to a
hierarchical EEG model assesses predictive or explanatory alignment, not a
physical intervention on participants. Software validation rejects this edge.

DoWhy refuters, when run, test robustness to specified perturbations. They do
not prove the graph correct, establish absence of unmeasured causes, or turn
alignment into mechanistic homology.

The real ERP CORE audit executes placebo-treatment, random-common-cause,
data-subset, bootstrap, and explicitly parameterized simulated-unobserved-
common-cause refuters for A→Y and each A→S model effect. The equal-participant
paired H1 and matched-target paired H2 estimates remain primary; the DoWhy
linear estimates document identification and sensitivity.

Graph-implied conditional-independence falsification is not claimed because
the released artifacts do not jointly observe the conceptual graph's latent
participant/item variation and measurement-choice nodes. Recording that
non-identification is preferable to presenting a partial observed graph as a
test of the full scientific DAG.
