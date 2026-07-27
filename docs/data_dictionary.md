# Analysis data dictionary

Variable roles are estimand-specific. A variable is not included merely
because it is available.

| Variable | Artifact field | Role in ERP CORE H1/H2 | Role in DERCo H3 | Current use |
|---|---|---|---|---|
| Experimental condition (A) | `condition`, encoded as `experimental_condition` | Randomized/counterbalanced treatment | Not the naturalistic H3 exposure | H1 and H2 contrasts |
| Human N400 outcome (Y) | `n400_mean_voltage_uv` | Measured outcome; more-negative is larger N400 | Measured outcome | Primary EEG outcome |
| Model prediction measure (S) | `target_surprisal_nats` | Deterministic model outcome for H2 | Predictive/explanatory exposure, not a physical treatment | H2, H3 |
| Participant (P) | `participant` | Repeated-measures design variable | Grouping factor and source of outcome heterogeneity | Equal participant weighting, grouped holdout, random intercept |
| Item identity (I) | `item` | Released ERP trials lack the authoritative word key; no fabricated join | Grouping factor and unmeasured item variation | Grouped holdout, random intercept |
| Model identity (M) | `model_id`, `model_revision` | Cause of model score only | Robustness/design variable | Separate model artifacts |
| Tokenizer/strategy (T) | `tokenizer_revision`, `probability_strategy`, `target_token_count` | Measurement/design choice for S | Measurement/design choice for S | Strategy sensitivity; token count is a lexical/tokenization control |
| Preprocessing choice (Q) | preprocessing run and configuration | Measurement choice affecting Y | Publisher and Cog-Surp measurement choices affecting Y | Versioned run identity and sensitivity |
| Human cloze surprisal | `human_cloze_surprisal_nats` | Not available for randomized ERP trials | Human predictability alternative and item-level precision/common-cause proxy | Separate H4 model and hierarchical covariate |
| Human response entropy | `human_response_entropy_nats` | Not available | Alternative uncertainty measure and precision variable | Separate H4 model and hierarchical covariate |
| Word frequency | `word_frequency` | Not joined without an authoritative ERP trial word | Observed item covariate affecting lexical processing and model score | Lexical control |
| Word length | `number_of_letters` | Not joined without trial words | Observed lexical precision/common-cause proxy | Lexical control |
| Word/context position | `word_position`, `context_word_count` | Design metadata only where known | Context/design variable affecting prediction and EEG | Lexical/context control |
| Trial rejection | `rejection_status`, peak-to-peak fields | Measurement-quality decision; conditioning is prespecified and sensitivity-audited | Publisher preprocessing quality decision | QC, never interpreted as a cognitive mediator |
| Behavioral accuracy | `behavior_accuracy` | Participant-quality criterion | Not used in DERCo H3 | Prespecified participant QC |

For randomized ERP CORE H1, A has no observed backdoor parent in the declared
DAG, so no adjustment is required for identification. Participant structure
improves precision. For DERCo H3, S is observational and the regression
coefficient is conditional/predictive; adding lexical variables does not turn
it into a causal effect. No current adjustment set treats a known mediator or
collider as a generic “control.”

