# ADR 0002: ERP CORE N400 is primary; synthetic EEG is non-evidential

- Status: Accepted
- Date: 2026-07-27

## Context

An empirical N400 benchmark needs real human EEG. Synthetic signals are useful
for deterministic CI and recovery tests but cannot support conclusions about
human cognition.

## Access and licensing evidence

ERP CORE is the primary dataset. The project website identifies a freely
available N400 word-pair judgment dataset and licenses the resources under
CC BY-SA 4.0:

- https://erpinfo.org/erp-core
- DOI landing identifier: https://doi.org/10.18115/D5JW4R
- N400 OSF resource: https://osf.io/29xpq/
- Paper: https://doi.org/10.1016/j.neuroimage.2020.117465

On 2026-07-27, anonymous direct downloads returned HTTP 200 for the N400
README, LICENSE, and `sub-001_task-N400_eeg.set` (4,806,240 bytes). The paper
reports archival raw, processed, and BIDS-compatible data for 40 participants.
The author site requires attribution and share-alike distribution of adapted
resources.

The BIDS `dataset_description.json` declares CC0, but the bundled `LICENSE`
and the author-maintained project page declare CC BY-SA 4.0. Until the
publisher resolves that inconsistency, Cog-Surp records both statements and
applies the more conservative CC BY-SA 4.0 terms to downloaded resources and
adaptations.

## Decision

Implement an ERP CORE adapter first and retain raw downloads immutably outside
version control. Record source identifiers, checksums, retrieval timestamps,
license, exclusions, events, montage, reference, and preprocessing in a
manifest. Cacheable subject-level downloads enable a bounded smoke slice before
the full cohort run.

`SyntheticN400Generator` fixtures may support tests, demos, recovery, power,
and sensitivity analysis only. Synthetic tables, figures, reports, and
dashboard pages must carry an explicit `data_status=synthetic` marker and a
persistent **Synthetic data** badge.

## Consequences

Empirical claims require an end-to-end ERP CORE artifact. CI remains small and
network-independent. Redistributed adapted dataset material must honor
CC BY-SA 4.0; raw EEG is never committed to this repository.
