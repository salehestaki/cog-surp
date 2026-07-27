# ADR 0001: Use a modular monolith

- Status: Accepted
- Date: 2026-07-27

## Context

Cog-Surp combines dataset adapters, EEG preprocessing, language-model scoring,
statistics, causal auditing, provenance, reporting, and a Streamlit reader.
The 30-day/300-hour scope requires reproducible local and batch execution, not
independently operated services.

## Decision

Build one installable Python package with ports-and-adapters boundaries.
Domain protocols isolate EEG datasets, surprisal backends, region-probability
strategies, storage, and run metadata. The CLI owns offline computation.
Streamlit contains presentation code only and reads completed artifacts.

## Consequences

One lockfile and release unit simplify reproducibility and cross-component
tests. Boundaries permit later extraction if operational evidence warrants it.
Kubernetes, task queues, microservices, and synchronous heavy dashboard work
are excluded from the first release.

