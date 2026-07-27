# Data policy

`data/raw/` contains immutable publisher downloads and is ignored by Git.
`data/derivatives/` contains transformed EEG and is also ignored. Every fetch
and derivative must have a checksummed manifest under `artifacts/runs/`.

ERP CORE resources have conflicting embedded license metadata: the BIDS
description says CC0, while the bundled license and author site say CC BY-SA
4.0. Cog-Surp conservatively applies CC BY-SA 4.0 and requires the Kappenman
et al. citation. See ADR 0002.

Never commit restricted EEG, participant-sensitive data, model weights, or
credentials.
