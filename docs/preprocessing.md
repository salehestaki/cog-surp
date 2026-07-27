# EEG preprocessing and outcome definitions

## ERP CORE

The primary pipeline reads EEGLAB continuous data, applies the standard 10-20
montage, marks the publisher EOG channels, resamples to 256 Hz, filters
0.1-30 Hz, detects and interpolates at most four extreme/flat EEG channels,
average-references, and performs deterministic 20-component extended-infomax
ICA on a 1-30 Hz fitting copy. Components associated with each recorded EOG
channel are removed. Because the low-pass is 30 Hz, the configured 60 Hz line
frequency lies outside the retained band and no separate notch is applied.

Target codes 211/212 are related and 221/222 are unrelated. Epochs span
-200-800 ms with a -200-0 ms baseline. Trials are rejected when any EEG
channel exceeds 200 microvolts peak-to-peak across the epoch or bipolar
horizontal/vertical EOG exceeds 100 microvolts in -200-200 ms. Participant
inclusion requires at least 75% behavioral accuracy, no more than 25% rejected
target trials, and at least 30 accepted trials per condition. The outcome is
mean CPz voltage from 300-500 ms.

The public OSF tree contains 39 participant folders (001-040 except 027).
Under the versioned automated rules, 34 enter the primary equal-participant
analysis and five remain in the all-public sensitivity analysis. This is a
Cog-Surp reprocessing cohort, not a claim that its automated exclusions exactly
reproduce every judgment in the publisher's final sample.

ERP CORE's public events do not expose randomized trial words. Its H1
condition contrast is valid, but trial/item H3 joins are prohibited.

## DERCo

DERCo releases publisher-preprocessed word epochs rather than continuous raw
recordings. Cog-Surp validates required metadata and channels, excludes the two
named publisher participants, removes only non-prediction epochs with empty
`WordID`, applies a -200-0 ms baseline, and averages 300-500 ms over Cz, CP1,
CP2, and Pz.

The outcome is `n400_mean_voltage_uv`. More-negative voltage means a larger
N400. The primary article-0 feature table contains 9,029 retained observations
from 20 participants and 559 model-scoreable items.

No source localization is attempted. ERP scalp maps are sensor-space
descriptions generated with canonical MNE plotting, not anatomical claims.

The secondary sensor-time analysis is explicitly `exploratory`. It applies a
two-sided within-participant spatiotemporal cluster permutation test to
unrelated-minus-related evokeds using EEG-channel adjacency, 512 permutations,
and a fixed seed over 0-800 ms. Cluster significance is not interpreted as an
exact effect onset, peak latency, neural source, or anatomical localization.
