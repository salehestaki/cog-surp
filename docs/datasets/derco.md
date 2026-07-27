# DERCo adapter

## Source and scope

DERCo is the Dublin EEG-based Reading Experiment Corpus. The publisher article
is Quach, Gurrin, and Healy (2024), *Scientific Data* 11:1104,
<https://doi.org/10.1038/s41597-024-03915-8>. Data are deposited at
<https://osf.io/rkqbu/> and preprocessing code is at
<https://github.com/Tayerquach/DERCo>.

The corpus contains 22 native-English-speaking adults reading five stories.
Words were presented at 200 ms per word and photodiode markers aligned EEG to
word presentation. EEG was acquired from 32 channels at 1,000 Hz. The first
three stories use classic RSVP and the remaining two use RSVP with flankers.
Human next-word predictions were collected separately from 100 respondents per
story word (500 people total across five stories).

## Exclusions

The dataset paper reports that two male participants were excluded for
excessive eye movements but does not name them. Later DERCo analyses explicitly
identify them as `QPF42` and `USQ95`, including
<https://openreview.net/pdf/c874381d3ec502345615060e745ced2189fd0d8c.pdf>.
The primary configuration names both and validates the count against the
publisher report.

## Released units and preprocessing

The adapter consumes publisher-preprocessed, article-level MNE Epochs FIF
files. The publisher pipeline includes 0.1-45 Hz filtering, common-average
reference, FASTER bad-channel handling, ICA/Picard with ICLabel, and
autoreject. Cog-Surp does not misrepresent these files as continuous raw data.

For the current analysis Cog-Surp applies the prespecified -200-0 ms baseline
and extracts mean voltage from 300-500 ms over Cz, CP1, CP2, and Pz.
Publisher rows outside the human-prediction task have empty `WordID` values and
are excluded explicitly. All analytical observations retain the publisher
`WordID`, which is the authoritative join to human predictions and model
scores.

## Cloze values

Raw exact-match cloze is recomputed from the downloaded response CSV for
audit. The publisher-corrected `p_cloze` attached to EEG metadata is canonical
for analysis; differences of up to 0.12 occur because the publisher corrected
known response/target issues. Full response-distribution entropy is computed
from normalized human responses as a second predictability measure.

## License

The OSF dataset does not declare a dataset license. The article is CC BY 4.0,
but Cog-Surp does not assume that the article license also licenses the
deposited EEG files. Dataset manifests therefore record `NOASSERTION`. Analyze
the data locally and do not redistribute it.
