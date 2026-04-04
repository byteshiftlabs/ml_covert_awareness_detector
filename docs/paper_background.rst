================
Paper Background
================

This page keeps a short, paper-first summary separate from the repository's own
model.


Question
========

Huang et al. (2018) asked whether a healthy volunteer under propofol sedation
could still show brain activity consistent with intentional command following
after losing the ability to squeeze a hand-held device on request.

The paper used the phrase *covert consciousness* for this mismatch:

- the person gives no outward behavioral response
- the brain still shows task-specific activity associated with following an instruction

That is a narrower claim than "the subject was fully awake," and it is also
narrower than "a classifier can detect consciousness."


Study
=====

The published proof-of-principle study recruited seven healthy volunteers.

- P01 was a saline control subject.
- P03 was excluded because of excessive head motion.
- Five propofol subjects remained in the reported analysis.

Propofol was infused in steps to predicted effect-site concentrations of 0, 0.4,
0.8, 1.2, 1.6, 2.0, and 2.4 ug/ml, with each step held for five minutes.
Behavioral responsiveness was tracked with an actual hand-squeeze task, which
let the authors define:

- PreLOR: before loss of responsiveness
- LOR: loss of responsiveness
- ROR: recovery of responsiveness


Tasks
=====

The paper did not train a classifier. It used task-based fMRI and asked subjects
to perform four tasks:

- tennis imagery
- navigation imagery
- squeeze imagery
- actual hand squeeze

The paper's main evidence standard was subject-level command following in the
fMRI signal, not model accuracy.

To reduce false positives, the authors ran pseudo-task analyses on resting-state
data and tightened the single-subject threshold until the estimated
false-positive rate fell below 5 percent. In the reported setup, that led to a
very strict individual-level threshold of voxel-level p = 1.E-15 with cluster
size 100 voxels.


Finding
=======

One subject, P04, showed robust task-locked activity during loss of
responsiveness:

- tennis imagery activated supplementary motor and premotor regions
- squeeze imagery activated premotor regions
- some imagery-related responses were delayed but still significant

A second subject, P07, showed more limited activation that the authors did not
treat as equally strong evidence on its own.

The paper's conclusion was a proof of principle:

- covert consciousness can occur in a healthy person under controlled propofol sedation
- anesthesia can provide a within-subject model for studying this dissociation
- the study was not designed to estimate prevalence or validate a clinical diagnostic test

The authors also reported that P04 had no explicit recall after scanning, which
they interpreted as consistent with the amnesic effects of propofol.


Repository Relation
===================

This repository is related to the paper, but it is not a direct reimplementation
of the 2018 analysis pipeline.

The paper's primary analysis:

- task-based fMRI
- single-subject inference
- imagery-region activation and selected connectivity summaries
- proof-of-principle case finding

This repository's primary analysis:

- preprocessed connectivity derivatives from the open dataset release
- connectivity matrices over seven labeled conditions
- engineered features, PCA-reduced connectivity, and XGBoost
- leave-one-subject-out subject classification

Some code paths in this repository are adapted from the later MATLAB reference
linked in the README, which belongs to the open-resource analysis codebase, not
to the 2018 paper alone.