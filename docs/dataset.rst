=======
Dataset
=======

This page summarizes the data used by the repository and the labels consumed by
the model.


Dataset
=======

:Dataset used by this repository: Michigan Human Anesthesia fMRI Dataset
:OpenNeuro ID: ds006623
:DOI: `10.18112/openneuro.ds006623.v1.0.0 <https://doi.org/10.18112/openneuro.ds006623.v1.0.0>`_
:Modeling set in this repository: 25 subjects from ``src/config.py``
:License: CC0 1.0 Universal (Public Domain)

The repository combines three related sources:

- the 2018 Huang et al. paper, which was a small task-fMRI proof-of-principle study
- the later open dataset and linked MATLAB reference, which define the 446-ROI connectivity workflow used here
- this repository, which builds a Python ML pipeline on top of those derivatives

The subject counts on this page refer to the current repository workflow, not to
the five analyzed propofol subjects reported in the 2018 paper.


Inputs
======

The project uses **preprocessed derivatives** only — not the raw scanner images
and not the original paper's AFNI task-fMRI preprocessing pipeline.

For each subject and scan, the code loads:

- **Timeseries file** (``*_timeseries.tsv``): A table where each row is a timepoint and each column is a brain region. We use the first 446 columns from the 4S456Parcels atlas.

- **Motion file** (``*_motion.tsv``): Head movement parameters for each timepoint. Column 8 contains framewise displacement (FD), which measures how much the head moved between consecutive timepoints.

The brain is divided with the **4S456Parcels** atlas. Following the linked
MATLAB reference workflow, the repository uses the first 446 regions.


Connectivity
============

For each scan, the code:

1. Loads the timeseries (446 brain regions over time).
2. Removes timepoints where the head moved too much (FD ≥ 0.8 mm).
3. Computes the Pearson correlation between every pair of brain regions.
4. Sets the diagonal to zero.

The result is a **446 × 446 connectivity matrix** — a symmetric table where each
cell indicates how strongly two brain regions are synchronised.


Labels
======

Each subject went through a sedation protocol with mental imagery tasks. The code
segments each subject's data into **7 conditions**.

These seven condition blocks come from the later MATLAB reference workflow used
for the open dataset, and they are the labels consumed by this repository's ML
pipeline.

The 2018 paper itself focused on task-fMRI responses across baseline, PreLOR,
LOR, ROR, and recovery periods rather than on this repository's final
classification matrix.

The current condition map is:

.. list-table::
   :header-rows: 1
   :widths: 5 25 15

   * - ID
     - Condition
     - Label
   * - 0
     - Resting state, run 1 (baseline)
     - Conscious
   * - 1
     - Imagery, run 1 (awake, pre-sedation)
     - Conscious
   * - 2
     - Imagery, run 2 before loss of responsiveness (preLOR)
     - Conscious
   * - 3
     - Imagery, runs 2–3 during loss of responsiveness (LOR)
     - **Unconscious**
   * - 4
     - Imagery, run 3 after return of responsiveness (postROR)
     - Conscious
   * - 5
     - Imagery, run 4 (recovery)
     - Conscious
   * - 6
     - Resting state, run 2 (recovery)
     - Conscious

For binary classification: condition 3 is labelled **unconscious** (0), all
others are labelled **conscious** (1).


Timing
======

**Loss of Responsiveness (LOR)** and **Return of Responsiveness (ROR)** times
are defined as frame indices taken from the linked MATLAB reference workflow.
They indicate where the subject stopped and resumed responding to auditory
commands.

The repository also skips **375 frames** around each transition when assembling
the LOR block because that is how the MATLAB reference constructs the
connectivity segments.

This is related to, but not identical with, the wording in the 2018 paper,
which described excluding uncertain transition periods because behavioral checks
were intermittent.

Subject ``sub-29`` has no postROR segment. Missing scans or fully censored
conditions are stored as NaN and skipped during training.


Subjects
========

The 25 subjects used (as defined in the code):

::

   sub-02  sub-03  sub-04  sub-05  sub-06  sub-07
   sub-11  sub-12  sub-13  sub-14  sub-15  sub-16
   sub-17  sub-18  sub-19  sub-20  sub-21  sub-22
   sub-23  sub-24  sub-25  sub-26  sub-27  sub-28
   sub-29

Subject sub-30 is excluded because the current repository workflow does not have
timing data for that subject in the linked reference workflow.


References
==========

**Original research paper:**

- Huang et al. (2018). *Scientific Reports*. `DOI: 10.1038/s41598-018-31436-z <https://doi.org/10.1038/s41598-018-31436-z>`_

**Linked MATLAB reference used for connectivity workflow provenance:**

- Jang et al. *An Open fMRI Resource for Studying Human Brain Function and Covert Consciousness Under Anesthesia* analysis code. `GitHub repository <https://github.com/janghw4/Anesthesia-fMRI-functional-connectivity-and-balance-calculation>`_

**Preprocessing tools:**

- BIDS specification: https://bids.neuroimaging.io/
- fMRIPrep: https://fmriprep.org/
- XCP-D: https://xcp-d.readthedocs.io/
