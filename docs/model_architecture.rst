=====
Model
=====

This page describes the repository's only model.

It does **not** describe the primary analysis used in Huang et al. (2018). That
paper relied on task-based fMRI activation during mental imagery, not on the
XGBoost classifier documented here.


Scope
=====

The paper and the repository answer related but different questions.

The 2018 paper:

- used task-based fMRI mental imagery
- made single-subject inferences with strict false-positive control
- treated task-locked activation during loss of responsiveness as the key signal

This repository:

- uses 446-ROI connectivity matrices derived from the open dataset release
- converts those matrices into engineered features and PCA components
- trains a cross-subject binary classifier with leave-one-subject-out validation

The repository can therefore be understood as an exploratory ML layer built on
the same research area, not as the paper's original decision procedure.


Pipeline
========

This codebase implements a single default classifier: **XGBoost**. The training
flow is:

1. assemble engineered summary features from each connectivity matrix
2. add per-subject deviation features relative to conscious conditions
3. reduce the full connectivity vector with PCA
4. train XGBoost on the combined feature matrix
5. choose a probability threshold that maximizes balanced accuracy


Evaluation
==========

Because we want the model to work on **new subjects it has never seen
before**, we evaluate it with a strict procedure called
Leave-One-Subject-Out (LOSO) cross-validation:

1. Pick one subject and set their data aside as the test set.
2. Train the model on all remaining subjects.
3. Test the model on the held-out subject.
4. Repeat for every subject.

This ensures the model is always tested on a person whose data it never saw
during training. It is a stronger check than random sample splitting because it
prevents the model from seeing the same person's patterns in both training and
test folds.


Conscious and unconscious samples may not appear in equal numbers. To
prevent class imbalance from biasing the classifier, the pipeline uses
SMOTE oversampling and class weighting with XGBoost so the model does not drift
toward the majority class.


Limits
======

This default model should not be confused with a validated consciousness test.

- It is not the same evidence standard used in the 2018 paper.
- It has not been established here as a clinical diagnostic tool.
- Its predictions depend on the repository's labeling, feature design, and subject split choices.


Run
===

.. code-block:: bash

   # Train and evaluate XGBoost with LOSO cross-validation
   python src/train.py

   # Quick smoke run on the first 5 subjects
   python src/train.py --max-subjects 5

This runs the default XGBoost pipeline, prints the metrics to stdout, and writes
a JSON summary under ``results/``.


Read Next
=========

- See :doc:`paper_background` for the original proof-of-principle study.
- See :doc:`feature_extraction` for how the repository builds model inputs.
- See :doc:`dataset` for the current subject-condition labeling workflow.
