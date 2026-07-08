============
Introduction
============

Overview
========

This repository is a small research codebase around covert-awareness-related
fMRI analysis. It is grounded in Huang et al. (2018), but it does not reproduce
the paper's primary task-fMRI analysis.


Paper
=====

In the source paper, *covert consciousness* meant task-specific mental imagery
activity despite loss of behavioral response. The published analysis included
five propofol subjects after one saline control and one motion exclusion.

For the paper-first summary, see :doc:`paper_background`.


Repository
==========

This repository uses preprocessed open derivatives, builds 446-ROI connectivity
matrices, extracts summary features plus PCA components, and trains one XGBoost
classifier with leave-one-subject-out validation.


Scope
=====

The paper's evidence standard and the repository's evidence standard are not
the same. This repository is research code, not a medical device or a validated
consciousness test.

.. danger::
    This repository is a research tool, not a medical device. It should not be
    used for diagnosis, treatment, or intraoperative decision-making.


Read Next
=========

1. :doc:`paper_background`
2. :doc:`dataset`
3. :doc:`feature_extraction`
4. :doc:`model_architecture`
