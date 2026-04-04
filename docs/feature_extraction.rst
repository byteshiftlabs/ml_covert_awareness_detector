==================
Feature Extraction
==================

This page summarizes the feature set used by the repository's classifier.


Scope
=====

Huang et al. (2018) used task-based fMRI activation as the main evidence
standard. This repository instead turns connectivity matrices into
cross-subject machine-learning inputs.


Inputs
======

The repository starts with a **446 x 446 connectivity matrix** for each
subject-condition block. Each entry says how strongly two brain regions vary
together across time after motion filtering.

Raw connectivity is too large to use directly as the only model input, so the
repository extracts smaller summaries and keeps a compressed version of the full
matrix.


Features
========

The repository uses four feature groups:

- **ISD**: a summary derived from the linked MATLAB reference. In this repository, ISD is efficiency minus clustering after principal-eigenvector regression.
- **Graph summaries**: mean degree, degree variability, mean strength, strength variability, and density.
- **Distribution summaries**: mean, standard deviation, skewness, kurtosis, quartiles, minimum, and maximum.
- **Connectivity vector**: the upper triangle of the connectivity matrix, containing roughly 99,000 pairwise values.


Training Matrix
===============

The default pipeline concatenates:

- 17 raw summary features are assembled for each sample
- 17 per-subject deviation features are added by comparing a sample to that subject's conscious baseline
- the high-dimensional connectivity vector is imputed and reduced with PCA
- the engineered summaries and PCA components are concatenated into the final training matrix


Interpretation
==============

A feature such as ISD can be a useful network summary, but this repository's
predictions come from the combined feature set. A high or low value in one
summary measure should not be interpreted on its own as direct evidence of
consciousness.


Read Next
=========

- See :doc:`paper_background` for the paper's actual claim.
- See :doc:`dataset` for how subject-condition blocks are assembled.
- See :doc:`model_architecture` for how the repository uses these features in XGBoost.
