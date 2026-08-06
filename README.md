# Covert Awareness Detector

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/Dataset-OpenNeuro%20ds006623-orange.svg)](https://openneuro.org/datasets/ds006623)

Machine learning pipeline for studying covert-awareness-related fMRI functional
connectivity during anesthesia.

> **Disclosure:** This software was developed with AI assistance under human supervision. It is actively being improved, validated, and documented.

## Overview

This repository contains a Python research pipeline built on the [Michigan Human Anesthesia fMRI Dataset](https://openneuro.org/datasets/ds006623) (OpenNeuro ds006623).

It does **not** reproduce the 2018 paper's main task-fMRI analysis. Instead, it trains a separate connectivity-based classifier on the public derivatives.

The ISD feature is a Python approximation based on the linked MATLAB reference. It keeps the same basic idea, `efficiency - clustering`, but the graph calculations are simpler, so the values will not exactly match the MATLAB/BCT output.

## Quick Start

This release requires Python 3.11+. It was verified with Python 3.12.3 on Ubuntu 24.04 WSL.

If `python3 --version` reports an older interpreter, install Python 3.11+ before creating `.venv`.

```bash
# Clone and set up
git clone https://github.com/byteshiftlabs/covert-awareness-detector.git
cd covert-awareness-detector
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt

# Quick smoke run: downloads only the first 5 supported subjects
./run_quick_training.sh

# Full run: downloads the supported dataset and trains the default model
./run_full_training.sh
```

## How It Works

The steps below describe the current Python classifier, not the original
paper's activation-based method.

The pipeline processes fMRI brain scans through several stages to classify consciousness states:

### 1. Data Input
We start with preprocessed brain scans from the Michigan Human Anesthesia fMRI Dataset. Each scan captures brain activity across hundreds of brain regions over time—think of it as a recording of which brain areas are "lighting up" together at each moment. We work with data from people who were scanned while transitioning between conscious and unconscious states under controlled anesthesia.

- **Source**: XCP-D preprocessed fMRI timeseries from OpenNeuro ds006623
- **What we have**: Brain activity measurements across 446 regions, recorded over time for 25 people in different states of consciousness

### 2. Feature Extraction
This is where we transform raw brain signals into meaningful patterns that distinguish consciousness from unconsciousness.

**Connectivity Matrices**: We measure how synchronized different brain regions are with each other. If two regions consistently activate together, they're "connected." This creates a map of functional connections across the entire brain—essentially a snapshot of how the brain's regions communicate.

**ISD (Integration-Segregation Difference)**: This measures the balance between integration (how efficiently information flows across the entire brain network) and segregation (how well distinct brain regions maintain specialized, local processing). ISD quantifies this by computing the difference between these two properties (efficiency minus clustering).

**Network Summary Statistics**: We also compute basic graph properties from the connectivity matrix—mean degree, strength, and density—to capture the overall topology of the brain network at each state.

### 3. Dimensionality Reduction (PCA)
Raw connectivity data is massive—nearly 100,000 individual connections between brain regions. Most of this information is redundant or noisy. **Principal Component Analysis (PCA)** is like finding the "essence" of the data: it identifies the main patterns that explain most of the variation, compressing the data down to the most important features while throwing away the noise. This prevents the model from overfitting to irrelevant details.

### 4. Model Training
We use **XGBoost**, a powerful machine learning algorithm that builds an "ensemble" of decision trees. Think of it as training many simple classifiers that each learn different aspects of the data, then combining their votes for a final prediction.

**Handling Class Imbalance with SMOTE**: Our dataset has more unconscious examples than conscious ones (people spend more time sedated). SMOTE (Synthetic Minority Oversampling) creates synthetic examples of the underrepresented class, ensuring the model learns to recognize both states equally well rather than just guessing "unconscious" most of the time.

**Leave-One-Subject-Out Cross-Validation**: We train the model on data from all subjects except one, then test it on the left-out subject. We repeat this for every subject. This ensures the model learns general patterns about consciousness, not just memorizing specific individuals' brain signatures.

### 5. Prediction
Give the trained model a new brain scan, and it outputs a probability: how likely is this person to be conscious or unconscious? The model draws on the full set of features it learned during training—connectivity patterns compressed via PCA, ISD metrics, and network summary statistics—to make its prediction.


## Project Structure

```
src/
  config.py               # Dataset paths, subject list, scan parameters
  data_loader.py          # Load timeseries, motion filtering, connectivity matrices
  download_dataset.py     # OpenNeuro dataset downloader
  features.py             # Approximate ISD, graph metrics, connectivity feature extraction
  train.py                # Full training pipeline: XGBoost + PCA + SMOTE
  validate_model.py       # Overfitting checks and permutation tests

tests/                    # pytest test suite
docs/                     # Sphinx documentation

run_full_training.sh      # Automated training pipeline (START HERE)
pyproject.toml            # Project metadata and dependencies
requirements.txt          # Minimum-spec dependencies
requirements-lock.txt     # Release-verified lockfile
```

## Requirements

- Python 3.11+
- Bash-compatible shell environment for the helper scripts
- Roughly 350 MB of free disk space for the quick run
- Roughly 1.8 GB of free disk space for the full download

## Default Model

The default training pipeline (`src/train.py` / `./run_full_training.sh`) trains and validates the **XGBoost** classifier only (full connectivity + PCA + SMOTE + threshold tuning).

For installation details and longer documentation, see [docs/](docs/) and [docs/installation.rst](docs/installation.rst).


## Acknowledgments

**Original Research**: Huang, Hudetz, Mashour et al. — University of Michigan  
**Dataset**: [OpenNeuro ds006623](https://openneuro.org/datasets/ds006623) (CC0 Public Domain)  
**MATLAB Reference**: [Jang et al.](https://github.com/janghw4/Anesthesia-fMRI-functional-connectivity-and-balance-calculation)  
**This Implementation**: Independent Python ML pipeline by byteshiftlabs, built with AI assistance

## Citation

```bibtex
@article{huang2018covert,
  title     = {Brain imaging reveals covert consciousness during behavioral unresponsiveness},
  author    = {Huang, Zirui and others},
  journal   = {Scientific Reports},
  volume    = {8},
  pages     = {13195},
  year      = {2018},
  doi       = {10.1038/s41598-018-31436-z}
}
```

## License

MIT License — see [LICENSE](LICENSE).  
Dataset: CC0 (Public Domain).

---

**Documentation**: [docs/](docs/)
