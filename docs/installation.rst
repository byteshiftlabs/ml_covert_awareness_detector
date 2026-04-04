============
Installation
============

This page covers the minimal setup needed to run the repository.


Environment
===========
The installation steps and default training pipeline have been tested on an
Ubuntu 24.04 WSL image.

If you are on Windows, install WSL first:

https://docs.microsoft.com/en-us/windows/wsl/install

The commands below assume a Linux-like shell environment with Python 3,
``venv``, and ``pip`` available.

.. code-block:: bash

   sudo apt update
   sudo apt install git python3 python3-venv python3-pip
   git clone https://github.com/byteshiftlabs/covert-awareness-detector.git
   cd covert-awareness-detector
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip setuptools wheel
   pip install -r requirements-lock.txt



Troubleshooting
===============

If you encounter issues:

.. code-block:: bash

   # Make sure virtual environment is activated
   which python  # Should point to .venv/bin/python
   
   # Reinstall dependencies if needed
   pip install --force-reinstall -r requirements-lock.txt





Next Steps
==========

Run ``python src/train.py`` or ``./run_full_training.sh`` to execute the
default XGBoost pipeline.

If you choose the minimum-spec install from ``requirements.txt`` instead of the
lockfile, run ``pip install .[docs]`` before ``make html``.
