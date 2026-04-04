"""Verify that importing train and validate_model does not execute pipelines."""

import importlib
import sys


def test_import_train_no_side_effects(capsys):
    """Importing train must not print training output or start computation."""
    sys.modules.pop('train', None)
    importlib.import_module('train')
    captured = capsys.readouterr()
    assert "ADVANCED CONSCIOUSNESS DETECTION" not in captured.out
    assert "Loading data" not in captured.out


def test_import_validate_model_no_side_effects(capsys):
    """Importing validate_model must not print validation output."""
    sys.modules.pop('validate_model', None)
    importlib.import_module('validate_model')
    captured = capsys.readouterr()
    assert "OVERFITTING VALIDATION" not in captured.out
    assert "Loading data" not in captured.out


def test_train_has_main_function():
    module = importlib.import_module('train')
    assert callable(getattr(module, 'main', None))


def test_validate_model_has_main_function():
    module = importlib.import_module('validate_model')
    assert callable(getattr(module, 'main', None))
