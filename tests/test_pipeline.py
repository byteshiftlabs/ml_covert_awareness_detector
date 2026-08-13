"""Tests for shared pipeline helpers."""

import json

import numpy as np
import pytest

from pipeline import build_feature_dataset, optimize_threshold, save_results, select_subjects, train_xgboost_classifier


def test_select_subjects_limits_ordered_subset():
    subjects = ["sub-01", "sub-02", "sub-03"]
    assert select_subjects(subjects, max_subjects=2) == ["sub-01", "sub-02"]


def test_build_feature_dataset_requires_at_least_one_subject():
    with pytest.raises(ValueError, match="At least one subject is required"):
        build_feature_dataset(subjects=[])


def test_optimize_threshold_returns_confusion_matrix_array():
    y_true = np.array([0, 0, 1, 1])
    y_probabilities = np.array([0.1, 0.4, 0.7, 0.9])

    threshold, metrics = optimize_threshold(y_true, y_probabilities)

    assert 0.1 <= threshold <= 0.9
    assert isinstance(metrics["confusion_matrix"], np.ndarray)
    assert metrics["confusion_matrix"].shape == (2, 2)


def test_save_results_writes_json(tmp_path):
    output_path = save_results(
        {
            "status": "ok",
            "metrics": {
                "balanced_acc": np.float64(0.75),
                "confusion_matrix": np.array([[1, 2], [3, 4]]),
            },
        },
        output_dir=tmp_path,
    )

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["metrics"]["balanced_acc"] == 0.75
    assert payload["metrics"]["confusion_matrix"] == [[1, 2], [3, 4]]


def test_train_xgboost_classifier_warns_when_smote_fails(monkeypatch):
    class FailingSMOTE:
        def __init__(self, *args, **kwargs):
            pass

        def fit_resample(self, x_train, y_train):
            raise ValueError("synthetic failure")

    class DummyClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, x_train, y_train, verbose=0):
            return self

        def predict_proba(self, x_test):
            return np.tile(np.array([[0.25, 0.75]]), (len(x_test), 1))

    monkeypatch.setitem(train_xgboost_classifier.__globals__, "SMOTE", FailingSMOTE)
    monkeypatch.setattr(train_xgboost_classifier.__globals__["xgb"], "XGBClassifier", DummyClassifier)

    x_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([0, 0, 1, 1])
    x_test = np.array([[1.5], [2.5]])

    with pytest.warns(RuntimeWarning, match="SMOTE resampling failed"):
        _, probabilities, _ = train_xgboost_classifier(x_train, y_train, x_test)

    assert np.allclose(probabilities, np.array([0.75, 0.75]))