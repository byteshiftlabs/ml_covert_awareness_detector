"""Tests for shared pipeline helpers."""

import json

import numpy as np
import pytest

import pipeline
from pipeline import (
    build_feature_dataset,
    fit_fold_features,
    optimize_threshold,
    save_results,
    select_subjects,
    train_xgboost_classifier,
)


def test_select_subjects_limits_ordered_subset():
    subjects = ["sub-01", "sub-02", "sub-03"]
    assert select_subjects(subjects, max_subjects=2) == ["sub-01", "sub-02"]


def test_build_feature_dataset_requires_at_least_one_subject():
    with pytest.raises(ValueError, match="At least one subject is required"):
        build_feature_dataset(subjects=[])


def test_build_feature_dataset_excludes_fully_missing_samples(monkeypatch):
    n_rois = 6

    def fake_load_subject_all_conditions(subject):
        rng = np.random.default_rng(0)
        matrices = np.zeros((7, n_rois, n_rois))
        for condition_idx in range(7):
            m = rng.random((n_rois, n_rois))
            m = (m + m.T) / 2
            np.fill_diagonal(m, 0)
            matrices[condition_idx] = m
        # Condition 4 is entirely missing for every subject in this fake.
        matrices[4] = np.full((n_rois, n_rois), np.nan)
        return matrices

    monkeypatch.setattr(pipeline, "load_subject_all_conditions", fake_load_subject_all_conditions)

    dataset = build_feature_dataset(subjects=["sub-01", "sub-02"])

    # 2 subjects x 6 valid conditions (condition 4 excluded) = 12 samples.
    assert dataset["labels"].shape[0] == 12
    assert dataset["subject_ids"].shape[0] == 12


def test_build_feature_dataset_raises_when_all_samples_missing(monkeypatch):
    n_rois = 6

    def fake_load_subject_all_conditions(_subject):
        return np.full((7, n_rois, n_rois), np.nan)

    monkeypatch.setattr(pipeline, "load_subject_all_conditions", fake_load_subject_all_conditions)

    with pytest.raises(ValueError, match="No valid samples found"):
        build_feature_dataset(subjects=["sub-01"])


def test_fit_fold_features_fits_pca_and_imputer_on_train_only():
    rng = np.random.default_rng(0)
    x_engineered_train = rng.random((10, 3))
    x_connectivity_train = rng.random((10, 20))
    x_engineered_test = rng.random((2, 3))
    x_connectivity_test = rng.random((2, 20))

    x_train, x_test, metadata = fit_fold_features(
        x_engineered_train, x_connectivity_train, x_engineered_test, x_connectivity_test
    )

    assert x_train.shape[0] == 10
    assert x_test.shape[0] == 2
    assert x_train.shape[1] == x_test.shape[1]
    assert metadata["connectivity_components"] <= x_connectivity_train.shape[0] - 1
    assert 0.0 <= metadata["connectivity_variance_explained"] <= 1.0 + 1e-9


def test_fit_fold_features_test_split_does_not_shift_train_pca():
    """Transforming a test split must not change the train-fitted PCA/imputer."""
    rng = np.random.default_rng(1)
    x_engineered_train = rng.random((10, 3))
    x_connectivity_train = rng.random((10, 20))
    x_engineered_test_a = rng.random((2, 3))
    x_connectivity_test_a = rng.random((2, 20))
    # A wildly different held-out split should not change the training-side output.
    x_engineered_test_b = rng.random((2, 3)) * 1000
    x_connectivity_test_b = rng.random((2, 20)) * 1000

    x_train_a, _, _ = fit_fold_features(
        x_engineered_train, x_connectivity_train, x_engineered_test_a, x_connectivity_test_a
    )
    x_train_b, _, _ = fit_fold_features(
        x_engineered_train, x_connectivity_train, x_engineered_test_b, x_connectivity_test_b
    )

    np.testing.assert_allclose(x_train_a, x_train_b)


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