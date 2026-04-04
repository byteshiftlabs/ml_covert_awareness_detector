"""Integration-style tests for the training CLI."""

import numpy as np

import train


def test_main_handles_list_confusion_matrix_in_reporting(monkeypatch, capsys, tmp_path):
    dataset = {
        "subjects": ["sub-01", "sub-02"],
        "subject_ids": np.array(["sub-01", "sub-01", "sub-02", "sub-02"]),
        "labels": np.array([0, 1, 0, 1]),
        "x_combined": np.array(
            [
                [0.0, 0.1],
                [0.2, 0.3],
                [0.4, 0.5],
                [0.6, 0.7],
            ]
        ),
        "engineered_feature_count": 2,
        "connectivity_components": 1,
        "connectivity_variance_explained": 0.5,
    }
    saved = {}

    def fake_build_feature_dataset(*, max_subjects=None, on_subject_loaded=None):
        assert max_subjects is None
        return dataset

    def fake_train_xgboost_classifier(_x_train, _y_train, x_test=None):
        probabilities = np.linspace(0.2, 0.8, num=len(x_test)) if x_test is not None else None
        return None, probabilities, None

    def fake_optimize_threshold(_y_true, _y_probabilities):
        return 0.55, {
            "balanced_acc": 0.75,
            "accuracy": 0.75,
            "recall_unconscious": 0.5,
            "recall_conscious": 1.0,
            "precision": 0.67,
            "f1": 0.8,
            "roc_auc": 0.9,
            "confusion_matrix": [[1, 1], [0, 2]],
        }

    def fake_save_results(payload, prefix="results", output_dir=None):
        saved["payload"] = payload
        return tmp_path / "results.json"

    monkeypatch.setattr(train, "build_feature_dataset", fake_build_feature_dataset)
    monkeypatch.setattr(train, "train_xgboost_classifier", fake_train_xgboost_classifier)
    monkeypatch.setattr(train, "optimize_threshold", fake_optimize_threshold)
    monkeypatch.setattr(train, "save_results", fake_save_results)

    train.main([])

    stdout = capsys.readouterr().out
    assert "FINAL RESULTS" in stdout
    assert "Detection: 1/2 unconscious states correctly identified" in stdout
    assert saved["payload"]["metrics"]["confusion_matrix"] == [[1, 1], [0, 2]]