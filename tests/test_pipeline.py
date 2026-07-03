"""Tests for shared pipeline helpers."""

import json

import numpy as np

from pipeline import optimize_threshold, save_results, select_subjects


def test_select_subjects_limits_ordered_subset():
    subjects = ["sub-01", "sub-02", "sub-03"]
    assert select_subjects(subjects, max_subjects=2) == ["sub-01", "sub-02"]


def test_optimize_threshold_returns_serializable_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_probabilities = np.array([0.1, 0.4, 0.7, 0.9])

    threshold, metrics = optimize_threshold(y_true, y_probabilities)

    assert 0.1 <= threshold <= 0.9
    assert isinstance(metrics["confusion_matrix"], list)
    json.dumps(metrics)


def test_save_results_writes_json(tmp_path):
    output_path = save_results({"status": "ok"}, output_dir=tmp_path)

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "ok"