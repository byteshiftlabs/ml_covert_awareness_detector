"""Tests for data loading helpers using synthetic data."""

import numpy as np
import pandas as pd
import pytest
from config import N_ROIS, FD_THRESHOLD, FD_COLUMN
from data_loader import filter_by_motion, compute_connectivity


class TestFilterByMotion:
    def test_keeps_good_timepoints(self):
        n_timepoints = 100
        timeseries = pd.DataFrame(
            np.random.default_rng(42).random((n_timepoints, N_ROIS))
        )
        # Create motion with 8 columns; FD in column 7 (0-indexed)
        motion_data = np.zeros((n_timepoints, 8))
        motion_data[:, FD_COLUMN] = 0.1  # all below threshold
        motion = pd.DataFrame(motion_data)

        result = filter_by_motion(timeseries, motion)
        assert len(result) == n_timepoints

    def test_removes_high_motion_timepoints(self):
        n_timepoints = 100
        timeseries = pd.DataFrame(
            np.random.default_rng(42).random((n_timepoints, N_ROIS))
        )
        motion_data = np.zeros((n_timepoints, 8))
        # First 50 good, last 50 bad
        motion_data[:50, FD_COLUMN] = 0.1
        motion_data[50:, FD_COLUMN] = 1.5
        motion = pd.DataFrame(motion_data)

        result = filter_by_motion(timeseries, motion)
        assert len(result) == 50

    def test_removes_all_if_all_bad(self):
        n_timepoints = 10
        timeseries = pd.DataFrame(
            np.random.default_rng(42).random((n_timepoints, N_ROIS))
        )
        motion_data = np.zeros((n_timepoints, 8))
        motion_data[:, FD_COLUMN] = 2.0  # all above threshold
        motion = pd.DataFrame(motion_data)

        result = filter_by_motion(timeseries, motion)
        assert len(result) == 0

    def test_threshold_boundary(self):
        """FD exactly at threshold should be excluded (< not <=)."""
        n_timepoints = 10
        timeseries = pd.DataFrame(
            np.random.default_rng(42).random((n_timepoints, N_ROIS))
        )
        motion_data = np.zeros((n_timepoints, 8))
        motion_data[:, FD_COLUMN] = FD_THRESHOLD  # exactly at threshold
        motion = pd.DataFrame(motion_data)

        result = filter_by_motion(timeseries, motion)
        assert len(result) == 0


class TestComputeConnectivity:
    def test_output_shape(self):
        timeseries = pd.DataFrame(
            np.random.default_rng(42).random((200, N_ROIS))
        )
        result = compute_connectivity(timeseries)
        assert result.shape == (N_ROIS, N_ROIS)

    def test_diagonal_is_zero(self):
        timeseries = pd.DataFrame(
            np.random.default_rng(42).random((200, N_ROIS))
        )
        result = compute_connectivity(timeseries)
        np.testing.assert_array_equal(np.diag(result), 0)

    def test_symmetric(self):
        timeseries = pd.DataFrame(
            np.random.default_rng(42).random((200, N_ROIS))
        )
        result = compute_connectivity(timeseries)
        np.testing.assert_array_almost_equal(result, result.T)

    def test_values_in_correlation_range(self):
        timeseries = pd.DataFrame(
            np.random.default_rng(42).random((200, N_ROIS))
        )
        result = compute_connectivity(timeseries)
        # Off-diagonal values should be in [-1, 1]
        mask = ~np.eye(N_ROIS, dtype=bool)
        assert np.all(result[mask] >= -1.0 - 1e-10)
        assert np.all(result[mask] <= 1.0 + 1e-10)

    def test_empty_timeseries_returns_nan(self):
        timeseries = pd.DataFrame(
            np.empty((0, N_ROIS))
        )
        result = compute_connectivity(timeseries)
        assert result.shape == (N_ROIS, N_ROIS)
        assert np.all(np.isnan(result))

    def test_small_matrix(self):
        """Verify with a small known case."""
        rng = np.random.default_rng(42)
        timeseries = pd.DataFrame(rng.random((50, 5)))
        result = compute_connectivity(timeseries)
        assert result.shape == (5, 5)
        assert np.all(np.diag(result) == 0)
