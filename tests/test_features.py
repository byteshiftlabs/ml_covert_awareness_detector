"""Tests for feature extraction on synthetic connectivity matrices."""

import numpy as np
import pytest
from features import (
    regress_principal_eigenvector,
    multilevel_efficiency,
    multilevel_clustering,
    compute_isd,
    extract_connectivity_features,
    extract_all_features,
)


@pytest.fixture
def identity_matrix():
    """Identity matrix (perfect self-correlation, no cross-correlation)."""
    return np.eye(10)


@pytest.fixture
def random_symmetric_matrix():
    """Random symmetric positive connectivity matrix."""
    rng = np.random.default_rng(42)
    m = rng.random((20, 20))
    m = (m + m.T) / 2
    np.fill_diagonal(m, 1.0)
    return m


@pytest.fixture
def nan_matrix():
    """Matrix with NaN rows/columns (simulates missing ROIs)."""
    m = np.ones((10, 10)) * 0.5
    np.fill_diagonal(m, 1.0)
    m[0, :] = np.nan
    m[:, 0] = np.nan
    return m


@pytest.fixture
def zero_matrix():
    """All-zero connectivity matrix."""
    return np.zeros((10, 10))


class TestRegressPrincipalEigenvector:
    def test_output_shape(self, random_symmetric_matrix):
        result = regress_principal_eigenvector(random_symmetric_matrix)
        assert result.shape == random_symmetric_matrix.shape

    def test_non_negative_output(self, random_symmetric_matrix):
        result = regress_principal_eigenvector(random_symmetric_matrix)
        valid = ~np.isnan(result)
        assert np.all(result[valid] >= 0)

    def test_handles_nan_rows(self, nan_matrix):
        result = regress_principal_eigenvector(nan_matrix)
        assert result.shape == nan_matrix.shape
        # NaN rows/cols should remain NaN
        assert np.isnan(result[0, 1])

    def test_empty_valid_returns_original(self):
        m = np.full((5, 5), np.nan)
        result = regress_principal_eigenvector(m)
        assert np.all(np.isnan(result))


class TestMultilevelEfficiency:
    def test_returns_scalar(self, random_symmetric_matrix):
        thresholds = np.logspace(-3, 0, 10)
        result = multilevel_efficiency(random_symmetric_matrix, thresholds)
        assert np.isscalar(result)

    def test_positive_for_connected_graph(self, random_symmetric_matrix):
        thresholds = np.logspace(-3, 0, 10)
        result = multilevel_efficiency(random_symmetric_matrix, thresholds)
        assert result > 0 or np.isnan(result)

    def test_nan_for_degenerate_input(self):
        m = np.full((2, 2), np.nan)
        thresholds = np.logspace(-3, 0, 10)
        result = multilevel_efficiency(m, thresholds)
        assert np.isnan(result)


class TestMultilevelClustering:
    def test_returns_scalar(self, random_symmetric_matrix):
        thresholds = np.logspace(-3, 0, 10)
        result = multilevel_clustering(random_symmetric_matrix, thresholds)
        assert np.isscalar(result)

    def test_non_negative(self, random_symmetric_matrix):
        thresholds = np.logspace(-3, 0, 10)
        result = multilevel_clustering(random_symmetric_matrix, thresholds)
        assert result >= 0 or np.isnan(result)


class TestComputeISD:
    def test_returns_three_values(self, random_symmetric_matrix):
        isd, eff, clust = compute_isd(random_symmetric_matrix)
        assert np.isscalar(isd)
        assert np.isscalar(eff)
        assert np.isscalar(clust)

    def test_isd_equals_efficiency_minus_clustering(self, random_symmetric_matrix):
        isd, eff, clust = compute_isd(random_symmetric_matrix)
        if not (np.isnan(isd) or np.isnan(eff) or np.isnan(clust)):
            assert abs(isd - (eff - clust)) < 1e-10

    def test_zero_matrix_computes(self, zero_matrix):
        isd, eff, clust = compute_isd(zero_matrix)
        # Should not raise; values may be NaN or 0
        assert np.isscalar(isd)


class TestExtractConnectivityFeatures:
    def test_output_length(self):
        n = 20
        m = np.random.default_rng(42).random((n, n))
        result = extract_connectivity_features(m)
        expected = n * (n - 1) // 2
        assert len(result) == expected

    def test_446_roi_length(self):
        m = np.zeros((446, 446))
        result = extract_connectivity_features(m)
        assert len(result) == 446 * 445 // 2  # 99,235


class TestExtractAllFeatures:
    def test_returns_dict(self, random_symmetric_matrix):
        result = extract_all_features(random_symmetric_matrix)
        assert isinstance(result, dict)

    def test_contains_isd_keys(self, random_symmetric_matrix):
        result = extract_all_features(random_symmetric_matrix)
        for key in ['isd', 'efficiency', 'clustering']:
            assert key in result

    def test_contains_graph_keys(self, random_symmetric_matrix):
        result = extract_all_features(random_symmetric_matrix)
        for key in ['mean_degree', 'std_degree', 'mean_strength',
                     'std_strength', 'density']:
            assert key in result

    def test_contains_statistical_keys(self, random_symmetric_matrix):
        result = extract_all_features(random_symmetric_matrix)
        for key in ['mean_conn', 'std_conn', 'skew_conn', 'kurtosis_conn',
                     'q25_conn', 'median_conn', 'q75_conn',
                     'max_conn', 'min_conn']:
            assert key in result

    def test_contains_connectivity_vector(self, random_symmetric_matrix):
        result = extract_all_features(random_symmetric_matrix)
        assert 'connectivity' in result
        n = random_symmetric_matrix.shape[0]
        assert len(result['connectivity']) == n * (n - 1) // 2

    def test_handles_zero_matrix(self, zero_matrix):
        result = extract_all_features(zero_matrix)
        assert result['mean_conn'] == 0.0
        assert result['density'] == 0.0
