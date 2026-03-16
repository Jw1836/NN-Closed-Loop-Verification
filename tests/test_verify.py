"""Tests for LyapunovProblem.verify() orchestrator."""

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from problems.bilinear_oscillator import BilinearProblem


@pytest.fixture
def problem():
    """Small bilinear problem — fast to enumerate."""
    return BilinearProblem(hidden_size=5)


class TestVerify:
    def test_returns_expected_keys(self, problem):
        results = problem.verify()
        assert set(results.keys()) >= {"origin", "positive", "decrease", "cells"}

    def test_cells_are_convex_hulls(self, problem):
        results = problem.verify()
        cells = results["cells"]
        assert isinstance(cells, list)
        assert len(cells) > 0
        assert all(isinstance(c, ConvexHull) for c in cells)

    def test_cells_tile_region(self, problem):
        """Cell areas should sum to the bounding-box area."""
        results = problem.verify()
        region_area = float(
            (problem.region[0, 1] - problem.region[0, 0])
            * (problem.region[1, 1] - problem.region[1, 0])
        )
        cell_area = sum(c.volume for c in results["cells"])
        assert cell_area == pytest.approx(region_area, abs=1e-6)

    def test_check_results_are_ndarray_or_none(self, problem):
        results = problem.verify()
        for key in ("origin", "positive", "decrease"):
            val = results[key]
            assert val is None or isinstance(val, np.ndarray)

    def test_early_exit_skips_later_checks(self):
        """With early_exit=True and a broken origin, positive/decrease stay None."""
        import torch

        prob = BilinearProblem(hidden_size=5)
        # Sabotage the output bias so V(0) != 0
        with torch.no_grad():
            list(prob.nn_lyapunov.network.children())[2].bias.fill_(5.0)
        prob.early_exit = True
        results = prob.verify()
        assert results["origin"] is not None
        # Later checks should not have run
        assert results["positive"] is None
        assert results["decrease"] is None
