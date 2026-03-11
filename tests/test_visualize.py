"""
Tests for the visualization module (scm.visualize).

Tests verify that each plot function:
  - Runs without error
  - Returns (fig, ax) or (fig, (ax1, ax2))
  - Saves to file when output_file is given
  - Handles edge cases (single zone, single point)
"""

import pytest
import numpy as np
import os
import tempfile

# Skip all tests if matplotlib not available
plt = pytest.importorskip("matplotlib.pyplot")

from scm.visualize import (
    plot_zone_map,
    plot_zone_map_with_payoff,
    plot_payoff_trajectory,
    plot_wage_trajectory,
    plot_price_trajectory,
    plot_allocation_pattern,
    plot_forest_diagram,
    plot_gradient_field,
)


@pytest.fixture
def tmpdir():
    """Temporary directory for output files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def zone_data():
    """Sample 3×4 zone grid data."""
    zone_grid = np.array([
        ['Z1', 'Z1', 'Z2', 'Z2'],
        ['Z1', 'Z2', 'Z2', 'Z3'],
        ['Z2', 'Z2', 'Z3', 'Z3'],
    ], dtype=object)
    param1 = np.array([0.5, 1.0, 1.5])
    param2 = np.array([0.2, 0.6, 1.0, 1.4])
    payoff_grid = np.random.rand(3, 4, 2)
    return zone_grid, param1, param2, payoff_grid


class TestPlotZoneMap:
    """Test plot_zone_map."""

    def test_basic_run(self, zone_data):
        fig, ax = plot_zone_map(
            zone_data[0], zone_data[1], zone_data[2])
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_save_to_file(self, zone_data, tmpdir):
        outfile = os.path.join(tmpdir, 'zone_map.png')
        fig, ax = plot_zone_map(
            zone_data[0], zone_data[1], zone_data[2],
            output_file=outfile)
        assert os.path.exists(outfile)
        plt.close(fig)

    def test_single_zone(self):
        """Grid with one zone should not error."""
        zone_grid = np.full((3, 3), 'Z1', dtype=object)
        fig, ax = plot_zone_map(
            zone_grid, np.array([1, 2, 3]), np.array([1, 2, 3]))
        assert fig is not None
        plt.close(fig)


class TestPlotZoneMapWithPayoff:
    """Test plot_zone_map_with_payoff."""

    def test_basic_run(self, zone_data):
        fig, axes = plot_zone_map_with_payoff(
            zone_data[0], zone_data[3], zone_data[1], zone_data[2],
            player=0)
        assert fig is not None
        assert len(axes) == 2
        plt.close(fig)

    def test_save_to_file(self, zone_data, tmpdir):
        outfile = os.path.join(tmpdir, 'zone_payoff.png')
        fig, _ = plot_zone_map_with_payoff(
            zone_data[0], zone_data[3], zone_data[1], zone_data[2],
            player=1, output_file=outfile)
        assert os.path.exists(outfile)
        plt.close(fig)


class TestPlotPayoffTrajectory:
    """Test plot_payoff_trajectory."""

    def test_basic_run(self):
        params = np.linspace(0, 2, 10)
        payoffs = np.random.rand(10, 2)
        fig, ax = plot_payoff_trajectory(params, payoffs)
        assert fig is not None
        plt.close(fig)

    def test_with_zone_labels(self):
        params = np.linspace(0, 2, 6)
        payoffs = np.random.rand(6, 2)
        zones = ['Z1', 'Z1', 'Z1', 'Z2', 'Z2', 'Z2']
        fig, ax = plot_payoff_trajectory(params, payoffs, zone_labels=zones)
        assert fig is not None
        plt.close(fig)

    def test_save_to_file(self, tmpdir):
        outfile = os.path.join(tmpdir, 'payoff.png')
        fig, ax = plot_payoff_trajectory(
            np.array([1, 2, 3]), np.array([[1, 2], [3, 4], [5, 6]]),
            output_file=outfile)
        assert os.path.exists(outfile)
        plt.close(fig)


class TestPlotWageTrajectory:
    """Test plot_wage_trajectory."""

    def test_basic_run(self):
        fig, ax = plot_wage_trajectory(
            np.linspace(0, 1, 5), np.random.rand(5, 3))
        assert fig is not None
        plt.close(fig)


class TestPlotPriceTrajectory:
    """Test plot_price_trajectory."""

    def test_basic_run(self):
        fig, ax = plot_price_trajectory(
            np.linspace(0, 1, 5), np.random.rand(5, 2))
        assert fig is not None
        plt.close(fig)


class TestPlotAllocationPattern:
    """Test plot_allocation_pattern."""

    def test_basic_run(self):
        X = np.array([[2.0, 1.5], [0.5, 3.0]])
        fig, ax = plot_allocation_pattern(X)
        assert fig is not None
        plt.close(fig)

    def test_with_labels(self):
        X = np.array([[2.0, 1.5], [0.5, 3.0]])
        fig, ax = plot_allocation_pattern(
            X, class_labels=['A', 'B'], good_labels=['g0', 'g1'])
        assert fig is not None
        plt.close(fig)

    def test_save(self, tmpdir):
        outfile = os.path.join(tmpdir, 'alloc.png')
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        fig, ax = plot_allocation_pattern(X, output_file=outfile)
        assert os.path.exists(outfile)
        plt.close(fig)


class TestPlotForestDiagram:
    """Test plot_forest_diagram."""

    def test_basic_run(self):
        X = np.array([[2.0, 1.5], [0.5, 3.0]])
        I = np.array([0, 1])
        J = np.array([0, 1])
        fig, ax = plot_forest_diagram(X, I, J)
        assert fig is not None
        plt.close(fig)

    def test_partial_active(self):
        """Only some classes/goods active."""
        X = np.array([[2.0, 0.0], [0.0, 3.0]])
        I = np.array([0])
        J = np.array([0])
        fig, ax = plot_forest_diagram(X, I, J)
        assert fig is not None
        plt.close(fig)


class TestPlotGradientField:
    """Test plot_gradient_field."""

    def test_basic_run(self):
        grad = np.random.randn(5, 5, 2)
        p1 = np.linspace(0, 1, 5)
        p2 = np.linspace(0, 1, 5)
        fig, ax = plot_gradient_field(grad, p1, p2)
        assert fig is not None
        plt.close(fig)

    def test_with_zone_background(self, zone_data):
        grad = np.random.randn(3, 4, 2)
        fig, ax = plot_gradient_field(
            grad, zone_data[1], zone_data[2], zone_grid=zone_data[0])
        assert fig is not None
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
