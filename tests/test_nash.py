"""
Tests for the Nash equilibrium finder (scm.nash).

Tests cover:
  1. best_response_direction: gradient shape, magnitude, direction sign
  2. best_response_search: line search finds improvement or stays put
  3. nash_iteration: convergence, output structure, payoff tracking
  4. find_nash_candidates: multi-start search, ranking, output format
"""

import pytest
import numpy as np
from scm.nash import (
    best_response_direction,
    best_response_search,
    nash_iteration,
    find_nash_candidates,
)
from scm.ccg import ccg_payoff


# ──────────────────────────────────────────────────────────────────────
# Test economies
# ──────────────────────────────────────────────────────────────────────

T_STD = np.array([[1.0, 0.0],
                   [1.0, 1.0]])
U_STD = np.array([[1.0, 0.8],
                   [0.8, 1.0]])
Y_STD = np.array([2.0, 4.0])
P_STD = np.array([1.0, 1.0])

T_SOAP = np.array([[0.2501, 0.0],
                    [0.25,   1.0]])
U_SOAP = np.array([[1.0, 1.0],
                    [1.0, 1.0]])
Y_SOAP = np.array([2.0, 4.0])
P_SOAP = np.array([2.0, 3.0])


class TestBestResponseDirection:
    """Test best_response_direction function."""

    def test_output_shapes(self):
        """Direction has shape (n,), magnitude is scalar."""
        direction, magnitude = best_response_direction(
            T_STD, U_STD, U_STD, Y_STD, P_STD, player=0)
        assert direction.shape == (2,)
        assert isinstance(magnitude, (float, np.floating))

    def test_direction_is_unit_or_zero(self):
        """Direction is either unit vector or zero vector."""
        direction, magnitude = best_response_direction(
            T_STD, U_STD, U_STD, Y_STD, P_STD, player=0)
        norm = np.linalg.norm(direction)
        assert norm < 1e-8 or abs(norm - 1.0) < 1e-8

    def test_magnitude_nonnegative(self):
        """Gradient magnitude is non-negative."""
        for player in range(2):
            _, mag = best_response_direction(
                T_STD, U_STD, U_STD, Y_STD, P_STD, player=player)
            assert mag >= 0

    def test_both_players(self):
        """Can compute direction for both players."""
        d0, m0 = best_response_direction(
            T_STD, U_STD, U_STD, Y_STD, P_STD, player=0)
        d1, m1 = best_response_direction(
            T_STD, U_STD, U_STD, Y_STD, P_STD, player=1)
        assert d0.shape == d1.shape == (2,)


class TestBestResponseSearch:
    """Test best_response_search (line search)."""

    def test_output_format(self):
        """Returns (best_row, best_payoff, step_size)."""
        best_row, best_payoff, step = best_response_search(
            T_STD, U_STD, U_STD, Y_STD, P_STD, player=0,
            n_steps=5, max_step=0.3)
        assert best_row.shape == (2,)
        assert isinstance(best_payoff, (float, np.floating))
        assert step >= 0

    def test_payoff_nondecreasing(self):
        """Line search should not decrease payoff from current point."""
        payoffs_base, _ = ccg_payoff(T_STD, U_STD, U_STD, Y_STD, P_STD)
        _, best_payoff, _ = best_response_search(
            T_STD, U_STD, U_STD, Y_STD, P_STD, player=0,
            n_steps=5, max_step=0.3)
        # Should be at least as good as base (within tolerance for numerical noise)
        assert best_payoff >= payoffs_base[0] - 1e-4

    def test_soap_market_player1(self):
        """Line search on soap market for player 1."""
        U_expr = np.array([[1.0, 1.0], [0.8, 1.0]])
        best_row, best_payoff, step = best_response_search(
            T_SOAP, U_SOAP, U_expr, Y_SOAP, P_SOAP, player=1,
            n_steps=8, max_step=0.5)
        assert best_row.shape == (2,)
        assert np.isfinite(best_payoff)


class TestNashIteration:
    """Test nash_iteration function."""

    def test_output_structure(self):
        """Result dict has all expected keys."""
        result = nash_iteration(
            T_STD, U_STD, U_STD, Y_STD, P_STD,
            max_iter=5, lr=0.1)
        assert 'profiles' in result
        assert 'payoffs' in result
        assert 'magnitudes' in result
        assert 'converged' in result
        assert 'n_iter' in result

    def test_profiles_list(self):
        """Profiles list has len = n_iter + 1."""
        result = nash_iteration(
            T_STD, U_STD, U_STD, Y_STD, P_STD,
            max_iter=5, lr=0.1)
        assert len(result['profiles']) == result['n_iter'] + 1
        for prof in result['profiles']:
            assert prof.shape == (2, 2)

    def test_payoffs_array(self):
        """Payoffs array has shape (n_iter+1, m)."""
        result = nash_iteration(
            T_STD, U_STD, U_STD, Y_STD, P_STD,
            max_iter=5, lr=0.1)
        assert result['payoffs'].shape[1] == 2  # m=2 classes
        assert np.all(np.isfinite(result['payoffs']))

    def test_magnitudes_array(self):
        """Magnitudes has right shape."""
        result = nash_iteration(
            T_STD, U_STD, U_STD, Y_STD, P_STD,
            max_iter=5, lr=0.1)
        assert result['magnitudes'].shape[1] == 2  # m=2 players

    def test_n_iter_bounded(self):
        """Number of iterations doesn't exceed max_iter."""
        result = nash_iteration(
            T_STD, U_STD, U_STD, Y_STD, P_STD,
            max_iter=10, lr=0.1)
        assert result['n_iter'] <= 10

    def test_convergence_reduces_gradient(self):
        """If converged, final gradient magnitude should be small."""
        result = nash_iteration(
            T_STD, U_STD, U_STD, Y_STD, P_STD,
            max_iter=100, lr=0.05, tol=1e-3)
        if result['converged']:
            assert result['magnitudes'][-1].max() < 1e-3


class TestFindNashCandidates:
    """Test find_nash_candidates multi-start search."""

    def test_output_format(self):
        """Returns list of dicts with expected keys."""
        candidates = find_nash_candidates(
            T_STD, U_STD, Y_STD, P_STD,
            n_restarts=2, max_iter=5)
        assert isinstance(candidates, list)
        assert len(candidates) == 2
        for c in candidates:
            assert 'U_expressed' in c
            assert 'payoffs' in c
            assert 'convergence_gap' in c
            assert 'n_iter' in c
            assert 'converged' in c

    def test_sorted_by_gap(self):
        """Candidates are sorted by convergence gap (ascending)."""
        candidates = find_nash_candidates(
            T_STD, U_STD, Y_STD, P_STD,
            n_restarts=3, max_iter=5)
        gaps = [c['convergence_gap'] for c in candidates]
        assert gaps == sorted(gaps)

    def test_payoffs_finite(self):
        """All candidate payoffs are finite."""
        candidates = find_nash_candidates(
            T_SOAP, U_SOAP, Y_SOAP, P_SOAP,
            n_restarts=2, max_iter=5)
        for c in candidates:
            assert np.all(np.isfinite(c['payoffs']))

    def test_U_expressed_shape(self):
        """Final U_expressed has correct shape."""
        candidates = find_nash_candidates(
            T_STD, U_STD, Y_STD, P_STD,
            n_restarts=2, max_iter=5)
        for c in candidates:
            assert c['U_expressed'].shape == (2, 2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
