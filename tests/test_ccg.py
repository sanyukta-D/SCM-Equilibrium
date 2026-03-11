"""
Tests for the Consumer Choice Game (CCG) module.

Tests cover:
  1. Basic payoff: U_expressed == U_true ⇒ payoff matches standard equilibrium
  2. FeigningU.m soap-market economy: CCG sweep over (alpha, beta)
  3. Payoff monotonicity: expressing higher utility for a good increases your
     allocation (and payoff changes predictably)
  4. Numerical gradient consistency: ∂payoff/∂U via finite differences
  5. Zone mapping: zone boundaries detected correctly
  6. Detailed payoff: wage/price/allocation decomposition
"""

import pytest
import numpy as np
from scm import solve_robust
from scm.ccg import (
    ccg_payoff, ccg_payoff_detailed, ccg_sweep,
    ccg_gradient, ccg_zone_map, extract_forest, zone_label,
)


# ──────────────────────────────────────────────────────────────────────
# Soap-market economy from FeigningU.m
# ──────────────────────────────────────────────────────────────────────
T_SOAP = np.array([[0.2501, 0.0],
                    [0.25,   1.0]])
Y_SOAP = np.array([2.0, 4.0])
P_SOAP = np.array([2.0, 3.0])
U_TRUE_SOAP = np.array([[1.0, 1.0],
                         [1.0, 1.0]])


# ──────────────────────────────────────────────────────────────────────
# Standard 2×2 economy (from paper §G.2.2 example 1)
# ──────────────────────────────────────────────────────────────────────
T_STD = np.array([[1.0, 0.0],
                   [1.0, 1.0]])
U_STD = np.array([[1.0, 0.8],
                   [0.8, 1.0]])
Y_STD = np.array([2.0, 4.0])
P_STD = np.array([1.0, 1.0])


class TestCCGPayoffBasic:
    """When U_expressed == U_true, CCG payoff == standard equilibrium payoff."""

    def test_identity_payoff(self):
        """U_expressed = U_true ⇒ CCG payoff matches solve_robust result."""
        payoffs, result = ccg_payoff(T_STD, U_STD, U_STD, Y_STD, P_STD)
        result_ref = solve_robust(T_STD, U_STD, Y_STD, P_STD)

        # Allocations should be the same
        X_ccg = result['X']
        X_ref = result_ref['X']
        np.testing.assert_allclose(X_ccg, X_ref, atol=1e-4)

        # Payoffs = U_true · X (row-wise dot product)
        expected = np.array([U_STD[i, :] @ X_ref[i, :] for i in range(2)])
        np.testing.assert_allclose(payoffs, expected, atol=1e-4)

    def test_payoff_nonnegative(self):
        """Payoffs should be non-negative for a well-defined economy."""
        payoffs, _ = ccg_payoff(T_STD, U_STD, U_STD, Y_STD, P_STD)
        assert np.all(payoffs >= -1e-6)

    def test_soap_identity_payoff(self):
        """Soap market: U_expressed = U_true ⇒ consistent payoff."""
        payoffs, result = ccg_payoff(T_SOAP, U_TRUE_SOAP, U_TRUE_SOAP,
                                      Y_SOAP, P_SOAP)
        X = result['X']
        expected = np.array([U_TRUE_SOAP[i, :] @ X[i, :] for i in range(2)])
        np.testing.assert_allclose(payoffs, expected, atol=1e-4)


class TestCCGPayoffDetailed:
    """Test the detailed payoff function with decomposition."""

    def test_detailed_matches_basic(self):
        """Detailed function returns same total payoff as basic."""
        pay1, _ = ccg_payoff(T_STD, U_STD, U_STD, Y_STD, P_STD)
        pay2, payoff_mat, wages, prices, quantities, X, zone = \
            ccg_payoff_detailed(T_STD, U_STD, U_STD, Y_STD, P_STD)
        np.testing.assert_allclose(pay1, pay2, atol=1e-6)

    def test_payoff_mat_sums_to_payoffs(self):
        """payoff_mat.sum(axis=1) == payoffs."""
        payoffs, payoff_mat, _, _, _, _, _ = \
            ccg_payoff_detailed(T_STD, U_STD, U_STD, Y_STD, P_STD)
        np.testing.assert_allclose(payoff_mat.sum(axis=1), payoffs, atol=1e-6)

    def test_zone_has_required_keys(self):
        """Zone dict should have I, J, status keys."""
        _, _, _, _, _, _, zone = \
            ccg_payoff_detailed(T_STD, U_STD, U_STD, Y_STD, P_STD)
        assert 'I' in zone
        assert 'J' in zone
        assert 'status' in zone


class TestCCGSweep:
    """
    Test CCG sweep — replicates the FeigningU.m structure.
    For the soap market, sweep over beta with fixed alpha.
    """

    def _make_U_func(self, alpha=1.0):
        """Create U_expressed = [[1, alpha], [beta, 1]]."""
        def U_func(params):
            beta = params['beta']
            return np.array([[1.0, alpha], [beta, 1.0]])
        return U_func

    def test_sweep_runs(self):
        """Sweep completes without error."""
        U_func = self._make_U_func(alpha=1.0)
        grid = [{'beta': b} for b in np.arange(0.3, 1.6, 0.3)]
        results = ccg_sweep(T_SOAP, U_TRUE_SOAP, Y_SOAP, P_SOAP,
                            U_func, grid)
        assert len(results) == len(grid)
        for r in results:
            assert 'payoffs' in r
            assert 'prices' in r
            assert r['payoffs'].shape == (2,)

    def test_sweep_identity_point(self):
        """When beta=1, alpha=1: U_expressed = U_true = [[1,1],[1,1]]."""
        U_func = self._make_U_func(alpha=1.0)
        grid = [{'beta': 1.0}]
        results = ccg_sweep(T_SOAP, U_TRUE_SOAP, Y_SOAP, P_SOAP,
                            U_func, grid)
        # At identity, payoffs should match direct equilibrium
        pay_direct, _ = ccg_payoff(T_SOAP, U_TRUE_SOAP, U_TRUE_SOAP,
                                    Y_SOAP, P_SOAP)
        np.testing.assert_allclose(results[0]['payoffs'], pay_direct, atol=1e-4)

    def test_sweep_varying_alpha(self):
        """Sweep over multiple alpha values, checking payoff shapes."""
        for alpha in [0.5, 0.75, 1.0, 1.5]:
            U_func = self._make_U_func(alpha=alpha)
            grid = [{'beta': b} for b in [0.3, 0.7, 1.0, 1.3]]
            results = ccg_sweep(T_SOAP, U_TRUE_SOAP, Y_SOAP, P_SOAP,
                                U_func, grid)
            assert len(results) == 4
            for r in results:
                assert r['payoffs'].shape == (2,)
                assert np.all(np.isfinite(r['payoffs']))


class TestCCGGradient:
    """Test numerical gradient (Jacobian) computation."""

    def test_gradient_shape(self):
        """Full Jacobian has shape (m, m, n)."""
        J = ccg_gradient(T_STD, U_STD, U_STD, Y_STD, P_STD)
        m, n = T_STD.shape
        assert J.shape == (m, m, n)

    def test_gradient_player_shape(self):
        """Single-player gradient has shape (m, n)."""
        grad = ccg_gradient(T_STD, U_STD, U_STD, Y_STD, P_STD, player=0)
        m, n = T_STD.shape
        assert grad.shape == (m, n)

    def test_gradient_finite(self):
        """Gradient entries should be finite."""
        J = ccg_gradient(T_STD, U_STD, U_STD, Y_STD, P_STD)
        assert np.all(np.isfinite(J))

    def test_gradient_consistency(self):
        """
        Check gradient by manual finite difference at one point.
        Perturb U[0,1] and compare payoff change to Jacobian entry.
        """
        eps = 1e-5
        U_plus = U_STD.copy()
        U_plus[0, 1] += eps
        U_minus = U_STD.copy()
        U_minus[0, 1] -= eps

        pay_plus, _ = ccg_payoff(T_STD, U_STD, U_plus, Y_STD, P_STD)
        pay_minus, _ = ccg_payoff(T_STD, U_STD, U_minus, Y_STD, P_STD)
        fd = (pay_plus - pay_minus) / (2 * eps)

        J = ccg_gradient(T_STD, U_STD, U_STD, Y_STD, P_STD, eps=eps)
        # J[:, 0, 1] should match fd
        np.testing.assert_allclose(J[:, 0, 1], fd, atol=1e-3)

    def test_gradient_soap_market(self):
        """Gradient computation on soap market economy."""
        U_expr = np.array([[1.0, 1.0], [0.5, 1.0]])
        J = ccg_gradient(T_SOAP, U_TRUE_SOAP, U_expr, Y_SOAP, P_SOAP)
        assert J.shape == (2, 2, 2)
        assert np.all(np.isfinite(J))


class TestCCGZoneMap:
    """Test zone mapping across parameter grids."""

    def test_zone_map_basic(self):
        """Zone map runs and returns correct shapes."""
        def U_func(params):
            return np.array([[1.0, params['alpha']],
                             [params['beta'], 1.0]])

        p1_grid = np.array([0.5, 1.0, 1.5])
        p2_grid = np.array([0.5, 1.0, 1.5])

        zone_grid, payoff_grid, wage_grid, forest_grid = ccg_zone_map(
            T_SOAP, U_TRUE_SOAP, Y_SOAP, P_SOAP,
            U_func, p1_grid, p2_grid,
            param1_name='alpha', param2_name='beta'
        )

        assert zone_grid.shape == (3, 3)
        assert payoff_grid.shape == (3, 3, 2)
        assert wage_grid.shape == (3, 3, 2)
        assert forest_grid.shape == (3, 3)

    def test_zone_labels_are_strings(self):
        """Zone labels should be strings like 'I={...}_J={...}'."""
        def U_func(params):
            return np.array([[1.0, params['alpha']],
                             [params['beta'], 1.0]])

        zone_grid, _, _, _ = ccg_zone_map(
            T_SOAP, U_TRUE_SOAP, Y_SOAP, P_SOAP,
            U_func, np.array([0.8, 1.2]), np.array([0.8, 1.2]),
            param1_name='alpha', param2_name='beta'
        )

        for i in range(2):
            for j in range(2):
                label = zone_grid[i, j]
                assert isinstance(label, str)
                assert 'I=' in label
                assert 'J=' in label


class TestCCGStrategicBehavior:
    """
    Test economic intuition: strategic preference misrepresentation
    should affect payoffs in predictable ways.
    """

    def test_overstating_preference_changes_allocation(self):
        """
        If class 0 overstates preference for good 1 (higher U[0,1]),
        its allocation of good 1 should increase.
        """
        U_base = U_STD.copy()
        U_high = U_STD.copy()
        U_high[0, 1] = 1.5  # overstate preference for good 1

        _, result_base = ccg_payoff(T_STD, U_STD, U_base, Y_STD, P_STD)
        _, result_high = ccg_payoff(T_STD, U_STD, U_high, Y_STD, P_STD)

        X_base = result_base['X']
        X_high = result_high['X']

        # Class 0 should get more of good 1 (or at least not less)
        assert X_high[0, 1] >= X_base[0, 1] - 1e-4

    def test_payoff_at_true_vs_distorted(self):
        """
        Playing U_true is not necessarily optimal (friction can help).
        Just verify both produce valid finite payoffs.
        """
        # True preferences
        pay_true, _ = ccg_payoff(T_STD, U_STD, U_STD, Y_STD, P_STD)

        # Distorted: class 1 overstates preference for good 0
        U_distort = U_STD.copy()
        U_distort[1, 0] = 1.2
        pay_distort, _ = ccg_payoff(T_STD, U_STD, U_distort, Y_STD, P_STD)

        assert np.all(np.isfinite(pay_true))
        assert np.all(np.isfinite(pay_distort))
        # Payoffs changed (not necessarily improved for everyone)
        assert not np.allclose(pay_true, pay_distort, atol=1e-6)


class TestCCGEdgeCases:
    """Edge cases and robustness."""

    def test_zero_utility_column(self):
        """If a good has zero utility for everyone, payoff is 0 for that good."""
        U_zero_col = np.array([[1.0, 0.0], [0.8, 0.0]])
        payoffs, result = ccg_payoff(T_STD, U_zero_col, U_STD, Y_STD, P_STD)
        # True utility for good 1 is 0, so contribution to payoff from good 1 is 0
        X = result['X']
        contribution_g1 = U_zero_col[:, 1] * X[:, 1]
        np.testing.assert_allclose(contribution_g1, 0.0, atol=1e-6)

    def test_same_expressed_as_true_is_fixed_point(self):
        """
        If playing U_expressed = U_true, the payoff is the 'honest' payoff.
        Calling it twice should give the same result.
        """
        p1, _ = ccg_payoff(T_STD, U_STD, U_STD, Y_STD, P_STD)
        p2, _ = ccg_payoff(T_STD, U_STD, U_STD, Y_STD, P_STD)
        np.testing.assert_allclose(p1, p2, atol=1e-8)


class TestForestExtraction:
    """Test Fisher forest extraction and zone labelling."""

    def test_extract_forest_shape(self):
        """Forest is a tuple of tuples; bpb_ordering is a list of lists."""
        result = solve_robust(T_STD, U_STD, Y_STD, P_STD)
        I, J = result['I'], result['J']
        X, p = result['X'], result['p']
        forest, bpb = extract_forest(U_STD, p, X, I, J)
        assert isinstance(forest, tuple)
        assert len(forest) == len(I)
        assert isinstance(bpb, list)
        assert len(bpb) == len(I)

    def test_forest_goods_are_active(self):
        """All goods in forest entries should be from active set J."""
        result = solve_robust(T_STD, U_STD, Y_STD, P_STD)
        I, J = result['I'], result['J']
        X, p = result['X'], result['p']
        forest, _ = extract_forest(U_STD, p, X, I, J)
        J_set = set(int(j) for j in J)
        for goods in forest:
            for g in goods:
                assert g in J_set

    def test_forest_matches_positive_allocation(self):
        """Forest entries should correspond to positive allocations."""
        result = solve_robust(T_STD, U_STD, Y_STD, P_STD)
        I, J = result['I'], result['J']
        X, p = result['X'], result['p']
        forest, _ = extract_forest(U_STD, p, X, I, J, tol=1e-6)
        for k, i in enumerate(I):
            for g in forest[k]:
                assert X[i, g] > 1e-6

    def test_zone_label_format(self):
        """Zone label has expected format."""
        I = np.array([0, 1])
        J = np.array([0, 1])
        label = zone_label(I, J)
        assert 'I={0,1}' in label
        assert 'J={0,1}' in label

    def test_zone_label_with_forest(self):
        """Zone label includes forest when provided."""
        I = np.array([0, 1])
        J = np.array([0, 1])
        forest = ((0, 1), (1,))
        label = zone_label(I, J, forest=forest)
        assert 'F=' in label
        assert '→' in label  # Unicode arrow

    def test_detailed_returns_forest(self):
        """ccg_payoff_detailed zone dict includes F and bpb_ordering."""
        _, _, _, _, _, _, zone = \
            ccg_payoff_detailed(T_STD, U_STD, U_STD, Y_STD, P_STD)
        assert 'F' in zone
        assert 'bpb_ordering' in zone
        assert isinstance(zone['F'], tuple)

    def test_sweep_returns_forest(self):
        """ccg_sweep results include forest and zone_label."""
        def U_func(params):
            return np.array([[1.0, params['alpha']], [0.8, 1.0]])
        grid = [{'alpha': 0.8}, {'alpha': 1.2}]
        results = ccg_sweep(T_STD, U_STD, Y_STD, P_STD, U_func, grid)
        for r in results:
            assert 'forest' in r
            assert 'zone_label' in r
            assert isinstance(r['forest'], tuple)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
