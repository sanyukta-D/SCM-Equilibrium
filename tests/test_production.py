"""Tests for the production LP and wage computation."""

import numpy as np
import pytest
from scm import solve_production, wages_from_prices


# ── Test 1: Production LP (paper section G.2.2, example 1, iteration 1) ──

def test_production_G22_quantities():
    """Production LP output matches paper section G.2.2 example 1."""
    T = np.array([
        [1.00, 0.40, 0.50],
        [0.50, 1.50, 0.25],
        [0.20, 0.35, 0.60],
    ])
    Y = np.array([10.0, 10.0, 10.0])
    p = np.array([1.0, 1.2, 1.3])
    q_expected = np.array([1.5, 3.85, 13.92])

    q, w, wages, J, I, revenue = solve_production(T, Y, p)

    np.testing.assert_allclose(q, q_expected, rtol=0.02, atol=0.05)
    assert set(J) == {0, 1, 2}, f"Expected all goods active, got J={J}"
    assert set(I) == {0, 1, 2}, f"Expected all labour active, got I={I}"
    assert abs(revenue - wages.sum()) < 0.1


# ── Test 8: Wages LP dual vs matrix-inverse cross-check ──

def test_wages_lp_vs_matrix_inverse():
    """LP dual wages match matrix-inverse wages."""
    T = np.array([
        [1.00, 0.40, 0.50],
        [0.50, 1.50, 0.25],
        [0.20, 0.35, 0.60],
    ])
    Y = np.array([10.0, 10.0, 10.0])
    p = np.array([1.0, 1.2, 1.3])

    q, w_lp, wages_lp, J, I, revenue = solve_production(T, Y, p)

    T_sub = T[np.ix_(I, J)]
    wages_inv = wages_from_prices(T_sub, p[J], Y[I])

    np.testing.assert_allclose(wages_lp[I], wages_inv, rtol=0.01, atol=0.05)
    assert abs(wages_lp.sum() - revenue) < 0.05
    assert abs(wages_inv.sum() - revenue) < 0.05
