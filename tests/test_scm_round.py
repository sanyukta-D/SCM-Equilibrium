"""Tests for one full SCM round (linear and PLC)."""

import numpy as np
import pytest
from scm import scm_round, scm_round_plc


# ── Test 5: Full round, paper section G.2.2 example 1, iteration 1 ──

def test_scm_round_G22():
    """One SCM round matches paper section G.2.2 example 1 (iteration 1)."""
    T = np.array([
        [1.00, 0.40, 0.50],
        [0.50, 1.50, 0.25],
        [0.20, 0.35, 0.60],
    ])
    U = np.array([
        [0.85, 0.50, 0.40],
        [0.40, 0.90, 0.45],
        [0.55, 0.40, 0.80],
    ])
    Y = np.array([10.0, 10.0, 10.0])
    p = np.array([1.0, 1.2, 1.3])

    p_new, q, W, X, I, J = scm_round(T, U, Y, p)

    # Production quantities
    np.testing.assert_allclose(q, [1.5, 3.85, 13.92], rtol=0.02, atol=0.05)

    # Wages
    np.testing.assert_allclose(W, [5.384, 2.831, 16.0], rtol=0.03, atol=0.05)

    # Money conservation
    assert abs(p_new @ q - W.sum()) < 0.05


# ── Test 15: PLC SCM round (one round, structural checks) ──

def test_plc_scm_round():
    """PLC round produces same q and W as linear (production LP is the same)."""
    T = np.array([
        [1.00, 0.40, 0.50],
        [0.50, 1.50, 0.25],
        [0.20, 0.35, 0.60],
    ])
    U1 = np.array([
        [0.85, 0.50, 0.40],
        [0.40, 0.90, 0.45],
        [0.55, 0.40, 0.80],
    ])
    U2 = U1 * 0.6
    L1 = np.array([
        [3.0, 2.0, 5.0],
        [1.5, 4.0, 3.0],
        [2.0, 1.5, 4.0],
    ])
    Y = np.array([10.0, 10.0, 10.0])
    p = np.array([1.0, 1.2, 1.3])

    p_new, q, W, X1, X2, I, J = scm_round_plc(T, U1, U2, L1, Y, p)

    # Money conservation
    assert abs(p_new @ q - W.sum()) < 0.05

    # Production and wages match linear version (production LP is unchanged)
    p_lin, q_lin, W_lin, _, _, _ = scm_round(T, U1, Y, p)
    np.testing.assert_allclose(q, q_lin, rtol=0.01, atol=0.05)
    np.testing.assert_allclose(W, W_lin, rtol=0.01, atol=0.05)

    # Segment-1 capacity not exceeded
    assert np.all(X1[np.ix_(I, J)] <= L1[np.ix_(I, J)] + 1e-5)
