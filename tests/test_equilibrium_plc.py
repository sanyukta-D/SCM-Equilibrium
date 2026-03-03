"""Tests for PLC tatonnement equilibrium computation."""

import numpy as np
import pytest
from scm import (
    compute_equilibrium, compute_equilibrium_plc,
    check_plc_equilibrium,
)


# ── Test 16: PLC equilibrium convergence + exhaustive condition check ──

def test_plc_equilibrium_2x2():
    """PLC 2x2 economy converges and satisfies all 11 conditions."""
    T  = np.array([[1.0, 0.0], [1.0, 1.0]])
    U1 = np.array([[1.0, 0.8], [0.8, 1.0]])
    U2 = U1 * 0.5
    L1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    Y  = np.array([2.0, 4.0])

    result = compute_equilibrium_plc(T, U1, U2, L1, Y, np.array([1.0, 1.0]),
                                      max_iter=300, tol=1e-7)

    assert result['status'] in ('converged', 'cycling')

    # All PLC equilibrium conditions
    checks, all_pass = check_plc_equilibrium(
        result, T, U1, U2, L1, Y, tol=1e-4, verbose=False
    )
    assert all_pass, f"Failed conditions: {[k for k, (p, _) in checks.items() if not p]}"

    # Production quantities should be close to linear version
    result_lin = compute_equilibrium(T, U1, Y, np.array([1.0, 1.0]),
                                      max_iter=200, tol=1e-7)
    np.testing.assert_allclose(result['q'], result_lin['q'], rtol=0.05, atol=0.1)
