# How Damped Tatonnement Works: The Math

## The Problem

Standard tatonnement iterates the SCM map:

    p_{t+1} = G(p_t)

where G(p) = SCM_round(p) computes one full production → wages → Fisher market → price update cycle.

At equilibrium, p* is a **fixed point**: G(p*) = p*.

When standard tatonnement **cycles** (as in our 3×3 linear example), the map overshoots the fixed point on each step. The iterates bounce between two (or more) price vectors, never settling down.

## The Fix: Damping

Damped tatonnement replaces the full step with a **convex combination**:

    p_{t+1} = (1 - α) · p_t + α · G(p_t)

where α ∈ (0, 1) is the **damping parameter** (step size).

Equivalently, writing F(p) = G(p) - p as the fixed-point residual:

    p_{t+1} = p_t + α · F(p_t)

This is **gradient-descent-like**: we move in the direction of the residual, but only by a fraction α.

## Why It Works: Contraction Mapping Perspective

Consider the linearization of G around the fixed point p*:

    G(p) ≈ G(p*) + J · (p - p*)  =  p* + J · (p - p*)

where J = ∂G/∂p |_{p*} is the Jacobian of the SCM map at equilibrium.

**Standard tatonnement converges if and only if all eigenvalues of J satisfy |λ_i| < 1** (the spectral radius ρ(J) < 1).

When tatonnement cycles, typically ρ(J) ≥ 1. In a 2-cycle, there exists an eigenvalue λ with |λ| > 1 (or λ ≈ -1 for a period-2 oscillation).

**Damped tatonnement** replaces J with:

    J_damped = (1 - α) · I + α · J

The eigenvalues of J_damped are:

    μ_i = (1 - α) + α · λ_i = 1 - α(1 - λ_i)

For convergence, we need |μ_i| < 1 for all eigenvalues λ_i.

## The Key Insight

If the original eigenvalues λ_i are real and the cycling is caused by an eigenvalue λ < -1 (overshoot), then:

    μ = 1 - α(1 - λ)

For λ = -1 (exact 2-cycle): μ = 1 - 2α

This is stable (|μ| < 1) for **any** α ∈ (0, 1).

More generally, for λ = -(1 + ε) where ε > 0:

    μ = 1 - α(2 + ε) = 1 - 2α - αε

We need |μ| < 1, i.e., -1 < 1 - α(2 + ε) < 1, which gives:

    α < 2 / (2 + ε)

So as long as the eigenvalue isn't too negative, a small enough α will stabilize the iteration.

## Optimal Damping

The convergence rate is determined by the largest |μ_i|. For the damped iteration:

    Rate = max_i |1 - α(1 - λ_i)|

The optimal α minimizes this over all eigenvalues. In our 3×3 cycling economy:

- Standard tatonnement (α=1) cycles → divergent eigenvalue
- α=0.5 converges but is slow (still close to the stability boundary)
- α=0.3 converges in ~190 iterations — good balance
- α=0.1 converges in ~560 iterations — too cautious

## Concrete Example

Our linear 3×3 cycling economy:

    T = [[4.18, 4.37, 0.53],    U = [[3.30, 5.24, 7.06],
         [2.80, 2.38, 1.50],         [3.70, 9.72, 9.63],
         [1.04, 2.02, 4.74]]         [2.59, 5.02, 3.08]]

    Y = [7.85, 5.37, 11.10]

**Standard tatonnement (α=1):**
- p_0 = [1, 1, 1]
- p_1 = [0.570, 1.105, 0.940]
- p_2 = [0.519, 1.006, 0.997]  ← state A
- p_3 = [0.559, 1.083, 0.953]  ← state B
- p_4 = [0.519, 1.006, 0.997]  ← state A (repeats!)
- The map has an eigenvalue near -1, causing a 2-cycle with amplitude ~0.077.

**Damped tatonnement (α=0.3):**
- p_0 = [1, 1, 1], fp_err = 0.43
- p_5 = [0.609, 1.023, 0.987], fp_err = 0.081
- p_10 = [0.542, 1.023, 0.987], fp_err = 0.014
- p_50 = [0.528, 1.023, 0.987], fp_err = 3.0e-6
- p_190 = [0.528, 1.023, 0.987], fp_err = 4.9e-8 ✓

**Equilibrium found:** p* ≈ [0.5280, 1.0228, 0.9870]
- Active goods: J = {1, 2}, Active labor: I = {1, 2}
- Class 0 is surplus labor (W_0 = 0)
- Class 1 buys only good 2 (BPB_max = 9.75)
- Class 2 buys only good 1 (BPB_max = 4.91)
- Money conservation: p·q = ΣW = 2.954

## Connection to Gradient Descent

Cheung, Cole & Devanur (2020) showed that for Fisher markets with CES utilities, tatonnement is equivalent to gradient descent on the **Eisenberg-Gale convex program**. Specifically:

    p_{t+1} = p_t - η · ∇Φ(p_t)

where Φ is a convex potential function and η is the step size.

Our damping parameter α plays the role of the learning rate η. When α is too large, gradient descent overshoots (oscillates); when α is small enough, it converges. The optimal rate balances speed against stability.

The full SCM map is more complex than a pure Fisher market (it includes production and wage computation), but the same principle applies: the SCM map can be viewed as an approximate gradient step, and damping controls the step size.

## When Damped Tatonnement Fails

Damped tatonnement does NOT solve all cases. It fails when:

1. **The fixed point doesn't exist** — e.g., prices must diverge because no equilibrium exists at finite prices.

2. **The eigenvalues are complex with large modulus** — damping reduces the real part but doesn't help with rotational instability. Need α → 0, making convergence infinitely slow.

3. **The map is discontinuous** — zone transitions (active set changes) create jumps in G(p). Near a zone boundary, the map can oscillate between zones, and damping may not stabilize across the discontinuity.

Our PLC toggling example exhibits case 3: the active set alternates between {0} and {0,1}, and the prices grow each cycle. No finite equilibrium may exist, or it may exist exactly on the zone boundary where damping can't reach it.

## Summary

| Method | Mechanism | When it works | When it fails |
|--------|-----------|---------------|---------------|
| Standard tatonnement (α=1) | Full step | ρ(J) < 1 | ρ(J) ≥ 1 (cycling/diverging) |
| Damped tatonnement (α<1) | Reduced step | Real eigenvalues with |λ|<1/α+1 | Complex eigenvalues, no fixed point |
| Anderson acceleration | Multi-step extrapolation | Smooth maps near convergence | Discontinuous maps |
| Newton/Broyden | Quadratic convergence | Near a solution | Far from solution, discontinuities |
| Homotopy | Path following | Always (theoretically) | Zone boundary discontinuities |
