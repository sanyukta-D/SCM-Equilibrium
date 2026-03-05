# Task 2: Alternative Equilibrium Computation When Tatonnement Fails

## 1. Problem Statement

Our SCM tatonnement computes equilibrium by iterating the SCM map: p_{t+1} = SCM_round(p_t). This converges for many economies but can cycle or diverge. We need alternatives for when it fails.

**Our specific settings (from the paper):**
- m labor classes, n goods
- Linear production technology (technology matrix T, m×n)
- Linear or PLC (piecewise-linear-concave) utilities
- Each labor class = one consumer with wage income
- The SCM map: Production LP → Wages (matrix inverse) → Budget normalization → Fisher Market (EG) → Inactive-good BPB pricing → New prices

**AD equivalence (Theorem 4.2):** SM equilibrium ↔ Arrow-Debreu equilibrium in AD(T,Y,U) with n goods, m households, n firms with ray technologies, and profit shares proportional to labor contribution.

---

## 2. What the Literature Says About Our Settings

### 2.1 Complexity Classification

The computational complexity depends heavily on the utility type:

| Setting | Complexity | Key Reference |
|---------|-----------|---------------|
| Linear utilities, exchange economy | **P** (polynomial) | Jain (2007); Duan & Mehlhorn (2015) |
| Linear utilities, constant-returns production | **P** (for constant #goods) | Devanur & Kannan (2008) |
| SPLC utilities, exchange economy | **PPAD-complete** | Chen, Dai, Du & Teng (2009) |
| SPLC utilities + SPLC production | **PPAD** | Garg & Vazirani (2014) |
| General PLC utilities, exchange | **FIXP-hard** | Etessami & Yannakakis (2010) |
| PLC utilities + polyhedral production | **∃R-complete** (existence) | Garg, Kannan & Vazirani (2015) |

Our model has **linear or PLC utilities** with **linear production** (ray technologies). This places us in the PPAD to FIXP range — hard in general, but with practical algorithms available.

### 2.2 Why Standard AD Algorithms Don't Directly Apply

Section C.4.2 of the paper explains why the LCP formulation of Garg & Vazirani (2014) cannot be extended to SCM:

- In the LCP formulation, each firm's production is independent: wages from firm j depend only on good j's price.
- In SCM, wages are **linear combinations of all active goods' prices** (w = T_I^{-1} · p_J, where I,J are the active sets). This creates cross-dependencies that break the LCP complementarity structure.
- The wage computation involves a matrix inverse of the active submatrix of T, coupling all goods together.

### 2.3 What Does Work: The Reduction to Exchange Economy

Garg, Kannan & Vazirani (2015) showed that any AD market with production can be **reduced to a pure exchange economy** such that equilibria are in 1-to-1 correspondence. When the original market has PLC utilities and polyhedral production, the exchange market has PLC utilities. This is theoretically important but the reduction may be complex to implement.

---

## 3. Practical Approaches

### 3.1 Damped Tatonnement (Easiest to Implement)

**Idea:** Replace p_{t+1} = SCM_round(p_t) with p_{t+1} = (1-α)·p_t + α·SCM_round(p_t) for step size α ∈ (0,1).

**Why it helps:** Standard tatonnement can overshoot and cycle. Damping reduces step size, which can convert divergent/cycling behavior into convergence — at the cost of slower convergence.

**Implementation:** One line change in `equilibrium.py`:
```python
p_new = (1 - alpha) * p_old + alpha * scm_round(p_old, ...)
```

**Adaptive damping:** Start with α=1 (standard tatonnement), and if the fixed-point error increases, halve α. This combines speed (when convergence is easy) with robustness (when it's not).

**Theoretical backing:** Cheung, Cole & Devanur (2020) showed that for CPF (Convex Potential Function) markets — which include Fisher markets with CES utilities — damped tatonnement is equivalent to gradient descent on a convex potential, guaranteeing convergence with sufficiently small step size. Our Fisher market sub-problem (EG program) falls in this class, though the full SCM map with production may not.

**Effort: Low. Priority: High.**

### 3.2 Anderson Acceleration (Moderate Effort, High Impact)

**Idea:** Instead of using just the latest iterate, use a linear combination of the last m iterates that minimizes the residual. This is the fixed-point analogue of GMRES for linear systems.

**How it works:**
- Store the last m+1 iterates: p_{t-m}, ..., p_t
- Compute residuals: r_k = SCM_round(p_k) - p_k
- Find coefficients θ that minimize ||Σ θ_k · r_k||
- Set p_{t+1} = Σ θ_k · SCM_round(p_k)

**Why it's promising for us:**
- Requires NO derivatives (the SCM map involves an LP solver, so Jacobians are not readily available)
- Works on any fixed-point iteration — drop-in replacement for our tatonnement loop
- Convergence is typically much faster than standard iteration, and more robust against cycling
- Well-studied: equivalent to GMRES for linear problems, multisecant methods for nonlinear

**Python implementations available:**
- `scipy.optimize.anderson` (built into SciPy)
- `cvxgrp/aa` (C with Python bindings, from Boyd's group at Stanford)
- `jaxopt.AndersonAcceleration` (JAX-based)

**Implementation sketch:**
```python
from scipy.optimize import anderson

def excess(p):
    return scm_round(p, T, Y, U) - p

p_eq = anderson(excess, p_init, M=5)  # M = memory depth
```

**Effort: Moderate. Priority: High.**

### 3.3 Newton's Method on the Fixed-Point Equation (Higher Effort)

**Idea:** Solve F(p) = SCM_round(p) - p = 0 using Newton's method: p_{t+1} = p_t - J_F(p_t)^{-1} · F(p_t).

**Advantages:** Quadratic convergence near a solution.

**Challenges for us:**
- The Jacobian J_F is not analytically available because SCM_round involves solving an LP (production) and a convex program (Fisher market/EG).
- Must use **finite differences** to approximate the Jacobian: ∂F_i/∂p_j ≈ (F(p + ε·e_j) - F(p)) / ε. This requires n+1 evaluations of SCM_round per Newton step.
- For small n (2-4 goods), this is feasible. For large n, it becomes expensive.
- Newton's method needs a good initial guess — can use the tatonnement iterates.

**Quasi-Newton variants:**
- **Broyden's method:** Approximates the Jacobian using rank-1 updates, requiring only 1 function evaluation per step (instead of n+1). Good balance of speed and convergence.
- **scipy.optimize.fsolve** or **scipy.optimize.root** with method='broyden1' or 'hybr' (Powell's hybrid method).

**Implementation sketch:**
```python
from scipy.optimize import root

def F(p):
    return scm_round(p, T, Y, U) - p

result = root(F, p_init, method='hybr')  # Powell's hybrid = Newton + trust region
p_eq = result.x
```

**Effort: Moderate. Priority: Medium.**

### 3.4 Homotopy Continuation (Robust, Higher Effort)

**Idea:** Start from an economy with a known equilibrium and continuously deform it into the target economy, tracking the equilibrium along the path.

**How it works:**
1. Define a "simple" economy (e.g., symmetric T, U) where equilibrium is trivially known (e.g., equal prices).
2. Parameterize: H(p, λ) = (1-λ)·F_simple(p) + λ·F_target(p) = 0
3. At λ=0, the solution is known. Track the solution curve as λ goes from 0 to 1.
4. Use predictor-corrector methods (predict with tangent direction, correct with Newton).

**Why it's relevant:**
- Homotopy methods are the standard tool in Applied General Equilibrium (AGE/CGE) modeling.
- They can find equilibria even when tatonnement diverges.
- They can potentially find ALL equilibria (multiple equilibria are possible in our setting).
- The paper (Section C.4) specifically mentions this as an open direction.

**Key references:**
- Eaves & Schmedders (1999), "General Equilibrium Models and Homotopy Methods"
- Pap (2011), "Computing Economic Equilibria by a Homotopy Method" (arXiv:1110.5144)
- Zangwill & Garcia (1981), the foundational text on homotopy methods

**Implementation:** Would need a homotopy continuation library. Python options include custom implementation using scipy's ODE integrators to follow the path, or interfacing with specialized software like PHCpack or Bertini (for polynomial systems).

**Effort: High. Priority: Medium (worth it for systematic exploration of equilibrium multiplicity).**

### 3.5 Scarf's Algorithm / Simplicial Methods (Theoretically Guaranteed)

**Idea:** Subdivide the price simplex into smaller simplices and use a combinatorial path-following method (based on Sperner's lemma) to find an approximate fixed point.

**Guarantees:** Always finds an approximate fixed point for any continuous function on a compact convex set — no assumptions on gross substitutability or contractivity needed.

**Drawbacks:**
- Convergence is slow: exponential in the number of goods in the worst case.
- The approximation quality depends on the subdivision granularity.
- Practical only for small n (2-5 goods).

**Recent advance (January 2025):** A new paper (arXiv:2501.10884) gives a faster algorithm using smoothed analysis, with runtime e^{O(n)}/ε instead of (1/ε)^{O(n)} — exponentially better in the approximation parameter.

**Relevance:** Since our current examples have 2-4 goods, Scarf's algorithm is actually feasible as a last resort. It guarantees finding an equilibrium (up to approximation) regardless of convergence issues.

**Effort: Medium. Priority: Low (last resort).**

### 3.6 Direct Optimization / Complementarity Formulation

**Idea:** Instead of iterating the SCM map, formulate the equilibrium conditions directly as a system of equations/inequalities and solve with an optimization solver.

**The equilibrium conditions (from Definition 3.1) form a mixed complementarity problem (MCP):**
1. Market clearing: Σ_i x_ij ≤ s_j, with equality if p_j > 0
2. Optimal consumption: x_i maximizes u_i(x_i) subject to p·x_i ≤ b_i
3. Profit maximization / zero-profit: p_j ≤ cost_j, with equality if s_j > 0
4. Budget = wage income

This can be written as an MCP and solved using PATH solver or similar.

**Python options:**
- Reformulate as a nonlinear program (NLP) and use scipy.optimize.minimize with constraints
- Use complementarity-aware solvers via pyomo + PATH

**Challenge:** The zone structure (active sets I, J) creates discontinuities. Within a zone, the system is smooth, but zone transitions are not. An MCP formulation handles this naturally through complementarity conditions.

**Effort: High. Priority: Medium.**

---

## 4. Recommended Implementation Order

Based on effort-to-impact ratio:

### Phase 1: Quick Wins (implement now)
1. **Damped tatonnement with adaptive step size** — one-line change, can rescue many non-convergent cases
2. **Anderson acceleration** — drop-in replacement using scipy.optimize.anderson, dramatically improves convergence

### Phase 2: Robust Solvers (implement next)
3. **Newton/quasi-Newton via scipy.optimize.root** — quadratic convergence, good for small-to-medium n
4. **Direct MCP formulation** — bypass tatonnement entirely, solve equilibrium conditions directly

### Phase 3: Heavy Machinery (for research exploration)
5. **Homotopy continuation** — find equilibria systematically, explore multiplicity
6. **Scarf's algorithm** — guaranteed but slow, last resort for small n

---

## 5. Key References

### Algorithms for AD Markets with Production
- Garg & Vazirani, "On Computability of Equilibria in Markets with Production" (SODA 2014) — LCP + Lemke for SPLC utilities + SPLC production
- Garg, Kannan & Vazirani, "Markets with Production: A Polynomial Time Algorithm and a Reduction to Pure Exchange" (EC 2015) — poly-time for constant #goods, reduction to exchange
- Devanur & Kannan, "Market Equilibria in Polynomial Time for Fixed Number of Goods or Agents" (FOCS 2008) — poly-time for fixed n or m

### Tatonnement Convergence
- Cheung, Cole & Devanur, "Tatonnement Beyond Gross Substitutes? Gradient Descent to the Rescue" (Games & Economic Behavior, 2020) — damped tatonnement as gradient descent on convex potential
- Goktas et al., "On the Convergence of Tatonnement for Linear Fisher Markets" (2024, arXiv:2406.12526) — step-size analysis for linear Fisher markets

### Fixed-Point Acceleration
- Walker & Ni, "Anderson Acceleration for Fixed-Point Iterations" (SIAM J. Numer. Anal., 2011) — foundational paper on Anderson acceleration
- Boyd et al., "Globally Convergent Type-I Anderson Acceleration for Non-Smooth Fixed-Point Iterations" (Stanford, cvxgrp/aa)

### Homotopy and Simplicial Methods
- Eaves & Schmedders, "General Equilibrium Models and Homotopy Methods" (J. Econ. Dynamics & Control, 1999)
- Pap, "Computing Economic Equilibria by a Homotopy Method" (arXiv:1110.5144, 2011)
- Scarf, "The Computation of Economic Equilibria" (Yale, 1973)

### Complexity
- Garg, Mehta, Sohoni & Vazirani, "A Complementary Pivot Algorithm for Market Equilibrium under SPLC Utilities" (SIAM J. Computing, 2015) — PPAD-complete but practical
- Chen et al., arXiv:2501.10884 (2025) — beating brute force for fixed-point computation
