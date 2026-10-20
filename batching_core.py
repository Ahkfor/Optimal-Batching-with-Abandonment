"""
batching_core.py -- Unified computational core for the numerical study of
"Optimal Batching with Abandonment: Threshold Policies Under Risk-Neutral
and Risk-Sensitive Criteria".

Model
-----
Discrete-time batch clearing.  State X_t = items in the buffer.

    X_{t+1} = X_t - Binomial(X_t, p) + Bernoulli(q)

Costs: fixed stopping (setup) cost alpha, per-item reward beta, holding
cost c per item-period, abandonment penalty gamma per abandoned item,
discount factor delta.  Effective per-item cost rate C = c + delta*gamma*p.

All solvers are pure NumPy/SciPy (the numba dependency of the original
scattered scripts is dropped).

Contents
--------
  Params, BASELINE                 parameter container + paper baseline
  binom_matrix                     padded binomial weights C(x,j) p^{x-j} (1-p)^j
  fg_recursion                     forward recursions for f and g (risk-neutral)
  t_recursion                      mean hitting-time recursion t(theta)=E_0[T_theta]
  reduced_objective_rn             v_rn(theta) = (beta*theta - alpha - f)/g
  theta_star_rn                    min{theta : Delta v_rn <= 0}
  frs_recursion / theta_star_rs    risk-sensitive (entropic) recursion, delta = 1
  robust_vi                        KL-robust risk-averse value iteration
  worst_case_joint                 joint (a,d) truncated-Gibbs worst case + marginals
  simulate_cycles                  Monte Carlo cycle-length simulation
  avg_objective / theta_star_avg   renewal-reward (average) criterion
  renewal_objective / theta_star_ren   discounted renewal criterion
  Phi / delta_crit                 phase function and critical discount factor
  vi_risk_neutral                  plain value iteration (cross-checks only)

IMPORTANT (g-stability, cf. Remark in the paper): at delta = 1 the g
recursion admits g = 1 identically, but the forward recursion loses
accuracy beyond theta ~ 150.  fg_recursion therefore substitutes the
theoretical g = 1 whenever delta == 1.

IMPORTANT (robust VI variants): the paper's Algorithm (robust-vi) states a
JOINT truncated-Gibbs tilt over (a, d) with value
    E_Q[phi] - (1/eta) D_KL(Q||P),        eta < 0.
The originally uploaded scripts instead (i) average the arrival atom d
under the nominal q BEFORE tilting (a marginal tilt over a only) and
(ii) evaluate the KL term with the opposite sign.  The published main-text
robust thresholds (51, 42, 27, 14) come from that legacy variant, while
the published eta-convergence table comes from the stated joint variant.
robust_vi exposes both via `tilt=` and `kl_sign=`; the default is the
paper-stated joint formulation.  See README.md, discrepancy note D2.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln
from dataclasses import dataclass, replace


# =============================================================================
# Parameters
# =============================================================================

@dataclass(frozen=True)
class Params:
    alpha: float = 100.0   # fixed stopping (setup) cost
    beta:  float = 10.0    # per-item reward
    c:     float = 0.01    # holding cost per item-period
    gamma: float = 0.5     # abandonment penalty per item
    p:     float = 0.01    # per-item abandonment probability
    q:     float = 0.8     # per-period arrival probability
    delta: float = 0.99    # discount factor


BASELINE = Params()


# =============================================================================
# Binomial weight matrix
# =============================================================================

def binom_matrix(X: int, p: float) -> np.ndarray:
    """W[x, j] = C(x, j) p^{x-j} (1-p)^j for 0 <= j <= x, else 0.

    With i = x - j abandonments this equals C(x,i) p^i (1-p)^{x-i}, the
    weights of the forward recursions.
    """
    x = np.arange(X + 1)[:, None]
    j = np.arange(X + 1)[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        lc = gammaln(x + 1) - gammaln(j + 1) - gammaln(x - j + 1)
        lp = (x - j) * np.log(p) + j * np.log(1 - p)
    W = np.where(j <= x, np.exp(lc + lp), 0.0)
    return np.nan_to_num(W, nan=0.0)


# =============================================================================
# Risk-neutral forward recursions for f and g
# =============================================================================

def fg_recursion(par: Params, X: int, W: np.ndarray | None = None):
    """Forward recursions

        f(x+1) = (C x + om_x f(x) - delta F(x)) / (delta (1-p)^x q)
        g(x+1) = (om_x g(x) - delta G(x))       / (delta (1-p)^x q)

    with om_x = 1 - delta (1-p)^x (1-q), C = c + delta gamma p, and
    F(x) = sum_{i=1}^x C(x,i) p^i (1-p)^{x-i} [q f(x+1-i) + (1-q) f(x-i)]
    (similarly G).  Boundary f(0) = 0, g(0) = 1; 1/g(theta) = E_0[delta^{T_theta}].

    At delta == 1 the theoretical g = 1 is substituted (see module docstring).

    Returns (f, g, ok): arrays of length X+1 and a boolean mask marking the
    numerically reliable prefix.
    """
    c, gam, p, q, delta = par.c, par.gamma, par.p, par.q, par.delta
    C = c + delta * gam * p
    if W is None:
        W = binom_matrix(X, p)
    f = np.zeros(X + 1)
    g = np.zeros(X + 1)
    ok = np.ones(X + 1, dtype=bool)
    g[0] = 1.0
    if delta == 1.0:
        g[:] = 1.0
    # Overflow past the numerically stable region is expected (Remark
    # g-stability); the ok mask marks the reliable prefix, so suppress the
    # runtime warnings and detect breakdown explicitly.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        for x in range(X):
            F = W[x, :x] @ (q * f[1:x + 1] + (1 - q) * f[:x])
            om = 1.0 - delta * (1 - p) ** x * (1 - q)
            f[x + 1] = (C * x + om * f[x] - delta * F) / (delta * (1 - p) ** x * q)
            if delta != 1.0:
                G = W[x, :x] @ (q * g[1:x + 1] + (1 - q) * g[:x])
                g[x + 1] = (om * g[x] - delta * G) / (delta * (1 - p) ** x * q)
                if not np.isfinite(g[x + 1]) or abs(g[x + 1]) < 1e-15:
                    ok[x + 1:] = False
                    break
            if not np.isfinite(f[x + 1]):
                ok[x + 1:] = False
                break
    return f, g, ok


def t_recursion(p: float, q: float, X: int) -> np.ndarray:
    """Mean hitting-time recursion (delta = 1): t(theta) = E_0[T_theta].

        t(x+1) = (om_x t(x) + 1 - F_t(x)) / ((1-p)^x q),
        om_x = 1 - (1-p)^x (1-q)   [delta = 1].
    """
    W = binom_matrix(X, p)
    t = np.zeros(X + 1)
    for x in range(X):
        Ft = W[x, :x] @ (q * t[1:x + 1] + (1 - q) * t[:x])
        om = 1.0 - (1 - p) ** x * (1 - q)
        t[x + 1] = (om * t[x] + 1.0 - Ft) / ((1 - p) ** x * q)
    return t


# =============================================================================
# Risk-neutral reduced objective and optimal threshold
# =============================================================================

def reduced_objective_rn(par: Params, X: int):
    """v_rn(theta) = (beta theta - alpha - f(theta)) / g(theta), theta = 0..X."""
    f, g, ok = fg_recursion(par, X)
    th = np.arange(X + 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        v = np.where(ok & (g != 0), (par.beta * th - par.alpha - f) / g, np.nan)
    return v, f, g, ok


def theta_star_rn(par: Params, X: int = 300) -> int:
    """theta*_rn = min{theta >= 0 : v(theta+1) <= v(theta)} (smallest optimum)."""
    v, _, _, ok = reduced_objective_rn(par, X)
    prev = -par.alpha  # v(0)
    for x in range(1, X + 1):
        if not ok[x] or not np.isfinite(v[x]):
            break
        if v[x] <= prev:
            return x - 1
        prev = v[x]
    return X


# =============================================================================
# Risk-sensitive (entropic) recursion, delta = 1
# =============================================================================

def frs_recursion(par: Params, eta: float, X: int):
    """Risk-sensitive cost potential f_rs at delta = 1, with xi = e^{eta f_rs}:

        xi(x+1) = (xi(x) om_rs(x) - F_rs(x)) / ((1-p)^x q)
        om_rs(x) = e^{eta c x} - (1-p)^x (1-q)
        F_rs(x)  = sum_{i=1}^x w_i e^{-eta gamma i}
                   [q xi(x+1-i) + (1-q) xi(x-i)].

    eta > 0 is risk-SEEKING in this sign convention (the certainty
    equivalent of costs is deflated); eta -> 0 recovers f (checked in
    verify_paper_numbers.py).

    Returns (f_rs, xi, ok).  For large eta*x the xi recursion overflows;
    the ok mask marks the reliable prefix (all paper thresholds lie well
    inside it).
    """
    c, gam, p, q = par.c, par.gamma, par.p, par.q
    W = binom_matrix(X, p)
    f = np.zeros(X + 1)
    xi = np.ones(X + 1)
    ok = np.ones(X + 1, dtype=bool)
    with np.errstate(over='ignore', invalid='ignore'):
        for x in range(X):
            i = np.arange(1, x + 1)
            j = x - i
            F = (W[x, j] * np.exp(-eta * gam * i)) @ (q * xi[j + 1] + (1 - q) * xi[j])
            om = np.exp(eta * c * x) - (1 - p) ** x * (1 - q)
            xi[x + 1] = (xi[x] * om - F) / ((1 - p) ** x * q)
            if not np.isfinite(xi[x + 1]) or xi[x + 1] <= 0:
                ok[x + 1:] = False
                break
            f[x + 1] = np.log(xi[x + 1]) / eta
    return f, xi, ok


def theta_star_rs(par: Params, eta: float, X: int = 400) -> int:
    """theta*_rs = min{theta : Delta f_rs(theta) >= beta} (risk-seeking, eta > 0)."""
    par1 = replace(par, delta=1.0)
    f, _, ok = frs_recursion(par1, eta, X)
    df = np.diff(f)
    for x in range(X):
        if not ok[x + 1]:
            break
        if df[x] >= par.beta:
            return x
    return X


# =============================================================================
# KL-robust risk-averse value iteration (delta = 1)
# =============================================================================

def robust_vi(par: Params, eta: float, kappa: float, N_max: int,
              tilt: str = 'joint', kl_sign: str = 'bellman',
              tol: float = 1e-10, maxit: int = 30000, V_init=None,
              check_every: int = 20):
    """Robust (likelihood-ratio cap kappa) risk-averse value iteration.

    Paper-stated Bellman equation (Algorithm robust-vi, delta = 1):

        V(x) = max{ beta x - alpha,
                    -c x + inf_{Q: dQ/dP <= kappa}
                           [ E_Q[ V(x - a + d) - gamma a ]
                             - (1/eta) D_KL(Q || P) ] }

    solved by the truncated-Gibbs likelihood ratio
        m*(a, d) = min(kappa, lambda_x e^{eta phi(a,d)}),
        phi(a, d) = V(x - a + d) - gamma a,
    with lambda_x fixed by E_P[m*] = 1 (batched bisection on log lambda).

    Parameters
    ----------
    tilt    : 'joint'    -- tilt over the joint (a, d) atom (paper statement;
                            default, and the variant behind the published
                            eta-convergence table);
              'marginal' -- average d under the nominal q first and tilt only
                            over a (the variant in the originally uploaded
                            scripts, behind the published main-text table).
    kl_sign : 'bellman'  -- evaluate the objective as stated,
                            E_Q[phi] - (1/eta) D_KL  (default);
              'legacy'   -- E_Q[phi] + (1/eta) D_KL, the sign used by the
                            originally uploaded scripts.

    The boundary condition V[N_max+1] = beta (N_max+1) - alpha is IMPOSED
    (the uploaded scripts clamped the next state at N_max instead; see
    README.md, discrepancy note D3).

    Returns (V, theta*_ra, n_iter): V of length N_max+2, the first state at
    which stopping is optimal, and the iteration count.
    """
    assert eta < 0, "robust_vi is the risk-averse case eta < 0"
    assert tilt in ('joint', 'marginal')
    assert kl_sign in ('bellman', 'legacy')
    alpha, beta = par.alpha, par.beta
    c, gam, p, q = par.c, par.gamma, par.p, par.q

    S = beta * np.arange(N_max + 2) - alpha
    V = np.zeros(N_max + 2) if V_init is None else V_init.copy()
    assert V.shape == (N_max + 2,)

    idx_x = np.arange(N_max + 1)[:, None]
    idx_a = np.arange(N_max + 1)[None, :]
    valid = idx_a <= idx_x
    A = np.where(valid, idx_a, 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        lw = (gammaln(idx_x + 1) - gammaln(A + 1) - gammaln(idx_x - A + 1)
              + A * np.log(p) + (idx_x - A) * np.log(1 - p))
    Pa = np.where(valid, np.exp(lw), 0.0)      # P(A = a | x)
    P0, P1 = Pa * (1 - q), Pa * q              # joint nominal P(a, d)
    xsa = np.clip(idx_x - A, 0, N_max + 1)
    xarr = np.arange(N_max + 1)
    logk = np.log(kappa)
    sgn = -1.0 if kl_sign == 'bellman' else 1.0   # cont = -cx + EQ + sgn*(1/eta)*DKL

    llog = np.zeros(N_max + 1)                  # warm-started log lambda_x
    V_old = V.copy()
    it = 0
    for it in range(maxit):
        if tilt == 'joint':
            phi0 = V[xsa] - gam * A
            phi1 = V[xsa + 1] - gam * A
            le0 = np.where(valid, eta * phi0, 0.0)
            le1 = np.where(valid, eta * phi1, 0.0)
            lo, hi = llog - 3.0, llog + 3.0
            for _ in range(28):
                mid = 0.5 * (lo + hi)
                m = ((P0 * np.exp(np.minimum(logk, mid[:, None] + le0))).sum(1)
                     + (P1 * np.exp(np.minimum(logk, mid[:, None] + le1))).sum(1))
                gt = m > 1.0
                hi = np.where(gt, mid, hi)
                lo = np.where(gt, lo, mid)
            llog = 0.5 * (lo + hi)
            lm0 = np.minimum(logk, llog[:, None] + le0)
            lm1 = np.minimum(logk, llog[:, None] + le1)
            m0, m1 = np.exp(lm0), np.exp(lm1)
            EQ = (P0 * m0 * phi0).sum(1) + (P1 * m1 * phi1).sum(1)
            DKL = (P0 * m0 * lm0).sum(1) + (P1 * m1 * lm1).sum(1)
        else:
            Wm = q * V[xsa + 1] + (1 - q) * V[xsa] - gam * A
            le = np.where(valid, eta * Wm, 0.0)
            lo, hi = llog - 3.0, llog + 3.0
            for _ in range(28):
                mid = 0.5 * (lo + hi)
                m = (Pa * np.exp(np.minimum(logk, mid[:, None] + le))).sum(1)
                gt = m > 1.0
                hi = np.where(gt, mid, hi)
                lo = np.where(gt, lo, mid)
            llog = 0.5 * (lo + hi)
            lm = np.minimum(logk, llog[:, None] + le)
            lr = np.exp(lm)
            EQ = (Pa * lr * Wm).sum(1)
            DKL = (Pa * lr * lm).sum(1)

        Vnew = np.maximum(S[:N_max + 1], -c * xarr + EQ + sgn * (1.0 / eta) * DKL)
        V = np.concatenate([Vnew, [S[N_max + 1]]])
        if it % check_every == check_every - 1:
            if np.max(np.abs(V - V_old)) < tol:
                break
            V_old = V.copy()

    diff = V - S
    th = next((x for x in range(N_max + 1) if diff[x] < 1e-9), N_max)
    return V, th, it + 1


def worst_case_joint(x: int, V: np.ndarray, par: Params, eta: float,
                     kappa: float):
    """Joint truncated-Gibbs worst case at state x (paper's eq. (ra-gibbs)).

    m*(a, d) = min(kappa, lambda_x e^{eta phi(a,d)}),
    phi(a, d) = V(x - a + d) - gamma a,  (a, d) ~ Binomial(x, p) x Bernoulli(q).

    Returns dict with the joint worst-case Q*(a, d), its abandonment
    marginal Q_A(a) = sum_d Q*(a, d)  (this is the paper's definition of the
    marginal worst-case measure: "averaging the joint truncated tilt over
    the arrival atom d"), the nominal marginal P_A(a), and the marginal
    likelihood ratio Q_A / P_A.
    """
    p, q, gam = par.p, par.q, par.gamma
    a = np.arange(x + 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        lw = (gammaln(x + 1) - gammaln(a + 1) - gammaln(x - a + 1)
              + a * np.log(p) + (x - a) * np.log(1 - p))
    PA = np.exp(lw)                              # nominal marginal over a
    P = np.stack([PA * (1 - q), PA * q], axis=1)  # joint nominal P(a, d)
    ns = np.clip(x - a, 0, len(V) - 1)
    phi = np.stack([V[ns] - gam * a, V[ns + 1] - gam * a], axis=1)

    le = eta * phi
    logk = np.log(kappa)
    lo, hi = -25.0, 25.0
    # bracket expansion
    for _ in range(60):
        f_lo = (P * np.exp(np.minimum(logk, lo + le))).sum()
        f_hi = (P * np.exp(np.minimum(logk, hi + le))).sum()
        if f_lo >= 1.0:
            lo -= 2.0
        elif f_hi <= 1.0:
            hi += 2.0
        else:
            break
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (P * np.exp(np.minimum(logk, mid + le))).sum() > 1.0:
            hi = mid
        else:
            lo = mid
    log_lam = 0.5 * (lo + hi)

    log_m = np.minimum(logk, log_lam + le)
    m = np.exp(log_m)
    Q = P * m                                    # joint worst case
    QA = Q.sum(axis=1)                           # marginal over d (paper def.)
    LR = QA / PA
    return dict(a=a, P_A=PA, Q_A=QA, LR=LR, Q_joint=Q, m_joint=m,
                log_lam=log_lam, phi=phi)


# =============================================================================
# Monte Carlo simulation of the batching cycle
# =============================================================================

def simulate_cycles(theta: int, p: float, q: float, n_paths: int = 100_000,
                    seed: int = 42) -> np.ndarray:
    """Vectorized simulation of T_theta = inf{t >= 0 : X_t >= theta}, X_0 = 0."""
    rng = np.random.default_rng(seed)
    Xs = np.zeros(n_paths, dtype=np.int64)
    T = np.full(n_paths, -1, dtype=np.int64)
    active = np.ones(n_paths, dtype=bool)
    t = 0
    while active.any():
        idx = active
        A = rng.binomial(Xs[idx], p)
        D = rng.binomial(1, q, size=int(idx.sum()))
        Xs[idx] = Xs[idx] - A + D
        t += 1
        hit = idx & (Xs >= theta)
        T[hit] = t
        active = Xs < theta
    return T


# =============================================================================
# Renewal criteria (Section: renewal extension)
# =============================================================================

def avg_objective(par: Params, X: int = 300):
    """Average-reward objective v_bar(theta) = N_1(theta) / t(theta) with
    N_1(theta) = beta theta - alpha - f(theta; delta=1)."""
    par1 = replace(par, delta=1.0)
    f, _, _ = fg_recursion(par1, X)
    t = t_recursion(par.p, par.q, X)
    th = np.arange(X + 1)
    N1 = par.beta * th - par.alpha - f
    with np.errstate(divide='ignore', invalid='ignore'):
        vbar = np.where(th > 0, N1 / t, np.nan)
    return vbar, N1, t


def theta_star_avg(par: Params, X: int = 300) -> int:
    vbar, _, _ = avg_objective(par, X)
    return int(np.nanargmax(vbar))


def renewal_objective(par: Params, X: int = 300):
    """Discounted renewal objective
        v^ren_delta(theta) = N_delta(theta) / (g(theta; delta) - 1),
        N_delta(theta) = beta theta - alpha - f(theta; delta),
    valid on the positive domain {g > 1}."""
    f, g, ok = fg_recursion(par, X)
    th = np.arange(X + 1)
    Nd = par.beta * th - par.alpha - f
    with np.errstate(divide='ignore', invalid='ignore'):
        vren = np.where(ok & (g > 1.0), Nd / (g - 1.0), np.nan)
    return vren, Nd, g, ok


def theta_star_ren(par: Params, X: int = 300) -> int:
    vren, _, _, _ = renewal_objective(par, X)
    return int(np.nanargmax(vren))


# =============================================================================
# Phase function and critical discount factor (beta-monotonicity)
# =============================================================================

def Phi(theta: int, par: Params) -> float:
    """Phi(theta; delta) = (theta+1)/g(theta+1) - theta/g(theta)."""
    _, g, _ = fg_recursion(par, theta + 2)
    return (theta + 1) / g[theta + 1] - theta / g[theta]


def delta_crit(theta: int, par: Params, lo: float = 0.5,
               hi: float = 1.0 - 1e-9, tol: float = 1e-10) -> float:
    """Critical discount factor delta_theta: the unique root of
    Phi(theta; delta) = 0 (negative below, positive above).  NaN if the
    root lies outside [lo, hi]."""
    from dataclasses import replace as _rep
    cb_lo = Phi(theta, _rep(par, delta=lo))
    cb_hi = Phi(theta, _rep(par, delta=hi))
    if cb_lo > 0 or cb_hi < 0:
        return np.nan
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if Phi(theta, _rep(par, delta=mid)) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# =============================================================================
# Plain risk-neutral value iteration (cross-checks only)
# =============================================================================

def vi_risk_neutral(par: Params, N_max: int, tol: float = 1e-12,
                    maxit: int = 5000):
    """Standard VI for V(x) = max{beta x - alpha, -C x + delta E[V(X')]}
    with imposed boundary V[N_max+1] = beta (N_max+1) - alpha.  Used to
    cross-check the reduced-objective recursions."""
    alpha, beta, delta = par.alpha, par.beta, par.delta
    C = par.c + delta * par.gamma * par.p
    p, q = par.p, par.q
    S = beta * np.arange(N_max + 2) - alpha
    V = S.copy()
    logp, log1p = np.log(p), np.log(1 - p)
    for it in range(maxit):
        cont = np.empty(N_max + 2)
        for x in range(N_max + 1):
            a = np.arange(x + 1)
            lw = (gammaln(x + 1) - gammaln(a + 1) - gammaln(x - a + 1)
                  + a * logp + (x - a) * log1p)
            w = np.exp(lw)
            ns = x - a
            cont[x] = -C * x + delta * (q * (w * V[ns + 1]).sum()
                                        + (1 - q) * (w * V[ns]).sum())
        cont[N_max + 1] = S[N_max + 1]
        Vnew = np.maximum(S, cont)
        if np.max(np.abs(Vnew - V)) < tol:
            V = Vnew
            break
        V = Vnew
    diff = V - S
    th = next((x for x in range(N_max + 1) if diff[x] < 1e-9), N_max)
    return V, th
