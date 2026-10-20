"""
verify_paper_numbers.py -- regression checks of every number quoted in the
paper's numerical section against the unified core.

Run:  python verify_paper_numbers.py           (all checks)
      python verify_paper_numbers.py fast      (skip the slow robust VI rows)

Each check prints OK / MISMATCH.  Exit code 0 iff all executed checks pass.
"""

import sys
import numpy as np
from dataclasses import replace

from batching_core import (Params, BASELINE, fg_recursion, t_recursion,
                           theta_star_rn, reduced_objective_rn, frs_recursion,
                           theta_star_rs, robust_vi, simulate_cycles,
                           avg_objective, renewal_objective, theta_star_avg,
                           theta_star_ren, delta_crit, vi_risk_neutral)

RESULTS = []


def check(name, got, want, tol=None):
    if tol is None:
        ok = got == want
        got_s, want_s = str(got), str(want)
    else:
        ok = abs(got - want) <= tol
        got_s, want_s = f"{got:.6g}", f"{want:.6g} (+/-{tol})"
    print(f"  [{'OK ' if ok else 'MISMATCH'}] {name}: got {got_s}, want {want_s}")
    RESULTS.append(ok)


FAST = len(sys.argv) > 1 and sys.argv[1] == 'fast'
X = 300
par = BASELINE

print("== risk-neutral reduced objective (baseline delta = 0.99) ==")
v, f, g, ok = reduced_objective_rn(par, X)
check("theta*_rn(0.99)", theta_star_rn(par, X), 43)
check("v_rn(theta*) = V_rn(0)", v[43], 137.9232, tol=5e-4)
check("N(43) = 330 - f(43)", par.beta * 43 - par.alpha - f[43], 295.0, tol=1.0)
check("E[delta^{T_43}] = 1/g(43)", 1.0 / g[43], 0.47, tol=0.01)
V_vi, th_vi = vi_risk_neutral(par, 400)
check("VI cross-check V(0)", V_vi[0], 137.9232, tol=5e-4)
check("VI cross-check theta*", th_vi, 43)

print("== sensitivity spot values (tab:baseline) ==")
check("alpha=1000", theta_star_rn(replace(par, alpha=1000.0), X), 91)
check("delta=0.9", theta_star_rn(replace(par, delta=0.9), X), 16)
check("delta=1.0", theta_star_rn(replace(par, delta=1.0), X), 73)
check("beta=1", theta_star_rn(replace(par, beta=1.0), X), 53)
check("beta=100", theta_star_rn(replace(par, beta=100.0), X), 41)

print("== renewal section (tab:renewal + text) ==")
t = t_recursion(par.p, par.q, X)
check("t(43)", t[43], 76.1, tol=0.1)
check("t(64)", t[64], 155.0, tol=0.1)
vbar, N1, _ = avg_objective(par, X)
th_avg = theta_star_avg(par, X)
check("theta*_avg", th_avg, 35)
check("vbar(theta*_avg)", vbar[th_avg], 4.11, tol=0.01)
check("loss at one-shot theta*_rn(1)=73",
      100 * (vbar[th_avg] - vbar[73]) / vbar[th_avg], 47.6, tol=0.1)
par99 = replace(par, delta=0.99)
vren, Nd, gd, okd = renewal_objective(par99, X)
th_ren = theta_star_ren(par99, X)
check("theta*_ren(0.99)", th_ren, 28)
check("v^ren(theta*_ren)", vren[th_ren], 317.4, tol=0.1)
check("loss at one-shot theta*_rn(0.99)=43",
      100 * (vren[th_ren] - vren[43]) / vren[th_ren], 18.3, tol=0.1)
if not FAST:
    mc = simulate_cycles(43, par.p, par.q, 100_000, seed=1).mean()
    check("MC E[T_43]", mc, 76.1, tol=0.5)

print("== risk-sensitive thresholds (delta = 1, risk-seeking) ==")
for eta, want in [(0.05, 92), (0.10, 116), (0.15, 141), (0.25, 195)]:
    check(f"theta*_rs(eta={eta})", theta_star_rs(par, eta, 400), want)

print("== risk-neutral limit of f_rs (eta -> 0+) ==")
# The two recursions agree in the limit eta -> 0, but each amplifies the
# O(eta) gap by 1/((1-p)^x q) > 1 per step (same instability as Remark
# g-stability), so the comparison is restricted to the stable prefix.
f_small, _, _ = frs_recursion(replace(par, delta=1.0), 1e-3, 120)
f_rn1, _, _ = fg_recursion(replace(par, delta=1.0), 120)
check("f_rs(eta=1e-3) ~ f_rn, max abs diff on x <= 50",
      float(np.max(np.abs(f_small[:51] - f_rn1[:51]))), 0.0, tol=0.1)

print("== robust risk-averse thresholds (kappa = 5, delta = 1) ==")
print("   published main-text row (51, 42, 27, 14) <- legacy solver variant")
if FAST:
    print("   [skipped in fast mode]")
else:
    Vw = None
    for eta, want in [(-0.05, 51), (-0.10, 42), (-0.20, 27), (-0.50, 14)]:
        Vw, th, _ = robust_vi(par, eta, 5.0, N_max=120, tilt='marginal',
                              kl_sign='legacy', V_init=Vw, maxit=4000)
        check(f"theta*_ra({eta}) legacy", th, want)
print("   paper-stated joint variant gives a DIFFERENT first cell:")
if FAST:
    print("   [skipped in fast mode]")
else:
    _, th, _ = robust_vi(par, -0.05, 5.0, N_max=120, tilt='joint',
                         kl_sign='bellman', maxit=4000)
    check("theta*_ra(-0.05) joint/bellman (documents discrepancy D2)",
          th, 61)

print("== eta-convergence row (tab:eta-convergence; joint/bellman variant) ==")
if FAST:
    print("   [skipped in fast mode]")
else:
    par_e = Params(alpha=10.0, beta=2.0, delta=1.0)
    check("theta*_rn (alpha=10, beta=2)", theta_star_rn(par_e, X), 47)
    Vw = None
    for eta, want in [(-0.50, 29), (-0.30, 35), (-0.10, 42), (-0.005, 47)]:
        Vw, th, _ = robust_vi(par_e, eta, 10.0, N_max=80, V_init=Vw,
                              maxit=4000)
        check(f"theta*_ra({eta})", th, want)

print("== phase boundaries ==")
check("delta_1", delta_crit(1, par), 0.56, tol=0.01)
check("delta_43 (p=0.005)", delta_crit(43, replace(par, p=0.005)), 0.9865,
      tol=5e-4)
check("delta_43 (p=0.01)", delta_crit(43, par), 0.9913, tol=5e-4)
check("delta_43 (p=0.02)", delta_crit(43, replace(par, p=0.02)), 0.9991,
      tol=5e-4)

print()
n_ok = sum(RESULTS)
print(f"{n_ok}/{len(RESULTS)} checks passed")
sys.exit(0 if n_ok == len(RESULTS) else 1)
