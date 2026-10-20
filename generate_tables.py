"""
generate_tables.py -- Table generators for the numerical section.

Covers
------
  tab:renewal          (MISSING from the original scripts -- added here)
  tab:baseline         baseline calibration + one-at-a-time sensitivity
  tab:eta-convergence  robust thresholds as eta -> 0- (appendix)

Each generator prints the LaTeX table body and writes it to
tables/<name>.tex.  Run:  python generate_tables.py
"""

import os
import numpy as np
from dataclasses import replace

from batching_core import (
    Params, BASELINE, fg_recursion, t_recursion, theta_star_rn,
    avg_objective, renewal_objective, theta_star_avg, theta_star_ren,
    robust_vi, simulate_cycles,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables')
os.makedirs(OUT, exist_ok=True)

X = 300  # recursion horizon (all optima lie far below)


# =============================================================================
# tab:renewal -- one-shot vs. renewal optima (the table requested)
# =============================================================================

def tab_renewal(par: Params = BASELINE, n_sim: int = 200_000, seed: int = 7):
    """
    Reproduces Table (tab:renewal):

        Criterion                        One-shot th*   Renewal th*  value      loss
        Average reward (delta=1)         73             35           vbar=4.11  47.6%
        Discounted renewal (delta=0.99)  43             28           vren=317.4 18.3%

    "One-shot theta*" is the ordinary risk-neutral optimum theta*_rn at the
    same delta; "loss" is the relative loss of the renewal objective when
    the one-shot threshold is used instead of the renewal-optimal one.
    Also cross-checks the text values t(43) = 76.1 and t(64) = 155.0 by
    Monte Carlo.
    """
    par_avg = replace(par, delta=1.0)
    par_ren = replace(par, delta=0.99)

    # --- average-reward row -------------------------------------------------
    vbar, N1, t = avg_objective(par, X)
    th_os_avg = theta_star_rn(par_avg, X)          # 73
    th_avg = theta_star_avg(par, X)                # 35
    loss_avg = 100.0 * (vbar[th_avg] - vbar[th_os_avg]) / vbar[th_avg]

    # --- discounted-renewal row ----------------------------------------------
    vren, Nd, g, ok = renewal_objective(par_ren, X)
    th_os_ren = theta_star_rn(par_ren, X)          # 43
    th_ren = theta_star_ren(par_ren, X)            # 28
    loss_ren = 100.0 * (vren[th_ren] - vren[th_os_ren]) / vren[th_ren]

    # --- Monte Carlo cross-check of mean cycle lengths quoted in the text ---
    t43 = t[43]
    t64 = t[64]
    mc43 = simulate_cycles(43, par.p, par.q, n_paths=n_sim, seed=seed).mean()
    mc64 = simulate_cycles(64, par.p, par.q, n_paths=n_sim, seed=seed + 1).mean()

    print("tab:renewal  (one-shot vs. renewal optima)")
    print(f"  average reward (delta=1)   : one-shot {th_os_avg}, renewal {th_avg}, "
          f"vbar = {vbar[th_avg]:.4f}, loss {loss_avg:.1f}%   (paper 73, 35, 4.11, 47.6%)")
    print(f"  discounted renewal (d=.99) : one-shot {th_os_ren}, renewal {th_ren}, "
          f"vren = {vren[th_ren]:.2f}, loss {loss_ren:.1f}%   (paper 43, 28, 317.4, 18.3%)")
    print(f"  E_0[T_43] = {t43:.2f} (MC {mc43:.2f})   (paper 76.1)")
    print(f"  E_0[T_64] = {t64:.2f} (MC {mc64:.2f})   (paper 155.0)")

    body = (
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Criterion & One-shot $\\theta^*$ & Renewal $\\theta^*$ & Renewal value "
        "& Loss at one-shot $\\theta^*$\\\\\n"
        "\\midrule\n"
        f"Average reward ($\\delta=1$)       & {th_os_avg} & {th_avg} & "
        f"$\\bar v={vbar[th_avg]:.2f}$             & ${loss_avg:.1f}\\%$\\\\\n"
        f"Discounted renewal ($\\delta=0.99$)& {th_os_ren} & {th_ren} & "
        f"$v^{{\\mathrm{{ren}}}}={vren[th_ren]:.1f}$ & ${loss_ren:.1f}\\%$\\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    path = os.path.join(OUT, 'tab_renewal.tex')
    with open(path, 'w') as fh:
        fh.write(body)
    print(f"  LaTeX written to {path}")
    return dict(th_os_avg=th_os_avg, th_avg=th_avg, vbar=vbar[th_avg],
                loss_avg=loss_avg, th_os_ren=th_os_ren, th_ren=th_ren,
                vren=vren[th_ren], loss_ren=loss_ren, t43=t43, t64=t64)


# =============================================================================
# tab:baseline -- baseline calibration and one-at-a-time sensitivity
# =============================================================================

def tab_baseline(par: Params = BASELINE):
    """theta*_rn under one-at-a-time parameter perturbations (Table tab:baseline
    spot values: alpha=1000 -> 91, delta=0.9 -> 16, delta=1 -> 73,
    beta=1 -> 53, beta=100 -> 41)."""
    rows = [("baseline", {}, 43),
            ("$\\alpha=1000$", dict(alpha=1000.0), 91),
            ("$\\delta=0.9$", dict(delta=0.9), 16),
            ("$\\delta=1$", dict(delta=1.0), 73),
            ("$\\beta=1$", dict(beta=1.0), 53),
            ("$\\beta=100$", dict(beta=100.0), 41)]
    print("tab:baseline  (sensitivity spot values)")
    out = []
    for name, kw, paper in rows:
        th = theta_star_rn(replace(par, **kw), X)
        flag = "OK" if th == paper else f"MISMATCH (paper {paper})"
        print(f"  {name:16s}: theta*_rn = {th:3d}   [{flag}]")
        out.append((name, th))
    body = ("\\begin{tabular}{lc}\n\\toprule\nCalibration & $\\theta^*_{rn}$\\\\\n"
            "\\midrule\n")
    body += "\n".join(f"{n} & {t}\\\\" for n, t in out)
    body += "\n\\bottomrule\n\\end{tabular}\n"
    path = os.path.join(OUT, 'tab_baseline.tex')
    with open(path, 'w') as fh:
        fh.write(body)
    print(f"  LaTeX written to {path}")
    return out


# =============================================================================
# tab:eta-convergence -- robust thresholds as eta -> 0- (appendix)
# =============================================================================

def tab_eta_convergence(etas=(-0.50, -0.30, -0.20, -0.15, -0.10,
                              -0.08, -0.06, -0.05, -0.01, -0.005)):
    """alpha=10, beta=2, kappa=10, remaining parameters baseline, delta=1;
    theta*_rn = 47.  IMPORTANT: the published row (29, 35, 38, 40, 42, 43,
    44, 44, 46, 47) is produced by the paper-stated JOINT tilt with the
    Bellman KL sign -- the defaults of robust_vi.  (The legacy variant in
    the originally uploaded scripts gives a different row; see README.md.)"""
    par = Params(alpha=10.0, beta=2.0, delta=1.0)
    kappa = 10.0
    th_rn = theta_star_rn(par, X)
    print(f"tab:eta-convergence  (theta*_rn = {th_rn}, paper 47)")
    paper = [-0.50, -0.30, -0.20, -0.15, -0.10, -0.08, -0.06, -0.05, -0.01, -0.005]
    paper_th = [29, 35, 38, 40, 42, 43, 44, 44, 46, 47]
    res, V = [], None
    for e in etas:  # warm start along the grid, most averse first
        V, th, _ = robust_vi(par, e, kappa, N_max=80, V_init=V, maxit=5000)
        res.append(th)
        ref = paper_th[paper.index(e)] if e in paper else None
        tag = "" if ref is None else f"   (paper {ref})"
        print(f"  eta = {e:+.3f}: theta*_ra = {th}{tag}")
    etas_tex = " & ".join(f"${e:g}$" for e in etas)
    ths_tex = " & ".join(str(t) for t in res)
    body = ("\\begin{tabular}{l" + "c" * len(etas) + "}\n\\toprule\n"
            f"$\\eta$ & {etas_tex}\\\\\n\\midrule\n"
            f"$\\theta^*_{{ra}}$ & {ths_tex}\\\\\n"
            f"$\\theta^*_{{rn}}$ & \\multicolumn{{{len(etas)}}}{{c}}{{{th_rn}}}\\\\\n"
            "\\bottomrule\n\\end{tabular}\n")
    path = os.path.join(OUT, 'tab_eta_convergence.tex')
    with open(path, 'w') as fh:
        fh.write(body)
    print(f"  LaTeX written to {path}")
    return res


if __name__ == '__main__':
    print("=" * 72)
    tab_renewal()
    print("=" * 72)
    tab_baseline()
    print("=" * 72)
    tab_eta_convergence()
