"""
generate_figures.py -- Unified figure generators for the numerical section.

One function per figure INCLUDED IN THE PAPER (see README.md for the full
mapping, incl. which uploaded scripts were superseded or dropped):

  fig:sensitivity       figure_E_sensitivity_primitives.png
  fig:operational       figure_G_operational_implications.pdf/.png
  fig:convergence       figure_B_three_policies_comparison.png
  fig:marginal_cost_rs  figure_FI_risk_premium_discount_panels.png
  fig:robust_pmf        probability_mass.png   (name used by the paper)
  fig:phase_diagram     figure_C_phase_diagram.pdf/.png

All computation goes through batching_core (pure NumPy/SciPy, no numba);
all styling goes through journal_style (single OR/MS style).

Run everything:   python generate_figures.py
Run one figure:   python generate_figures.py sensitivity
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import replace

from batching_core import (Params, BASELINE, fg_recursion, theta_star_rn,
                           frs_recursion, theta_star_rs, robust_vi,
                           worst_case_joint, simulate_cycles, t_recursion,
                           delta_crit)
from journal_style import (apply_style, panel_label, open_marker,
                           GSCALE, DASH, DASHES, MARKERS,
                           K_BLACK, K_DARK, K_MED, K_LIGHT,
                           FILL_HI, FILL_LO, BAR_FACE, BAR_EDGE)

apply_style()

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)


def _save(fig, stem):
    png = os.path.join(FIGDIR, stem + '.png')
    pdf = os.path.join(FIGDIR, stem + '.pdf')
    fig.savefig(png, bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  saved {png}\n  saved {pdf}")


# =============================================================================
# fig:sensitivity  (was generate_figure_E in generate_new_figures.py)
# =============================================================================

def fig_sensitivity(par: Params = BASELINE, N_max: int = 150):
    """2x4 panel: theta*_rn vs. alpha, beta, c, gamma, p, q, delta
    (grids exactly as in Table tab:baseline)."""
    print("fig:sensitivity -- computing ...")
    panels = [
        ('alpha', np.logspace(1, 3, 21),     r'Setup Cost, $\alpha$'),
        ('beta',  np.logspace(0, 2, 21),     r'Marginal Reward, $\beta$'),
        ('c',     np.logspace(-3, -1, 21),   r'Holding Cost, $c$'),
        ('gamma', np.logspace(-1, 1, 21),    r'Abandonment Penalty, $\gamma$'),
        ('p',     np.logspace(-3, -1, 21),   r'Abandonment Probability, $p$'),
        ('q',     np.linspace(0.5, 1.0, 21), r'Arrival Probability, $q$'),
        ('delta', np.linspace(0.9, 1.0, 21), r'Discount Factor, $\delta$'),
    ]
    results = {}
    for name, grid, _ in panels:
        results[name] = np.array(
            [theta_star_rn(replace(par, **{name: float(v)}), N_max)
             for v in grid], dtype=float)
        print(f"  {name}: theta* in [{results[name].min():.0f}, "
              f"{results[name].max():.0f}]")

    fig, axes = plt.subplots(2, 4, figsize=(13, 5.2))
    axes = axes.flatten()
    log_scaled = {'alpha', 'beta', 'c', 'gamma', 'p'}
    for ax, (name, grid, xlabel) in zip(axes, panels):
        thetas = results[name]
        ax.plot(grid, thetas, color=K_BLACK, linewidth=1.1, marker='o',
                markersize=3, markerfacecolor='white', markeredgewidth=0.8)
        base_val = getattr(par, name)
        ax.axvline(x=base_val, color=K_MED, linestyle=DASH['dotted'],
                   linewidth=0.9)
        ax.plot(base_val, thetas[np.argmin(np.abs(grid - base_val))],
                'o', color=K_BLACK, markersize=5, markerfacecolor='white',
                markeredgewidth=1.1, zorder=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$\theta^*_{rn}$')
        if name in log_scaled:
            ax.set_xscale('log')
        ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[-1].axis('off')
    fig.tight_layout()
    _save(fig, 'figure_E_sensitivity_primitives')


# =============================================================================
# fig:operational  (was generate_figure_G_operational.py -- clean, ported)
# =============================================================================

def fig_operational(par: Params = BASELINE, n_sim: int = 100_000, seed: int = 42):
    """(a) v_rn(theta) = N(theta) * E[delta^{T_theta}];
    (b) Monte Carlo distribution of the cycle length T_theta."""
    print("fig:operational -- computing ...")
    N_max = 200
    f, g, ok = fg_recursion(par, N_max)
    ts = theta_star_rn(par, N_max)
    v_star = (par.beta * ts - par.alpha - f[ts]) / g[ts]
    print(f"  theta*_rn = {ts} (paper 43), v_rn(theta*) = {v_star:.2f} "
          f"(paper 137.92)")

    theta_axis = np.arange(0, 83)   # beyond ~80, f grows geometrically
    N_nominal = par.beta * theta_axis - par.alpha - f[theta_axis]
    E_discount = 1.0 / g[theta_axis]
    v_rn = N_nominal * E_discount

    thetas = {'low': ts // 2, 'base': ts, 'high': ts + ts // 2}
    cycles = {tag: simulate_cycles(th, par.p, par.q, n_sim, seed)
              for tag, th in thetas.items()}
    for tag, th in thetas.items():
        m, s = cycles[tag].mean(), cycles[tag].std()
        print(f"  theta = {th}: E[T] = {m:.1f}, SD = {s:.1f}, CV = {s/m:.3f}")

    LS_MAIN = dict(color=K_BLACK, linestyle=DASH['solid'], linewidth=1.2)
    LS_NOM = dict(color=K_DARK, linestyle=DASH['dashed'], linewidth=1.1)
    LS_DISC = dict(color=K_MED, linestyle=DASH['dashdot'], linewidth=1.1)

    fig = plt.figure(figsize=(10, 4.2))
    gs = GridSpec(1, 2, figure=fig, wspace=0.32,
                  left=0.075, right=0.925, top=0.87, bottom=0.15)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # ---- panel (a) ----
    ax1.axhline(y=0, color='0.6', linewidth=0.7, zorder=1)
    ax1.plot(theta_axis, N_nominal, zorder=3, **LS_NOM,
             label=r'$N(\theta) = \beta\theta - \alpha - f(\theta)$')
    ax1.plot(theta_axis, v_rn, zorder=4, **LS_MAIN,
             label=r'$v_{rn}(\theta) = N(\theta)\,\mathbb{E}[\delta^{T_\theta}]$')
    ax1b = ax1.twinx()
    ax1b.plot(theta_axis, E_discount, zorder=2, **LS_DISC,
              label=r'$\mathbb{E}[\delta^{T_\theta}] = 1/g(\theta)$')
    ax1b.set_ylabel(r'Expected discount factor, $\mathbb{E}[\delta^{T_\theta}]$',
                    fontsize=9.5, color=K_MED)
    ax1b.set_ylim(0, 1.05)
    ax1b.spines['top'].set_visible(False)
    ax1b.tick_params(axis='y', colors=K_MED, direction='out')
    ax1.axvline(x=ts, color=K_MED, linestyle=DASH['dotted'], linewidth=0.9,
                ymax=0.90, zorder=1)
    ax1.plot(ts, v_rn[ts], 'o', color=K_BLACK, markersize=4.5,
             markerfacecolor='white', markeredgewidth=1.1, zorder=6)
    ax1.annotate(r'$\theta^*_{rn} = %d$' % ts,
                 xy=(ts, v_rn[ts]), xytext=(ts + 7, v_rn[ts] + 95),
                 fontsize=9, color=K_BLACK,
                 arrowprops=dict(arrowstyle='->', color=K_BLACK, lw=0.7))
    ax1.text(0.97, 0.965,
             r'$v_{rn}(\theta) = (\beta\theta - \alpha - f(\theta)) \cdot'
             r' \mathbb{E}[\delta^{T_\theta}]$'
             r'  =  (accrual) $\times$ (erosion)',
             transform=ax1.transAxes, fontsize=8, ha='right', va='top')
    ax1.set_xlabel(r'Batching threshold, $\theta$')
    ax1.set_ylabel('Value')
    panel_label(ax1, '(a) The batching trade-off: accrual vs. discount '
                     'erosion')
    ax1.set_xlim(0, 82)
    ax1.set_ylim(-600, 550)
    ax1.spines['top'].set_visible(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='lower left',
               bbox_to_anchor=(0.015, 0.02), handlelength=3.0,
               borderaxespad=0.3)

    # ---- panel (b) ----
    t_max = max(c.max() for c in cycles.values())
    bins = np.arange(0, t_max + 6, 3)
    m_b, s_b = cycles['base'].mean(), cycles['base'].std()
    ax2.hist(cycles['base'], bins=bins, density=True, color=BAR_FACE,
             edgecolor=BAR_EDGE, linewidth=0.5, zorder=1,
             label=rf'$\theta = {thetas["base"]}$ ($\theta^*$): '
                   rf'$\mathbb{{E}}[T]={m_b:.0f}$, CV$={s_b/m_b:.2f}$')
    m_l, s_l = cycles['low'].mean(), cycles['low'].std()
    ax2.hist(cycles['low'], bins=bins, density=True, histtype='step',
             color=K_BLACK, linewidth=1.2, linestyle=DASH['solid'], zorder=3,
             label=rf'$\theta = {thetas["low"]}$: '
                   rf'$\mathbb{{E}}[T]={m_l:.0f}$, CV$={s_l/m_l:.2f}$')
    m_h, s_h = cycles['high'].mean(), cycles['high'].std()
    ax2.hist(cycles['high'], bins=bins, density=True, histtype='step',
             color=K_DARK, linewidth=1.2, linestyle=DASH['dashed'], zorder=3,
             label=rf'$\theta = {thetas["high"]}$: '
                   rf'$\mathbb{{E}}[T]={m_h:.0f}$, CV$={s_h/m_h:.2f}$')
    ax2.text(0.97, 0.96,
             'Longer accumulation:\nlonger and more\nvariable cycles',
             transform=ax2.transAxes, fontsize=8, ha='right', va='top')
    ax2.set_xlabel(r'Batching cycle length, $T_\theta$ (periods)')
    ax2.set_ylabel('Density')
    panel_label(ax2, r'(b) Distribution of the batching cycle $T_\theta$ '
                     r'($10^5$ simulated paths)')
    ax2.set_xlim(0, t_max)
    ax2.legend(loc='upper right', handlelength=3.0,
               bbox_to_anchor=(1.0, 0.74), borderaxespad=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    _save(fig, 'figure_G_operational_implications')


# =============================================================================
# fig:convergence  (was generate_figure_B in generate_all_figures.py --
#                   REWRITTEN: the uploaded version used alpha=0, beta=2,
#                   q=0.9 and an implicit eta_rs=1 solver, none of which is
#                   documented in the paper)
# =============================================================================

# Calibration note (README, D4): the paper caption for fig:convergence
# documents only kappa = 10 and the convergence abscissa p = 0.02.  We use
# the baseline calibration at delta = 1 (the criterion shared by all three
# policies in this figure) and the representative risk parameters below;
# adjust ETA_RS / ETA_RA here once the paper text states them.
ETA_RS_PANEL = 0.10    # risk-seeking eta for panel (a)
ETA_RA_PANEL = -0.30   # risk-averse eta for panel (a)
KAPPA_CONV = 10.0      # documented in the caption
P_CONV = 0.02          # documented in the caption


def fig_convergence(par: Params = BASELINE):
    """(a) theta*_rn, theta*_rs, theta*_ra vs. abandonment probability p;
    (b) convergence of theta*_rs(eta), theta*_ra(eta) to theta*_rn as
    eta -> 0 at fixed p = 0.02."""
    par = replace(par, delta=1.0)
    print("fig:convergence -- computing (this is the slow figure: "
          "one robust VI per grid point) ...")

    # ---- panel (a): thresholds vs. p --------------------------------------
    p_values = np.logspace(-2, -1, 12)
    th_rn, th_rs, th_ra = [], [], []
    V_warm = None
    for pv in p_values:
        par_p = replace(par, p=float(pv))
        t_rn = theta_star_rn(par_p, 300)
        t_rs = theta_star_rs(par_p, ETA_RS_PANEL, 400)
        V_warm, t_ra, _ = robust_vi(par_p, ETA_RA_PANEL, KAPPA_CONV,
                                    N_max=t_rn + 1, V_init=None, maxit=3000)
        th_rn.append(t_rn)
        th_rs.append(t_rs)
        th_ra.append(t_ra)
        print(f"  p = {pv:.3f}: rn {t_rn}, rs {t_rs}, ra {t_ra}")
    th_rn, th_rs, th_ra = map(np.array, (th_rn, th_rs, th_ra))

    # ---- panel (b): convergence in eta at p = 0.02 -------------------------
    par_c = replace(par, p=P_CONV)
    th_rn_c = theta_star_rn(par_c, 300)
    eta_pos = np.logspace(np.log10(0.01), np.log10(0.25), 6)
    eta_neg = -np.logspace(np.log10(0.01), np.log10(0.5), 6)
    th_rs_eta = [theta_star_rs(par_c, float(e), 400) for e in eta_pos]
    th_ra_eta, V = [], None
    for e in sorted(eta_neg):       # warm start, most averse first
        V, t, _ = robust_vi(par_c, float(e), KAPPA_CONV, N_max=th_rn_c + 1,
                            V_init=V, maxit=4000)
        th_ra_eta.append(t)
    th_ra_eta = th_ra_eta[::-1]     # back to ascending |eta| order of eta_neg
    print(f"  convergence at p = {P_CONV}: theta*_rn = {th_rn_c}")

    # ---- draw ---------------------------------------------------------------
    fig, (ax, ax_eta) = plt.subplots(1, 2, figsize=(11, 4.2),
                                     gridspec_kw={'width_ratios': [1.1, 1.0]})

    ax.plot(p_values, th_rn, color=K_BLACK, linestyle=DASH['solid'],
            linewidth=1.3,
            label=r'$\theta^*_{rn}$ (risk-neutral)')
    ax.plot(p_values, th_rs, color=K_DARK, linestyle=DASH['dashed'],
            linewidth=1.1,
            label=rf'$\theta^*_{{rs}}$ (risk-seeking, $\eta={ETA_RS_PANEL}$)')
    ax.plot(p_values, th_ra, color=K_MED, linestyle=DASH['dashdot'],
            linewidth=1.1,
            label=rf'$\theta^*_{{ra}}$ (risk-averse, $\eta={ETA_RA_PANEL}$, '
                  rf'$\kappa={KAPPA_CONV:g}$)')
    ax.set_xscale('log')
    ax.set_xlabel(r'Abandonment probability, $p$')
    ax.set_ylabel(r'Optimal threshold, $\theta^*$')
    panel_label(ax, '(a) Three optimal policies')
    ax.legend(loc='upper left', handlelength=3.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax_eta.axhline(y=th_rn_c, color=K_BLACK, linewidth=1.2,
                   label=rf'$\theta^*_{{rn}} = {th_rn_c}$')
    ax_eta.plot(eta_pos, th_rs_eta, color=K_DARK, linestyle=DASH['dashed'],
                linewidth=1.1, marker='o', markersize=3.5,
                markerfacecolor='white', markeredgewidth=0.9,
                label=r'$\theta^*_{rs}(\eta)$, $\eta > 0$')
    ax_eta.plot(-eta_neg, th_ra_eta, color=K_MED,
                linestyle=DASH['dashdot'], linewidth=1.1, marker='s',
                markersize=3.5, markerfacecolor='white', markeredgewidth=0.9,
                label=rf'$\theta^*_{{ra}}(\eta)$, $\eta < 0$, '
                      rf'$\kappa={KAPPA_CONV:g}$')
    ax_eta.set_xscale('log')
    ax_eta.set_xlabel(r'$|\eta|$')
    ax_eta.set_ylabel(r'Optimal threshold, $\theta^*$')
    panel_label(ax_eta, rf'(b) Risk-neutral limit as $\eta \to 0$ '
                        rf'($p={P_CONV}$)')
    ax_eta.legend(loc='center right', handlelength=3.0)
    ax_eta.spines['top'].set_visible(False)
    ax_eta.spines['right'].set_visible(False)

    fig.tight_layout()
    _save(fig, 'figure_B_three_policies_comparison')


# =============================================================================
# fig:marginal_cost_rs  (was generate_figure_FI in generate_new_figures.py)
# =============================================================================

def _rn_gap_exact(par: Params, th: int):
    """Exact risk-neutral value function on {0..th} at delta = 1 under the
    threshold policy stopping at th: solves the (th x th) Poisson system
    (as in the uploaded generate_figure_FI).  Returns (V_rn, gap)."""
    c, gam, p, q = par.c, par.gamma, par.p, par.q
    alpha, beta = par.alpha, par.beta
    C = c + gam * p   # delta = 1
    from scipy.special import gammaln as _gl
    M = np.zeros((th, th))
    bvec = np.zeros(th)
    for x in range(th):
        M[x, x] = 1.0
        bvec[x] = -C * x
        a = np.arange(x + 1)
        w = np.exp(_gl(x + 1) - _gl(a + 1) - _gl(x - a + 1)
                   + a * np.log(p) + (x - a) * np.log(1 - p))
        for branch, prb in ((1, q), (0, 1 - q)):
            ns = x - a + branch
            over = ns >= th
            bvec[x] += (w[over] * prb).sum() * (beta * th - alpha)
            M[x, ns[~over]] -= (w[~over] * prb)
    V_rn = np.concatenate([np.linalg.solve(M, bvec), [beta * th - alpha]])
    gap = V_rn - (beta * np.arange(th + 1) - alpha)
    return V_rn, gap


def fig_marginal_cost_rs(par: Params = BASELINE, N_max_F: int = 220,
                         ra_variant: dict = dict(tilt='marginal',
                                                 kl_sign='legacy')):
    """(a) Delta f_rs(theta; eta) crossing beta (risk-seeking premium);
    (b) closing of the stopping gap V_ra(x) - (beta x - alpha).

    NOTE (README, D2): the PUBLISHED panel (b) was computed with the legacy
    solver of the uploaded scripts (marginal tilt + legacy KL sign), so the
    default `ra_variant` reproduces it.  Pass
    `ra_variant=dict(tilt='joint', kl_sign='bellman')` for the paper-stated
    algorithm instead -- the qualitative picture is identical but the
    eta = -0.05 threshold moves (51 -> 61)."""
    par = replace(par, delta=1.0)
    alpha, beta = par.alpha, par.beta
    print("fig:marginal_cost_rs -- computing ...")

    th = theta_star_rn(par, N_max_F)          # 73 at baseline, delta = 1
    f_rn, _, _ = fg_recursion(par, N_max_F + 1)
    _, gap_rn = _rn_gap_exact(par, th)
    print(f"  theta*_rn (delta=1) = {th} (paper 73)")

    eta_rs_values = [0.05, 0.1, 0.15, 0.25]
    rs_curves, crossings = {}, {}
    for eta in eta_rs_values:
        f_rs, _, _ = frs_recursion(par, eta, N_max_F + 1)
        rs_curves[eta] = np.diff(f_rs[:N_max_F + 2])
        crossings[eta] = theta_star_rs(par, eta, N_max_F + 100)
        print(f"  eta={eta}: theta*_rs = {crossings[eta]}")

    kappa = 5.0
    eta_ra_values = [-0.05, -0.1, -0.2, -0.5]
    gaps, thetas_ra = {}, {}
    V = None
    for eta in sorted(eta_ra_values):      # warm start: most averse first
        V, t_ra, _ = robust_vi(par, eta, kappa, N_max=th, V_init=V,
                               maxit=6000, **ra_variant)
        gaps[eta] = V[:th + 1] - (beta * np.arange(th + 1) - alpha)
        thetas_ra[eta] = t_ra
        print(f"  eta={eta}: theta*_ra = {t_ra}")

    # ---- draw ---------------------------------------------------------------
    mkrs = MARKERS
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    xF = np.arange(0, N_max_F + 1)
    axL.plot(xF, np.diff(f_rn[:N_max_F + 2]), color=K_BLACK,
             linestyle=DASH['solid'], linewidth=1.3,
             label=r'$\Delta f$ (risk-neutral)')
    for k, eta in enumerate(eta_rs_values):
        axL.plot(xF, rs_curves[eta], linewidth=1.0, markevery=8,
                 color=GSCALE[(k + 1) % 4], linestyle=DASHES[(k + 1) % 4],
                 marker=MARKERS[(k + 1) % 4], markersize=3,
                 markerfacecolor='white', markeredgewidth=0.8,
                 label=rf'$\Delta f_{{rs}}$, $\eta = {eta}$')
    axL.axhline(y=beta, color=K_MED, linestyle=DASH['solid'], linewidth=0.9)
    axL.text(3, beta * 0.95, r'$\beta$ (marginal reward)',
             fontsize=8, ha='left', va='top', color=K_MED)
    axL.plot(th, beta, 'o', color=K_BLACK, markersize=4.5,
             markerfacecolor='white', markeredgewidth=1.1, zorder=5)
    axL.annotate(r'$\theta^*_{rn}$', xy=(th, beta - 0.1), xytext=(th, 0.4),
                 fontsize=9, color=K_BLACK, ha='center',
                 arrowprops=dict(arrowstyle='->', color=K_BLACK, lw=0.7))
    for k, eta in enumerate(eta_rs_values):
        t = crossings[eta]
        axL.plot(t, beta, marker=MARKERS[(k + 1) % 4], color=GSCALE[(k + 1) % 4],
                 markersize=4.5, zorder=5,
                 markerfacecolor='white', markeredgewidth=1.0, linestyle='none')
        axL.annotate(rf'$\theta^*_{{rs}}({eta})$', xy=(t, beta - 0.1),
                     xytext=(t, 0.4), fontsize=8, color=GSCALE[(k + 1) % 4],
                     ha='center',
                     arrowprops=dict(arrowstyle='->',
                                     color=GSCALE[(k + 1) % 4], lw=0.7))
    axL.text(0.02, 0.61,
             r'$\Delta f_{rs}(\theta) \leq \Delta f(\theta)$:' + '\n' +
             r'risk seeking deflates marginal costs,' + '\n' +
             r'so the crossing with $\beta$ shifts right:' + '\n' +
             r'$\theta^*_{rs}(\eta) \geq \theta^*_{rn}$, increasing' + '\n' +
             r'in $\eta$.',
             transform=axL.transAxes, fontsize=7.5, ha='left', va='top')
    axL.set_xlabel(r'Threshold, $\theta$')
    axL.set_ylabel(r'Marginal Cost, $\Delta f(\theta)$')
    panel_label(axL, '(a) Risk-seeking premium: marginal-cost crossing')
    axL.set_xlim(0, N_max_F)
    axL.set_ylim(0, beta * 1.3)
    axL.legend(loc='upper left', handlelength=2.6)
    axL.spines['top'].set_visible(False)
    axL.spines['right'].set_visible(False)

    axR.plot(np.arange(0, th + 1), gap_rn, color=K_BLACK,
             linestyle=DASH['solid'], linewidth=1.3,
             label=r'Risk-Neutral, $\eta \to 0$')
    for k, eta in enumerate(eta_ra_values):
        t_end = thetas_ra[eta] + 1
        axR.plot(np.arange(0, t_end + 1), gaps[eta][:t_end + 1],
                 linewidth=1.0, markevery=6,
                 color=GSCALE[(k + 1) % 4], linestyle=DASHES[(k + 1) % 4],
                 marker=MARKERS[(k + 1) % 4], markersize=3,
                 markerfacecolor='white', markeredgewidth=0.8,
                 label=rf'$\eta = {eta}$, $\kappa = {kappa}$')
    axR.axhline(y=0, color=K_MED, linestyle=DASH['solid'], linewidth=0.9)
    axR.plot(th, 0, 'o', color=K_BLACK, markersize=4.5,
             markerfacecolor='white', markeredgewidth=1.1, zorder=5)
    axR.annotate(r'$\theta^*_{rn}$', xy=(th, 6), xytext=(th, -52),
                 fontsize=9, color=K_BLACK, ha='center',
                 arrowprops=dict(arrowstyle='->', color=K_BLACK, lw=0.7))
    for k, eta in enumerate(eta_ra_values):
        t = thetas_ra[eta]
        axR.plot(t, 0, marker=MARKERS[(k + 1) % 4], color=GSCALE[(k + 1) % 4],
                 markersize=4.5, zorder=5,
                 markerfacecolor='white', markeredgewidth=1.0, linestyle='none')
        axR.annotate(rf'$\theta^*_{{ra}}({eta})$', xy=(t, 6), xytext=(t, -52),
                     fontsize=8, color=GSCALE[(k + 1) % 4], ha='center',
                     arrowprops=dict(arrowstyle='->',
                                     color=GSCALE[(k + 1) % 4], lw=0.7))
    axR.text(0.97, 0.62,
             r'No scalar recursion exists in the robust case:' + '\n' +
             r'the threshold is the first zero of the stopping gap' + '\n' +
             r'$V_{ra}(x) - (\beta x - \alpha)$. Stronger aversion' + '\n' +
             r'closes the gap earlier: $\theta^*_{ra}(\eta) \leq '
             r'\theta^*_{rn}$,' + '\n' +
             r'decreasing in $|\eta|$.',
             transform=axR.transAxes, fontsize=7.5, ha='right', va='top')
    axR.set_xlabel(r'Buffer State, $x$')
    axR.set_ylabel(r'Stopping Gap, $V(x) - (\beta x - \alpha)$')
    panel_label(axR, '(b) Risk-averse discount: closing of the stopping gap')
    axR.set_xlim(0, th + 17)
    axR.set_ylim(-70, gap_rn[0] * 1.12)
    axR.legend(loc='upper right', handlelength=2.6)
    axR.spines['top'].set_visible(False)
    axR.spines['right'].set_visible(False)

    fig.tight_layout()
    _save(fig, 'figure_FI_risk_premium_discount_panels')


# =============================================================================
# fig:robust_pmf  (was worst_case_measure.py -- REWRITTEN:
#   (i)  saved under the filename the paper includes: probability_mass.png;
#   (ii) the worst-case measure is now computed as the paper states: the
#        JOINT truncated-Gibbs tilt over (a, d), marginalised over the
#        arrival atom d -- the uploaded script solved a marginal-only tilt,
#        see README, D5;
#   (iii) imposed boundary V[N_max+1] = beta (N_max+1) - alpha instead of
#        clamping.
#   Calibration alpha=10, beta=4, q=0.9 (undocumented in the paper, which
#   states only x = 30, p = 0.04, eta = -0.3 -- see README, D5) is kept
#   from the uploaded script so the figure remains comparable.)
# =============================================================================

def fig_robust_pmf():
    par = Params(alpha=10.0, beta=4.0, c=0.01, gamma=0.5, p=0.04, q=0.9,
                 delta=1.0)
    eta, x_state, N_max = -0.3, 30, 100
    kappa_values = [1.5, 3.0, 5.0]
    a_max_display = 8
    print("fig:robust_pmf -- computing ...")

    V_dict, wc = {}, {}
    for kappa in kappa_values:
        V_dict[kappa], th, _ = robust_vi(par, eta, kappa, N_max=N_max,
                                         maxit=6000)
        wc[kappa] = worst_case_joint(x_state, V_dict[kappa], par, eta, kappa)
        print(f"  kappa = {kappa}: theta*_ra = {th}, "
              f"sum Q_A = {wc[kappa]['Q_A'].sum():.6f}")

    a_plot = np.arange(0, a_max_display + 1)
    pmf_nominal = wc[kappa_values[0]]['P_A'][:a_max_display + 1]
    kstyle = {1.5: (K_DARK, DASH['dashed'], 'o'),
              3.0: (K_MED, DASH['dashdot'], 's'),
              5.0: (K_BLACK, DASH['dotted'], '^')}

    fig = plt.figure(figsize=(10, 4.0))
    gs = GridSpec(1, 2, figure=fig, wspace=0.30,
                  left=0.075, right=0.98, top=0.90, bottom=0.15)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.bar(a_plot, pmf_nominal, width=0.55, color=BAR_FACE,
            edgecolor=BAR_EDGE, linewidth=0.6, zorder=1,
            label=r'Nominal $\mathbb{P}_{x}^{(A)}$')
    for kappa in kappa_values:
        col, ls, mk = kstyle[kappa]
        ax1.plot(a_plot, wc[kappa]['Q_A'][:a_max_display + 1],
                 color=col, linewidth=1.1, marker=mk,
                 markersize=4, linestyle=ls, zorder=3,
                 markerfacecolor='white', markeredgewidth=0.9,
                 label=rf'Robust $\mathbb{{Q}}^*_{{x}}$, $\kappa={kappa}$')
    ax1.set_xlabel(r'Number of Abandonments, $a$')
    ax1.set_ylabel('Probability Mass')
    panel_label(ax1, '(a) Worst-case probability mass function')
    ax1.set_xlim(-0.6, a_max_display + 0.6)
    ax1.set_xticks(a_plot)
    ax1.set_ylim(bottom=-0.005)
    ax1.legend(loc='upper right', handlelength=2.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    for kappa in kappa_values:
        col, ls, mk = kstyle[kappa]
        ax2.plot(a_plot, wc[kappa]['LR'][:a_max_display + 1],
                 color=col, linewidth=1.1, marker=mk,
                 markersize=4, linestyle=ls, zorder=3,
                 markerfacecolor='white', markeredgewidth=0.9,
                 label=rf'$\kappa = {kappa}$')
    ax2.axhline(y=1.0, color=K_MED, linestyle=DASH['dotted'], linewidth=0.9)
    ax2.set_xlabel(r'Number of Abandonments, $a$')
    ax2.set_ylabel(r'Likelihood Ratio $\mathbb{Q}^*_{x} / \mathbb{P}_{x}^{(A)}$')
    panel_label(ax2, '(b) Likelihood ratio')
    ax2.set_xlim(-0.6, a_max_display + 0.6)
    ax2.set_xticks(a_plot)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper right', handlelength=2.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    _save(fig, 'probability_mass')


# =============================================================================
# fig:phase_diagram  (was generate_figure_C_enriched.py -- clean, ported;
#                     replaces the FAKE hand-drawn boundary of the old
#                     generate_figure_C, see README, D6)
# =============================================================================

def _sign_field(beta_grid, delta_grid, par: Params, N_max: int):
    """Z[i, j] = Phi(theta*(beta_j, delta_i); delta_i)."""
    from batching_core import binom_matrix
    W = binom_matrix(N_max, par.p)   # shared across all delta rows
    Z = np.full((len(delta_grid), len(beta_grid)), np.nan)
    for i, d in enumerate(delta_grid):
        par_d = replace(par, delta=float(d))
        f, g, ok = fg_recursion(par_d, N_max, W=W)
        for j, b in enumerate(beta_grid):
            prev = -par.alpha   # v(0) = (beta*0 - alpha - f(0)) / g(0)
            t = N_max
            for x in range(1, N_max + 1):
                if not ok[x]:
                    t = x - 1
                    break
                vx = (b * x - par.alpha - f[x]) / g[x]
                if vx <= prev:
                    t = x - 1
                    break
                prev = vx
            if 1 <= t and t + 1 <= N_max and ok[t + 1]:
                Z[i, j] = (t + 1) / g[t + 1] - t / g[t]
    return Z


def _extract_boundary(beta_grid, delta_grid, Z, smooth_window=17):
    from scipy.signal import savgol_filter
    bd = np.full(len(beta_grid), np.nan)
    for j in range(len(beta_grid)):
        col = Z[:, j]
        for i in range(len(delta_grid) - 1):
            if (np.isfinite(col[i]) and np.isfinite(col[i + 1])
                    and col[i] < 0 <= col[i + 1]):
                bd[j] = (delta_grid[i] - col[i]
                         * (delta_grid[i + 1] - delta_grid[i])
                         / (col[i + 1] - col[i]))
    valid = np.isfinite(bd)
    if valid.sum() >= smooth_window:
        bd_s = bd.copy()
        bd_s[valid] = savgol_filter(bd[valid], smooth_window, 2)
        bd = bd_s
    return bd


def fig_phase_diagram(par: Params = BASELINE):
    print("fig:phase_diagram -- computing (sign field on a (beta, delta) "
          "grid; a few minutes) ...")
    ts_base = theta_star_rn(par, 400)

    # panel (a): exact delta_theta curves, varying p
    theta_axis = np.arange(1, 71)
    p_specs = [(0.005, 'p5'), (par.p, 'base'), (0.02, 'p20')]
    dt_curves = {}
    for p_val, tag in p_specs:
        dt_curves[tag] = np.array([
            delta_crit(int(t), replace(par, p=p_val)) for t in theta_axis])
        print(f"  delta_theta curve done: p = {p_val}")

    # panel (b): (beta, delta) boundaries, varying q
    beta_grid = np.linspace(2, 25, 80)
    delta_grid = np.linspace(0.985, 0.9999, 300)
    q_specs = [(0.6, 'q6'), (par.q, 'base'), (1.0, 'q10')]
    bd_curves = {}
    for q_val, tag in q_specs:
        Z = _sign_field(beta_grid, delta_grid, replace(par, q=q_val), 500)
        bd_curves[tag] = _extract_boundary(beta_grid, delta_grid, Z)
        print(f"  (beta, delta) boundary done: q = {q_val}")

    LS_BASE = dict(color=K_BLACK, linestyle=DASH['solid'], linewidth=1.3)
    LS_VAR1 = dict(color=K_DARK, linestyle=DASH['dashed'], linewidth=1.1)
    LS_VAR2 = dict(color=K_MED, linestyle=DASH['dashdot'], linewidth=1.1)
    LS_VAR3 = dict(color=K_MED, linestyle=DASH['dotted'], linewidth=1.1)

    fig = plt.figure(figsize=(10, 4.2))
    gs = GridSpec(1, 2, figure=fig, wspace=0.28,
                  left=0.075, right=0.98, top=0.87, bottom=0.15)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    def t(d):
        return 1.0 - d

    y_top, y_bot = 3e-4, 0.55
    ax1.fill_between(theta_axis, t(dt_curves['base']), y_top,
                     color=FILL_HI, edgecolor='none')
    ax1.fill_between(theta_axis, y_bot, t(dt_curves['base']),
                     color=FILL_LO, edgecolor='none')
    ax1.plot(theta_axis, t(dt_curves['p5']), **LS_VAR1, label=r'$p = 0.005$')
    ax1.plot(theta_axis, t(dt_curves['base']), **LS_BASE,
             label=r'$p = 0.01$ (baseline)')
    ax1.plot(theta_axis, t(dt_curves['p20']), **LS_VAR2, label=r'$p = 0.02$')
    ax1.axvline(x=ts_base, color=K_MED, linestyle=DASH['dotted'],
                linewidth=0.9)
    ax1.axhline(y=t(par.delta), color=K_MED, linestyle=DASH['dotted'],
                linewidth=0.9)
    ax1.plot(ts_base, t(par.delta), 'o', color=K_BLACK, markersize=4.5,
             markerfacecolor='white', markeredgewidth=1.1, zorder=5)
    ax1.annotate(r'baseline $(\theta^*_{rn}, \delta)$',
                 xy=(ts_base, t(par.delta)), xytext=(ts_base + 4, t(0.974)),
                 fontsize=8.5, color=K_BLACK,
                 arrowprops=dict(arrowstyle='->', color=K_BLACK, lw=0.7))
    ax1.text(0.2, 0.935,
             'Sign-positive regime\n' + r'$\partial\theta^*/\partial\beta > 0$',
             transform=ax1.transAxes, fontsize=9, ha='center', va='center',
             color=K_BLACK)
    ax1.text(0.42, 0.25,
             'Sign-negative regime\n' + r'$\partial\theta^*/\partial\beta < 0$',
             transform=ax1.transAxes, fontsize=9, ha='center', va='center',
             color=K_BLACK)
    ax1.set_yscale('log')
    delta_ticks = [0.55, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
    ax1.set_yticks([t(d) for d in delta_ticks])
    ax1.set_yticklabels([f'{d:g}' for d in delta_ticks])
    ax1.set_xlabel(r'Threshold, $\theta$')
    ax1.set_ylabel(r'Discount factor, $\delta$  (log scale in $1-\delta$)')
    panel_label(ax1, r'(a) Exact critical boundary $\delta_\theta$ '
                     r'($q = 0.8$; varying $p$)')
    ax1.set_xlim(1, 70)
    ax1.set_ylim(y_bot, y_top)
    ax1.legend(loc='lower right', handlelength=3.0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    d_lo, d_hi = 0.988, 1.0
    ax2.fill_between(beta_grid, bd_curves['base'], d_hi,
                     color=FILL_HI, edgecolor='none')
    ax2.fill_between(beta_grid, d_lo, bd_curves['base'],
                     color=FILL_LO, edgecolor='none')
    ax2.plot(beta_grid, bd_curves['q6'], **LS_VAR1, label=r'$q = 0.6$')
    ax2.plot(beta_grid, bd_curves['base'], **LS_BASE,
             label=r'$q = 0.8$ (baseline)')
    ax2.plot(beta_grid, bd_curves['q10'], **LS_VAR3, label=r'$q = 1.0$')
    ax2.axvline(x=par.beta, color=K_MED, linestyle=DASH['dotted'],
                linewidth=0.9)
    ax2.axhline(y=par.delta, color=K_MED, linestyle=DASH['dotted'],
                linewidth=0.9)
    ax2.plot(par.beta, par.delta, 'o', color=K_BLACK, markersize=4.5,
             markerfacecolor='white', markeredgewidth=1.1, zorder=5)
    ax2.annotate(r'baseline $(\beta, \delta)$',
                 xy=(par.beta, par.delta),
                 xytext=(par.beta + 1.6, par.delta + 0.0011),
                 fontsize=8.5, color=K_BLACK,
                 arrowprops=dict(arrowstyle='->', color=K_BLACK, lw=0.7))
    ax2.text(0.62, 0.82,
             'Sign-positive regime\n' + r'$\partial\theta^*/\partial\beta > 0$',
             transform=ax2.transAxes, fontsize=9, ha='center', va='center',
             color=K_BLACK)
    ax2.text(0.78, 0.10,
             'Sign-negative regime\n' + r'$\partial\theta^*/\partial\beta < 0$',
             transform=ax2.transAxes, fontsize=9, ha='center', va='center',
             color=K_BLACK)
    ax2.set_xlabel(r'Unit reward, $\beta$')
    ax2.set_ylabel(r'Discount factor, $\delta$')
    panel_label(ax2, r'(b) Reward monotonicity phase diagram '
                     r'($p = 0.01$; varying $q$)')
    ax2.set_xlim(2, 25)
    ax2.set_ylim(d_lo, d_hi)
    ax2.legend(loc='upper left', handlelength=3.0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    _save(fig, 'figure_C_phase_diagram')


# =============================================================================
# Dispatch
# =============================================================================

ALL = dict(sensitivity=fig_sensitivity,
           operational=fig_operational,
           convergence=fig_convergence,
           marginal_cost_rs=fig_marginal_cost_rs,
           robust_pmf=fig_robust_pmf,
           phase_diagram=fig_phase_diagram)

if __name__ == '__main__':
    which = sys.argv[1:] or list(ALL)
    for name in which:
        print("=" * 72)
        ALL[name]()
    print("=" * 72)
    print("All requested figures generated.")
