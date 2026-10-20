# Unified numerical-study code

Companion code for *Optimal Batching with Abandonment: Threshold Policies
Under Risk-Neutral and Risk-Sensitive Criteria*.  This package consolidates
the five scattered scripts (`generate_all_figures.py`,
`generate_new_figures.py`, `generate_figure_G_operational.py`,
`generate_figure_C_enriched.py`, `worst_case_measure.py`) into one
NumPy/SciPy-only codebase (the numba dependency is dropped), fixes the
errors found in them, and adds the table generators that were missing —
including **tab:renewal**.

## Layout

```
numerical_study/
  batching_core.py         all solvers (see module docstring)
  journal_style.py       single journal style (Operations Research
                         conventions: strictly grayscale, curves told apart
                         by gray level + dash pattern + open marker, light
                         gray region shading, plain black annotations,
                         thin spines, frameless legends, plain "(a)" panels)
  generate_figures.py      one function per figure in the paper
  generate_tables.py       one function per table in the paper
  verify_paper_numbers.py  regression checks of every quoted number
  figures/                 generated figures land here
  tables/                  generated LaTeX table bodies land here
```

Quick start:

```bash
python verify_paper_numbers.py fast   # ~30 s; full mode ~10 min
python generate_tables.py             # renewal + baseline + eta-convergence
python generate_figures.py sensitivity operational      # pick any subset
python generate_figures.py            # all six (phase_diagram is slow)
```

## Figure / table mapping

| Paper label | Content | Generator | Output file | Replaces |
|---|---|---|---|---|
| fig:sensitivity | θ*_rn vs. 7 primitives | `fig_sensitivity` | figure_E_sensitivity_primitives.png/.pdf | `generate_figure_E` |
| fig:operational | batching trade-off + cycle-length MC | `fig_operational` | figure_G_operational_implications.png/.pdf | `generate_figure_G_operational.py` (clean, ported) |
| fig:convergence | three policies + η→0 limit | `fig_convergence` | figure_B_three_policies_comparison.png/.pdf | `generate_figure_B` (**rewritten**, see D1/D4) |
| fig:marginal_cost_rs | Δf_rs crossing + stopping gap | `fig_marginal_cost_rs` | figure_FI_risk_premium_discount_panels.png/.pdf | `generate_figure_FI` (ported; see D2/D3) |
| fig:robust_pmf | worst-case PMF + likelihood ratio | `fig_robust_pmf` | probability_mass.png/.pdf | `worst_case_measure.py` (**rewritten**, see D5) |
| fig:phase_diagram | exact δ_θ boundaries | `fig_phase_diagram` | figure_C_phase_diagram.png/.pdf | `generate_figure_C_enriched.py` (ported); old `generate_figure_C` **dropped** (D6) |
| tab:renewal | one-shot vs. renewal optima | `tab_renewal` | tables/tab_renewal.tex | **was missing — added** |
| tab:baseline | sensitivity spot values | `tab_baseline` | tables/tab_baseline.tex | **was missing — added** |
| tab:eta-convergence | θ*_ra as η→0⁻ | `tab_eta_convergence` | tables/tab_eta_convergence.tex | **was missing — added** |

All figure/table code the paper uses is now present **except** the
discounted risk-sensitive appendix: `tab:discounted-rs` and the
app:disc-numerics validation (Methods A/B; quoted values V_rs(0) =
658.930146, θ*_rs = 116) had no code in the uploads and are **not**
implemented here — see "Still missing" below.

## Which uploaded code was wrong — discrepancy notes

- **D1 — wrong calibration in the old fig:convergence.**
  `generate_figure_B` used α=0, β=2, q=0.9 (baseline is α=100, β=10,
  q=0.8) and computed the risk-seeking curve with `theta_rs_fast`, which
  hard-codes η=1. The rewrite uses the baseline calibration at δ=1 with
  explicit η (D4).

- **D2 — two different robust solvers behind two published tables.**
  The paper's Algorithm (robust-vi) states a *joint* truncated-Gibbs tilt
  over (a, d) with value E_Q[φ] − (1/η) D_KL.  The uploaded scripts
  instead (i) average the arrival atom d under the nominal q *before*
  tilting (a marginal tilt over a) and (ii) evaluate the KL term with the
  opposite sign.  Consequences, reproduced numerically here:
  - the **main-text robust row** θ*_ra = 51, 42, 27, 14 (η = −0.05, −0.10,
    −0.20, −0.50, κ=5) comes from the scripts' legacy variant;
  - the **η-convergence table** (29, 35, 38, 40, 42, 43, 44, 44, 46, 47)
    comes from the paper-stated joint variant;
  - the two are inconsistent: the stated algorithm gives 61 (not 51) in
    the first main-text cell.
  `robust_vi(tilt=..., kl_sign=...)` implements both; the default is the
  paper-stated `tilt='joint', kl_sign='bellman'`.  **Action needed:**
  either regenerate the main-text row with the stated algorithm (and
  update fig:marginal_cost_rs panel (b), which also used the legacy
  solver) or amend the algorithm statement.

- **D3 — boundary condition.**  All uploaded robust-VI codes clamp the
  arrival successor at `min(N_max, x−a+1)`; the Algorithm states the
  imposed boundary V(N_max+1) = β(N_max+1) − α.  `robust_vi` uses the
  imposed boundary (verified to leave every published threshold
  unchanged).

- **D4 — fig:convergence caption is underspecified.**  The caption
  documents only κ=10 and the convergence abscissa p=0.02.  The rewrite
  uses baseline parameters at δ=1, η_rs=0.10, η_ra=−0.30 (module-level
  constants `ETA_RS_PANEL`, `ETA_RA_PANEL` in `generate_figures.py`) —
  confirm these against the paper text and adjust.

- **D5 — fig:robust_pmf.**  Three fixes: (i) the paper includes
  `probability_mass.png` but the script saved
  `figure_robust_pmf_and_likelihood_ratio_v3.png` — the generator now
  writes `probability_mass.png/.pdf`; (ii) the script solved a
  marginal-only tilt over a, while the paper defines the marginal worst
  case by averaging the *joint* truncated tilt over the arrival atom d —
  `worst_case_joint` now implements the paper's definition; (iii) the
  calibration α=10, β=4, q=0.9 (and N_max) is not documented in the paper,
  which states only x=30, p=0.04, η=−0.3 — kept from the script for
  comparability, but should be added to the caption or text.

- **D6 — fake phase boundary dropped.**  The old `generate_figure_C`
  drew the boundary from a hand-crafted logistic curve
  (`0.55 + 0.35/(1+exp(-0.3*(beta-12)))`).  It is replaced by the exact
  bisection on Φ(θ;δ) = (θ+1)/g(θ+1) − θ/g(θ) = 0 (already in
  `generate_figure_C_enriched.py`; verified: δ_1 ≈ 0.558, δ_43 =
  0.9865/0.9913/0.9991 for p = 0.005/0.01/0.02).

- **D7 — obsolete material (not ported).**  Figures A (heuristic scaled
  curves), D, F, G (risk-neutral limit), H (continuous time), I
  (superseded by F+I); `theta_rs_fast` / `theta_rs_eta_fast` (implicit
  η=1, inexact truncation for risk-seeking); all numba kernels.

- **D8 — g-stability (paper remark).**  At δ=1 the g recursion admits
  g≡1 but loses accuracy beyond θ≈150; `fg_recursion` substitutes the
  theoretical g≡1 when δ=1.  The f recursions share the same 1/((1−p)^x q)
  amplification — every recursion returns an `ok` mask marking the
  reliable prefix, and all paper thresholds lie well inside it.

## Still missing (no code in the uploads, not implemented here)

1. **tab:discounted-rs** and the app:disc-numerics validation
   ("Methods A/B") for the *discounted* risk-sensitive criterion
   (quoted: V_rs(0) = 658.930146, θ*_rs = 116).  Needs a discounted
   entropic value iteration plus the two independent validation methods —
   substantial; flagged for the authors.
2. The paper's robust main-text table under the **stated** algorithm
   (once D2 is resolved, `tab_*`/`fig_marginal_cost_rs` can be rerun with
   `tilt='joint', kl_sign='bellman'`).

## Verification

`verify_paper_numbers.py` reproduces every number quoted in the text:
V_rn(0)=137.9232, θ*_rn ∈ {43, 91, 16, 73, 53, 41}, N(43)≈295,
E[δ^{T_43}]≈0.47, t(43)=76.1, t(64)=155.0 (plus a Monte Carlo
cross-check), the full tab:renewal row (35/4.11/47.6%, 28/317.4/18.3%),
θ*_rs ∈ {92, 116, 141, 195}, both robust rows (legacy 51/42/27/14 and
joint 29/35/…/47), and the phase boundaries (δ_1≈0.56, δ_43 row).
Full mode additionally runs the slow robust-VI rows; `fast` skips them.
