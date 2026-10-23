# Numerical-study code
## Layout

```
numerical_study/
  batching_core.py         all solvers (see module docstring)
  journal_style.py       single journal style
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

| Paper label | Content | Generator | Output file|
|---|---|---|---|
| fig:sensitivity | θ*_rn vs. 7 primitives | `fig_sensitivity` | figure_E_sensitivity_primitives.png/.pdf |
| fig:operational | batching trade-off + cycle-length MC | `fig_operational` | figure_G_operational_implications.png/.pdf |
| fig:convergence | three policies + η→0 limit | `fig_convergence` | figure_B_three_policies_comparison.png/.pdf |
| fig:marginal_cost_rs | Δf_rs crossing + stopping gap | `fig_marginal_cost_rs` | figure_FI_risk_premium_discount_panels.png/.pdf |
| fig:robust_pmf | worst-case PMF + likelihood ratio | `fig_robust_pmf` | probability_mass.png/.pdf |
| fig:phase_diagram | exact δ_θ boundaries | `fig_phase_diagram` | figure_C_phase_diagram.png/.pdf |
| tab:renewal | one-shot vs. renewal optima | `tab_renewal` | tables/tab_renewal.tex 
| tab:baseline | sensitivity spot values | `tab_baseline` | tables/tab_baseline.tex
| tab:eta-convergence | θ*_ra as η→0⁻ | `tab_eta_convergence` | tables/tab_eta_convergence.tex
