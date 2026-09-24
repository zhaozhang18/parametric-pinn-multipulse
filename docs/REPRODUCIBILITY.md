# Reproducibility guide

## Scope

The repository provides the final implementation entry points, deterministic
amplitude-combination splits, formal summary outputs, and clean command wrappers.
Exact full reproduction additionally requires the final checkpoint/data archive
listed under `checkpoints/`.

## Paper-to-code map

| Paper item | Public entry point(s) | Included audit output |
|---|---|---|
| Table I: AP-PINNs | `scripts/ap_pinn/run_pipeline.py`, `evaluate_forward.py`, `invert.py` | `results/formal_summaries/ap_pinn/` |
| Table II: MP-PINN/CNN/DDNN | `scripts/mp_pinn/*`, `scripts/baselines/*` | `results/paper_tables/table2_model_comparison.csv` |
| Table III: distance extrapolation | `scripts/distance/compare_ranges.py` | `results/paper_tables/table3_distance_extrapolation.csv` |
| Table IV: extension budgets | `scripts/distance/run_budget_experiment.py`, `evaluate_budgets.py` | `results/paper_tables/table4_range_extension.csv` |
| Table V: beta2 forward | `scripts/beta2_mp_pinn/evaluate_*.py` | `results/formal_summaries/beta2/forward_*` |
| Table VI: beta2 inverse | `scripts/beta2_mp_pinn/invert.py` | `results/formal_summaries/beta2/inverse/` |
| Figs. 2--9 | `scripts/figures/figure2.py` ... `figure9.py` | per-sample archive required for Figs. 4--9 |

## Validation levels

1. **Syntax validation**: `python tools/verify_release.py`.
2. **Interface validation**: run each public script with `--help`.
3. **Smoke test**: run a reduced sample/grid configuration for each model family.
4. **Formal reproduction**: use final checkpoints and full evaluation settings.

The code uses the historical variable/CSV name `D` in some implementation files for
compatibility with saved experiments. In the manuscript, the same dimensionless
quantity is denoted by the normalized second-order dispersion parameter
`beta2_tilde`; the model name remains beta2-MP-PINN.
