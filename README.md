# Predicting and Inverting Nonlinear Multi-Pulse Propagation in Optical Fibers with a Single Parametric Physics-Informed Neural Network

Official implementation for the manuscript "Predicting and Inverting Nonlinear Multi-Pulse Propagation in Optical Fibers with a Single Parametric Physics-Informed Neural Network", including AP-PINN, MP-PINN, and β₂-MP-PINN models for nonlinear multi-pulse propagation in optical fibers.

## What is included

- **AP-PINNs** for prescribed pulse numbers and continuous-amplitude inversion;
- **MP-PINN** with a fixed sparse eight-slot representation for `K=1,...,8`;
- **beta2-MP-PINN** conditioned on normalized second-order dispersion;
- frozen-forward inverse reconstruction with candidate pulse-number search and BIC;
- CNN and DDNN baselines;
- propagation-distance extrapolation and fine-tuning experiments;
- scripts for manuscript Figs. 2--9;
- deterministic split CSVs and compact formal result summaries.

## Repository layout

```text
scripts/                 Clean, stable public entry points
src/implementation/      Exact selected experiment implementations and dependencies
data/                     Canonical deduplicated train/test combination lists
results/paper_tables/     Manuscript Tables I--VI in CSV form
results/formal_summaries/ Small final evaluation summaries used to audit the tables
checkpoints/              Required-checkpoint manifest (binaries released separately)
docs/                     Reproducibility, provenance, and release notes
tools/                    Cross-platform validation utilities
paper_figures/            Default destination for regenerated figures
```

## Installation

Python 3.8 or newer is recommended. Create an isolated environment:

```powershell
conda create -n parametric-pinn python=3.8 -y
conda activate parametric-pinn
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CUDA build, install the PyTorch wheel appropriate for your CUDA driver before
installing the remaining packages.

Validate the release:

```powershell
python tools/verify_release.py
```

## Clean entry points

Every public wrapper forwards all command-line arguments to the exact implementation
script recorded in `docs/IMPLEMENTATION_MAP.csv`.

```powershell
# AP-PINN
python scripts/ap_pinn/run_pipeline.py --help
python scripts/ap_pinn/evaluate_forward.py --help
python scripts/ap_pinn/invert.py --help

# MP-PINN
python scripts/mp_pinn/train.py --help
python scripts/mp_pinn/evaluate_forward.py --help
python scripts/mp_pinn/invert.py --help

# beta2-MP-PINN
python scripts/beta2_mp_pinn/train.py --help
python scripts/beta2_mp_pinn/continue_training.py --help
python scripts/beta2_mp_pinn/evaluate_unseen_amplitudes.py --help
python scripts/beta2_mp_pinn/evaluate_seen_amplitudes.py --help
python scripts/beta2_mp_pinn/invert.py --help

# Baselines and distance extension
python scripts/baselines/train_evaluate_forward.py --help
python scripts/baselines/train_inverse_cnn.py --help
python scripts/baselines/evaluate_inverse.py --help
python scripts/distance/run_budget_experiment.py --help
python scripts/distance/evaluate_budgets.py --help
```

Run commands from the repository root. Pass explicit output, run, dataset, and
checkpoint paths rather than relying on machine-specific defaults.

## Data and checkpoints

The Git repository includes the deterministic amplitude-combination lists, but not
SSFM field tensors, per-sample metric streams, or model binaries. See:

- [`data/README.md`](data/README.md)
- [`checkpoints/REQUIRED_CHECKPOINTS.csv`](checkpoints/REQUIRED_CHECKPOINTS.csv)
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)

**Checkpoint/data archive:** `TO_BE_ADDED_BEFORE_PUBLICATION`

After the archive is uploaded, replace the placeholder above with a permanent URL and
SHA-256 checksum.

## Reproducing manuscript outputs

Compact published values are provided under [`results/paper_tables`](results/paper_tables).
The exact code mapping and required assets are listed in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

Figure wrappers:

```powershell
python scripts/figures/figure2.py --help
python scripts/figures/figure3.py --help
python scripts/figures/figure4.py --help
python scripts/figures/figure5.py --help
python scripts/figures/figure6.py --help
python scripts/figures/figure7.py --help
python scripts/figures/figure8.py --help
python scripts/figures/figure9.py --help
```

Figures 4--9 require final checkpoints and/or per-sample evaluation records from the
separate archival asset.

## Terminology note

Some implementation filenames, command-line options, and saved CSV columns retain the
historical symbol `D` for backward compatibility. It denotes the same dimensionless
normalized second-order dispersion parameter written as `beta2_tilde` in the
manuscript. The model name is beta2-MP-PINN.

## Citation

Use [`CITATION.cff`](CITATION.cff). Replace the repository URL, DOI, and publication
metadata after the article and repository are public.

## License

This project is released under the MIT License; see [LICENSE](LICENSE).

## Maintenance status

This repository is released for reproducibility of the associated manuscript and is
not actively maintained. Issues and pull requests may not receive responses.


## Code Availability

The source code for the proposed parametric PINN framework is publicly available at:

https://github.com/USERNAME/parametric-pinn-multipulse

Please replace `USERNAME` with the GitHub account name after repository creation.
