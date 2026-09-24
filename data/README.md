# Data included in the repository

This repository contains only deterministic train/test **combination lists** used by
the paper. Full SSFM propagation tensors are generated on demand by the scripts and
are not stored in Git.

- `ap_pinn/M2`--`M8`: seen and unseen PAM4 combinations for each prescribed pulse number.
- `mp_pinn`: the canonical sparse-eight-slot split shared by the MP-PINN,
  beta2-MP-PINN, and propagation-distance experiments.

The beta2-conditioned and distance-extension experiments reuse the same amplitude
split, so duplicate copies from individual experiment folders were removed. This
reduces the public data directory from 273 duplicated CSV files to 32 canonical files
without changing any split contents.


## Code Availability

The source code for the proposed parametric PINN framework is publicly available at:

https://github.com/USERNAME/parametric-pinn-multipulse

Please replace `USERNAME` with the GitHub account name after repository creation.
