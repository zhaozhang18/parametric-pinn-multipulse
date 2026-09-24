# Model checkpoints

Checkpoint binaries are not included in this Git repository. The required files and
SHA-256 values available from the research workspace are listed in
`REQUIRED_CHECKPOINTS.csv`.

Recommended publication method:

1. Upload the selected checkpoint bundle to a GitHub Release, Zenodo, or an
   institutional repository.
2. Add the permanent URL and archive SHA-256 to the root `README.md`.
3. Preserve the relative paths listed in the manifest, or pass explicit checkpoint
   paths to the command-line interfaces.

Do **not** publish all historical checkpoints; only the final paper models are needed.

Relative paths in `REQUIRED_CHECKPOINTS.csv` use Windows separators (`\`). On POSIX
systems, replace them with `/` when reconstructing the archive layout.


## Code Availability

The source code for the proposed parametric PINN framework is publicly available at:

https://github.com/USERNAME/parametric-pinn-multipulse

Please replace `USERNAME` with the GitHub account name after repository creation.
