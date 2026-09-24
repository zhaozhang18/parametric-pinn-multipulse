# Release validation

The release candidate was validated as follows:

- all Python files under `src/` and `scripts/` passed `compileall`;
- all 27 stable public wrappers returned successfully for `--help`;
- the cross-platform credential scan in `tools/verify_release.py` found no obvious secrets;
- the Figure 2 wrapper completed a real smoke run and generated both expected PNG files;
- the original Windows release verifier was also reported to pass under Python 3.8.20.

Full numerical training/evaluation was not rerun during packaging because the final
model checkpoints and per-sample field archives are intentionally distributed
separately. Run reduced smoke tests after those assets are uploaded, as listed in
`RELEASE_CHECKLIST.md`.
