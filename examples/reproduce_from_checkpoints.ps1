# Run from the repository root after downloading checkpoints.
$Device = "cuda"
$RunsRoot = "D:\path\to\downloaded_runs"

# MP-PINN forward evaluation
python scripts/mp_pinn/evaluate_forward.py `
  --runs-root $RunsRoot `
  --pinn-checkpoint "$RunsRoot\universal_sparse8_K1to8_highK_try1\sparse8_forward_pinn.pt" `
  --device $Device `
  --summary-only

# MP-PINN inverse interface
python scripts/mp_pinn/invert.py --help

# beta2-MP-PINN forward and inverse interfaces
python scripts/beta2_mp_pinn/evaluate_unseen_amplitudes.py --help
python scripts/beta2_mp_pinn/invert.py --help
