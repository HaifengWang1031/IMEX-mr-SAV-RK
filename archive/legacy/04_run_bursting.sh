#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda_base="${CONDA_BASE:-/Users/wanghaifeng/miniconda3}"

source "$conda_base/etc/profile.d/conda.sh"
conda activate pde

cd "$script_dir"

# Pass shared m, Re, final time T, and number of grid points N to all experiments, e.g.
# ./04_run_bursting.sh --m 4 --Re 50 --T 5000 --N 256
# Shared settings come from experiments/bursting/config.json, then these CLI options.
# --rerun creates new attempts; otherwise compatible completed results are reused.
# Each of the three cases prints its own run directory and persistent log path.
echo "Running adaptive SDIRK2_mr_SAV experiment..."
python -u 04_run_bursting.py "$@" \
    --mode adaptive \
    --M SDIRK2_mr_SAV

echo "Running fixed-step SDIRK2_mr_SAV experiment with tau=5e-4..."
python -u 04_run_bursting.py "$@" \
    --mode fix \
    --M SDIRK2_mr_SAV \
    --tau 5e-4

echo "Running fixed-step SDIRK2_mr_SAV experiment with tau=1e-3..."
python -u 04_run_bursting.py "$@" \
    --mode fix \
    --M SDIRK2_mr_SAV \
    --tau 1e-3

echo "All three bursting cases completed or reused."
