#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda_base="${CONDA_BASE:-/Users/wanghaifeng/miniconda3}"

source "$conda_base/etc/profile.d/conda.sh"
conda activate pde

cd "$script_dir"

# Pass shared m, Re, final time T, and number of grid points N to all experiments, e.g.
# ./04_run_bursting.sh --m 4 --Re 50 --T 5000 --N 128
# If omitted, 04_run_bursting.py uses the defaults m=3, Re=30, and T=10000.
echo "Running adaptive SDIRK2_mr_SAV experiment..."
python 04_run_bursting.py "$@" \
    --mode adaptive \
    --M SDIRK2_mr_SAV

echo "Running fixed-step SDIRK2_mr_SAV experiment with tau=5e-4..."
python 04_run_bursting.py "$@" \
    --mode fix \
    --M SDIRK2_mr_SAV \
    --tau 5e-4

echo "Running fixed-step SDIRK2_mr_SAV experiment with tau=1e-3..."
python 04_run_bursting.py "$@" \
    --mode fix \
    --M SDIRK2_mr_SAV \
    --tau 1e-3

echo "Both bursting experiments completed."
