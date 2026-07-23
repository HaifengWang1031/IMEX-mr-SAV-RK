#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda_base="${CONDA_BASE:-/Users/wanghaifeng/miniconda3}"

source "$conda_base/etc/profile.d/conda.sh"
conda activate pde

cd "$script_dir"

echo "Running adaptive SDIRK2_mr_SAV experiment..."
python 04_run_bursting.py "$@" \
    --mode adaptive \
    --M SDIRK2_mr_SAV \
    --m 3 \
    --Re 30

echo "Running fixed-step SDIRK2_mr_SAV experiment with tau=5e-4..."
python 04_run_bursting.py "$@" \
    --mode fix \
    --M SDIRK2_mr_SAV \
    --tau 5e-4 \
    --m 3 \
    --Re 30

echo "Both bursting experiments completed."