#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$script_dir/04_run_bursting.sh" --m 2 --Re 20 --T 10000 --N 64
"$script_dir/04_run_bursting.sh" --m 2 --Re 30 --T 10000 --N 64
"$script_dir/04_run_bursting.sh" --m 3 --Re 30 --T 10000 --N 64
"$script_dir/04_run_bursting.sh" --m 3 --Re 40 --T 10000 --N 64
"$script_dir/04_run_bursting.sh" --m 4 --Re 40 --T 10000 --N 64
"$script_dir/04_run_bursting.sh" --m 4 --Re 50 --T 10000 --N 64
