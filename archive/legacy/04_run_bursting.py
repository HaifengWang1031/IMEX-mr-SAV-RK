"""Backward-compatible command entry; see experiments/bursting/README.md."""
from experiments.bursting.run import main


if __name__ == "__main__":
    raise SystemExit(main())
