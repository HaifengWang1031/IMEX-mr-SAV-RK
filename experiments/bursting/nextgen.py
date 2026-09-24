"""Compatibility entry for the unified nextgen experiment workflow."""
from experiments.common.workflow import main
if __name__ == "__main__":
    raise SystemExit(main("bursting"))
