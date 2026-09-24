"""Inventory legacy artifacts in place; never load numerical arrays or infer parameters."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
LEGACY_DIRS = ("data", "fig", "figures", "logs", "output", "tmp")
GENERATED_SUFFIXES = {".json", ".npz", ".h5", ".hdf5", ".tex", ".pdf", ".png"}


def collect(root):
    """Return metadata only. A source directory or filename is not provenance proof."""
    paths = set()
    for name in LEGACY_DIRS:
        directory = root / name
        if directory.is_symlink():
            continue
        for parent, dirs, files in os.walk(directory, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not (Path(parent) / d).is_symlink())
            paths.update(Path(parent) / name for name in files if name != ".DS_Store")
    paths.update(p for p in root.iterdir() if p.suffix in GENERATED_SUFFIXES)
    # This existing experiment stores generated output alongside source/config files.
    # Index candidates as unclassified; do not label all JSON or Markdown as output.
    transition = root / "archive/legacy"
    for parent, dirs, files in os.walk(transition, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not (Path(parent) / d).is_symlink())
        paths.update(Path(parent) / name for name in files
                     if Path(name).suffix in GENERATED_SUFFIXES | {".md"})
    records = []
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file():
            continue
        stat = path.stat()
        records.append({"path": path.relative_to(root).as_posix(),
                        "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    counts = Counter(p["path"].split("/")[0] for p in records)
    return {
        "schema_version": 1,
        "indexed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": list(LEGACY_DIRS) + ["root artifact candidates", "archive/legacy metadata candidates"],
        "interpretation": "Metadata snapshot only, not a content hash or completion check. Experiment attribution is in docs/experiment-inventory.md. Parameters and completion are unknown unless verified separately. mtime is not computation time. Symlinks are excluded.",
        "file_count": len(records), "total_bytes": sum(p["size_bytes"] for p in records),
        "counts_by_top_level": dict(sorted(counts.items())), "files": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "archive/legacy-index.json")
    args = parser.parse_args()
    inventory = collect(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Only the regenerable index is replaced; none of the listed files are written.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                     prefix=".legacy-index-", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(inventory, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    try:
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Indexed {inventory['file_count']} files ({inventory['total_bytes']} bytes): {args.output}")


if __name__ == "__main__":
    main()
