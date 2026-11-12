"""Invoke the unchanged central level-0 conversion function in a subprocess."""
from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--central-scripts", required=True, type=Path)
    parser.add_argument("--pedestals", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpus", required=True, type=int)
    args = parser.parse_args()
    if args.cpus < 1:
        parser.error("--cpus must be positive")
    sys.path.insert(0, str(args.central_scripts))
    import PrepareLevel0CalibParams as central

    with args.pedestals.open(encoding="utf-8") as handle:
        modules = json.load(handle)
    if not isinstance(modules, dict) or not modules:
        raise ValueError("Central pedestal JSON is empty or invalid")
    tasks = [(name, {"ped": str(args.pedestals)}, False) for name in modules]
    # Same central conversion as its CLI; typed CPU count avoids its argparse bug.
    with multiprocessing.get_context("spawn").Pool(args.cpus) as pool:
        payload = dict(pool.map(central.buildLevel0CalibParams, tasks))
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    central.JS.saveAsJson(str(args.output), payload)


if __name__ == "__main__":
    main()
