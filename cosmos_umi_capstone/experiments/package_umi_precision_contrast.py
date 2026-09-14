"""CLI wrapper for publishing the Task 4 lightweight review bundle."""
from __future__ import annotations

import argparse
import json

try:
    from .analyze_umi_precision_contrast import analyze_run
except ImportError:
    from analyze_umi_precision_contrast import analyze_run


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Package an analyzed UMI precision run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # ``analyze_run`` binds the package to the raw manifest and publishes it
    # atomically without replacement.  A revised package must use a new
    # revision directory; there is no destructive overwrite escape hatch.
    result = analyze_run(args.run_dir, args.output_dir)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0 if result.get("status") == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "write_precision_artifacts"]
