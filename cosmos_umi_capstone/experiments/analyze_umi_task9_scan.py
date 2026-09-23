"""Recompute one completed Task 9 stratum from saved float tensors.

This command never loads a Cosmos model or changes an in-progress run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .umi_task9_spectrum import analyze_stratum
except ImportError:
    from umi_task9_spectrum import analyze_stratum


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = analyze_stratum(args.scan_dir)
    compact = {
        "status": result["status"],
        "scene": result["scene_identity"]["scene"],
        "seed": result["seed"],
        "spaces": {
            name: {
                "interpretation": value["interpretation"],
                "half_step_pass_count": value["half_step_pass_count"],
                "half_step_total": value["half_step_total"],
                "k90": value["k90"], "k95": value["k95"], "k99": value["k99"],
                "effective_rank": value["effective_rank"],
            }
            for name, value in result["spaces"].items()
        },
    }
    print(json.dumps(compact, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
