"""Validate and collate six Task 9 spectra without pooling response matrices."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collate_scans(scan_dirs: Sequence[str | Path], output_dir: str | Path,
                  *, expected_scenes: int = 3, expected_seeds: tuple[int, ...] = (0, 1)) -> dict:
    if len(scan_dirs) != expected_scenes * len(expected_seeds):
        raise ValueError("the complete preregistered scene-by-seed matrix is required")
    entries: dict[tuple[int, int], tuple[dict, dict]] = {}
    common: dict | None = None
    scene_bindings: dict[int, dict] = {}
    for raw_root in scan_dirs:
        root = Path(raw_root)
        config = _read_json(root / "config.json")
        status = _read_json(root / "run_status.json")
        summary = _read_json(root / "analysis" / "scan_summary.json")
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
        identity_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if status.get("status") != "COMPLETE" or status.get("identity_hash") != identity_hash:
            raise ValueError(f"incomplete or identity-mismatched scan: {root}")
        if status.get("completed") != len(config["plan"]):
            raise ValueError(f"incomplete call plan: {root}")
        if summary.get("status") != "COMPLETE" or summary.get("scene_identity") != config["identity"]:
            raise ValueError(f"stale or mismatched per-stratum analysis: {root}")
        scene = config["identity"]["scene"]
        scene_id = int(scene["record_index"])
        seed = int(config["seed"])
        if seed not in expected_seeds or summary.get("seed") != seed or (scene_id, seed) in entries:
            raise ValueError("invalid seed or duplicate scene/seed stratum")
        stable = {
            "mask_hash": config["mask_hash"],
            "direction_hashes": config["direction_hashes"],
            "plan": config["plan"],
            "direction_seed": config["identity"]["direction_seed"],
            "model": config["identity"]["checkpoint"]["content_manifest_sha256"],
            "vae": config["identity"]["vae"]["sha256"],
            "framework_commit": config["identity"]["framework_commit"],
            "contract": config["identity"]["task8_contract"],
        }
        if common is None:
            common = stable
        elif stable != common:
            raise ValueError("scene/seed strata do not share the preregistered model, directions, or plan")
        binding = {"base_hash": config["base_hash"],
                   "prompt": config["identity"]["prompt"],
                   "action_chunk0": config["identity"]["action_chunk0"],
                   "scene": scene,
                   "data_sha256": config["identity"]["data"]["sha256"]}
        if scene_id in scene_bindings and scene_bindings[scene_id] != binding:
            raise ValueError("same-scene seed pair has differing input, action, or prompt")
        scene_bindings[scene_id] = binding
        entries[(scene_id, seed)] = (config, summary)
    if len(scene_bindings) != expected_scenes or any((scene_id, seed) not in entries
                                                   for scene_id in scene_bindings for seed in expected_seeds):
        raise ValueError("missing scene or seed stratum")
    rows: list[dict] = []
    for (scene_id, seed), (_config, summary) in sorted(entries.items()):
        for space, result in summary["spaces"].items():
            k95 = result["k95"]
            residuals = result["heldout_relative_residual_by_k"]
            values = [] if k95 is None or residuals is None else [
                value for value in residuals[k95 - 1] if value is not None]
            rows.append({
                "record_index": scene_id, "seed": seed, "space": space,
                "interpretation": result["interpretation"],
                "half_step_pass_count": result["half_step_pass_count"],
                "half_step_total": result["half_step_total"],
                "observed_rank": result["observed_rank"],
                "k90": result["k90"], "k95": k95, "k99": result["k99"],
                "effective_rank": result["effective_rank"],
                "unresolved_energy_fraction": result["unresolved_energy_fraction"],
                "max_heldout_residual_at_k95": max(values) if values else None,
            })
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("matrix summary output directory must be new and empty")
    output.mkdir(parents=True, exist_ok=True)
    result = {"status": "COMPLETE", "strata": len(entries), "scenes": sorted(scene_bindings),
              "seeds": list(expected_seeds), "matrix_pooled": False,
              "scope": "distinct observed Bridge episodes, fixed action chunk within each scene",
              "rows": rows}
    (output / "matrix_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True,
                                                         allow_nan=False) + "\n", encoding="utf-8")
    with (output / "stratum_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Task 9: cross-scene and cross-seed response spectra", "",
             "Six spectra were validated separately; no response matrices were pooled.", "",
             "| Episode record | Seed | Space | Half-step pass | k95 | Effective rank | Max held-out residual at k95 |",
             "|---:|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['record_index']} | {row['seed']} | {row['space']} | "
                     f"{row['half_step_pass_count']}/{row['half_step_total']} | {row['k95']} | "
                     f"{row['effective_rank']} | {row['max_heldout_residual_at_k95']} |")
    lines += ["", "Ranks describe only the 32 sampled training directions per stratum. "
              "Distinct episodes and two noise seeds are replication conditions, not a population estimate.", ""]
    (output / "matrix_report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, action="append", required=True,
                        help="Repeat exactly six times, one completed scene/seed stratum each")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = collate_scans(args.scan_dir, args.output_dir)
    print(json.dumps({"status": result["status"], "strata": result["strata"],
                      "matrix_pooled": result["matrix_pooled"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
