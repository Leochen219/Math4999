"""Read-only independent arithmetic and file-integrity audit of one Task 9 scan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def audit(root: Path) -> dict:
    root = root.resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    status = json.loads((root / "run_status.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "analysis" / "scan_summary.json").read_text(encoding="utf-8"))
    if status["status"] != "COMPLETE" or len(config["plan"]) != status["completed"]:
        raise ValueError("scan incomplete")
    manifest = root / "MANIFEST.sha256"
    verified_files = 0
    for line in manifest.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        target = (root / relative).resolve()
        target.relative_to(root)
        observed = hashlib.sha256(target.read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"manifest mismatch: {relative}")
        verified_files += 1
    base = np.load(root / "base_condition.npy", allow_pickle=False)
    mask = np.load(root / "mask.npy", allow_pickle=False).astype(bool)
    scale = float(np.sqrt(np.mean(base[mask].astype(np.float64) ** 2)))
    directions = {name: np.load(root / f"direction_{name}.npy", allow_pickle=False)
                  for name in config["direction_hashes"]}
    train_ids = sorted(name for name in directions if name.startswith("train_"))
    holdout_ids = sorted(name for name in directions if name.startswith("holdout_"))
    if (len(train_ids), len(holdout_ids)) != (32, 8):
        raise ValueError("expected 32 training and eight held-out directions")

    def sample(sample_id: str, filename: str) -> np.ndarray:
        return np.load(root / "samples" / sample_id / filename, allow_pickle=False)

    result = {"verified_manifest_files": verified_files, "samples": status["completed"], "spaces": {}}
    for space, filename in (("predicted_latent", "predicted_latent.npy"),
                            ("feedback_condition", "next_condition.npy"),
                            ("float_rgb_final", "decoded_final_rgb.npy")):
        columns = []
        for name in train_ids:
            plus_input = sample(f"{name}_plus", "condition.npy").astype(np.float64)
            minus_input = sample(f"{name}_minus", "condition.npy").astype(np.float64)
            actual_step = float(np.mean((plus_input - minus_input)[mask]
                                        * directions[name][mask].astype(np.float64)) / 2)
            if actual_step <= 0 or not np.isclose(actual_step, 0.003 * scale, rtol=1e-3):
                raise ValueError(f"unexpected actual paired step: {name}")
            positive = sample(f"{name}_plus", filename).astype(np.float64)
            negative = sample(f"{name}_minus", filename).astype(np.float64)
            columns.append(((positive - negative) / (2 * actual_step)).reshape(-1))
        matrix = np.column_stack(columns)
        singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
        reported = np.asarray(summary["spaces"][space]["singular_values"], dtype=np.float64)
        if not np.allclose(singular, reported, rtol=1e-10, atol=1e-10):
            raise ValueError(f"independent singular values differ: {space}")
        repeat_rms = float(np.sqrt(np.mean((sample("baseline_post", filename).astype(np.float64)
                                          - sample("baseline_pre", filename).astype(np.float64)) ** 2)))
        if not np.isclose(repeat_rms, summary["spaces"][space]["baseline_noise_rms"],
                          rtol=0, atol=1e-12):
            raise ValueError(f"independent baseline RMS differs: {space}")
        result["spaces"][space] = {"largest_singular_value": float(singular[0]),
                                    "repeat_rms": repeat_rms,
                                    "k95": summary["spaces"][space]["k95"]}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.scan_dir), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
