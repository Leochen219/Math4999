"""Independent, mmap-based audit of Task 10 endpoint RGB results.

This deliberately does not import the experiment runner or analyzer. It reads
the atomic .npy files directly and recomputes the preregistered endpoint metric.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

RECORDS = (1, 2, 3, 6, 7, 14)
PAIRS = ((0, 1), (2, 3))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(difference * difference)))


def verify(root: Path) -> dict:
    rows = {}
    with (root / "analysis" / "stratum_metrics.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows[(int(row["record_index"]), tuple(json.loads(row["seed_pair"])))] = row
    if set(rows) != {(index, pair) for index in RECORDS for pair in PAIRS}:
        raise ValueError("metric table does not contain exactly the 12 preregistered strata")
    audited = []
    largest_metric_gap = 0.0
    for index in RECORDS:
        input_file = root / "preflight" / f"record_{index:02d}" / "inputs.npz"
        with np.load(input_file, allow_pickle=False) as payload:
            truth16 = np.asarray(payload["rgb_float32"][16], dtype=np.float32)
            truth32 = np.asarray(payload["rgb_float32"][32], dtype=np.float32)
            actions = np.asarray(payload["actions_normalized"], dtype=np.float32).reshape(2, 16, 10)
        for pair in PAIRS:
            formal = root / "records" / f"record_{index:02d}" / f"seeds_{pair[0]}_{pair[1]}" / "formal"
            samples = formal / "samples"
            metadata = {}
            for call in ("G0", "G0_repeat", "TF2", "AR2"):
                folder = samples / call
                status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
                if status.get("status") != "success":
                    raise ValueError(f"{index}/{pair}/{call} is not an atomic success")
                meta = json.loads((folder / "record.json").read_text(encoding="utf-8"))
                expected_seed = pair[0] if call.startswith("G0") else pair[1]
                noise = meta["generation"]["noise_evidence"]
                if (noise["seed"] != expected_seed or noise["prepare_seed"] != expected_seed
                        or meta["generation"]["sampler_generator_seeds"] != [expected_seed] * 30):
                    raise ValueError(f"{index}/{pair}/{call} consumed the wrong seed")
                action = np.load(folder / "record_action.npy", mmap_mode="r")
                if not np.array_equal(action, actions[0 if call.startswith("G0") else 1]):
                    raise ValueError(f"{index}/{pair}/{call} action does not match preflight")
                metadata[call] = meta
            if (sha256(samples / "G0" / "record_output_full.npy") !=
                    sha256(samples / "G0_repeat" / "record_output_full.npy")):
                raise ValueError(f"{index}/{pair} G0 repeat latent differs")
            if (sha256(samples / "G0" / "record_decoded_rgb_full.npy") !=
                    sha256(samples / "G0_repeat" / "record_decoded_rgb_full.npy")):
                raise ValueError(f"{index}/{pair} G0 repeat RGB differs")
            if metadata["TF2"]["prediction_noise_hash"] != metadata["AR2"]["prediction_noise_hash"]:
                raise ValueError(f"{index}/{pair} second-chunk noise is unpaired")
            g0_encoded = np.load(samples / "G0" / "record_encoded_condition.npy", mmap_mode="r")
            ar_input = np.load(samples / "AR2" / "record_condition_input_fp32.npy", mmap_mode="r")
            if not np.array_equal(g0_encoded, ar_input):
                raise ValueError(f"{index}/{pair} AR2 did not consume the G0 float feedback latent")
            g0 = np.load(samples / "G0" / "record_generated_rgb.npy", mmap_mode="r")
            tf = np.load(samples / "TF2" / "record_generated_rgb.npy", mmap_mode="r")
            ar = np.load(samples / "AR2" / "record_generated_rgb.npy", mmap_mode="r")
            if any(array.shape != (3, 16, 256, 256) or array.dtype != np.float32 for array in (g0, tf, ar)):
                raise ValueError(f"{index}/{pair} generated RGB shape/dtype mismatch")
            values = {"g0_rmse": rmse(g0[:, -1], truth16),
                      "tf_rmse": rmse(tf[:, -1], truth32),
                      "ar_rmse": rmse(ar[:, -1], truth32)}
            values["delta_feedback_rmse"] = values["ar_rmse"] - values["tf_rmse"]
            for name, value in values.items():
                gap = abs(value - float(rows[(index, pair)][name]))
                largest_metric_gap = max(largest_metric_gap, gap)
                if gap > 1e-12:
                    raise ValueError(f"{index}/{pair}/{name} independent metric mismatch: {gap}")
            audited.append({"record_index": index, "seed_pair": list(pair), **values,
                            "repeat_exact": True, "action_noise_pairing": True,
                            "feedback_condition_exact": True})
            del g0, tf, ar, g0_encoded, ar_input
    means = [float(np.mean([row["delta_feedback_rmse"] for row in audited if row["record_index"] == index]))
             for index in RECORDS]
    official = json.loads((root / "analysis" / "comparison.json").read_text(encoding="utf-8"))
    if abs(float(np.mean(means)) - official["episode_mean_delta_feedback_rmse"]) > 1e-12:
        raise ValueError("independent episode mean differs from primary analysis")
    return {"status": "INDEPENDENT_AUDIT_PASS", "strata": audited,
            "episode_means": dict(zip(map(str, RECORDS), means)),
            "episode_mean_delta_feedback_rmse": float(np.mean(means)),
            "max_absolute_metric_discrepancy": largest_metric_gap,
            "scope": "six episodes, two seed schedules; second-chunk float32 RGB final frame"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = verify(Path(args.run_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "episode_mean_delta_feedback_rmse", "max_absolute_metric_discrepancy")}))
