"""CPU tests for Task 9 cross-stratum, non-pooled reporting."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from analyze_umi_task9_matrix import collate_scans


def _write_stratum(root: Path, scene: int, seed: int, *, action: str = "fixed") -> None:
    analysis = root / "analysis"
    analysis.mkdir(parents=True)
    identity = {
        "scene": {"record_index": scene, "episode_id": scene + 100},
        "prompt": f"scene-{scene}", "action_chunk0": {"normalized_sha256": action},
        "data": {"sha256": f"data-{scene}"}, "direction_seed": 20260923,
        "checkpoint": {"content_manifest_sha256": "model"},
        "vae": {"sha256": "vae"}, "framework_commit": "framework",
        "task8_contract": {"GDE_precision": "float32"},
    }
    config = {"identity": identity, "seed": seed, "base_hash": f"base-{scene}",
              "mask_hash": "mask", "direction_hashes": {"train_00": "direction"},
              "plan": [{"sample_id": "baseline_pre"}]}
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    status = {"status": "COMPLETE", "completed": 1,
              "identity_hash": hashlib.sha256(canonical.encode()).hexdigest()}
    space = {"interpretation": "LOCAL_JACOBIAN_CANDIDATE", "half_step_pass_count": 1,
             "half_step_total": 1, "observed_rank": 1, "k90": 1, "k95": 1, "k99": 1,
             "effective_rank": 1.0, "unresolved_energy_fraction": 0.0,
             "heldout_relative_residual_by_k": [[0.1]]}
    summary = {"status": "COMPLETE", "scene_identity": identity, "seed": seed,
               "spaces": {"predicted_latent": space, "feedback_condition": space,
                          "float_rgb_final": space}}
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "run_status.json").write_text(json.dumps(status), encoding="utf-8")
    (analysis / "scan_summary.json").write_text(json.dumps(summary), encoding="utf-8")


class MatrixSummaryTests(unittest.TestCase):
    def test_six_strata_are_reported_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scans = []
            for scene in (1, 2, 3):
                for seed in (0, 1):
                    scan = root / f"scene_{scene}_seed_{seed}"
                    _write_stratum(scan, scene, seed)
                    scans.append(scan)
            result = collate_scans(scans, root / "matrix")
            self.assertEqual(result["strata"], 6)
            self.assertFalse(result["matrix_pooled"])
            self.assertEqual(len(result["rows"]), 18)
            self.assertTrue((root / "matrix" / "stratum_metrics.csv").is_file())
            with self.assertRaises(FileExistsError):
                collate_scans(scans, root / "matrix")

    def test_seed_pair_action_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scans = []
            for scene in (1, 2, 3):
                for seed in (0, 1):
                    scan = root / f"scene_{scene}_seed_{seed}"
                    _write_stratum(scan, scene, seed,
                                    action="changed" if (scene, seed) == (2, 1) else "fixed")
                    scans.append(scan)
            with self.assertRaisesRegex(ValueError, "same-scene seed pair"):
                collate_scans(scans, root / "matrix")

    def test_missing_stratum_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "complete preregistered"):
                collate_scans([], Path(temporary) / "matrix")


if __name__ == "__main__":
    unittest.main()
