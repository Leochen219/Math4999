"""CPU contracts for the preregistered response-spectrum experiment."""
import unittest
import tempfile
from pathlib import Path

import numpy as np

from umi_task9_spectrum import (
    build_call_plan,
    central_response,
    generate_direction_bank,
    spectrum_metrics,
    run_samples,
    analyze_stratum,
)


class DirectionBankTests(unittest.TestCase):
    def test_frozen_masked_bank_is_reproducible_and_rms_orthonormal(self):
        mask = np.zeros((2, 3, 5), dtype=bool)
        mask[:, 0:2, :] = True
        first = generate_direction_bank(mask, seed=20260923, train_count=8, holdout_count=4)
        second = generate_direction_bank(mask, seed=20260923, train_count=8, holdout_count=4)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.array_equal(first[:, ~mask], np.zeros((12, 10), dtype=np.float32)))
        active = first[:, mask].astype(np.float64)
        self.assertTrue(np.allclose(active @ active.T / active.shape[1], np.eye(12), atol=2e-7))

    def test_too_few_active_coordinates_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "active"):
            generate_direction_bank(np.ones(3, dtype=bool), seed=1, train_count=3, holdout_count=1)


class PlanTests(unittest.TestCase):
    def test_first_stratum_has_162_calls_and_half_step_for_every_independent_direction(self):
        plan = build_call_plan(train_count=32, holdout_count=8, alpha=0.003,
                               calibration_alpha=0.0015)
        self.assertEqual(len(plan), 162)
        self.assertEqual(plan[0]["sample_id"], "baseline_pre")
        self.assertEqual(plan[-1]["sample_id"], "baseline_post")
        self.assertEqual(len({item["sample_id"] for item in plan}), 162)
        self.assertEqual(sum(item["kind"] == "holdout" for item in plan), 16)
        self.assertEqual(sum(item["kind"] == "calibration" for item in plan), 80)
        self.assertEqual({item["alpha"] for item in plan if item["kind"] == "calibration"}, {0.0015})
        self.assertEqual({item["direction_id"] for item in plan if item["kind"] == "calibration"},
                         {f"train_{i:02d}" for i in range(32)} | {f"holdout_{i:02d}" for i in range(8)})

    def test_default_calibration_uses_exact_half_step(self):
        plan = build_call_plan()
        self.assertEqual({item["alpha"] for item in plan if item["kind"] == "calibration"}, {0.0015})


class SpectrumTests(unittest.TestCase):
    def test_central_response_uses_sign_paired_float_arrays(self):
        plus = np.array([1.003, 2.006], dtype=np.float32)
        minus = np.array([0.997, 1.994], dtype=np.float32)
        self.assertTrue(np.allclose(central_response(plus, minus, 0.003), [1.0, 2.0], atol=2e-5))
        with self.assertRaisesRegex(ValueError, "shape"):
            central_response(plus, minus[:1], 0.003)

    def test_rank_two_energy_effective_rank_and_heldout_projection(self):
        train = np.diag([3.0, 2.0, 0.0]).astype(np.float32)
        heldout = np.array([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32).T
        result = spectrum_metrics(train, heldout)
        self.assertEqual(result["observed_rank"], 2)
        self.assertAlmostEqual(result["cumulative_energy"][0], 9.0 / 13.0, places=12)
        self.assertEqual(result["k90"], 2)
        self.assertEqual(result["k95"], 2)
        self.assertAlmostEqual(result["effective_rank"], np.exp(-9/13*np.log(9/13)-4/13*np.log(4/13)), places=12)
        self.assertAlmostEqual(result["heldout_relative_residual_by_k"][1][0], 0.0, places=12)
        self.assertAlmostEqual(result["heldout_relative_residual_by_k"][1][1], 0.0, places=12)

    def test_zero_response_has_na_rank_and_heldout_zero_denominator(self):
        result = spectrum_metrics(np.zeros((4, 3), dtype=np.float32),
                                  np.zeros((4, 1), dtype=np.float32))
        self.assertEqual(result["status"], "NO_NONZERO_RESPONSE")
        self.assertIsNone(result["effective_rank"])
        self.assertIsNone(result["heldout_relative_residual_by_k"])

    def test_sub_resolution_singular_value_does_not_inflate_effective_rank(self):
        train = np.diag([1.0, 1e-6]).astype(np.float64)
        result = spectrum_metrics(train, np.eye(2), singular_resolution=1e-4)
        self.assertEqual(result["observed_rank"], 1)
        self.assertEqual(result["k99"], 1)
        self.assertEqual(len(result["singular_values"]), 2)


class CompactRunnerTests(unittest.TestCase):
    def test_float_outputs_are_saved_atomically_and_resume_never_recomputes(self):
        base = np.ones((1, 2, 1, 2, 2), dtype=np.float32)
        mask = np.ones_like(base, dtype=bool)
        direction = np.ones_like(base, dtype=np.float32)
        plan = [
            {"sample_id": "baseline_pre", "kind": "baseline", "direction_id": None, "alpha": 0.0, "sign": 0},
            {"sample_id": "train_00_plus", "kind": "train", "direction_id": "train_00", "alpha": 0.003, "sign": 1},
            {"sample_id": "train_00_minus", "kind": "train", "direction_id": "train_00", "alpha": 0.003, "sign": -1},
            {"sample_id": "baseline_post", "kind": "baseline", "direction_id": None, "alpha": 0.0, "sign": 0},
        ]
        calls = []

        def step(condition, seed):
            calls.append((condition.copy(), seed))
            predicted = np.repeat(condition, 4, axis=2)
            return {"predicted_latent": predicted.copy(),
                    "decoded_last_rgb": np.full((3, 2, 2), float(condition.mean()), dtype=np.float32),
                    "next_condition_fp32": 2 * condition,
                    "prediction_noise_hash": "paired-noise",
                    "actual_consumed_condition": condition.copy(),
                    "step_evidence": {"action_hash": "frozen-action", "condition_steps": 30}}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            outcome = run_samples(root, base, mask, {"train_00": direction}, plan,
                                  seed=0, identity={"scene": "synthetic"}, step=step)
            self.assertEqual(outcome["status"], "COMPLETE")
            self.assertEqual(len(calls), 4)
            self.assertEqual(np.load(root / "samples" / "train_00_plus" / "next_condition.npy").dtype, np.float32)
            record = __import__("json").loads((root / "samples" / "train_00_plus" / "record.json").read_text())
            self.assertEqual(record["step_evidence"], {"action_hash": "frozen-action", "condition_steps": 30})
            self.assertEqual(len(list((root / "samples").iterdir())), 4)
            run_samples(root, base, mask, {"train_00": direction}, plan,
                        seed=0, identity={"scene": "synthetic"}, step=step, resume=True)
            self.assertEqual(len(calls), 4)
            with self.assertRaisesRegex(ValueError, "identity"):
                run_samples(root, base, mask, {"train_00": direction}, plan,
                            seed=0, identity={"scene": "changed"}, step=step, resume=True)

    def test_nonpaired_noise_is_hard_failure(self):
        base = np.ones((1, 1, 1, 2, 2), dtype=np.float32)
        mask = np.ones_like(base, dtype=bool)
        plan = [{"sample_id": name, "kind": "baseline", "direction_id": None, "alpha": 0.0, "sign": 0}
                for name in ("baseline_pre", "baseline_post")]
        count = 0

        def step(condition, seed):
            nonlocal count
            count += 1
            return {"predicted_latent": condition.copy(),
                    "decoded_last_rgb": np.ones((3, 2, 2), dtype=np.float32),
                    "next_condition_fp32": condition.copy(),
                    "prediction_noise_hash": str(count),
                    "actual_consumed_condition": condition.copy()}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            with self.assertRaisesRegex(ValueError, "noise"):
                run_samples(root, base, mask, {}, plan,
                            seed=0, identity={"scene": "synthetic"}, step=step)
            self.assertEqual(__import__("json").loads((root / "run_status.json").read_text())["status"], "BLOCKED")
            self.assertTrue((root / "samples" / "baseline_pre" / "status.json").is_file())


class OfflineAnalysisTests(unittest.TestCase):
    def test_linear_rank_two_map_has_zero_noise_and_consistent_half_step(self):
        base = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(1, 1, 1, 2, 2)
        mask = np.ones_like(base, dtype=bool)
        bank = generate_direction_bank(mask, seed=123, train_count=2, holdout_count=1)
        directions = {"train_00": bank[0], "train_01": bank[1], "holdout_00": bank[2]}
        plan = build_call_plan(train_count=2, holdout_count=1)
        matrix = np.array([[1.0, 2.0, 0.0, 1.0], [0.0, 1.0, 3.0, 0.0]], dtype=np.float32)

        def step(condition, seed):
            value = matrix @ condition.reshape(-1)
            feedback = np.array([value[0], value[1], value[0] + value[1], 0.0], dtype=np.float32).reshape(base.shape)
            return {"predicted_latent": value, "decoded_last_rgb": np.full((3, 2, 2), value[0], dtype=np.float32),
                    "next_condition_fp32": feedback, "prediction_noise_hash": "same",
                    "actual_consumed_condition": condition.copy()}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            run_samples(root, base, mask, directions, plan, seed=0,
                        identity={"scene": "linear"}, step=step)
            summary = analyze_stratum(root)
            self.assertEqual(summary["status"], "COMPLETE")
            self.assertEqual(summary["spaces"]["predicted_latent"]["n_train"], 2)
            self.assertEqual(summary["spaces"]["predicted_latent"]["half_step_pass_count"], 3)
            self.assertEqual(summary["spaces"]["feedback_condition"]["half_step_pass_count"], 3)
            self.assertEqual(summary["spaces"]["predicted_latent"]["baseline_noise_rms"], 0.0)
            self.assertLess(max(summary["spaces"]["predicted_latent"]["heldout_relative_residual_by_k"][-1]), 1e-5)
            self.assertTrue((root / "analysis" / "predicted_latent_spectrum.png").is_file())
            self.assertTrue((root / "analysis" / "feedback_condition_spectrum.svg").is_file())
            self.assertTrue((root / "analysis" / "experiment_report.md").is_file())
            self.assertTrue((root / "MANIFEST.sha256").is_file())
            config_path = root / "config.json"
            original_config = config_path.read_bytes()
            changed = __import__("json").loads(original_config)
            changed["identity"]["scene"] = "another-scene"
            config_path.write_text(__import__("json").dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity hash"):
                analyze_stratum(root)
            config_path.write_bytes(original_config)
            altered = np.load(root / "base_condition.npy", allow_pickle=False)
            altered.flat[0] += 1.0
            np.save(root / "base_condition.npy", altered, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "immutable configuration"):
                analyze_stratum(root)


if __name__ == "__main__":
    unittest.main()
