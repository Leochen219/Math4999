"""CPU contracts for the preregistered real-scene feedback study."""
import unittest

import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from task8_frozen.run_umi_task8_experiment import run_task8
from test_run_umi_task8_experiment import Task8RunnerTests
from task8_frozen.task8_formal import build_binding

from umi_task10_real_scenes import (
    LOCKED_RECORDS,
    SEED_PAIRS,
    aggregate_episodes,
    analyze_stratum_records,
    build_call_plan,
    compare_endpoint_rgb,
    forecast_disk_after_calls,
    locked_preflight_ranks,
    select_locked_records,
    verify_call_records,
)


class SelectionTests(unittest.TestCase):
    def test_locked_selection_is_ordered_and_never_substitutes(self):
        inventory = [
            {"record_index": i, "frames": 47, "has_language": True,
             "primary_jpeg_nonempty": True, "sequence_flags_aligned": True}
            for i in range(15)
        ]
        self.assertEqual(tuple(row["record_index"] for row in select_locked_records(inventory)), LOCKED_RECORDS)
        inventory[6]["frames"] = 32
        with self.assertRaisesRegex(ValueError, "record 6"):
            select_locked_records(inventory)

    def test_two_seed_schedules_produce_48_formal_calls(self):
        self.assertEqual(SEED_PAIRS, ((0, 1), (2, 3)))
        plans = [build_call_plan(*pair) for _ in LOCKED_RECORDS for pair in SEED_PAIRS]
        self.assertEqual(sum(len(plan) for plan in plans), 48)
        self.assertEqual([row["call"] for row in plans[0]], ["G0", "G0_repeat", "TF2", "AR2"])
        self.assertEqual([row["seed"] for row in plans[-1]], [2, 2, 3, 3])
        self.assertEqual(plans[-1][2]["action_source"], plans[-1][3]["action_source"])

    def test_frozen_preflight_ranks_skip_repeated_scene_and_bad_language(self):
        self.assertEqual(locked_preflight_ranks(), (0, 1, 2, 3, 4, 7))

    def test_measured_disk_forecast_keeps_five_gib_reserve(self):
        self.assertAlmostEqual(forecast_disk_after_calls(75, 80 * 1024**2, 48), 70.125)
        with self.assertRaisesRegex(ValueError, "reserve"):
            forecast_disk_after_calls(6, 80 * 1024**2, 48)

    def test_task8_runner_accepts_only_explicitly_enabled_seeded_plan(self):
        fixture = Task8RunnerTests()
        plan = build_call_plan(2, 3)
        with TemporaryDirectory() as folder:
            result = run_task8(Path(folder), execute_call=fixture._formal_result,
                               release=True, binding=fixture._binding(),
                               resource_snapshot=fixture._snapshot(), plan=plan,
                               allow_custom_plan=True)
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["completed_samples"], ["G0", "G0_repeat", "TF2", "AR2"])

    def test_binding_hash_changes_with_actual_seed_schedule(self):
        with TemporaryDirectory() as folder:
            model = Path(folder) / "model"; model.write_bytes(b"m")
            vae = Path(folder) / "vae"; vae.write_bytes(b"v")
            data = Path(folder) / "data"; data.write_bytes(b"d")
            kwargs = {"code_paths": [model], "model_path": model, "vae_path": vae,
                      "input_path": data, "actions": np.zeros((2, 16, 10), np.float32),
                      "metadata": {}, "framework_commit": "a" * 40}
            first = build_binding(**kwargs, seed_pair=(0, 1))
            second = build_binding(**kwargs, seed_pair=(2, 3))
            self.assertNotEqual(first["noise"], second["noise"])


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.records = {
            "G0": {"condition_input_fp32": np.array([1], np.float32), "action": np.array([2], np.float32),
                   "prediction_noise_hash": "a", "output_full": np.array([3], np.float32),
                   "decoded_rgb_full": np.array([4], np.float32), "encoded_condition": np.array([5], np.float32)},
            "G0_repeat": {"condition_input_fp32": np.array([1], np.float32), "action": np.array([2], np.float32),
                          "prediction_noise_hash": "a", "output_full": np.array([3], np.float32),
                          "decoded_rgb_full": np.array([4], np.float32), "encoded_condition": np.array([5], np.float32)},
            "TF2": {"condition_input_fp32": np.array([6], np.float32), "action": np.array([7], np.float32),
                    "prediction_noise_hash": "b"},
            "AR2": {"condition_input_fp32": np.array([5], np.float32), "action": np.array([7], np.float32),
                    "prediction_noise_hash": "b"},
        }

    def test_exact_repeat_and_feedback_pair_pass(self):
        verify_call_records(self.records)

    def test_mismatched_second_step_noise_stops(self):
        self.records["AR2"]["prediction_noise_hash"] = "wrong"
        with self.assertRaisesRegex(ValueError, "noise"):
            verify_call_records(self.records)

    def test_stale_feedback_condition_stops(self):
        self.records["AR2"]["condition_input_fp32"][0] = 9
        with self.assertRaisesRegex(ValueError, "feedback"):
            verify_call_records(self.records)


class FrozenRuntimeSeedTests(unittest.TestCase):
    def test_frozen_feedback_runtime_routes_seed_two_and_three_to_noise_and_sampler(self):
        from task8_frozen.umi_task7_runtime import FeedbackRuntime
        from test_umi_task7_runtime import _Encoder, _fixture
        runtime, _, z0, mask, v0, condition_shape = _fixture()
        runtime.decode_fp32 = lambda full_latent, *, precision: {
            "raw_output": np.zeros((1, 3, 3, 2, 2), dtype=np.float32),
            "operation_count": 1, "operation_dtypes": ["float32"], "invocations": 1,
        }
        feedback = FeedbackRuntime(runtime, encoder=_Encoder(condition_shape), z0=z0,
                                   mask=mask, condition_indexes=runtime.inputs.geometry.condition_indexes,
                                   v0=v0, seed=0)
        condition = feedback.extract_condition(z0)
        second = feedback.step(condition, 2)
        third = feedback.step(condition, 3)
        self.assertEqual(second["evidence"]["prepare_seed"], 2)
        self.assertEqual(third["evidence"]["prepare_seed"], 3)
        self.assertEqual(second["evidence"]["sampler_seeds"], [2] * 30)
        self.assertEqual(third["evidence"]["sampler_seeds"], [3] * 30)
        self.assertNotEqual(second["prediction_noise_hash"], third["prediction_noise_hash"])

class AnalysisTests(unittest.TestCase):
    def test_endpoint_delta_does_not_use_whole_chunk(self):
        truth = np.zeros((3, 2, 2), np.float32)
        tf = np.zeros((3, 16, 2, 2), np.float32)
        ar = np.zeros_like(tf)
        tf[:, :15] = 1
        ar[:, :15] = 0
        tf[:, 15] = 0.1
        ar[:, 15] = 0.2
        result = compare_endpoint_rgb(tf, ar, truth)
        self.assertAlmostEqual(result["tf_rmse"], 0.1, places=6)
        self.assertAlmostEqual(result["ar_rmse"], 0.2, places=6)
        self.assertAlmostEqual(result["delta_feedback_rmse"], 0.1, places=6)

    def test_episodes_not_seed_rows_are_statistical_units(self):
        rows = [{"record_index": i, "seed_pair": list(pair), "delta_feedback_rmse": float(i + seed)}
                for i in LOCKED_RECORDS for seed, pair in enumerate(SEED_PAIRS)]
        result = aggregate_episodes(rows)
        self.assertEqual(result["episode_count"], 6)
        self.assertEqual(result["seed_pair_count"], 12)
        self.assertAlmostEqual(result["episodes"][0]["mean_delta_feedback_rmse"], 1.5)

    def test_full_stratum_reports_endpoint_and_matched_latent_mask(self):
        truth = np.zeros((33, 3, 2, 2), np.float32)
        g0 = np.zeros((3, 16, 2, 2), np.float32)
        tf = np.full_like(g0, 0.1)
        ar = np.full_like(g0, 0.2)
        base = {"condition_input_fp32": np.array([1, 2], np.float32),
                "action": np.array([1], np.float32), "prediction_noise_hash": "n0",
                "output_full": np.array([1], np.float32), "decoded_rgb_full": np.array([1], np.float32),
                "encoded_condition": np.array([0.1, 9], np.float32), "generated_rgb": g0}
        records = {"G0": base, "G0_repeat": dict(base),
                   "TF2": {"condition_input_fp32": np.array([0, 0], np.float32),
                           "action": np.array([2], np.float32), "prediction_noise_hash": "n1",
                           "encoded_condition": np.array([0.1, 9], np.float32), "generated_rgb": tf},
                   "AR2": {"condition_input_fp32": base["encoded_condition"],
                           "action": np.array([2], np.float32), "prediction_noise_hash": "n1",
                           "encoded_condition": np.array([0.2, 9], np.float32), "generated_rgb": ar}}
        summary, frames = analyze_stratum_records(records, truth, np.zeros(2, np.float32),
                                                   np.zeros(2, np.float32), np.array([True, False]),
                                                   record_index=1, seed_pair=(0, 1))
        self.assertAlmostEqual(summary["delta_feedback_rmse"], 0.1, places=6)
        self.assertAlmostEqual(summary["tf2_latent_rms"], 0.1, places=6)
        self.assertAlmostEqual(summary["ar2_latent_rms"], 0.2, places=6)
        self.assertAlmostEqual(summary["squared_error_identity_residual"], 0.0, places=12)
        self.assertEqual(len(frames), 16)
        self.assertEqual(frames[-1]["truth_frame_index"], 32)


if __name__ == "__main__":
    unittest.main()
