import json
import unittest

import numpy as np

from umi_task6_primitives import (
    ALPHAS,
    DIRECTION_IDS,
    SOURCE_COMMIT,
    STATE_CATALOG,
    build_decoder_replay_plan,
    build_generation_plan,
    build_run_status,
    parse_action,
    pilot_groups,
    preprocess_frame,
    stable_hash,
    evaluate_resources,
)


class CatalogAndPlansTests(unittest.TestCase):
    def test_catalog_is_pinned_and_pilot_only_selects_bridge_zero_seed_zero(self):
        self.assertEqual(SOURCE_COMMIT, "2b17a2413bd86b2cf9b03823637108851e4ddf2d")
        self.assertEqual(set(STATE_CATALOG), {"umi_reference", "bridge_0", "bridge_384"})
        self.assertEqual(STATE_CATALOG["bridge_0"].video_path, "inputs/action/bridge_20260501_0.mp4")
        self.assertEqual(STATE_CATALOG["bridge_384"].action_path, "inputs/action/bridge_20260501_384.json")
        self.assertEqual([(g.state, g.seed) for g in pilot_groups()], [("bridge_0", 0)])

    def test_generation_and_decoder_plans_have_exact_approved_counts_and_identity(self):
        plan = build_generation_plan("bridge_0", 0)
        self.assertEqual(len(plan), 32)
        self.assertEqual(len({row["sample_id"] for row in plan}), 32)
        self.assertEqual(ALPHAS, (0.001, 0.003, 0.01))
        self.assertEqual(DIRECTION_IDS, ("v0", "v1", "v2", "u01", "u12"))
        self.assertEqual(plan[0]["kind"], "baseline")
        self.assertEqual(plan[-1]["kind"], "baseline")
        replay = build_decoder_replay_plan("bridge_0", 0)
        self.assertEqual(len(replay), 16)
        self.assertEqual(sum(x["kind"] == "perturbation" for x in replay), 12)
        self.assertEqual({x["decode_precision"] for x in replay}, {"native_bf16", "temporary_fp32"})
        self.assertEqual([x["direction_id"] for x in replay if x["kind"] == "perturbation"], ["v0"] * 12)


class PreprocessTests(unittest.TestCase):
    def test_aspect_preserving_resize_and_reflection_padding_wide_tall_square_odd(self):
        for shape in ((2, 5, 3), (5, 2, 3), (3, 3, 3), (3, 5, 1)):
            frame = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
            result = preprocess_frame(frame, size=8)
            self.assertEqual(result.shape, (8, 8, shape[2]))
            self.assertEqual(result.dtype, np.uint8)
            self.assertGreaterEqual(result.min(), frame.min())
            self.assertLessEqual(result.max(), frame.max())

    def test_reflection_padding_is_true_reflect_without_edge_repeat(self):
        frame = np.array([[[1], [2], [3], [4]], [[5], [6], [7], [8]]], dtype=np.uint8)
        result = preprocess_frame(frame, size=4)
        self.assertTrue(np.array_equal(result[:, :, 0], [[5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4]]))

    def test_float_dtype_and_range_are_preserved(self):
        frame = np.linspace(0, 1, 15, dtype=np.float32).reshape(3, 5, 1)
        result = preprocess_frame(frame, size=8)
        self.assertEqual(result.dtype, np.float32)
        self.assertGreaterEqual(float(result.min()), 0)
        self.assertLessEqual(float(result.max()), 1)


class InputAndResourceTests(unittest.TestCase):
    def test_action_requires_exact_shape_finite_numeric_and_hash_is_stable(self):
        action = parse_action([[float(i + j) for j in range(10)] for i in range(16)])
        self.assertEqual(action.shape, (16, 10))
        self.assertEqual(action.dtype, np.float32)
        self.assertEqual(stable_hash(action), stable_hash(action.copy()))
        for invalid in (np.zeros((15, 10)), np.zeros((16, 9)), np.full((16, 10), np.nan), np.full((16, 10), np.inf)):
            with self.subTest(shape=invalid.shape):
                with self.assertRaises(ValueError):
                    parse_action(invalid)

    def test_hard_stop_dominates_warning_and_reason_is_machine_stable(self):
        result = evaluate_resources({"gpu_used_gib": 80, "gpu_free_gib": 10, "gpu_reserved_gib": 70,
                                     "ram_available_gib": 250, "rss_gib": 170, "swap_used_gib": 1,
                                     "disk_free_gib": 4, "forecast_free_gib": 2,
                                     "gpu_cleanup_growth_gib": 3, "ram_cleanup_growth_gib": 11,
                                     "consecutive_growth_samples": 2})
        self.assertEqual(result["status"], "HARD_STOP")
        self.assertIn(result["reason_code"], {"GPU_USED_HIGH", "GPU_FREE_LOW", "RAM_AVAILABLE_LOW", "SWAP_IN_USE", "DISK_FREE_LOW"})
        self.assertTrue(result["hard_stop_reasons"])
        warning = evaluate_resources({"gpu_used_gib": 61, "gpu_free_gib": 34, "gpu_reserved_gib": 1,
                                      "ram_available_gib": 399, "rss_gib": 101, "swap_used_gib": 0,
                                      "disk_free_gib": 7, "forecast_free_gib": 5.5,
                                      "gpu_cleanup_growth_gib": 0, "ram_cleanup_growth_gib": 0,
                                      "consecutive_growth_samples": 0})
        self.assertEqual(warning["status"], "WARNING")

    def test_run_status_is_canonical_and_contains_terminal_evidence(self):
        payload = build_run_status("AWAITING_REVIEW", reason_code="PILOT_COMPLETE",
                                   completed=["baseline_pre"], failed=[], skipped=["baseline_post"],
                                   resource_snapshots={"gpu": {"used_gib": 1}},
                                   hashes={"code": "c", "model": "m", "config": "f", "direction": "d", "input": "i", "noise": "n"})
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertEqual(payload["status"], "AWAITING_REVIEW")
        self.assertEqual(payload["reason_code"], "PILOT_COMPLETE")
        self.assertEqual(payload["completed_samples"], ["baseline_pre"])
        self.assertEqual(payload["last_resource_snapshots"]["gpu"]["used_gib"], 1)
        self.assertEqual(set(payload["hashes"]), {"code", "model", "config", "direction", "input", "noise"})


if __name__ == "__main__":
    unittest.main()
