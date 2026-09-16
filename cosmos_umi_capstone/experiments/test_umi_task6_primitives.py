import json
import unittest
from unittest.mock import patch

import numpy as np

from umi_task6_primitives import (
    ALPHAS,
    DIRECTION_IDS,
    SOURCE_COMMIT,
    STATE_CATALOG,
    build_decoder_replay_plan,
    build_generation_plan,
    build_run_status,
    TERMINAL_STATUSES,
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

    def test_group_layer_accepts_only_exact_integer_seed_zero_or_one(self):
        for bad in (True, False, "0", 0.0, 1.5, 2, -1):
            with self.subTest(seed=bad):
                with self.assertRaises(ValueError):
                    build_generation_plan("bridge_0", bad)


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
        self.assertTrue(np.array_equal(result[:, :, 0], [[1, 2, 3, 4], [5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]]))

    def test_float_dtype_and_range_are_preserved(self):
        frame = np.linspace(0, 1, 15, dtype=np.float32).reshape(3, 5, 1)
        result = preprocess_frame(frame, size=8)
        self.assertEqual(result.dtype, np.float32)
        self.assertGreaterEqual(float(result.min()), 0)
        self.assertLessEqual(float(result.max()), 1)

    def test_no_upscale_and_official_rounding_right_bottom_reflect_and_edge(self):
        frame = np.arange(15, dtype=np.float32).reshape(3, 5, 1)
        # target 8x8: scale is capped at one (no upscale), then right/bottom padding only.
        result = preprocess_frame(frame, size=(8, 8))
        self.assertEqual(result.shape, (8, 8, 1))
        self.assertTrue(np.array_equal(result[:3, :5, 0], frame[:, :, 0]))
        # 5x5 -> 8x8 has 3px right/bottom reflect padding.
        square = np.arange(25, dtype=np.uint8).reshape(5, 5, 1)
        reflected = preprocess_frame(square, size=8)
        self.assertTrue(np.array_equal(reflected[:5, :5], square))
        self.assertTrue(np.array_equal(reflected[5:, :5, 0], square[3:0:-1, :, 0]))
        # If either padding is at least the resized dimension, official code
        # uses edge mode for both axes.
        edge = preprocess_frame(np.arange(4, dtype=np.uint8).reshape(1, 4, 1), size=8)
        self.assertTrue(np.all(edge[1:, :, 0] == edge[0:1, :, 0]))

    def test_downscale_calls_explicit_backend_once(self):
        calls = []
        frame = np.zeros((8, 8, 1), dtype=np.float32)
        def backend(value, width, height):
            calls.append((value.shape, width, height))
            return np.ones((height, width, 1), dtype=np.float32)
        result = preprocess_frame(frame, size=4, resize_backend=backend)
        self.assertEqual(calls, [((8, 8, 1), 4, 4)])
        self.assertTrue(np.all(result == 1))


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

    def test_nonfinite_and_malformed_resource_values_fail_closed(self):
        base = {"gpu_used_gib": 0, "gpu_free_gib": 100, "gpu_reserved_gib": 0,
                "ram_available_gib": 600, "rss_gib": 1, "swap_used_gib": 0,
                "disk_free_gib": 20, "forecast_free_gib": 20}
        for key, value in (("gpu_used_gib", float("inf")), ("gpu_free_gib", float("-inf")),
                           ("ram_available_gib", "not-a-number"), ("disk_free_gib", float("nan")),
                           ("consecutive_growth_samples", "bad"), ("gpu_consecutive_growth_samples", float("inf"))):
            with self.subTest(key=key):
                sample = dict(base); sample[key] = value
                result = evaluate_resources(sample)
                self.assertEqual(result["status"], "HARD_STOP")
                self.assertEqual(result["reason_code"], "RESOURCE_SNAPSHOT_NONFINITE")

    def test_resource_boundaries_and_all_stop_reasons(self):
        base = {"gpu_used_gib": 0, "gpu_free_gib": 100, "gpu_reserved_gib": 0,
                "ram_available_gib": 600, "rss_gib": 1, "swap_used_gib": 0,
                "disk_free_gib": 20, "forecast_free_gib": 20,
                "gpu_peak_allocated_gib": 0, "gpu_peak_nvml_used_gib": 0}
        for key, value, code in (("gpu_used_gib", 75.01, "GPU_USED_HIGH"),
                                  ("gpu_free_gib", 19.99, "GPU_FREE_LOW"),
                                  ("gpu_reserved_gib", 65.01, "GPU_RESERVED_HIGH"),
                                  ("gpu_peak_allocated_gib", 35.01, "GPU_SMOKE_PEAK_ALLOCATED_HIGH"),
                                  ("gpu_peak_nvml_used_gib", 45.01, "GPU_SMOKE_PEAK_USED_HIGH"),
                                  ("ram_available_gib", 299.99, "RAM_AVAILABLE_LOW"),
                                  ("rss_gib", 160.01, "RAM_RSS_HIGH"),
                                  ("swap_used_gib", 0.01, "SWAP_IN_USE"),
                                  ("disk_free_gib", 4.99, "DISK_FREE_LOW")):
            with self.subTest(key=key):
                sample = dict(base); sample[key] = value
                phase = "resource-smoke" if "peak" in key else "pilot"
                starting = key in ("disk_free_gib", "free_disk_gib")
                self.assertEqual(evaluate_resources(sample, phase=phase, starting_new_sample=starting)["reason_code"], code)
        self.assertEqual(evaluate_resources({**base, "gpu_used_gib": 1.01}, phase="start")["reason_code"], "GPU_START_USED_HIGH")
        self.assertEqual(evaluate_resources({**base, "ram_available_gib": 499.99}, phase="start")["reason_code"], "RAM_START_AVAILABLE_LOW")
        self.assertEqual(evaluate_resources({**base, "disk_free_gib": 9.99}, phase="start")["reason_code"], "DISK_START_FREE_LOW")
        self.assertEqual(evaluate_resources({**base, "gpu_cleanup_growth_gib": 2.01, "consecutive_growth_samples": 2})["reason_code"], "GPU_CLEANUP_GROWTH")
        self.assertEqual(evaluate_resources({**base, "ram_cleanup_growth_gib": 10.01, "consecutive_growth_samples": 2})["reason_code"], "RAM_CLEANUP_GROWTH")

    def test_resource_forecast_oom_monitor_and_full_matrix_gates(self):
        base = {"gpu_used_gib": 0, "gpu_free_gib": 100, "gpu_reserved_gib": 0,
                "ram_available_gib": 600, "rss_gib": 1, "swap_used_gib": 0,
                "disk_free_gib": 20, "forecast_free_gib": 20}
        self.assertEqual(evaluate_resources({**base, "cuda_oom": True})["reason_code"], "CUDA_OOM")
        self.assertEqual(evaluate_resources({**base, "monitor_failure": "poll failed"})["reason_code"], "MONITOR_FAILURE")
        warning = evaluate_resources({**base, "mean_success_sample_bytes": 5 * 1024**3,
                                      "remaining_samples": 3, "disk_free_gib": 24.0})
        self.assertEqual(warning["reason_code"], "DISK_FORECAST_WARNING")
        full = evaluate_resources({**base, "disk_free_gib": 20,
                                   "remaining_matrix_estimate_gib": 12, "full_matrix_launch": True})
        self.assertEqual(full["reason_code"], "DISK_FULL_MATRIX_INSUFFICIENT")

    def test_disk_stop_only_applies_when_starting_a_new_sample_and_forecast_aliases_match(self):
        base = {"gpu_used_gib": 0, "gpu_free_gib": 100, "gpu_reserved_gib": 0,
                "ram_available_gib": 600, "rss_gib": 1, "swap_used_gib": 0}
        for key in ("disk_free_gib", "free_disk_gib"):
            with self.subTest(key=key):
                snapshot = {**base, key: 4.99, "mean_success_sample_bytes": 1 * 1024**3, "remaining_samples": 1}
                self.assertEqual(evaluate_resources(snapshot, starting_new_sample=True)["reason_code"], "DISK_FREE_LOW")
                self.assertEqual(evaluate_resources(snapshot, starting_new_sample=False)["status"], "WARNING")
                self.assertNotEqual(evaluate_resources(snapshot, starting_new_sample=False)["reason_code"], "DISK_FREE_LOW")
        for count in (1.5, -1, "3", True, None, float("inf")):
            with self.subTest(count=count):
                result = evaluate_resources({**base, "disk_free_gib": 20, "remaining_samples": count})
                self.assertEqual(result["reason_code"], "RESOURCE_SNAPSHOT_NONFINITE")
        for mean in (-1, "x", None, float("inf"), float("nan")):
            with self.subTest(mean=mean):
                result = evaluate_resources({**base, "disk_free_gib": 20, "remaining_samples": 1, "mean_success_sample_bytes": mean})
                self.assertEqual(result["reason_code"], "RESOURCE_SNAPSHOT_NONFINITE")

    def test_exact_resource_boundaries_and_independent_growth_counters(self):
        safe = {"gpu_used_gib": 75, "gpu_free_gib": 20, "gpu_reserved_gib": 65,
                "gpu_peak_allocated_gib": 35, "gpu_peak_nvml_used_gib": 45,
                "ram_available_gib": 300, "rss_gib": 160, "swap_used_gib": 0,
                "disk_free_gib": 5, "forecast_free_gib": 6}
        safe_result = evaluate_resources(safe, starting_new_sample=True)
        self.assertNotIn("DISK_FREE_LOW", {x["code"] for x in safe_result["hard_stop_reasons"]})
        self.assertEqual(evaluate_resources({**safe, "gpu_used_gib": 60, "gpu_free_gib": 35,
                                              "ram_available_gib": 400, "rss_gib": 100,
                                              "disk_free_gib": 8, "forecast_free_gib": 6})["status"], "OK")
        gpu_only = evaluate_resources({**safe, "gpu_cleanup_growth_gib": 2.01,
                                       "gpu_consecutive_growth_samples": 2,
                                       "ram_cleanup_growth_gib": 11,
                                       "ram_consecutive_growth_samples": 1})
        self.assertEqual(gpu_only["reason_code"], "GPU_CLEANUP_GROWTH")
        ram_only = evaluate_resources({**safe, "gpu_cleanup_growth_gib": 2.01,
                                       "gpu_consecutive_growth_samples": 1,
                                       "ram_cleanup_growth_gib": 11,
                                       "ram_consecutive_growth_samples": 2})
        self.assertEqual(ram_only["reason_code"], "RAM_CLEANUP_GROWTH")

    def test_phase_specific_start_smoke_and_pilot_gates(self):
        loaded = {"gpu_used_gib": 20, "gpu_free_gib": 80, "gpu_reserved_gib": 20,
                  "ram_available_gib": 600, "rss_gib": 1, "swap_used_gib": 0,
                  "disk_free_gib": 20, "forecast_free_gib": 20,
                  "gpu_peak_allocated_gib": 35.01, "gpu_peak_nvml_used_gib": 45.01}
        self.assertEqual(evaluate_resources(loaded, phase="resource-smoke")["reason_code"], "GPU_SMOKE_PEAK_ALLOCATED_HIGH")
        self.assertEqual(evaluate_resources(loaded, phase="pilot")["status"], "OK")
        self.assertEqual(evaluate_resources({**loaded, "gpu_used_gib": 1.01}, phase="startup")["reason_code"], "GPU_START_USED_HIGH")
        self.assertEqual(evaluate_resources({**loaded, "gpu_used_gib": 0, "ram_available_gib": 499.99}, phase="preload")["reason_code"], "RAM_START_AVAILABLE_LOW")
        self.assertEqual(evaluate_resources({**loaded, "gpu_used_gib": 0, "disk_free_gib": 9.99}, phase="startup")["reason_code"], "DISK_START_FREE_LOW")

    def test_resize_failure_is_explicit_and_not_silently_nearest(self):
        frame = np.zeros((8, 8, 1), dtype=np.float32)
        real_import = __import__
        def unavailable(name, *args, **kwargs):
            if name == "torch" or name.startswith("torchvision"):
                raise ImportError("simulated unavailable official resize")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=unavailable), self.assertRaisesRegex(RuntimeError, "official.*resize"):
            preprocess_frame(frame, size=4)
        with self.assertRaisesRegex(RuntimeError, "official.*resize"):
            preprocess_frame(frame, size=4, resize_backend=lambda *_: (_ for _ in ()).throw(RuntimeError("official resize failed")))
        def internal_type_error(*_args):
            raise TypeError("backend computation failed")
        with self.assertRaisesRegex(TypeError, "backend computation failed"):
            preprocess_frame(frame, size=4, resize_backend=internal_type_error)

    def test_full_matrix_requires_estimate_and_equality_fails(self):
        base = {"gpu_used_gib": 0, "gpu_free_gib": 100, "gpu_reserved_gib": 0,
                "ram_available_gib": 600, "rss_gib": 1, "swap_used_gib": 0,
                "disk_free_gib": 18, "forecast_free_gib": 20}
        for estimate in (None, False, "0", -1, "bad", float("inf"), float("nan")):
            sample = dict(base)
            sample["remaining_matrix_estimate_gib"] = estimate
            result = evaluate_resources(sample, phase="full-matrix")
            self.assertEqual(result["reason_code"], "DISK_FULL_MATRIX_INSUFFICIENT")
        self.assertEqual(evaluate_resources({**base, "remaining_matrix_estimate_gib": 0}, phase="full-matrix")["status"], "OK")
        self.assertEqual(evaluate_resources({**base, "disk_free_gib": 18,
                                             "remaining_matrix_estimate_gib": 10}, phase="full-matrix")["reason_code"], "DISK_FULL_MATRIX_INSUFFICIENT")
        self.assertEqual(evaluate_resources({**base, "disk_free_gib": 18.001,
                                             "remaining_matrix_estimate_gib": 10}, phase="full-matrix")["status"], "OK")

    def test_exact_start_smoke_and_growth_boundaries_are_safe(self):
        startup = {"gpu_used_gib": 1, "gpu_free_gib": 100, "gpu_reserved_gib": 65,
                   "ram_available_gib": 500, "rss_gib": 160, "swap_used_gib": 0,
                   "disk_free_gib": 10, "forecast_free_gib": 6}
        self.assertFalse(evaluate_resources(startup, phase="startup")["hard_stop_reasons"])
        smoke = {**startup, "gpu_used_gib": 20, "gpu_peak_allocated_gib": 35,
                 "gpu_peak_nvml_used_gib": 45}
        self.assertFalse(evaluate_resources(smoke, phase="resource-smoke")["hard_stop_reasons"])
        growth = {**startup, "gpu_cleanup_growth_gib": 2, "gpu_consecutive_growth_samples": 2,
                  "ram_cleanup_growth_gib": 10, "ram_consecutive_growth_samples": 2}
        self.assertFalse(evaluate_resources(growth)["hard_stop_reasons"])

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

    def test_run_status_rejects_nonterminal_unknown_or_invalid_hash_contract(self):
        hashes = {key: "x" for key in ("code", "model", "config", "direction", "input", "noise")}
        for status in TERMINAL_STATUSES:
            self.assertEqual(build_run_status(status, hashes=hashes)["status"], status)
        for status in ("PREFLIGHT", "UNKNOWN", ""):
            with self.assertRaises(ValueError):
                build_run_status(status, hashes=hashes)
        for bad in ({}, {**hashes, "extra": "x"}, {**hashes, "code": ""}, {**hashes, "code": None}):
            with self.assertRaises(ValueError):
                build_run_status("AWAITING_REVIEW", hashes=bad)


if __name__ == "__main__":
    unittest.main()
