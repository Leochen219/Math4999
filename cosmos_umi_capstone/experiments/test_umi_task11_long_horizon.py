"""CPU contracts for Task 11 five-chunk real-trajectory routing and metrics."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from umi_task11_long_horizon import (
    HorizonTrajectory,
    build_schedule_call_plan,
    compare_exact_repeat,
    compute_endpoint_rgb_metrics,
    compute_error_geometry,
    compute_masked_error_geometry,
    execute_schedule,
    evaluate_resource_gate,
    lock_run_identity,
    task8_array_hash,
    validate_call_result,
)


def _trajectory() -> HorizonTrajectory:
    # Small spatial dimensions keep this CPU fixture light; production
    # preflight separately locks the admitted RGB geometry to 256 x 256.
    rgb = np.broadcast_to(np.arange(81, dtype=np.float32)[:, None, None, None],
                          (81, 3, 2, 2)).copy() / 80.0
    states = np.zeros((81, 7), dtype=np.float32)
    original = np.zeros((80, 7), dtype=np.float32)
    raw = np.stack([np.full((16, 10), index, dtype=np.float32) for index in range(5)])
    normalized = raw / 5.0
    return HorizonTrajectory(record_index=15, episode_id="record-15", rgb=rgb, states=states,
                             original_actions=original, raw_actions=raw,
                             normalized_actions=normalized, prompt="sweep into pile")


class TrajectoryWindowTests(unittest.TestCase):
    def test_windows_use_exact_contiguous_real_observations_and_two_official_chunks(self):
        trajectory = _trajectory()
        expected = {
            1: (0, 0, (0, 1)),
            2: (0, 1, (0, 1)),
            3: (16, 1, (1, 2)),
            4: (32, 1, (2, 3)),
            5: (48, 1, (3, 4)),
        }
        for horizon, (start, local_chunk, action_chunks) in expected.items():
            with self.subTest(horizon=horizon):
                window = trajectory.window(horizon)
                self.assertEqual(window.global_frame_indexes, tuple(range(start, start + 33)))
                self.assertEqual(window.global_action_chunks, action_chunks)
                self.assertEqual(window.local_chunk_index, local_chunk)
                self.assertEqual(window.rgb.shape, (33, 3, 2, 2))
                self.assertAlmostEqual(float(window.rgb[0, 0, 0, 0]), start / 80.0, places=7)
                self.assertAlmostEqual(float(window.rgb[-1, 0, 0, 0]), (start + 32) / 80.0, places=7)
                np.testing.assert_array_equal(window.actions[0], trajectory.normalized_actions[action_chunks[0]])
                np.testing.assert_array_equal(window.actions[1], trajectory.normalized_actions[action_chunks[1]])

    def test_fixed_record_and_eighty_transition_boundary_fail_closed(self):
        trajectory = _trajectory()
        with self.assertRaisesRegex(ValueError, "record 15"):
            HorizonTrajectory(record_index=14, episode_id="wrong", rgb=trajectory.rgb, states=trajectory.states,
                              original_actions=trajectory.original_actions, raw_actions=trajectory.raw_actions,
                              normalized_actions=trajectory.normalized_actions, prompt=trajectory.prompt)
        with self.assertRaisesRegex(ValueError, "80 transitions"):
            HorizonTrajectory(record_index=15, episode_id="short", rgb=trajectory.rgb, states=trajectory.states,
                              original_actions=trajectory.original_actions[:79], raw_actions=trajectory.raw_actions,
                              normalized_actions=trajectory.normalized_actions, prompt=trajectory.prompt)


class CallPlanAndEvidenceTests(unittest.TestCase):
    def test_schedules_route_locked_seeds_to_g0_repeat_and_matching_tf_ar_horizons(self):
        first = build_schedule_call_plan(0)
        second = build_schedule_call_plan(1)
        self.assertEqual(len(first), 10)
        self.assertEqual(len(second), 10)
        self.assertEqual([(row["call"], row["horizon"], row["mode"]) for row in first], [
            ("G0", 1, "common"), ("G0_repeat", 1, "repeat"),
            ("TF", 2, "teacher_forced"), ("AR", 2, "autoregressive"),
            ("TF", 3, "teacher_forced"), ("AR", 3, "autoregressive"),
            ("TF", 4, "teacher_forced"), ("AR", 4, "autoregressive"),
            ("TF", 5, "teacher_forced"), ("AR", 5, "autoregressive"),
        ])
        self.assertEqual([row["seed"] for row in first], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        self.assertEqual([row["seed"] for row in second], [5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
        self.assertEqual([row["global_condition_frame"] for row in first if row["call"] == "TF"],
                         [16, 32, 48, 64])

    def test_call_result_requires_actual_seed_action_consumption_noise_and_all_step_condition(self):
        action = np.arange(160, dtype=np.float32).reshape(16, 10)
        condition = np.array([[1.0, 2.0]], dtype=np.float32)
        mask = np.array([[True, False]])
        carrier = np.array([[1.0, 0.0]], dtype=np.float32)
        action_hash = "74565be3c14cac652b6e4c1ed186d46da1344e56826745706abc85cfe1152e5d"
        token_hash = "44e964c69932da99ab4cf5374c934eac8d35367f676d0c5097acec4d13fa8585"
        noise_hash = "b" * 64
        spec = {"sample_id": "TF_h2_seed1", "seed": 1, "horizon": 2, "mode": "teacher_forced"}
        result = {
            "action": action.copy(), "action_hash": action_hash,
            "packed_action_token_hashes": [token_hash] * 30,
            "action_consumption": {"all_steps_match": True, "steps": 30,
                                   "expected_token_hash": token_hash,
                                   "consumed_token_hashes": [token_hash] * 30},
            "provenance": {"action_evidence": {
                "action_hash": action_hash, "effective_action_hash": token_hash,
                "effective_action": {"dtype": "float32", "shape": [16, 64],
                                     "sha256": "cefbf5fe2acfae4dd7377be7b0124d73b70ea2cade9dada870784c91baaffebe"}}},
            "prediction_noise_hash": noise_hash,
            "condition_input_fp32": condition.copy(),
            "condition_steps_fp32": np.repeat(carrier[None], 30, axis=0),
            "generation": {"sampler_generator_seeds": [1] * 30,
                           "noise_evidence": {"seed": 1, "prepare_seed": 1}},
        }
        evidence = validate_call_result(spec, result, expected_action=action,
                                        expected_condition_input=condition,
                                        expected_condition_carrier=carrier,
                                        condition_mask=mask)
        self.assertEqual(evidence["noise_hash"], noise_hash)
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_call_result({**spec, "seed": 2}, result, expected_action=action,
                                 expected_condition_input=condition, expected_condition_carrier=carrier,
                                 condition_mask=mask)
        changed = {**result, "condition_steps_fp32": np.repeat(np.zeros_like(carrier)[None], 30, axis=0)}
        with self.assertRaisesRegex(ValueError, "condition"):
            validate_call_result(spec, changed, expected_action=action, expected_condition_input=condition,
                                 expected_condition_carrier=carrier, condition_mask=mask)

    def test_repeat_gate_compares_full_float_outputs_and_consumed_action_evidence(self):
        reference = {"output_full": np.array([1.0], dtype=np.float32),
                     "generated_rgb": np.array([2.0], dtype=np.float32),
                     "decoded_rgb_full": np.array([2.0], dtype=np.float32),
                     "decoded_last_rgb": np.array([2.0], dtype=np.float32),
                     "encoded_condition": np.array([3.0], dtype=np.float32),
                     "condition_input_fp32": np.array([4.0], dtype=np.float32),
                     "action": np.array([5.0], dtype=np.float32),
                     "prediction_noise_hash": "a" * 64,
                     "packed_action_token_hashes": ["b" * 64] * 30}
        compare_exact_repeat(reference, {key: value.copy() if isinstance(value, np.ndarray) else value
                                         for key, value in reference.items()})
        changed = dict(reference, generated_rgb=np.array([2.0001], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "repeat"):
            compare_exact_repeat(reference, changed)

    def test_schedule_feeds_ar_its_own_previous_float_endpoint_and_tf_real_endpoints(self):
        trajectory = _trajectory()
        condition_mask = np.ones((1, 2), dtype=bool)
        def encode(frame):
            return np.array([[frame[0, 0, 0], frame[1, 0, 0]]], dtype=np.float32)
        truth_conditions = {frame: encode(trajectory.rgb[frame]) for frame in (0, 16, 32, 48, 64, 80)}
        seen = []

        def live_call(spec, window):
            action = window.actions[spec["local_action_chunk_index"]]
            condition_rgb = spec["condition_rgb"]
            condition = encode(condition_rgb)
            value = 0.125 if spec["horizon"] == 1 else (0.2 + spec["horizon"] / 100.0
                                                        + (0.01 if spec["call"] == "AR" else 0.0))
            generated = np.full((3, 16, 2, 2), value, dtype=np.float32)
            final = generated[:, -1].copy()
            encoded = encode(final)
            carrier = condition.copy()
            action_hash = task8_array_hash(action)
            padded = np.zeros((16, 64), dtype=np.float32)
            padded[:, :10] = action
            padded_hash = task8_array_hash(padded)
            descriptor_hash = hashlib.sha256(
                (str(padded.dtype) + "\0" + repr(tuple(padded.shape)) + "\0").encode("ascii")
                + padded.tobytes(order="C")).hexdigest()
            noise_hash = hashlib.sha256(f"noise-{spec['seed']}".encode()).hexdigest()
            result = {
                "output_full": np.full((1,), spec["seed"], dtype=np.float32),
                "generated_rgb": generated, "decoded_rgb_full": np.concatenate(
                    [generated[:, :1], generated], axis=1), "decoded_last_rgb": final,
                "encoded_condition": encoded, "condition_input_fp32": condition,
                "condition_steps_fp32": np.repeat(carrier[None, :, :], 30, axis=0),
                "action": action.copy(), "action_hash": action_hash,
                "packed_action_token_hashes": [padded_hash] * 30,
                "action_consumption": {"all_steps_match": True, "steps": 30,
                                       "expected_token_hash": padded_hash,
                                       "consumed_token_hashes": [padded_hash] * 30},
                "prediction_noise_hash": noise_hash,
                "generation": {"sampler_generator_seeds": [spec["seed"]] * 30,
                               "noise_evidence": {"seed": spec["seed"], "prepare_seed": spec["seed"]}},
                "provenance": {"condition_source": spec["condition_source"],
                    "condition_rgb_sha256": hashlib.sha256(
                        (str(condition_rgb.dtype) + "\0" + repr(tuple(condition_rgb.shape)) + "\0").encode("ascii")
                        + np.ascontiguousarray(condition_rgb).tobytes(order="C")).hexdigest(),
                    "action_evidence": {"action_hash": action_hash, "effective_action_hash": padded_hash,
                        "effective_action": {"dtype": "float32", "shape": [16, 64],
                                             "sha256": descriptor_hash}}},
            }
            seen.append((dict(spec), window, result))
            return result

        result = execute_schedule(trajectory, 0, live_call, truth_conditions=truth_conditions,
                                  condition_mask=condition_mask,
                                  embed_condition=lambda value: np.array(value, copy=True))
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(len(seen), 10)
        tf_h2 = seen[2]
        ar_h2 = seen[3]
        tf_h3 = seen[4]
        ar_h3 = seen[5]
        np.testing.assert_array_equal(tf_h2[0]["condition_rgb"], trajectory.rgb[16])
        np.testing.assert_array_equal(seen[0][2]["condition_input_fp32"], truth_conditions[0])
        np.testing.assert_array_equal(tf_h2[2]["condition_input_fp32"], truth_conditions[16])
        np.testing.assert_array_equal(ar_h2[0]["condition_rgb"], seen[0][2]["decoded_last_rgb"])
        np.testing.assert_array_equal(ar_h2[2]["condition_input_fp32"], seen[0][2]["encoded_condition"])
        np.testing.assert_array_equal(tf_h3[0]["condition_rgb"], trajectory.rgb[32])
        np.testing.assert_array_equal(tf_h3[2]["condition_input_fp32"], truth_conditions[32])
        np.testing.assert_array_equal(ar_h3[0]["condition_rgb"], ar_h2[2]["decoded_last_rgb"])
        np.testing.assert_array_equal(ar_h3[2]["condition_input_fp32"], ar_h2[2]["encoded_condition"])

        def bad_decoded_window(spec, window):
            result = live_call(spec, window)
            if spec["call"] == "G0":
                result["decoded_rgb_full"] = result["decoded_rgb_full"].copy()
                result["decoded_rgb_full"][:, 4] += np.float32(0.1)
            return result

        with self.assertRaisesRegex(ValueError, "16-frame float chunk"):
            execute_schedule(trajectory, 0, bad_decoded_window, truth_conditions=truth_conditions,
                             condition_mask=condition_mask,
                             embed_condition=lambda value: np.array(value, copy=True))

    def test_schedule_requires_shared_truth_encoding_for_all_six_endpoints(self):
        trajectory = _trajectory()
        with self.assertRaisesRegex(ValueError, "truth endpoints"):
            execute_schedule(trajectory, 0, lambda *_: {}, truth_conditions={16: np.ones((1,), dtype=np.float32)},
                             condition_mask=np.ones((1,), dtype=bool), embed_condition=lambda value: value)


class MetricAndResourceTests(unittest.TestCase):
    def test_rgb_metrics_and_tf_ar_error_identity_use_time_aligned_endpoint_values(self):
        truth = np.array([[[0.0, 0.5]], [[0.5, 0.5]], [[1.0, 0.5]]], dtype=np.float32)
        tf = np.array([[[0.5, 0.5]], [[0.5, 0.5]], [[1.0, 0.5]]], dtype=np.float32)
        ar = np.array([[[0.25, 0.5]], [[0.75, 0.5]], [[1.0, 0.5]]], dtype=np.float32)
        metrics = compute_endpoint_rgb_metrics(tf, truth)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(0.25 / 6.0), places=7)
        self.assertAlmostEqual(metrics["mae"], 0.5 / 6.0, places=7)
        self.assertAlmostEqual(metrics["psnr"], -20.0 * np.log10(metrics["rmse"]), places=7)
        geometry = compute_error_geometry(tf, ar, truth)
        self.assertAlmostEqual(geometry["squared_error_difference"],
                               geometry["cross_term_2dot_over_n"] + geometry["feedback_change_squared"], places=14)
        self.assertAlmostEqual(geometry["identity_residual"], 0.0, places=14)
        mask = np.array([[[True, False]], [[True, False]], [[True, False]]])
        latent_geometry = compute_masked_error_geometry(tf, ar, truth, mask)
        self.assertAlmostEqual(latent_geometry["identity_residual"], 0.0, places=14)
        exact = compute_endpoint_rgb_metrics(truth, truth)
        self.assertIsNone(exact["psnr"])
        self.assertEqual(exact["psnr_status"], "infinite_exact_match")

    def test_adjacent_ar_metrics_include_scalar_endpoint_rmse_delta(self):
        from umi_task11_long_horizon import adjacent_ar_change
        truth = np.zeros((3, 1, 2), dtype=np.float32)
        previous = np.full_like(truth, 0.25)
        current = np.full_like(truth, 0.5)
        metrics = adjacent_ar_change(previous, current, truth, truth)
        self.assertAlmostEqual(metrics["previous_endpoint_rmse"], 0.25)
        self.assertAlmostEqual(metrics["current_endpoint_rmse"], 0.5)
        self.assertAlmostEqual(metrics["endpoint_rmse_delta"], 0.25)

    def test_resource_gate_keeps_strict_gpu_cgroup_swap_and_five_gib_forecast(self):
        safe = {"gpu_used_gib": 2.0, "gpu_free_gib": 90.0, "gpu_reserved_gib": 2.0,
                "gpu_peak_allocated_gib": 10.0, "gpu_peak_nvml_used_gib": 12.0,
                "ram_available_gib": 600.0, "rss_gib": 10.0, "swap_used_gib": 0.0,
                "disk_free_gib": 20.0, "cgroup_memory_limited": True,
                "cgroup_memory_limit_gib": 110.0, "cgroup_memory_current_gib": 18.0,
                "cgroup_memory_free_gib": 92.0}
        gate = evaluate_resource_gate(safe, phase="formal", remaining_calls=20, sample_bytes=1024 * 1024)
        self.assertGreaterEqual(gate["snapshot"]["forecast_free_gib"], 5.0)
        for changes, message in (({"gpu_free_gib": 19.9}, "GPU_FREE_LOW"),
                                 ({"cgroup_memory_free_gib": 9.9,
                                   "cgroup_memory_current_gib": 100.1}, "CGROUP_MEMORY_HEADROOM_CRITICAL"),
                                 ({"swap_used_gib": 0.01}, "SWAP_IN_USE")):
            with self.subTest(changes=changes), self.assertRaisesRegex(RuntimeError, message):
                evaluate_resource_gate({**safe, **changes}, phase="formal",
                                        remaining_calls=20, sample_bytes=1024 * 1024)
        with self.assertRaisesRegex(RuntimeError, "DISK_FORECAST_LOW"):
            evaluate_resource_gate({**safe, "disk_free_gib": 5.1}, phase="formal",
                                   remaining_calls=20, sample_bytes=10 * 2**30)


class ResumeIdentityTests(unittest.TestCase):
    def test_resume_is_bound_to_the_full_input_model_and_code_identity(self):
        identity = {"record_index": 15, "source_sha256": "a" * 64,
                    "seed_schedules": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
                    "code_sha256": {"runner": "b" * 64}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_run_identity(root, identity, resume=False)
            with self.assertRaisesRegex(FileExistsError, "resume"):
                lock_run_identity(root, identity, resume=False)
            self.assertEqual(lock_run_identity(root, identity, resume=True), identity)
            changed = {**identity, "seed_schedules": [[0, 1, 2, 3, 5], [5, 6, 7, 8, 9]]}
            with self.assertRaisesRegex(ValueError, "identity"):
                lock_run_identity(root, changed, resume=True)


if __name__ == "__main__":
    unittest.main()
