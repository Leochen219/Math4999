from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from task8_frozen.run_umi_task8_experiment import (
    BlockedExecution,
    FORMAL_CALL_PLAN,
    ResourceStop,
    ResumeMismatch,
    Task8SampleStore,
    _array_hash,
    build_formal_call_plan,
    evaluate_task8_resources,
    run_task8,
)


class Task8RunnerTests(unittest.TestCase):
    @staticmethod
    def _binding():
        keys = ("code", "model", "vae", "config", "data", "actions", "preprocessing", "noise")
        return {key: ("a" * 63 + format(index, "x")) for index, key in enumerate(keys)}

    @staticmethod
    def _snapshot(disk=20.0):
        return {"gpu_used_gib": 0, "gpu_free_gib": 25, "gpu_reserved_gib": 0,
                "gpu_peak_allocated_gib": 0, "gpu_peak_nvml_used_gib": 0,
                "ram_available_gib": 400, "rss_gib": 1, "swap_used_gib": 0,
                "disk_free_gib": disk, "cgroup_memory_limited": False,
                "cgroup_memory_limit_gib": None, "cgroup_memory_current_gib": None,
                "cgroup_memory_free_gib": None}

    @staticmethod
    def _formal_result(spec, *, noise="b" * 64, condition=None, action_hash=None,
                       output_offset=0.0, token_hash="c" * 64):
        generated = np.full((3, 17, 2, 2), 0.25 + output_offset, dtype=np.float32)
        condition = np.full((2, 2), 0.5, dtype=np.float32) if condition is None else np.asarray(condition, np.float32)
        action = np.full((16, 10), 0.1, dtype=np.float32)
        return {"output_full": np.full((2, 2), 0.75 + output_offset, dtype=np.float32),
                "generated_rgb": generated, "decoded_last_rgb": generated[:, -1].copy(),
                "condition_input_fp32": condition.copy(), "condition_rgb": generated[:, -1].copy(),
                "encoded_condition": condition.copy(),
                "action": action, "action_hash": action_hash or _array_hash(action),
                "prediction_noise_hash": noise,
                "packed_action_token_hashes": [token_hash] * 30,
                "action_consumption": {"all_steps_match": True, "steps": 30,
                                        "expected_token_hash": token_hash,
                                        "consumed_token_hashes": [token_hash] * 30},
                "precision": {"G": "float32", "D": "float32", "E": "float32"}}

    def _run_kwargs(self, temp, execute, **extra):
        kwargs = {"execute_call": execute, "release": True, "binding": self._binding(),
                  "resource_snapshot": self._snapshot()}
        kwargs.update(extra)
        return kwargs

    def test_formal_plan_has_exact_four_calls(self):
        plan = build_formal_call_plan()
        self.assertEqual([row["sample_id"] for row in plan], [row["sample_id"] for row in FORMAL_CALL_PLAN])
        self.assertEqual([(row["condition_source"], row["chunk_index"], row["seed"]) for row in plan],
                         [("real_x0", 0, 0), ("real_x0", 0, 0), ("real_x16", 1, 1), ("g0_float_last_fp32", 1, 1)])

    def test_plan_mutation_is_rejected_even_when_sample_ids_are_unchanged(self):
        plan = build_formal_call_plan()
        plan[2]["seed"] = 0
        with self.assertRaises(ResumeMismatch):
            run_task8(tempfile.mkdtemp(), execute_call=lambda spec: {}, release=True,
                      binding=self._binding(), resource_snapshot=self._snapshot(), plan=plan)

    def test_task8_formal_disk_policy_keeps_five_gib_reserve_not_preload_ten(self):
        result = evaluate_task8_resources(self._snapshot(disk=6.0), phase="formal", starting_new_sample=True)
        self.assertNotEqual(result["status"], "HARD_STOP")
        self.assertNotEqual(evaluate_task8_resources(self._snapshot(disk=6.0), phase="preload", starting_new_sample=True)["status"], "HARD_STOP")
        high_gpu = self._snapshot(disk=20.0); high_gpu["gpu_used_gib"] = 2.0
        with self.assertRaises(ResourceStop):
            evaluate_task8_resources(high_gpu, phase="preload", starting_new_sample=True)

    def test_generation_requires_explicit_release_and_resource_snapshot(self):
        with self.assertRaises(BlockedExecution):
            run_task8(tempfile.mkdtemp(), execute_call=lambda spec: {}, release=False)

    def test_resources_stop_below_disk_reserve(self):
        snapshot = {"gpu_used_gib": 0, "gpu_free_gib": 10, "gpu_reserved_gib": 0,
                    "gpu_peak_allocated_gib": 0, "gpu_peak_nvml_used_gib": 0,
                    "ram_available_gib": 20, "rss_gib": 1, "swap_used_gib": 0,
                    "disk_free_gib": 4.9, "cgroup_memory_limited": False,
                    "cgroup_memory_limit_gib": None, "cgroup_memory_current_gib": None,
                    "cgroup_memory_free_gib": None}
        with self.assertRaises(ResourceStop):
            evaluate_task8_resources(snapshot, phase="formal", starting_new_sample=True)

    def test_sample_store_does_not_overwrite_success_on_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            store = Task8SampleStore(temp)
            store.write_success("G0", {"status": "success"})
            self.assertEqual(store.prepare("G0", resume=True), "skip")
            self.assertEqual(store.prepare("G0", resume=True), "skip")

    def test_cpu_executor_can_run_only_when_released(self):
        seen = []
        def execute(spec):
            seen.append(spec["sample_id"])
            return self._formal_result(spec)
        run_dir = tempfile.mkdtemp()
        result = run_task8(run_dir, **self._run_kwargs(run_dir, execute))
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(seen, ["G0_real_x0_seed0", "G0_repeat_real_x0_seed0", "TF2_real_x16_seed1", "AR2_g0_float_last_fp32_seed1"])
        status = json.loads((Path(run_dir) / "run_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["completed_samples"], seen)

    def test_post_sample_callback_runs_after_publication_and_reports_remaining_calls(self):
        run_dir = Path(tempfile.mkdtemp())
        observed = []

        def execute(spec):
            return self._formal_result(spec)

        def after_sample(sample_id, remaining):
            sample_path = run_dir / "samples" / sample_id
            self.assertTrue((sample_path / "status.json").is_file())
            status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
            self.assertIsNone(status["current_sample"])
            observed.append((sample_id, remaining))

        run_task8(run_dir, **self._run_kwargs(run_dir, execute), post_sample_callback=after_sample)
        self.assertEqual([remaining for _, remaining in observed], [3, 2, 1, 0])

    def test_post_sample_callback_failure_is_durable_and_stops_before_next_call(self):
        run_dir = Path(tempfile.mkdtemp())
        calls = []

        def execute(spec):
            calls.append(spec["call"])
            return self._formal_result(spec)

        def after_sample(sample_id, remaining):
            raise ResourceStop("fixture post-cleanup stop")

        with self.assertRaises(ResourceStop):
            run_task8(run_dir, **self._run_kwargs(run_dir, execute), post_sample_callback=after_sample)
        self.assertEqual(calls, ["G0"])
        status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "RESOURCE_STOP")
        self.assertEqual(status["completed_samples"], ["G0_real_x0_seed0"])

    def test_binding_requires_all_eight_sha256_identities(self):
        with self.assertRaises(BlockedExecution):
            run_task8(tempfile.mkdtemp(), execute_call=lambda spec: {}, release=True,
                      binding={"code": "c" * 64}, resource_snapshot=self._snapshot())

    def test_failure_writes_atomic_run_status_with_completed_prefix(self):
        seen = []
        def execute(spec):
            seen.append(spec["sample_id"])
            if len(seen) == 2:
                raise RuntimeError("fixture failure")
            return self._formal_result(spec)
        run_dir = tempfile.mkdtemp()
        with self.assertRaises(RuntimeError):
            run_task8(run_dir, **self._run_kwargs(run_dir, execute))
        status = json.loads((Path(run_dir) / "run_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "FAILED")
        self.assertEqual(status["completed_samples"], ["G0_real_x0_seed0"])
        self.assertEqual(status["current_sample"], "G0_repeat_real_x0_seed0")

    def test_first_executor_failure_marks_generation_started(self):
        def execute(spec):
            raise RuntimeError("first-call failure")
        run_dir = tempfile.mkdtemp()
        with self.assertRaises(RuntimeError):
            run_task8(run_dir, **self._run_kwargs(run_dir, execute))
        status = json.loads((Path(run_dir) / "run_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "FAILED")
        self.assertTrue(status["generation_started"])

    def test_repeat_requires_condition_latent_and_decoded_bitwise_identity(self):
        def execute(spec):
            result = self._formal_result(spec)
            if spec["call"] == "G0_repeat":
                result["condition_input_fp32"] = result["condition_input_fp32"] + 1.0
            return result
        run_dir = tempfile.mkdtemp()
        with self.assertRaises(ResumeMismatch):
            run_task8(run_dir, **self._run_kwargs(run_dir, execute))

    def test_tf_ar_require_paired_noise_and_action_consumption(self):
        def execute(spec):
            return self._formal_result(spec, noise=("b" if spec["call"] != "AR2" else "d") * 64)
        run_dir = tempfile.mkdtemp()
        with self.assertRaises(ResumeMismatch):
            run_task8(run_dir, **self._run_kwargs(run_dir, execute))

    def test_ar_condition_must_be_g0_decoded_last_frame(self):
        def execute(spec):
            result = self._formal_result(spec)
            if spec["call"] == "AR2":
                result["condition_input_fp32"] = result["condition_input_fp32"] + 1.0
            return result
        run_dir = tempfile.mkdtemp()
        with self.assertRaises(ResumeMismatch):
            run_task8(run_dir, **self._run_kwargs(run_dir, execute))

    def test_resume_rehydrates_g0_feedback_frame_before_ar(self):
        first_calls = []
        def first_execute(spec):
            first_calls.append(spec["call"])
            if spec["call"] == "TF2":
                raise RuntimeError("pause after G0")
            return self._formal_result(spec)
        run_dir = tempfile.mkdtemp()
        with self.assertRaises(RuntimeError):
            run_task8(run_dir, **self._run_kwargs(run_dir, first_execute))
        resumed_calls = []
        ar_request = {}
        def resume_execute(spec):
            resumed_calls.append(spec["call"])
            if spec["call"] == "AR2":
                ar_request.update(spec)
            return self._formal_result(spec)
        result = run_task8(run_dir, **self._run_kwargs(run_dir, resume_execute, resume=True))
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(resumed_calls, ["TF2", "AR2"])
        np.testing.assert_array_equal(ar_request["condition_rgb"], np.full((3, 2, 2), 0.25, np.float32))


if __name__ == "__main__":
    unittest.main()
