from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from task8_frozen.task8_formal import (
    FormalBlocked,
    FormalError,
    VerifiedFormalExecutor,
    build_binding,
    _save_gt_conditions,
    _task8_source_paths,
    TASK8_SOURCE_MODULES,
    run_formal,
    validate_smoke_run,
    _array_hash,
)
from task8_frozen.run_umi_task8_experiment import Task8SampleStore


class _FakeMonitor:
    def __init__(self):
        self.samples = []
        self.capture_rows = []
        self.stopped = False
        self._monitor = self
        self.capture_disk_gib = 20.0

    def start(self):
        return self

    def stop(self):
        self.stopped = True

    def capture_sample(self, sample_id, phase, remaining, run_dir=None):
        row = {"sample_id": sample_id, "phase": phase, "remaining_samples": remaining,
               "mean_success_sample_bytes": 100, "decision_status": "OK",
               **self.check(phase="formal", starting_new_sample=False)}
        row["disk_free_gib"] = self.capture_disk_gib
        self.capture_rows.append(row)
        return row

    def check(self, *, phase, starting_new_sample):
        snapshot = {"gpu_used_gib": 0, "gpu_free_gib": 25, "gpu_reserved_gib": 0,
                    "gpu_peak_allocated_gib": 0, "gpu_peak_nvml_used_gib": 0,
                    "ram_available_gib": 400, "rss_gib": 1, "swap_used_gib": 0,
                    "disk_free_gib": 20, "cgroup_memory_limited": False,
                    "cgroup_memory_limit_gib": None, "cgroup_memory_current_gib": None,
                    "cgroup_memory_free_gib": None, "phase": phase,
                    "starting_new_sample": starting_new_sample}
        self.samples.append(snapshot)
        return snapshot


class _FakeLiveExecutor:
    def __init__(self, size=2):
        self._last_g0_frame = None
        self.size = size

    def __call__(self, spec):
        frame = np.full((3, self.size, self.size), 0.25, np.float32)
        condition = np.full((1, 2), 0.5, np.float32)
        action = np.full((16, 10), 0.1, np.float32)
        return {"output_full": np.full((2, 2), 0.75, np.float32),
                "generated_rgb": np.full((3, 16, self.size, self.size), 0.25, np.float32),
                "decoded_last_rgb": frame, "condition_rgb": frame.copy(),
                "condition_input_fp32": condition.copy(), "encoded_condition": condition.copy(),
                "action": action, "action_hash": _array_hash(action), "prediction_noise_hash": "b" * 64,
                "packed_action_token_hashes": ["c" * 64] * 30,
                "action_consumption": {"all_steps_match": True, "steps": 30,
                                        "expected_token_hash": "c" * 64,
                                        "consumed_token_hashes": ["c" * 64] * 30},
                "precision": {"G": "float32", "D": "float32", "E": "float32"},
                "provenance": {"condition_source": spec["condition_source"]}}


class FormalTask8Tests(unittest.TestCase):
    def _make_cpu_fixture(self, root: Path, *, cleanup=None):
        run_root = root / "formal-run"; smoke = root / "smoke"
        inputs = root / "inputs.npz"; metadata = root / "metadata.json"
        rgb = np.zeros((33, 3, 256, 256), np.float32); actions = np.zeros((2, 16, 10), np.float32)
        np.savez(inputs, rgb=rgb, actions=actions, prompt=np.asarray("put block"),
                 provenance_json=np.asarray('{"source":"fixture"}'))
        metadata.write_text(json.dumps({"selected": {"language": "put block"}}), encoding="utf-8")
        framework = root / "framework"; framework.mkdir(); (framework / "code.py").write_text("fixture", encoding="utf-8")
        checkpoint = root / "checkpoint"; checkpoint.write_bytes(b"model")
        vae = root / "vae"; vae.write_bytes(b"vae")
        store = Task8SampleStore(smoke / "samples")
        action = np.zeros((16, 10), np.float32); token = "c" * 64
        smoke_result = {"output_full": np.zeros((2, 2), np.float32),
                        "generated_rgb": np.zeros((3, 16, 2, 2), np.float32),
                        "decoded_last_rgb": np.zeros((3, 2, 2), np.float32),
                        "encoded_condition": np.zeros((1, 2), np.float32),
                        "condition_input_fp32": np.zeros((1, 2), np.float32),
                        "prediction_noise_hash": "b" * 64,
                        "packed_action_token_hashes": [token] * 30,
                        "action_consumption": {"all_steps_match": True, "steps": 30,
                                                "consumed_token_hashes": [token] * 30},
                        "action": action, "action_hash": _array_hash(action),
                        "precision": {"G": "float32"}}
        store.write_success("engineering_smoke", smoke_result)
        sample_bytes = sum(path.stat().st_size for path in (smoke / "samples" / "engineering_smoke").rglob("*")
                           if path.is_file())
        (smoke / "run_status.json").write_text(json.dumps({
            "status": "ENGINEERING_SMOKE_COMPLETE", "measured_sample_bytes": sample_bytes,
            "forecast_free_gib_after_four_calls": 10.0
        }), encoding="utf-8")
        context = types.SimpleNamespace(
            encoder=types.SimpleNamespace(encode=lambda frame, precision: {
                "arrays": {"actual_output": np.asarray([[0.5, 1.5]], np.float32)},
                "evidence": {"actual_output_dtype": "float32"}}),
            feedback=types.SimpleNamespace(mask=np.asarray([[True, False]], bool),
                                          condition_indexes=(0,), temporal_axis=0),
            cleanup=cleanup or (lambda: None),
        )
        monitor = _FakeMonitor()
        config = {"run_root": run_root, "smoke_dir": smoke, "inputs_npz": inputs,
                  "metadata_json": metadata, "framework_root": framework,
                  "checkpoint": checkpoint, "vae": vae}
        return config, context, monitor

    def _run_cpu_formal(self, config, context, monitor):
        with patch("task8_frozen.task8_formal._framework_commit", return_value="f" * 40):
            return run_formal(config, release=True, context_loader=lambda **kwargs: context,
                              monitor_factory=lambda path: monitor,
                              executor_factory=lambda context, monitor: _FakeLiveExecutor(size=256))

    def test_release_gate_prevents_context_loader(self):
        with self.assertRaises(FormalBlocked):
            from task8_frozen.task8_formal import run_formal
            run_formal({}, release=False, context_loader=lambda: self.fail("must not load"))

    def test_binding_contains_exact_eight_sha256_identities(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {name: root / name for name in ("model", "vae", "inputs")}
            for path in paths.values():
                path.write_bytes(path.name.encode())
            binding = build_binding(code_paths=[root / "model"], model_path=paths["model"],
                                    vae_path=paths["vae"], input_path=paths["inputs"],
                                    actions=np.zeros((2, 16, 10), np.float32), metadata={"x": 1},
                                    framework_commit="f" * 40)
        self.assertEqual(set(binding), {"code", "model", "vae", "config", "data", "actions", "preprocessing", "noise"})
        self.assertTrue(all(isinstance(value, str) and len(value) == 64 for value in binding.values()))

    def test_support_module_content_and_framework_commit_are_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_root = root / "experiments"; source_root.mkdir()
            for module in TASK8_SOURCE_MODULES:
                (source_root / f"{module}.py").write_text(f"source:{module}", encoding="utf-8")
            model = root / "model"; model.write_bytes(b"model")
            vae = root / "vae"; vae.write_bytes(b"vae")
            data = root / "input"; data.write_bytes(b"input")
            kwargs = {"code_paths": _task8_source_paths(source_root), "model_path": model, "vae_path": vae,
                      "input_path": data, "actions": np.zeros((2, 16, 10), np.float32),
                      "metadata": {}, "framework_commit": "a" * 40}
            first = build_binding(**kwargs)
            (source_root / "umi_task7_encoder.py").write_text("changed support implementation", encoding="utf-8")
            support_changed = build_binding(**kwargs)
            framework_changed = build_binding(**{**kwargs, "framework_commit": "b" * 40})
        self.assertNotEqual(first["code"], support_changed["code"])
        self.assertNotEqual(first["code"], framework_changed["code"])

    def test_cuda_synchronization_failure_is_not_swallowed(self):
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True,
                                       synchronize=lambda: (_ for _ in ()).throw(RuntimeError("CUDA launch failed")))
        )
        wrapper = VerifiedFormalExecutor(_FakeLiveExecutor())
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "CUDA launch failed"):
                wrapper._sync_and_capture("G0")

    def test_ground_truth_encoder_saves_condition_only_mask_matching_output(self):
        carrier_mask = np.zeros((2, 3, 2, 2), dtype=bool)
        carrier_mask[:, 1, :, :] = True

        class Encoder:
            def encode(self, frame, precision):
                return {"arrays": {"actual_output": np.ones((2, 1, 2, 2), np.float32)},
                        "evidence": {"actual_output_dtype": "float32"}}

        context = types.SimpleNamespace(
            encoder=Encoder(),
            feedback=types.SimpleNamespace(mask=carrier_mask, condition_indexes=(1,), temporal_axis=1),
        )
        batch = types.SimpleNamespace(rgb=np.zeros((33, 3, 2, 2), np.float32))
        with tempfile.TemporaryDirectory() as temp:
            gt16, gt32 = _save_gt_conditions(context, batch, Path(temp))
            with np.load(Path(temp) / "ground_truth_conditions.npz", allow_pickle=False) as saved:
                condition_mask = saved["condition_mask"]
                full_mask = saved["full_condition_mask"]
        self.assertEqual(condition_mask.shape, gt16.shape)
        self.assertEqual(condition_mask.shape, gt32.shape)
        self.assertEqual(int(condition_mask.sum()), 8)
        self.assertEqual(full_mask.shape, carrier_mask.shape)

    def test_wrapper_rejects_ar_frame_before_setting_live_state(self):
        live = _FakeLiveExecutor()
        wrapper = VerifiedFormalExecutor(live)
        g0 = wrapper({"call": "G0", "condition_source": "real_x0"})
        self.assertIsNotNone(g0)
        with self.assertRaises(FormalError):
            wrapper({"call": "AR2", "condition_source": "g0_float_last_fp32",
                     "condition_rgb": np.zeros((3, 2, 2), np.float32)})
        self.assertTrue(np.array_equal(live._last_g0_frame, np.full((3, 2, 2), 0.25, np.float32)) is False)

    def test_smoke_requires_complete_status_and_verified_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "run_status.json").write_text(json.dumps({"status": "BLOCKED"}), encoding="utf-8")
            with self.assertRaises(FormalError):
                validate_smoke_run(root)

    def test_smoke_recorded_disk_forecast_is_not_used_as_fresh_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            action = np.zeros((16, 10), np.float32); token = "c" * 64
            result = {"output_full": np.zeros((2, 2), np.float32),
                      "generated_rgb": np.zeros((3, 16, 2, 2), np.float32),
                      "decoded_last_rgb": np.zeros((3, 2, 2), np.float32),
                      "encoded_condition": np.zeros((2, 2), np.float32),
                      "condition_input_fp32": np.zeros((2, 2), np.float32),
                      "prediction_noise_hash": "b" * 64,
                      "packed_action_token_hashes": [token] * 30,
                      "action_consumption": {"all_steps_match": True, "steps": 30,
                                              "consumed_token_hashes": [token] * 30},
                      "action": action, "action_hash": _array_hash(action),
                      "precision": {"G": "float32"}}
            Task8SampleStore(root / "samples").write_success("engineering_smoke", result)
            sample_bytes = sum(path.stat().st_size for path in (root / "samples" / "engineering_smoke").rglob("*")
                               if path.is_file())
            (root / "run_status.json").write_text(json.dumps({
                "status": "ENGINEERING_SMOKE_COMPLETE", "measured_sample_bytes": sample_bytes,
                "forecast_free_gib_after_four_calls": 0.1
            }), encoding="utf-8")
            validated = validate_smoke_run(root)
        self.assertEqual(validated["measured_sample_bytes"], sample_bytes)

    def test_cpu_orchestrator_keeps_formal_root_empty_until_runner_starts(self):
        with tempfile.TemporaryDirectory() as temp:
            config, context, monitor = self._make_cpu_fixture(Path(temp))
            result = self._run_cpu_formal(config, context, monitor)
            run_root = Path(config["run_root"])
            self.assertEqual(result["formal"]["formal_calls"], 4)
            self.assertTrue((run_root / "formal" / "run_status.json").is_file())
            self.assertTrue(monitor.stopped)
            self.assertEqual([row["remaining_samples"] for row in monitor.capture_rows], [3, 2, 1, 0])
            self.assertEqual(result["analysis"]["conditions"]["G0_x16"]["rms"], 0.0)
            with np.load(run_root / "formal" / "ground_truth_conditions.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["condition_mask"].shape, (1, 2))
            self.assertTrue((run_root / "task8_formal_status.json").is_file())

    def test_cleanup_failure_stops_monitor_and_writes_run_level_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            config, context, monitor = self._make_cpu_fixture(
                Path(temp), cleanup=lambda: (_ for _ in ()).throw(RuntimeError("cleanup fixture failure")))
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                self._run_cpu_formal(config, context, monitor)
            self.assertTrue(monitor.stopped)
            run_root = Path(config["run_root"])
            status = json.loads((run_root / "task8_formal_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "FAILED")
            self.assertEqual(status["stage"], "cleanup")
            formal_status = json.loads((run_root / "formal" / "run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(formal_status["status"], "COMPLETE")

    def test_post_cleanup_disk_forecast_stops_before_next_call_and_keeps_published_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            config, context, monitor = self._make_cpu_fixture(Path(temp))
            monitor.capture_disk_gib = 5.0
            with self.assertRaisesRegex(RuntimeError, "DISK_FORECAST_LOW"):
                self._run_cpu_formal(config, context, monitor)
            run_root = Path(config["run_root"])
            status = json.loads((run_root / "formal" / "run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "RESOURCE_STOP")
            self.assertEqual(status["completed_samples"], ["G0_real_x0_seed0"])
            self.assertTrue((run_root / "formal" / "samples" / "G0_real_x0_seed0" / "status.json").is_file())
            self.assertTrue(monitor.stopped)

    def test_resume_binding_mismatch_is_rejected_before_any_write_or_model_load(self):
        with tempfile.TemporaryDirectory() as temp:
            config, context, monitor = self._make_cpu_fixture(Path(temp))
            self._run_cpu_formal(config, context, monitor)
            run_root = Path(config["run_root"])
            binding_path = run_root / "setup" / "binding.json"
            prior_binding = binding_path.read_bytes()
            status_path = run_root / "task8_formal_status.json"
            prior_status = status_path.read_bytes()
            metadata_path = Path(config["metadata_json"])
            metadata_path.write_text(json.dumps({"selected": {"language": "different prompt"}}), encoding="utf-8")
            resumed_config = {**config, "resume": True}
            with patch("task8_frozen.task8_formal._framework_commit", return_value="f" * 40):
                with self.assertRaisesRegex(FormalBlocked, "resume binding differs"):
                    run_formal(resumed_config, release=True,
                               context_loader=lambda **kwargs: self.fail("must reject before model load"))
            self.assertEqual(binding_path.read_bytes(), prior_binding)
            self.assertEqual(status_path.read_bytes(), prior_status)
            self.assertFalse((run_root / "setup" / "attempts").exists())


if __name__ == "__main__":
    unittest.main()
