"""CPU-only Task 11 runner gates; these tests never import Torch or generate."""
from __future__ import annotations

import unittest
import json
import hashlib
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np

import run_umi_task11_long_horizon as runner
from run_umi_task11_long_horizon import parse_args, validate_cli_contract, validate_smoke_status


def _argv(stage: str = "preflight") -> list[str]:
    values = ["--stage", stage, "--run-dir", "/tmp/task11-five-chunk",
              "--input-run", "/tmp/task11-preflight-input",
              "--dataset-root", "/tmp/bridge", "--framework-root", "/tmp/framework",
              "--checkpoint", "/tmp/model", "--vae", "/tmp/vae.pth",
              "--normalizer-stats", "/tmp/normalizer.json",
              "--official-parity-report", "/tmp/parity.json",
              "--record-index", "15", "--horizon-chunks", "5", "--seed-schedules", "both"]
    if stage != "preflight":
        values.append("--release")
    return values


class RunnerGateTests(unittest.TestCase):
    def test_stage_contract_requires_explicit_release_and_matching_preflight_input(self):
        preflight = parse_args(_argv("preflight"))
        validate_cli_contract(preflight)
        smoke = parse_args(_argv("smoke"))
        validate_cli_contract(smoke)
        smoke.input_run = "/tmp/task11-five-chunk"
        with self.assertRaisesRegex(ValueError, "input-run"):
            validate_cli_contract(smoke)

    def test_only_complete_identity_bound_smoke_with_measured_sample_is_admitted(self):
        passed = {"status": "ENGINEERING_SMOKE_COMPLETE", "run_identity_sha256": "a" * 64,
                  "smoke_calls": 1, "measured_sample_bytes": 1024,
                  "resource_gate_status": "PASS", "sample_id": "smoke_G0"}
        validate_smoke_status(passed, "a" * 64)
        for changes, message in (({"status": "FAILED"}, "complete"),
                                 ({"run_identity_sha256": "b" * 64}, "identity"),
                                 ({"smoke_calls": 0}, "one call"),
                                 ({"measured_sample_bytes": 0}, "footprint"),
                                 ({"resource_gate_status": "HARD_STOP"}, "resource"),
                                 ({"sample_id": "other"}, "sample identity")):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                validate_smoke_status({**passed, **changes}, "a" * 64)

    def test_model_stage_requires_explicit_generation_free_preflight_receipt(self):
        identity = {"id": "preflight"}
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            input_run = str(run_dir / "preflight")
            expected = {"status": "PREFLIGHT_LOCKED_GENERATION_NOT_PERFORMED",
                        "run_identity_sha256": runner._canonical_sha(identity),
                        "input_run": input_run, "report_sha256": "a" * 64,
                        "trajectory_npz_sha256": "b" * 64}
            path = run_dir / "preflight_status.json"
            path.write_text(json.dumps(expected))
            runner._verify_preflight_status(run_dir, identity, input_run, "a" * 64, "b" * 64)
            path.write_text(json.dumps({**expected, "status": "PREFLIGHT_FAILED"}))
            with self.assertRaisesRegex(ValueError, "generation-free preflight"):
                runner._verify_preflight_status(run_dir, identity, input_run, "a" * 64, "b" * 64)

    def test_runtime_failure_before_generation_is_terminal_and_does_not_claim_generation(self):
        from test_umi_task11_long_horizon import _trajectory

        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args(_argv("smoke"))
            args.run_dir = str(Path(temporary) / "five_chunk")
            args.resume = False
            with patch.object(runner, "task8_input_for_window",
                              side_effect=lambda trajectory, window: types.SimpleNamespace(actions=window.actions,
                                  rgb=window.rgb, prompt=trajectory.prompt)), \
                 patch.object(runner, "_verify_runtime", side_effect=RuntimeError("no locked runtime")):
                with self.assertRaisesRegex(RuntimeError, "locked runtime"):
                    runner.run_smoke(args, {"id": "test"}, _trajectory(), Path("unused.npz"))
            status = json.loads((Path(args.run_dir) / "smoke" / "run_status.json").read_text())
            self.assertEqual(status["status"], "FAILED")
            self.assertFalse(status["generation_started"])
            self.assertEqual(status["smoke_calls"], 0)

    def test_smoke_cpu_fixture_binds_frozen_live_imports_and_atomic_sample_store(self):
        """Exercise the one-call smoke orchestration with a fake CPU boundary only."""
        from test_umi_task11_long_horizon import _trajectory

        class FakeTorch:
            class cuda:
                @staticmethod
                def empty_cache(): return None

        class Feedback:
            mask = np.ones((1, 1, 1, 1, 1), dtype=bool)
            def embed_condition(self, value): return np.asarray(value, dtype=np.float32).copy()

        class Encoder:
            def encode(self, frame, *, precision):
                return {"arrays": {"actual_output": np.full((1, 1, 1, 1, 1), frame[0, 0, 0], dtype=np.float32)}}

        class Context:
            def __init__(self): self.feedback, self.encoder, self.batch = Feedback(), Encoder(), None
            def cleanup(self): return None

        def live_hash(value):
            value = np.ascontiguousarray(value)
            digest = hashlib.sha256()
            digest.update(str(value.dtype).encode()); digest.update(b"\0")
            digest.update(repr(tuple(value.shape)).encode()); digest.update(b"\0")
            digest.update(value.tobytes(order="C"))
            return digest.hexdigest()

        class MonitorInner:
            def capture_sample(self, sample_id, phase, remaining, *, run_dir):
                return {"decision_status": "PASS", "gpu_used_gib": 2.0, "gpu_free_gib": 90.0,
                    "gpu_reserved_gib": 2.0, "gpu_peak_allocated_gib": 10.0,
                    "gpu_peak_nvml_used_gib": 12.0, "ram_available_gib": 600.0,
                    "rss_gib": 10.0, "swap_used_gib": 0.0, "disk_free_gib": 20.0,
                    "cgroup_memory_limited": True, "cgroup_memory_limit_gib": 110.0,
                    "cgroup_memory_current_gib": 18.0, "cgroup_memory_free_gib": 92.0,
                    "mean_success_sample_bytes": 1024}

        class Monitor:
            def __init__(self, run_dir, *, gpu_index): self._monitor = MonitorInner()
            def start(self): return self
            def check(self, **kwargs):
                return {"status": "PASS", "gpu_used_gib": 0.5, "gpu_free_gib": 90.0,
                    "gpu_reserved_gib": 2.0, "gpu_peak_allocated_gib": 10.0,
                    "gpu_peak_nvml_used_gib": 12.0, "ram_available_gib": 600.0,
                    "rss_gib": 10.0, "swap_used_gib": 0.0, "disk_free_gib": 20.0,
                    "cgroup_memory_limited": True, "cgroup_memory_limit_gib": 110.0,
                    "cgroup_memory_current_gib": 18.0, "cgroup_memory_free_gib": 92.0}
            def stop(self): return None

        class Executor:
            def __init__(self, context, **kwargs): self.context = context
            def __call__(self, spec):
                action = self.context.batch.actions[0]
                condition_rgb = self.context.batch.rgb[0]
                condition = self.context.encoder.encode(condition_rgb, precision="temporary_fp32")["arrays"]["actual_output"]
                generated = np.full((3, 16, 2, 2), 0.25, dtype=np.float32)
                final = generated[:, -1].copy()
                encoded = self.context.encoder.encode(final, precision="temporary_fp32")["arrays"]["actual_output"]
                padded = np.zeros((16, 64), dtype=np.float32); padded[:, :10] = action
                action_hash, padded_hash = runner.task8_array_hash(action), runner.task8_array_hash(padded)
                return {"output_full": np.ones((1,), dtype=np.float32), "generated_rgb": generated,
                    "decoded_rgb_full": np.concatenate([generated[:, :1], generated], axis=1),
                    "decoded_last_rgb": final, "encoded_condition": encoded,
                    "condition_input_fp32": condition.copy(),
                    "condition_steps_fp32": np.repeat(condition[None], 30, axis=0),
                    "action": action.copy(), "action_hash": action_hash,
                    "packed_action_token_hashes": [padded_hash] * 30,
                    "action_consumption": {"all_steps_match": True, "steps": 30,
                        "expected_token_hash": padded_hash, "consumed_token_hashes": [padded_hash] * 30},
                    "prediction_noise_hash": "a" * 64,
                    "generation": {"sampler_generator_seeds": [0] * 30,
                        "noise_evidence": {"seed": 0, "prepare_seed": 0}},
                    "precision": {"G": "float32", "D": "float32", "E": "float32"},
                    "provenance": {"condition_source": "real_x0",
                        "condition_rgb_sha256": live_hash(condition_rgb),
                        "action_evidence": {"effective_action_hash": padded_hash,
                            "effective_action": {"dtype": "float32", "shape": [16, 64],
                                                 "sha256": live_hash(padded)}}}}

        live_module = types.ModuleType("task8_frozen.task8_live")
        live_module.Task8LiveExecutor, live_module.Task8ResourceMonitor = Executor, Monitor
        live_module.load_task8_live = lambda **kwargs: Context()
        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args(_argv("smoke")); args.run_dir = str(Path(temporary) / "five_chunk")
            with patch.dict(sys.modules, {"task8_frozen.task8_live": live_module}), \
                 patch.object(runner, "_verify_runtime", return_value=FakeTorch), \
                 patch.object(runner, "task8_input_for_window",
                              side_effect=lambda trajectory, window: types.SimpleNamespace(actions=window.actions,
                                  rgb=window.rgb, prompt=trajectory.prompt)):
                status = runner.run_smoke(args, {"id": "cpu-smoke"}, _trajectory(), Path("unused.npz"))
            self.assertEqual(status["status"], "ENGINEERING_SMOKE_COMPLETE")
            verified = runner._verify_smoke_artifact(Path(args.run_dir) / "smoke",
                                                     runner._canonical_sha({"id": "cpu-smoke"}))
            self.assertEqual(verified["smoke_calls"], 1)

    def test_model_and_framework_hash_pins_are_mandatory(self):
        runner._assert_pinned_model_identity(
            "0a4b762014f9fe3e3e8e13db204a14995d139c7669f58dd786dd6263ca291a9d",
            "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36",
            "ffa9c6b60a6b04b2fae337577bc6cbd8a93c39f5")
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            runner._assert_pinned_model_identity("0" * 64, runner.EXPECTED_VAE_SHA256,
                                                  runner.EXPECTED_FRAMEWORK_COMMIT)

    def test_incomplete_formal_resume_rejects_status_identity_before_rewriting(self):
        from test_umi_task11_long_horizon import _trajectory

        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args(_argv("formal"))
            args.run_dir = temporary
            args.resume = True
            formal_dir = Path(temporary) / "formal"
            formal_dir.mkdir()
            prior = {"status": "FAILED", "run_identity_sha256": "different", "sample_ids": []}
            status_path = formal_dir / "run_status.json"
            status_path.write_text(json.dumps(prior))
            with self.assertRaisesRegex(ValueError, "resume status identity"):
                runner.run_formal(args, {"id": "current"}, _trajectory(), {}, Path("unused.npz"))
            self.assertEqual(json.loads(status_path.read_text()), prior)

    def test_condition_only_mask_uses_recorded_runtime_axis_and_indexes(self):
        full = np.zeros((1, 2, 3, 2, 2), dtype=bool)
        full[:, :, 1, 0, 0] = True
        selected = runner.condition_only_mask(full, temporal_axis=2, condition_indexes=(1,))
        self.assertEqual(selected.shape, (1, 2, 1, 2, 2))
        np.testing.assert_array_equal(selected, np.take(full, [1], axis=2))
        with self.assertRaisesRegex(ValueError, "runtime condition geometry"):
            runner.condition_only_mask(full, temporal_axis=9, condition_indexes=(1,))

    def test_analysis_emits_paired_ar_minus_tf_rgb_and_latent_rms_deltas(self):
        rows = [
            {"schedule_index": 0, "horizon_chunks": 2, "mode": "TF", "rmse": 0.2,
             "condition_latent_rms": 0.4},
            {"schedule_index": 0, "horizon_chunks": 2, "mode": "AR", "rmse": 0.5,
             "condition_latent_rms": 0.1},
        ]
        paired = runner.paired_horizon_error_deltas(rows)
        self.assertEqual(len(paired), 1)
        self.assertAlmostEqual(paired[0]["rgb_rmse_ar_minus_tf"], 0.3)
        self.assertAlmostEqual(paired[0]["condition_latent_rms_ar_minus_tf"], -0.3)

    def test_runtime_resource_gate_preserves_frozen_ok_or_warning_status(self):
        safe = {"gpu_used_gib": 0.5, "gpu_free_gib": 90.0, "gpu_reserved_gib": 1.0,
                "gpu_peak_allocated_gib": 1.0, "gpu_peak_nvml_used_gib": 1.0,
                "ram_available_gib": 600.0, "rss_gib": 1.0, "swap_used_gib": 0.0,
                "disk_free_gib": 70.0, "cgroup_memory_limited": True,
                "cgroup_memory_limit_gib": 110.0, "cgroup_memory_current_gib": 18.0,
                "cgroup_memory_free_gib": 92.0}
        gate = runner.evaluate_resource_gate(safe, phase="formal", remaining_calls=20,
                                             sample_bytes=1024 * 1024)
        self.assertIn(gate["status"], {"OK", "WARNING"})
        with self.assertRaisesRegex(RuntimeError, "CGROUP_MEMORY_HEADROOM_CRITICAL"):
            runner.evaluate_resource_gate({**safe, "cgroup_memory_free_gib": 9.0,
                                           "cgroup_memory_current_gib": 101.0},
                                          phase="formal", remaining_calls=20,
                                          sample_bytes=1024 * 1024)

    def test_formal_cpu_integration_runs_postcleanup_callback_for_all_twenty_calls(self):
        """Bind the real schedule loop to a fake live boundary without Torch/model generation."""
        from test_umi_task11_long_horizon import _trajectory

        captured: list[str] = []
        class FakeTorch:
            class cuda:
                @staticmethod
                def empty_cache():
                    return None

        class FakeResourceStop(RuntimeError):
            pass

        class FakeSampleStore:
            def __init__(self, root):
                self.root = Path(root)

        class FakeMonitorInner:
            def capture_sample(self, sample_id, phase, remaining, *, run_dir):
                captured.append(sample_id)
                return {"decision_status": "PASS", "disk_free_gib": 20.0,
                        "gpu_used_gib": 2.0, "gpu_free_gib": 90.0,
                        "gpu_reserved_gib": 2.0, "gpu_peak_allocated_gib": 10.0,
                        "gpu_peak_nvml_used_gib": 12.0, "ram_available_gib": 600.0,
                        "rss_gib": 10.0, "swap_used_gib": 0.0,
                        "cgroup_memory_limited": True, "cgroup_memory_limit_gib": 110.0,
                        "cgroup_memory_current_gib": 18.0, "cgroup_memory_free_gib": 92.0,
                        "mean_success_sample_bytes": 1024}

        class FakeMonitor:
            def __init__(self, run_dir, *, gpu_index):
                self.run_dir = Path(run_dir)
                self._monitor = FakeMonitorInner()
            def start(self): return self
            def check(self, **kwargs):
                return {"status": "PASS", "gpu_used_gib": 0.5, "gpu_free_gib": 90.0,
                        "gpu_reserved_gib": 2.0, "gpu_peak_allocated_gib": 10.0,
                        "gpu_peak_nvml_used_gib": 12.0, "ram_available_gib": 600.0,
                        "rss_gib": 10.0, "swap_used_gib": 0.0, "disk_free_gib": 20.0,
                        "cgroup_memory_limited": True, "cgroup_memory_limit_gib": 110.0,
                        "cgroup_memory_current_gib": 18.0, "cgroup_memory_free_gib": 92.0}
            def stop(self): return None

        class FakeFeedback:
            mask = np.ones((1, 1, 1, 1, 1), dtype=bool)
            z0 = np.zeros_like(mask, dtype=np.float32)
            condition_indexes = (0,)
            temporal_axis = 2
            def embed_condition(self, value): return np.asarray(value, dtype=np.float32).copy()

        class FakeEncoder:
            def encode(self, frame, *, precision):
                value = np.full((1, 1, 1, 1, 1), frame[0, 0, 0], dtype=np.float32)
                return {"arrays": {"actual_output": value},
                        "evidence": {"precision": precision}}

        class FakeContext:
            def __init__(self):
                self.feedback = FakeFeedback()
                self.encoder = FakeEncoder()
                self.batch = None
                self.cleaned = False
            def cleanup(self): self.cleaned = True

        class FakeExecutor:
            def __init__(self, context, **kwargs): self.context = context
            def __call__(self, spec):
                action = np.asarray(self.context.batch.actions[int(spec["chunk_index"])], dtype=np.float32)
                condition = self.context.encoder.encode(spec["condition_rgb"], precision="temporary_fp32")["arrays"]["actual_output"]
                mode_offset = np.float32(0.01 if spec["call"] == "AR" else 0.0)
                value = np.float32(0.1 + int(spec["seed"]) / 100.0) + mode_offset
                generated = np.full((3, 16, 2, 2), value, dtype=np.float32)
                final = generated[:, -1].copy()
                encoded = self.context.encoder.encode(final, precision="temporary_fp32")["arrays"]["actual_output"]
                padded = np.zeros((16, 64), dtype=np.float32)
                padded[:, :10] = action
                def live_hash(array):
                    array = np.ascontiguousarray(array)
                    digest = hashlib.sha256()
                    digest.update(str(array.dtype).encode("ascii")); digest.update(b"\0")
                    digest.update(repr(tuple(array.shape)).encode("ascii")); digest.update(b"\0")
                    digest.update(array.tobytes(order="C"))
                    return digest.hexdigest()
                action_hash = runner.task8_array_hash(action)
                padded_hash = runner.task8_array_hash(padded)
                noise_hash = hashlib.sha256(f"noise-{spec['seed']}".encode()).hexdigest()
                carrier = self.context.feedback.embed_condition(condition)
                return {"output_full": np.asarray([spec["seed"]], dtype=np.float32),
                    "generated_rgb": generated,
                    "decoded_rgb_full": np.concatenate([generated[:, :1], generated], axis=1),
                    "decoded_last_rgb": final, "encoded_condition": encoded,
                    "condition_input_fp32": condition.copy(),
                    "condition_steps_fp32": np.repeat(carrier[None], 30, axis=0),
                    "action": action.copy(), "action_hash": action_hash,
                    "packed_action_token_hashes": [padded_hash] * 30,
                    "action_consumption": {"all_steps_match": True, "steps": 30,
                        "expected_token_hash": padded_hash, "consumed_token_hashes": [padded_hash] * 30},
                    "prediction_noise_hash": noise_hash,
                    "generation": {"sampler_generator_seeds": [spec["seed"]] * 30,
                        "noise_evidence": {"seed": spec["seed"], "prepare_seed": spec["seed"]}},
                    "precision": {"G": "float32", "D": "float32", "E": "float32"},
                    "provenance": {"condition_source": spec["condition_source"],
                        "condition_rgb_sha256": live_hash(spec["condition_rgb"]),
                        "action_evidence": {"action_hash": action_hash,
                            "effective_action_hash": padded_hash,
                            "effective_action": {"dtype": "float32", "shape": [16, 64],
                                                 "sha256": live_hash(padded)}}}}

        live_module = types.ModuleType("task8_frozen.task8_live")
        live_module.Task8LiveExecutor = FakeExecutor
        live_module.Task8ResourceMonitor = FakeMonitor
        live_module.load_task8_live = lambda **kwargs: FakeContext()
        formal_module = types.ModuleType("task8_frozen.task8_formal")
        formal_module._framework_commit = lambda *args: "commit"
        formal_module._task8_source_paths = lambda: []
        formal_module.validate_smoke_run = lambda *args, **kwargs: None

        def fake_analysis(formal_dir, trajectory, truth_conditions, condition_mask, identity):
            output = Path(formal_dir) / "analysis_attempts" / "attempt_001"
            output.mkdir(parents=True)
            summary = {"status": "PASS", "run_identity_sha256": runner._canonical_sha(identity),
                       "endpoint_metrics_per_seed": [{} for _ in range(18)],
                       "per_frame_metrics": [{} for _ in range(288)],
                       "tf_ar_error_geometry_rgb": [{} for _ in range(8)],
                       "tf_ar_error_geometry_condition_masked_latent": [{} for _ in range(8)],
                       "adjacent_ar_endpoint_changes": [{} for _ in range(8)],
                       "tf_ar_endpoint_error_deltas": [{} for _ in range(8)]}
            (output / "task11_horizon_metrics.json").write_text(json.dumps(summary))
            (output / "endpoint_metrics.csv").write_text("sample_id\n")
            (output / "per_frame_metrics.csv").write_text("sample_id\n")
            (output / "tf_ar_error_geometry.json").write_text("{}\n")
            (output / "report.md").write_text("CPU fixture\n")
            return {"status": "PASS", "output_dir": str(output),
                "task11_horizon_metrics_sha256": runner.file_sha256(output / "task11_horizon_metrics.json"),
                "endpoint_metrics_sha256": runner.file_sha256(output / "endpoint_metrics.csv"),
                "per_frame_metrics_sha256": runner.file_sha256(output / "per_frame_metrics.csv"),
                "tf_ar_error_geometry_sha256": runner.file_sha256(output / "tf_ar_error_geometry.json"),
                "report_sha256": runner.file_sha256(output / "report.md")}

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "five_chunk"
            smoke_dir = run_dir / "smoke"
            smoke_dir.mkdir(parents=True)
            identity = {"id": "test"}
            identity_hash = runner._canonical_sha(identity)
            (smoke_dir / "run_status.json").write_text(json.dumps({
                "status": "ENGINEERING_SMOKE_COMPLETE", "run_identity_sha256": identity_hash,
                "smoke_calls": 1, "sample_id": "smoke_G0", "measured_sample_bytes": 1024,
                "resource_gate_status": "PASS"}))
            args = parse_args(_argv("formal"))
            args.run_dir = str(run_dir)
            args.resume = False
            args.framework_root = "/tmp/framework"
            args.checkpoint = "/tmp/model"
            args.vae = "/tmp/vae"
            args.horizon_chunks = 5
            input_npz = Path(temporary) / "trajectory.npz"
            input_npz.write_bytes(b"cpu integration fixture")
            with patch.dict(sys.modules, {"task8_frozen.task8_live": live_module,
                                         "task8_frozen.task8_formal": formal_module}), \
                 patch.object(runner, "_verify_runtime", return_value=FakeTorch), \
                 patch.object(runner, "_verify_smoke_artifact", return_value={"measured_sample_bytes": 1024}), \
                 patch.object(runner, "task8_input_for_window",
                              side_effect=lambda trajectory, window: types.SimpleNamespace(actions=window.actions,
                                  rgb=window.rgb, prompt=trajectory.prompt)), \
                  \
                 patch.object(runner, "_write_horizon_analysis", side_effect=fake_analysis):
                result = runner.run_formal(args, identity, _trajectory(), {}, input_npz)
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(len(captured), 21)  # 20 formal calls and final cleanup capture
            self.assertEqual(captured[:20], [f"formal_call_{index:02d}" for index in range(20)])
            self.assertEqual(len(result["sample_ids"]), 20)
            artifact = Path(result["horizon_analysis"]["output_dir"]) / "task11_horizon_metrics.json"
            artifact.write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "analysis artifact hash mismatch"):
                runner._verify_completed_run(Path(args.run_dir) / "formal", identity)


if __name__ == "__main__":
    unittest.main()
