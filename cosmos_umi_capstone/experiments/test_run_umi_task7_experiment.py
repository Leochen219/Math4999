import json
import sys
import tempfile
import time
import weakref
import unittest
from unittest import mock
from contextlib import ExitStack
from pathlib import Path

import numpy as np

import run_umi_task7_experiment as runner


def _safe_snapshot():
    return {"gpu_used_gib": 0.0, "gpu_free_gib": 100.0, "gpu_reserved_gib": 0.0,
            "gpu_peak_allocated_gib": 0.0, "gpu_peak_nvml_used_gib": 0.0,
            "ram_available_gib": 500.0, "rss_gib": 0.0, "swap_used_gib": 0.0,
            "disk_free_gib": 100.0, "cgroup_memory_limited": False,
            "cgroup_memory_limit_gib": None, "cgroup_memory_current_gib": None,
            "cgroup_memory_free_gib": None}


def _a_record():
    return {"encoded_condition": np.zeros((1,), dtype=np.float32), "encoder": {"fixture": True},
            "evidence": {"operation_counts": {"G": 0, "D": 0, "E": 1}}}


class StaticTestMonitor:
    def __init__(self):
        self.last_resources = _safe_snapshot()
        self.captures = []
        self.stopped = False
    def check(self, **kwargs):
        return {"status": "OK", "snapshot": dict(self.last_resources)}
    def capture_sample(self, *args, **kwargs):
        self.captures.append(kwargs.get("phase", args[1] if len(args) > 1 else ""))
        return {"decision_status": "OK"}
    def stop(self):
        self.stopped = True


class Task7RunnerTests(unittest.TestCase):
    def _decoder_record_fixture(self, root, *, precision="temporary_fp32"):
        sample_name = "baseline_pre"
        replay_id = f"bridge_0__seed_0__{sample_name}__temporary_fp32"
        sample = Path(root) / replay_id
        sample.mkdir(parents=True)
        arrays = {
            "decoder_input_full_latent.npy": np.zeros((1,), dtype=np.float32),
            "decoded_final_float32.npy": np.zeros((3, 2, 2), dtype=np.float32),
            "direct_condition_latent_float32.npy": np.zeros((1,), dtype=np.float32),
        }
        hashes = {}
        for name, value in arrays.items():
            np.save(sample / name, value)
            hashes[name] = runner.sha256_file(sample / name)
        (sample / "record.json").write_text(json.dumps({
            "status": "success",
            "artifact_sha256": hashes,
            "spec": {
                "decode_precision": precision,
                "sample_id": "bridge_0__seed_0__baseline_pre",
                "replay_id": replay_id,
                "seed": 0,
                "state": "bridge_0",
            },
        }), encoding="utf-8")
        return sample, replay_id

    def test_real_decoder_record_fixture_is_accepted_and_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            sample, replay_id = self._decoder_record_fixture(temp)
            record = runner._artifact_status(sample, runner.REQUIRED_DECODER_ARTIFACTS,
                                             status_filename="record.json")
            runner._validate_decoder_record(record, sample_name="baseline_pre", replay_id=replay_id)
            self.assertEqual(record["status"], "success")

    def test_real_decoder_record_fixture_hash_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            sample, _ = self._decoder_record_fixture(temp)
            with (sample / "decoded_final_float32.npy").open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(runner.ResumeMismatch, "hash mismatch"):
                runner._artifact_status(sample, runner.REQUIRED_DECODER_ARTIFACTS,
                                        status_filename="record.json")

    def test_real_decoder_record_fixture_wrong_precision_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            sample, replay_id = self._decoder_record_fixture(temp, precision="native")
            record = runner._artifact_status(sample, runner.REQUIRED_DECODER_ARTIFACTS,
                                             status_filename="record.json")
            with self.assertRaisesRegex(runner.ResumeMismatch, "decode_precision"):
                runner._validate_decoder_record(record, sample_name="baseline_pre", replay_id=replay_id)

    def _real_monitor(self, root):
        gpu = {"gpu_used_gib": 0.0, "gpu_free_gib": 100.0, "gpu_reserved_gib": 1.0,
               "gpu_allocated_gib": 1.0, "gpu_peak_allocated_gib": 2.0,
               "gpu_peak_reserved_gib": 3.0, "gpu_peak_nvml_used_gib": 4.0}
        ram = {"ram_available_gib": 500.0, "rss_gib": 1.0, "swap_used_gib": 0.0}
        disk = {"disk_free_gib": 100.0, "cgroup_memory_limited": False,
                "cgroup_memory_limit_gib": None, "cgroup_memory_current_gib": None,
                "cgroup_memory_free_gib": None}
        return runner.ResourceMonitor(root, gpu_sampler=lambda: dict(gpu),
                                      ram_sampler=lambda: dict(ram), disk_sampler=lambda: dict(disk),
                                      sleep=lambda seconds: time.sleep(min(seconds, 0.001)))

    def test_real_resource_monitor_uses_smoke_forecast_and_publishes_after_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); run = base / "run"; stage = run / "stages" / "A"
            smoke = runner.Task7SampleStore(run / "smoke" / "samples")
            smoke.write_success("smoke_baseline", {"record": _a_record()},
                                operation_counts={"G": 1, "D": 1, "E": 1})
            monitor = self._real_monitor(base / "monitor"); monitor.start()
            events = []
            try:
                result = runner.run_stage("A", stage, execute=lambda spec: _a_record(),
                                          binding=runner.test_binding("A"), monitor=monitor,
                                          cleanup_sample=lambda: events.append("release") or {"released": True},
                                          reset_sample_peak=lambda: events.append("peak_reset"))
            finally:
                monitor.stop()
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["completed_count"], 16)
            self.assertEqual(events[0], "peak_reset")
            self.assertIn("release", events)
            phases = [row["phase"] for row in monitor._sample_rows]
            self.assertEqual(phases[:3], ["pre_call", "post_call", "post_cleanup"])
            self.assertTrue((stage / "samples" / "A_baseline_pre_native" / "status.json").is_file())

    def test_run_stage_releases_output_record_before_post_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            observed = []
            def execute(spec):
                array = np.ones((4,), dtype=np.float32)
                observed.append(weakref.ref(array))
                return {"encoded_condition": array, "encoder": {"fixture": True},
                        "evidence": {"operation_counts": {"G": 0, "D": 0, "E": 1}}}
            def cleanup():
                import gc
                gc.collect()
                self.assertIsNone(observed[-1]())
                return {"released": True}
            result = runner.run_stage("A", temp, execute=execute,
                                      binding=runner.test_binding("A"), monitor=runner.StaticMonitor(_safe_snapshot()),
                                      cleanup_sample=cleanup)
            self.assertEqual(result["status"], "COMPLETE")

    def test_strict_gpu_sampler_fails_closed_and_preserves_reserved_peak(self):
        class Cuda:
            @staticmethod
            def is_available(): return True
            @staticmethod
            def memory_allocated(_): return 2 * 2**30
            @staticmethod
            def memory_reserved(_): return 3 * 2**30
            @staticmethod
            def max_memory_allocated(_): return 4 * 2**30
            @staticmethod
            def max_memory_reserved(_): return 5 * 2**30
        torch = type("Torch", (), {"cuda": Cuda})
        with mock.patch.dict(sys.modules, {"torch": torch}):
            sample = runner._strict_gpu_sampler(lambda: {"gpu_used_gib": 1.0}, 0)()
            self.assertEqual(sample["gpu_peak_reserved_gib"], 5.0)
            self.assertEqual(sample["gpu_reserved_gib"], 3.0)
        class BrokenCuda:
            @staticmethod
            def is_available(): return True
            @staticmethod
            def memory_allocated(_): raise RuntimeError("late torch failure")
        broken = type("Torch", (), {"cuda": BrokenCuda})
        with mock.patch.dict(sys.modules, {"torch": broken}):
            failed = runner._strict_gpu_sampler(lambda: {"gpu_used_gib": 1.0}, 0)()
            self.assertIn("monitor_failure", failed)

    def test_reset_peak_calls_torch_reset_for_each_sample(self):
        calls = []
        class Cuda:
            @staticmethod
            def is_available(): return True
            @staticmethod
            def reset_peak_memory_stats(index): calls.append(index)
        torch = type("Torch", (), {"cuda": Cuda})
        with mock.patch.dict(sys.modules, {"torch": torch}):
            runner._reset_sample_peak_memory()
        self.assertEqual(calls, [0])

    def _a_executor_fixture(self, output):
        root = Path(tempfile.mkdtemp()); decoder = root / "decoder"
        sample = decoder / "bridge_0__seed_0__baseline_pre__temporary_fp32"
        sample.mkdir(parents=True)
        np.save(sample / "decoded_final_float32.npy", np.zeros((3, 2, 2), dtype=np.float32))
        np.save(sample / "direct_condition_latent_float32.npy", np.zeros((1, 1, 1, 1, 2), dtype=np.float32))
        class Encoder:
            def encode(self, frame, precision="native"):
                return {"arrays": {"actual_output": output}, "evidence": {"operation_count": 1}}
        return root, runner.make_stage_executor("A", source={"decoder_root": str(decoder)},
                                                feedback=type("Feedback", (), {"encoder": Encoder()})())

    def test_a_encoder_exception_preserves_actual_e_count(self):
        with tempfile.TemporaryDirectory() as temp:
            class Encoder:
                def encode(self, frame, precision="native"):
                    error = RuntimeError("encoder failed")
                    error.capture = {"operation_counts": {"G": 0, "D": 0, "E": 1}}
                    raise error
            root = Path(temp); decoder = root / "decoder"
            sample = decoder / "bridge_0__seed_0__baseline_pre__temporary_fp32"; sample.mkdir(parents=True)
            np.save(sample / "decoded_final_float32.npy", np.zeros((3, 2, 2), dtype=np.float32))
            executor = runner.make_stage_executor("A", source={"decoder_root": str(decoder)},
                                                    feedback=type("Feedback", (), {"encoder": Encoder()})())
            result = runner.run_stage("A", root / "stage", execute=executor,
                                      binding=runner.test_binding("A"), monitor=runner.StaticMonitor(_safe_snapshot()))
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["attempt_counts"], {"G": 0, "D": 0, "E": 1})

    def test_a_post_encode_shape_failure_preserves_encoder_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root, executor = self._a_executor_fixture(np.zeros((1, 1, 1, 1, 3), dtype=np.float32))
            result = runner.run_stage("A", root / "stage", execute=executor,
                                      binding=runner.test_binding("A"), monitor=runner.StaticMonitor(_safe_snapshot()))
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["attempt_counts"], {"G": 0, "D": 0, "E": 1})

    def test_smoke_stop_cleanup_failure_marks_root_terminal(self):
        class Monitor(StaticTestMonitor):
            def stop(self):
                self.stopped = True
                raise RuntimeError("monitor stop failed")
        root, factory, monitor, patches = self._main_fixture(monitor=Monitor())
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner, "_code_digest", return_value="code"))
            for patch in patches: stack.enter_context(patch)
            with self.assertRaises(RuntimeError):
                runner.main(["--stage", "smoke", "--run-dir", str(root), "--raw-root", str(root.parent / "source"),
                             "--decoder-root", str(root.parent / "source"), "--launch-contract", str(root / "contract")],
                            factory=factory, monitor=monitor)
        root_status = json.loads((root / "run_status.json").read_text(encoding="utf-8"))
        self.assertEqual(root_status["status"], "FAILED")
        self.assertEqual(root_status["reason_code"], "RuntimeError")
        self.assertEqual(root_status["operation_counts"], {"G": 1, "D": 1, "E": 1})
        smoke_status = json.loads((root / "smoke" / "run_status.json").read_text(encoding="utf-8"))
        self.assertNotEqual(smoke_status["status"], "SMOKE_COMPLETE")

    def test_smoke_success_cannot_be_launched_again_without_resume(self):
        root, factory, monitor, patches = self._main_fixture()
        args = ["--stage", "smoke", "--run-dir", str(root), "--raw-root", str(root.parent / "source"),
                "--decoder-root", str(root.parent / "source"), "--launch-contract", str(root / "contract")]
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner, "_code_digest", return_value="code"))
            for patch in patches: stack.enter_context(patch)
            self.assertEqual(runner.main(args, factory=factory, monitor=monitor), 0)
            built = factory.built
            with self.assertRaises(runner.ResumeMismatch):
                runner.main(args, factory=factory, monitor=monitor)
        self.assertEqual(factory.built, built)

    def test_smoke_feedback_exception_preserves_partial_capture_and_counts(self):
        class FailingFeedback:
            def __init__(self, *args, **kwargs):
                pass
            def extract_condition(self, value):
                return np.asarray(value).copy()
            def step(self, condition, step):
                error = RuntimeError("feedback failed after partial work")
                error.capture = {"operation_counts": {"G": 1, "D": 1, "E": 0},
                                 "diagnostic_marker": "partial-gd"}
                raise error

        root, factory, monitor, patches = self._main_fixture()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner, "_code_digest", return_value="code"))
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(mock.patch("umi_task7_runtime.FeedbackRuntime", FailingFeedback))
            with self.assertRaises(RuntimeError):
                runner.main(["--stage", "smoke", "--run-dir", str(root), "--raw-root", str(root.parent / "source"),
                             "--decoder-root", str(root.parent / "source"), "--launch-contract", str(root / "contract")],
                            factory=factory, monitor=monitor)
        status = json.loads((root / "run_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["operation_counts"], {"G": 1, "D": 1, "E": 0})
        self.assertEqual(status["error_operation_counts"], {"G": 1, "D": 1, "E": 0})
        self.assertEqual(status["error_capture"]["diagnostic_marker"], "partial-gd")

    def test_formal_failure_root_preserves_stage_attempt_counts(self):
        root, factory, monitor, patches = self._main_fixture()
        (root / "contract").write_text("contract", encoding="utf-8")
        args = ["--stage", "A", "--run-dir", str(root), "--raw-root", str(root.parent / "source"),
                "--decoder-root", str(root.parent / "source"), "--launch-contract", str(root / "contract")]
        failed = {"status": "FAILED", "reason": "post validation", "attempt_counts": {"G": 2, "D": 1, "E": 1},
                  "completed_counts": {"G": 0, "D": 0, "E": 0}}
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner, "_code_digest", return_value="code"))
            stack.enter_context(mock.patch.object(runner, "run_stage", return_value=failed))
            for patch in patches: stack.enter_context(patch)
            self.assertEqual(runner.main(args, factory=factory, monitor=monitor), 1)
        status = json.loads((root / "run_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["operation_counts"], failed["attempt_counts"])
        self.assertEqual(status["stage_status"]["attempt_counts"], failed["attempt_counts"])

    def test_setup_encoder_observer_counts_failures_and_restores_method(self):
        calls = []
        class RawEncoder:
            def encode(self, value):
                calls.append(value)
                if value == "bad": raise RuntimeError("setup encode failed")
                return value
        raw = RawEncoder()
        adapter = type("Adapter", (), {"encoder": raw})()
        original_loader = lambda **kwargs: {"encoder": adapter}
        factory = type("Factory", (), {})()
        factory.loader = original_loader
        evidence, restore = runner._install_setup_encoder_observer(factory)
        payload = factory.loader()
        self.assertTrue(evidence["installed"])
        self.assertEqual(payload["encoder"].encoder.encode("ok"), "ok")
        with self.assertRaises(RuntimeError): payload["encoder"].encoder.encode("bad")
        self.assertEqual(evidence["encode_calls"], 2)
        self.assertEqual(len(evidence["encode_failures"]), 1)
        restore()
        self.assertIs(factory.loader, original_loader)
        self.assertTrue(evidence["ownership_restored"])
        self.assertEqual(raw.encode("ok2"), "ok2")

    def test_setup_encoder_observer_restores_class_slots_and_instance_overrides(self):
        class RawEncoder:
            def encode(self, value):
                if value == "bad":
                    raise RuntimeError("class encode failed")
                return value

        class Factory:
            def loader(self):
                return {"encoder": type("Adapter", (), {"encoder": RawEncoder()})()}

        factory = Factory()
        evidence, restore = runner._install_setup_encoder_observer(factory)
        payload = factory.loader()
        self.assertEqual(payload["encoder"].encoder.encode("ok"), "ok")
        with self.assertRaises(RuntimeError):
            payload["encoder"].encoder.encode("bad")
        self.assertEqual(evidence["encode_calls"], 2)
        self.assertEqual(len(evidence["encode_failures"]), 1)
        restore()
        self.assertTrue(evidence["ownership_restored"])
        self.assertNotIn("loader", factory.__dict__)
        self.assertNotIn("encode", payload["encoder"].encoder.__dict__)

        raw = RawEncoder()
        original_encode = lambda value: "instance:" + value
        raw.encode = original_encode
        adapter = type("Adapter", (), {"encoder": raw})()
        original_loader = lambda: {"encoder": adapter}
        factory = type("Factory", (), {})()
        factory.loader = original_loader
        evidence, restore = runner._install_setup_encoder_observer(factory)
        self.assertEqual(factory.loader()["encoder"].encoder.encode("ok"), "instance:ok")
        restore()
        self.assertIs(factory.__dict__["loader"], original_loader)
        self.assertIs(raw.__dict__["encode"], original_encode)

    def test_setup_evidence_is_stage_and_attempt_scoped(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = runner._persist_setup_evidence(root, "smoke", {"encode_calls": 1})
            second = runner._persist_setup_evidence(root, "A", {"encode_calls": 2})
            retry = runner._persist_setup_evidence(root, "A", {"encode_calls": 3})
            self.assertNotEqual(first, second)
            self.assertNotEqual(second, retry)
            self.assertEqual(json.loads(first.read_text(encoding="utf-8"))["encode_calls"], 1)
            self.assertEqual(json.loads(second.read_text(encoding="utf-8"))["encode_calls"], 2)
            self.assertEqual(json.loads(retry.read_text(encoding="utf-8"))["encode_calls"], 3)

    def test_complete_resume_accepts_verified_interrupted_attempt_and_is_noop(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []
            first = True

            def execute(spec):
                calls.append(spec["sample_id"])
                nonlocal first
                if first:
                    first = False
                    error = KeyboardInterrupt("operator stop")
                    error.capture = {"operation_counts": {"G": 1, "D": 0, "E": 0}}
                    raise error
                return _a_record()

            stage = Path(temp) / "stage"
            runner.run_stage("A", stage, execute=execute, binding=runner.test_binding("A"),
                             monitor=runner.StaticMonitor(_safe_snapshot()))
            resumed = runner.run_stage("A", stage, execute=execute, binding=runner.test_binding("A"),
                                       monitor=runner.StaticMonitor(_safe_snapshot()), resume=True)
            self.assertEqual(resumed["status"], "COMPLETE")
            before = {path: path.read_bytes() for path in (stage / "run_status.json", stage / "samples" / "A_baseline_pre_native" / "status.json")}
            calls_before = len(calls)
            checked = runner.run_stage("A", stage, execute=lambda _spec: self.fail("complete resume executed"),
                                       binding=runner.test_binding("A"), monitor=runner.StaticMonitor(_safe_snapshot()), resume=True)
            self.assertEqual(checked["status"], "COMPLETE")
            self.assertEqual(len(calls), calls_before)
            self.assertEqual({path: path.read_bytes() for path in before}, before)

            status = json.loads((stage / "run_status.json").read_text(encoding="utf-8"))
            status["failed_attempts"].append(dict(status["failed_attempts"][0]))
            (stage / "run_status.json").write_text(json.dumps(status), encoding="utf-8")
            with self.assertRaises(runner.ResumeMismatch):
                runner.run_stage("A", stage, execute=lambda _spec: _a_record(),
                                 binding=runner.test_binding("A"), monitor=runner.StaticMonitor(_safe_snapshot()), resume=True)

    def test_precheck_complete_resume_allows_verified_interrupted_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = root / "stages" / "A"
            first = True

            def execute(_spec):
                nonlocal first
                if first:
                    first = False
                    error = KeyboardInterrupt("operator stop")
                    error.capture = {"operation_counts": {"G": 1, "D": 0, "E": 0}}
                    raise error
                return _a_record()

            runner.run_stage("A", stage, execute=execute, binding=runner.test_binding("A"),
                             monitor=runner.StaticMonitor(_safe_snapshot()))
            runner.run_stage("A", stage, execute=execute, binding=runner.test_binding("A"),
                             monitor=runner.StaticMonitor(_safe_snapshot()), resume=True)
            contract = root / "contract.json"
            contract.write_text(json.dumps({"code_bundle_sha256": "c" * 64}), encoding="utf-8")
            status_path = stage / "run_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["binding"] = {**status["binding"],
                                  "config": {"stage": "A", "plan": runner.build_stage_plan("A")},
                                  "launch_contract_sha256": runner.sha256_file(contract),
                                  "model": "model", "vae": "vae", "framework": "framework"}
            status_path.write_text(json.dumps(status), encoding="utf-8")
            args = runner.parse_args(["--stage", "A", "--run-dir", str(root), "--resume",
                                      "--raw-root", str(root / "raw"), "--decoder-root", str(root / "decoder"),
                                      "--launch-contract", str(contract)])
            with mock.patch.object(runner, "validate_task7_sources", return_value={"source_tree_sha256": "fixture-source"}), \
                 mock.patch.object(runner, "_code_digest", return_value="fixture-task7-code"), \
                 mock.patch.object(runner, "_load_contract", return_value={"checkpoint_identity": {"sha256": "model"},
                     "vae_sha256": "vae", "framework_commit": "framework"}):
                self.assertEqual(runner._precheck_resume(root, args), 0)

    def _complete_resume_fixture(self, temp):
        root = Path(temp); stage = root / "stages" / "A"
        runner.run_stage("A", stage, execute=lambda spec: _a_record(),
                         binding=runner.test_binding("A"), monitor=runner.StaticMonitor(_safe_snapshot()))
        contract = root / "contract.json"
        contract.write_text(json.dumps({"code_bundle_sha256": "c" * 64}), encoding="utf-8")
        status_path = stage / "run_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        binding = dict(status["binding"])
        binding.update({"config": {"stage": "A", "plan": runner.build_stage_plan("A")},
                        "launch_contract_sha256": runner.sha256_file(contract), "model": "model",
                        "vae": "vae", "framework": "framework"})
        status["binding"] = binding
        status_path.write_text(json.dumps(status), encoding="utf-8")
        args = runner.parse_args(["--stage", "A", "--run-dir", str(root), "--resume",
                                  "--raw-root", str(root / "raw"), "--decoder-root", str(root / "decoder"),
                                  "--launch-contract", str(contract)])
        return root, stage, contract, args, status

    def test_complete_resume_rejects_contract_tamper_and_duplicate_sample_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root, stage, contract, args, status = self._complete_resume_fixture(temp)
            source = {"source_tree_sha256": "fixture-source"}
            patches = [mock.patch.object(runner, "validate_task7_sources", return_value=source),
                       mock.patch.object(runner, "_code_digest", return_value="fixture-task7-code"),
                       mock.patch.object(runner, "_load_contract", return_value={"code_bundle_sha256": "c" * 64,
                           "checkpoint_identity": {"sha256": "model"}, "vae_sha256": "vae", "framework_commit": "framework"})]
            with ExitStack() as stack:
                for patch in patches: stack.enter_context(patch)
                contract.write_text(json.dumps({"code_bundle_sha256": "d" * 64}), encoding="utf-8")
                with self.assertRaises(runner.ResumeMismatch): runner._precheck_resume(root, args)
            status["binding"]["launch_contract_sha256"] = runner.sha256_file(contract)
            status["completed_samples"].append(status["completed_samples"][0])
            stage_status = stage / "run_status.json"
            stage_status.write_text(json.dumps(status), encoding="utf-8")
            with mock.patch.object(runner, "validate_task7_sources", return_value=source), \
                 mock.patch.object(runner, "_code_digest", return_value="fixture-task7-code"), \
                 mock.patch.object(runner, "_load_contract", return_value={"code_bundle_sha256": "d" * 64,
                     "checkpoint_identity": {"sha256": "model"}, "vae_sha256": "vae", "framework_commit": "framework"}):
                with self.assertRaises(runner.ResumeMismatch): runner._precheck_resume(root, args)

    def test_c_baseline_encoded_z2_mismatch_is_counted_and_rejected(self):
        z1 = np.zeros((1, 1, 1, 1, 2), dtype=np.float32)
        z2 = np.ones_like(z1)
        mask = np.ones_like(z1, dtype=bool)
        class Feedback:
            def __init__(self): self.mask = mask
            def extract_condition(self, value): return np.asarray(value).copy()
            def step(self, condition, step):
                return {"encoded_condition": np.full_like(z1, 2.0),
                        "evidence": {"operation_counts": {"G": 1, "D": 1, "E": 1},
                                     "prediction_noise_hash": "seed1"}}
        source = {"analysis_gate": {"a_scientific_pass": True, "b_engineering_pass": True,
                                     "b_repeatability_pass": True, "source_sha256": "s", "code_sha256": "c"},
                  "source_tree_sha256": "s", "task7_code_sha256": "c", "z1": z1,
                  "b_z2": z2, "b_seed1_noise_hash": "seed1", "delta1_directions": {}}
        executor = runner.make_stage_executor("C", source=source, feedback=Feedback())
        spec = next(item for item in runner.build_stage_plan("C") if item["kind"] == "baseline_pre")
        with self.assertRaises(runner.ResumeMismatch) as caught:
            executor(spec)
        self.assertEqual(caught.exception.capture["operation_counts"], {"G": 1, "D": 1, "E": 1})

    def test_c_skip_records_resume_as_skipped_not_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            def execute(spec):
                if spec["kind"] == "perturbation":
                    raise runner.SkipSample("SKIPPED_C_ZERO_RAY", "zero ray")
                return {"full_latent": np.zeros((1,), dtype=np.float32),
                        "predicted_latent": np.zeros((1,), dtype=np.float32),
                        "encoded_condition": np.zeros((1,), dtype=np.float32),
                        "condition_input_fp32": np.zeros((1,), dtype=np.float32),
                        "actual": {key: np.zeros((1,), dtype=np.float32) for key in
                                    ("prepared_condition", "initial_condition", "reference_condition", "first_condition", "last_condition")}
                                  | {"condition_steps": np.zeros((30, 1), dtype=np.float32)},
                        "evidence": {"operation_counts": {"G": 1, "D": 1, "E": 1}}}
            stage = Path(temp) / "stage"
            first = runner.run_stage("C", stage, execute=execute, binding=runner.test_binding("C"),
                                     monitor=runner.StaticMonitor(_safe_snapshot()), allowed_skips=True)
            second = runner.run_stage("C", stage, execute=execute, binding=runner.test_binding("C"),
                                      monitor=runner.StaticMonitor(_safe_snapshot()), allowed_skips=True, resume=True)
            self.assertEqual(first["status"], "COMPLETE")
            self.assertEqual(first["skipped_count"], 36)
            self.assertEqual(second["skipped_count"], 36)
            self.assertEqual(second["completed_count"], 2)

    def test_resume_complete_rejects_source_or_code_identity_change_before_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); contract = root / "contract.json"
            contract.write_text(json.dumps({"code_bundle_sha256": "c" * 64}), encoding="utf-8")
            runner.Task7SampleStore(root / "smoke" / "samples").write_success(
                "smoke_baseline", {"record": _a_record()}, operation_counts={"G": 1, "D": 1, "E": 1})
            status = {"status": "SMOKE_COMPLETE", "engineering": True,
                      "formal_counts": {"G": 0, "D": 0, "E": 0},
                      "observed_counts": {"G": 1, "D": 1, "E": 1},
                      "binding": {"source_tree_sha256": "old-source", "task7_code_sha256": "old-code",
                                  "launch_contract_sha256": runner.sha256_file(contract), "config": {"stage": "smoke"}}}
            (root / "smoke").mkdir(exist_ok=True)
            (root / "smoke" / "run_status.json").write_text(json.dumps(status), encoding="utf-8")
            args = runner.parse_args(["--stage", "smoke", "--run-dir", str(root), "--resume",
                                      "--raw-root", str(root / "raw"), "--decoder-root", str(root / "decoder"),
                                      "--launch-contract", str(contract)])
            with mock.patch.object(runner, "validate_task7_sources", return_value={"source_tree_sha256": "new-source"}), \
                 mock.patch.object(runner, "_code_digest", return_value="new-code"):
                with self.assertRaises(runner.ResumeMismatch):
                    runner._precheck_resume(root, args)

    def test_smoke_complete_is_not_reused_after_root_cleanup_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); smoke = root / "smoke"
            runner.Task7SampleStore(smoke / "samples").write_success(
                "smoke_baseline", {"record": _a_record()}, operation_counts={"G": 1, "D": 1, "E": 1})
            (smoke / "run_status.json").write_text(json.dumps({"status": "SMOKE_COMPLETE"}), encoding="utf-8")
            (root / "run_status.json").write_text(json.dumps({"status": "FAILED", "reason_code": "CLEANUP_FAILURE"}), encoding="utf-8")
            args = runner.parse_args(["--stage", "smoke", "--run-dir", str(root), "--resume"])
            with self.assertRaises(runner.ResumeMismatch):
                runner._precheck_resume(root, args)
            args_no_resume = runner.parse_args(["--stage", "smoke", "--run-dir", str(root)])
            with self.assertRaises(runner.ResumeMismatch):
                runner._precheck_resume(root, args_no_resume)

    def test_owned_monitor_paths_are_stage_scoped_and_not_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"; root.mkdir(); root_b = Path(temp) / "run-b"; root_b.mkdir(); created = []
            class FakeMonitor:
                def __init__(self, monitor_root, **kwargs):
                    created.append(Path(monitor_root)); Path(monitor_root).mkdir(parents=True, exist_ok=True)
                    self.last_resources = _safe_snapshot()
                def start(self): return self
                def check(self, **kwargs): return {"status": "HARD_STOP", "reason_code": "TEST_STOP"}
                def stop(self): pass
            def args_for(run_root, stage):
                return ["--stage", stage, "--run-dir", str(run_root), "--raw-root", str(run_root.parent / "raw"),
                        "--decoder-root", str(run_root.parent / "decoder"), "--launch-contract", str(run_root / "contract")]
            with mock.patch.object(runner, "validate_task7_sources", return_value={"source_tree_sha256": "s"}), \
                 mock.patch.object(runner, "_load_contract", return_value={"code_bundle_sha256": "c" * 64}), \
                 mock.patch.object(runner, "_code_digest", return_value="code"), \
                 mock.patch.object(runner, "ResourceMonitor", FakeMonitor):
                for stage, run_root in (("A", root), ("B", root_b)):
                    with self.assertRaises(runner.ResourceStop):
                        runner.main(args_for(run_root, stage))
            self.assertEqual(created[0], root / "monitor" / "a")
            self.assertEqual(created[1], root_b / "monitor" / "b")
            self.assertEqual(len(set(created)), 2)

    def test_cli_exposes_all_task7_bindings(self):
        args = runner.parse_args([
            "--stage", "A", "--run-dir", "run", "--resume", "--raw-root", "raw",
            "--decoder-root", "decoder", "--framework-root", "framework",
            "--checkpoint", "checkpoint", "--vae", "vae", "--action", "action",
            "--video", "video", "--launch-contract", "contract.json",
            "--task5-root", "task5", "--gpu-index", "0",
        ])
        self.assertEqual(args.stage, "A")
        self.assertTrue(args.resume)
        self.assertEqual(args.framework_root, "framework")
        self.assertEqual(args.checkpoint, "checkpoint")
        self.assertEqual(args.gpu_index, 0)

    def test_plans_have_exact_formal_counts_and_fixed_order(self):
        self.assertEqual(len(runner.build_stage_plan("A")), 16)
        self.assertEqual(runner.stage_counts("A"), {"G": 0, "D": 0, "E": 16})
        self.assertEqual(len(runner.build_stage_plan("B")), 16)
        self.assertEqual(runner.stage_counts("B"), {"G": 16, "D": 16, "E": 16})
        plan = runner.build_stage_plan("C")
        self.assertEqual(len(plan), 38)
        self.assertEqual(runner.stage_counts("C"), {"G": 38, "D": 38, "E": 38})
        self.assertEqual(plan[0]["kind"], "baseline_pre")
        self.assertEqual(plan[-1]["kind"], "baseline_post")
        self.assertEqual([row["beta"] for row in plan if row["kind"] == "perturbation"],
                         [0.1, 0.1, 0.2, 0.2, 0.4, 0.4] * 6)

    def test_resource_policy_requires_complete_live_snapshot(self):
        with self.assertRaises(runner.ResourceStop):
            runner.require_resource_snapshot({"gpu_used_gib": 0.0}, phase="stage")
        runner.require_resource_snapshot(_safe_snapshot(), phase="preload")
        snapshot = _safe_snapshot()
        snapshot.update({"gpu_used_gib": 76.0})
        with self.assertRaises(runner.ResourceStop):
            runner.require_resource_snapshot(snapshot, phase="stage")
        snapshot = _safe_snapshot()
        snapshot.update({"cgroup_memory_limited": True, "cgroup_memory_limit_gib": 100,
                         "cgroup_memory_current_gib": 95, "cgroup_memory_free_gib": 5})
        with self.assertRaises(runner.ResourceStop):
            runner.require_resource_snapshot(snapshot, phase="stage")

    def test_store_keeps_failed_attempt_and_verifies_success_on_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            store = runner.Task7SampleStore(Path(temp) / "samples")
            first = store.write_failure("sample", {"error": "oom"}, operation_counts={"G": 1, "D": 0, "E": 0})
            self.assertTrue((first / "status.json").is_file())
            self.assertEqual(store.prepare("sample", resume=True), "run")
            self.assertTrue(any(path.name.startswith("sample.attempt.") for path in store.root.iterdir()))
            store.write_success("sample", {"value": np.ones((2,), dtype=np.float32)}, required_files=("record.json",))
            self.assertEqual(store.prepare("sample", resume=True), "skip")
            (store.root / "sample" / "record.json").write_text("tampered", encoding="utf-8")
            with self.assertRaises(runner.ResumeMismatch):
                store.prepare("sample", resume=True)

    def test_run_stage_uses_observed_failed_attempt_counts_and_never_infers(self):
        with tempfile.TemporaryDirectory() as temp:
            def execute(spec):
                error = RuntimeError("mid-call")
                error.capture = {"operation_counts": {"G": 1, "D": 0, "E": 0}}
                raise error

            result = runner.run_stage("A", temp, execute=execute,
                                      binding=runner.test_binding("A"),
                                      monitor=runner.StaticMonitor(_safe_snapshot()))
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["attempt_counts"], {"G": 1, "D": 0, "E": 0})
            self.assertEqual(result["completed_count"], 0)
            self.assertTrue((Path(temp) / "samples" / "A_baseline_pre_native.attempt.001" / "status.json").is_file())

    def test_interrupted_run_resumes_and_accumulates_attempt_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            first = True
            def execute(spec):
                nonlocal first
                if first:
                    first = False
                    error = KeyboardInterrupt("operator stop")
                    error.capture = {"operation_counts": {"G": 1, "D": 0, "E": 0}}
                    raise error
                return _a_record()
            interrupted = runner.run_stage("A", temp, execute=execute,
                                           binding=runner.test_binding("A"),
                                           monitor=runner.StaticMonitor(_safe_snapshot()))
            self.assertEqual(interrupted["status"], "RUNNING")
            self.assertEqual(interrupted["attempt_counts"], {"G": 1, "D": 0, "E": 0})
            resumed = runner.run_stage("A", temp, execute=execute,
                                       binding=runner.test_binding("A"), resume=True,
                                       monitor=runner.StaticMonitor(_safe_snapshot()))
            self.assertEqual(resumed["status"], "COMPLETE")
            self.assertEqual(resumed["completed_count"], 16)
            self.assertEqual(resumed["attempt_counts"], {"G": 1, "D": 0, "E": 16})
            checked = runner.run_stage("A", temp, execute=execute,
                                       binding=runner.test_binding("A"), resume=True,
                                       monitor=runner.StaticMonitor(_safe_snapshot()))
            self.assertEqual(checked["attempt_counts"], resumed["attempt_counts"])

    def test_formal_post_capture_stop_preserves_attempted_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            class Monitor(StaticTestMonitor):
                def capture_sample(self, *args, **kwargs):
                    phase = kwargs.get("phase", "")
                    self.captures.append(phase)
                    if phase == "post_call":
                        return {"decision_status": "HARD_STOP", "reason_code": "POST_GATE"}
                    return {"decision_status": "OK"}
            result = runner.run_stage("A", temp,
                execute=lambda spec: _a_record(),
                binding=runner.test_binding("A"), monitor=Monitor())
            self.assertEqual(result["status"], "RESOURCE_STOP")
            self.assertEqual(result["attempt_counts"], {"G": 0, "D": 0, "E": 1})
            self.assertEqual(result["completed_count"], 0)

    def test_run_directory_must_not_be_inside_read_only_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            run = raw / "new-run"
            with self.assertRaises(runner.ResumeMismatch):
                runner.validate_run_roots(run, raw_root=raw, decoder_root=root / "decoder")

    def test_preflight_is_generation_free_and_records_bindings(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            called = []
            with self.assertRaises(runner.BlockedExecution):
                runner.main(["--stage", "preflight", "--run-dir", str(run)], factory=lambda *_a, **_k: called.append(True))
            self.assertEqual(called, [])

    def _main_fixture(self, *, record_full=None, monitor=None):
        base = Path(tempfile.mkdtemp()); root = base / "run"; root.mkdir(); source_root = base / "source"; source_root.mkdir()
        (root / "contract").write_text("fixture-contract", encoding="utf-8")
        z0 = np.zeros((1, 1, 1, 1, 4), dtype=np.float32)
        mask = np.zeros_like(z0, dtype=bool); mask[..., :2] = True
        v0 = np.zeros_like(z0); v0[..., 0] = 1.0
        consumed = z0.copy(); expected = np.ones_like(z0)
        source = {"raw_root": str(source_root), "decoder_root": str(source_root), "source_tree_sha256": "source",
                  "task7_code_sha256": "code", "z0_sha256": runner._array_hash(z0),
                  "mask_sha256": runner._array_hash(mask), "v0_sha256": runner._array_hash(v0)}
        contract = {"checkpoint_identity": {"sha256": "model"}, "vae_sha256": "vae",
                    "framework_commit": "framework", "code_bundle_sha256": "c" * 64}

        Geometry = type("Geometry", (), {"mask": mask, "condition_indexes": (0,)})
        Inputs = type("Inputs", (), {"z0": z0, "geometry": Geometry(), "directions": {"v0": v0}})

        class Adapter:
            runtime = object()

        class FakeFeedback:
            def __init__(self, *args, **kwargs): self.mask = mask; self.encoder = FakeEncoder()
            def extract_condition(self, value): return np.asarray(value).copy()
            def step(self, condition, step):
                full = expected.copy() if record_full is None else np.asarray(record_full).copy()
                return {"full_latent": full, "encoded_condition": np.zeros((1, 1, 1, 1, 4), np.float32),
                        "evidence": {"operation_counts": {"G": 1, "D": 1, "E": 1}}}

        class FakeEncoder:
            def __init__(self, *args, **kwargs): pass
            def encode(self, frame, precision="native"):
                return {"arrays": {"actual_output": expected.copy()},
                        "evidence": {"actual_encoder_input_dtype": "float32", "actual_output_dtype": "float32",
                                     "operation_count": 1, "dispatch_observed": True}}

        class Factory:
            def __init__(self): self.built = 0; self.unloaded = 0
            def build(self): self.built += 1; return (Adapter(), Inputs(), object())
            def unload(self): self.unloaded += 1

        fake_monitor = monitor or StaticTestMonitor()
        patches = [mock.patch.object(runner, "validate_task7_sources", return_value=source),
                   mock.patch.object(runner, "_load_contract", return_value=contract),
                   mock.patch.object(runner, "_source_condition", return_value=(z0, consumed)),
                   mock.patch.object(runner, "_source_sample_root", return_value=source_root),
                   mock.patch.object(runner, "_load_array", side_effect=lambda path: expected.copy() if ("output_full" in str(path) or "direct_condition" in str(path)) else (mask.copy() if "mask" in str(path) else v0.copy())),
                   mock.patch("umi_task7_runtime.FeedbackRuntime", FakeFeedback),
                   mock.patch("umi_task7_encoder.FeedbackEncoder", FakeEncoder)]
        return root, Factory(), fake_monitor, patches

    def test_main_preload_hardstop_does_not_build_factory(self):
        class Monitor(StaticTestMonitor):
            def check(self, **kwargs): raise runner.ResourceStop("preload")
        root, factory, monitor, patches = self._main_fixture(monitor=Monitor())
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner, "_code_digest", return_value="code"))
            for patch in patches: stack.enter_context(patch)
            with self.assertRaises(runner.ResourceStop):
                runner.main(["--stage", "smoke", "--run-dir", str(root), "--raw-root", str(root.parent / "source"), "--decoder-root", str(root.parent / "source"), "--launch-contract", str(root / "contract")], factory=factory, monitor=monitor)
        self.assertEqual(factory.built, 0)

    def test_main_stage_a_executes_all_sixteen_cpu_fixture_samples(self):
        root, factory, monitor, patches = self._main_fixture()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner, "_code_digest", return_value="code"))
            stack.enter_context(mock.patch.object(runner, "_source_frame", return_value=np.zeros((3, 1, 1, 1), dtype=np.float32)))
            stack.enter_context(mock.patch.object(runner, "_decoder_sample_root", return_value=root.parent / "source"))
            for patch in patches: stack.enter_context(patch)
            result = runner.main(["--stage", "A", "--run-dir", str(root), "--raw-root", str(root.parent / "source"),
                                  "--decoder-root", str(root.parent / "source"), "--launch-contract", str(root / "contract")],
                                 factory=factory, monitor=monitor)
            self.assertEqual(result, 0, json.loads((root / "stages" / "A" / "run_status.json").read_text(encoding="utf-8")))
        status = json.loads((root / "stages" / "A" / "run_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "COMPLETE")
        self.assertEqual(status["completed_count"], 16)
        self.assertEqual(factory.unloaded, 1)

    def test_main_smoke_rejects_historical_g_mismatch_and_unloads(self):
        root, factory, monitor, patches = self._main_fixture(record_full=np.zeros((1, 1, 1, 1, 4), np.float32))
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner, "_code_digest", return_value="code"))
            for patch in patches: stack.enter_context(patch)
            with self.assertRaises(runner.ResumeMismatch):
                runner.main(["--stage", "smoke", "--run-dir", str(root), "--raw-root", str(root.parent / "source"), "--decoder-root", str(root.parent / "source"), "--launch-contract", str(root / "contract")], factory=factory, monitor=monitor)
        self.assertEqual(factory.unloaded, 1)
        self.assertTrue((root / "run_status.json").is_file())

    def test_main_smoke_post_capture_peak_violation_is_rejected(self):
        class Monitor(StaticTestMonitor):
            def capture_sample(self, *args, **kwargs):
                self.captures.append(kwargs.get("phase", args[1] if len(args) > 1 else ""))
                if self.captures[-1] == "post_call":
                    return {"decision_status": "HARD_STOP", "reason_code": "GPU_SMOKE_PEAK_ALLOCATED_HIGH"}
                return {"decision_status": "OK"}
        root, factory, monitor, patches = self._main_fixture(monitor=Monitor())
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner, "_code_digest", return_value="code"))
            for patch in patches: stack.enter_context(patch)
            with self.assertRaises(runner.ResourceStop):
                runner.main(["--stage", "smoke", "--run-dir", str(root), "--raw-root", str(root.parent / "source"), "--decoder-root", str(root.parent / "source"), "--launch-contract", str(root / "contract")], factory=factory, monitor=monitor)
        self.assertIn("post_call", monitor.captures)
        root_status = json.loads((root / "run_status.json").read_text(encoding="utf-8"))
        self.assertEqual(root_status["operation_counts"], {"G": 1, "D": 1, "E": 1})

    def test_preflight_without_production_bindings_is_not_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(runner.BlockedExecution):
                runner.main(["--stage", "preflight", "--run-dir", temp])

    def test_main_never_writes_failure_status_into_protected_source_root(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / "raw"; raw.mkdir()
            sentinel = raw / "run_status.json"; sentinel.write_text("sentinel", encoding="utf-8")
            with self.assertRaises(runner.ResumeMismatch):
                runner.main(["--stage", "preflight", "--run-dir", str(raw),
                             "--raw-root", str(raw), "--decoder-root", str(raw)])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "sentinel")

    def test_c_probe_uses_mask_rms_and_normalized_ray(self):
        z1 = np.zeros((1, 1, 1, 1, 4), dtype=np.float32)
        mask = np.zeros_like(z1, dtype=bool); mask[..., :2] = True
        direction = np.zeros_like(z1); direction[..., :2] = (3.0, 4.0)
        seen = []
        class Feedback:
            def __init__(self): self.mask = mask
            def extract_condition(self, full): return full.copy()
            def step(self, condition, step): seen.append(condition.copy()); return {"evidence": {"operation_counts": {"G": 1, "D": 1, "E": 1}}}
        source = {"analysis_gate": {"a_scientific_pass": True, "b_engineering_pass": True, "b_repeatability_pass": True, "source_sha256": "s", "code_sha256": "c"},
                  "source_tree_sha256": "s", "task7_code_sha256": "c", "z1": z1, "delta1_directions": {"delta1_00": direction}}
        executor = runner.make_stage_executor("C", source=source, feedback=Feedback())
        spec = next(item for item in runner.build_stage_plan("C") if item.get("direction_id") == "delta1_00" and item.get("beta") == .1 and item.get("sign") == 1)
        executor(spec)
        np.testing.assert_array_equal(seen[0][..., :2], np.asarray([[[[[.3, .4]]]]], dtype=np.float32))

    def test_c_executor_runs_all_38_actual_ray_probes_and_gate(self):
        z1 = np.zeros((1, 1, 1, 1, 2), dtype=np.float32)
        mask = np.ones_like(z1, dtype=bool)
        directions = {f"delta1_{index:02d}": np.full_like(z1, index + 1, dtype=np.float32) for index in range(6)}
        seen = []
        class Feedback:
            def __init__(self): self.mask = mask
            def extract_condition(self, value): return np.asarray(value).copy()
            def step(self, condition, step):
                seen.append(np.array(condition, copy=True))
                return {"evidence": {"operation_counts": {"G": 1, "D": 1, "E": 1}}}
        source = {"analysis_gate": {"a_scientific_pass": True, "b_engineering_pass": True, "b_repeatability_pass": True,
                                     "source_sha256": "s", "code_sha256": "c"},
                  "source_tree_sha256": "s", "task7_code_sha256": "c", "z1": z1,
                  "delta1_directions": directions}
        executor = runner.make_stage_executor("C", source=source, feedback=Feedback())
        plan = runner.build_stage_plan("C")
        for spec in plan:
            executor(spec)
        self.assertEqual(len(seen), 38)
        self.assertTrue(np.array_equal(seen[0], z1))
        self.assertTrue(np.array_equal(seen[-1], z1))
        index = 0
        for spec in plan[1:-1]:
            direction = directions[spec["direction_id"]]
            expected = np.float32(spec["sign"] * spec["beta"]) * direction
            np.testing.assert_array_equal(seen[index + 1], expected)
            index += 1

    def test_b_step1_resume_loads_only_verified_step0_condition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_root = root / "source"; source_root.mkdir()
            condition0 = np.zeros((1, 1, 1, 1, 2), dtype=np.float32)
            condition1 = np.full_like(condition0, 2.0)
            # Approved FeedbackRuntime schema: conditions are extracted from a
            # full carrier, and ``actual`` stores full carriers.  Keep the
            # temporal axis distinct so a condition-only mock cannot pass.
            full = np.zeros((1, 1, 3, 1, 2), dtype=np.float32)
            mask = np.zeros_like(full, dtype=bool)
            mask[:, :, 0, :, :] = True
            source = {"raw_root": str(source_root), "decoder_root": str(source_root)}
            class Feedback:
                def __init__(self): self.seen = []; self.mask = mask
                def extract_condition(self, value): return np.asarray(value)[:, :, :1, :, :].copy()
                def embed_condition(self, value):
                    carrier = np.zeros_like(full)
                    carrier[:, :, :1, :, :] = np.asarray(value)
                    return carrier
                def step(self, condition, step):
                    condition = np.asarray(condition).copy(); self.seen.append(condition)
                    encoded = condition1.copy()
                    carrier = self.embed_condition(condition)
                    return {"full_latent": full.copy(), "encoded_condition": encoded,
                            "next_condition_fp32": encoded.copy(),
                            "condition_input_fp32": condition.copy(),
                            "actual": {key: carrier.copy() for key in ("prepared_condition", "initial_condition", "reference_condition", "first_condition", "last_condition")}
                                      | {"condition_steps": np.repeat(carrier[None], 30, axis=0)},
                            "evidence": {"operation_counts": {"G": 1, "D": 1, "E": 1},
                                         "prediction_noise_hash": f"noise-{step}"}}
            feedback = Feedback()
            spec0 = next(item for item in runner.build_stage_plan("B") if item["sample_id"] == "B_baseline_pre_step_0")
            spec1 = next(item for item in runner.build_stage_plan("B") if item["sample_id"] == "B_baseline_pre_step_1")
            real_load_array = runner._load_array
            def load_array(path, **kwargs):
                return full.copy() if Path(path).parent == source_root else real_load_array(path, **kwargs)
            with mock.patch.object(runner, "_source_condition", return_value=(condition0.copy(), condition0.copy())), \
                 mock.patch.object(runner, "_source_sample_root", return_value=source_root), \
                 mock.patch.object(runner, "_load_array", side_effect=load_array):
                executor = runner.make_stage_executor("B", source=source, feedback=feedback, run_dir=root)
                record0 = executor(spec0)
                store = runner.Task7SampleStore(root / "stages" / "B" / "samples")
                store.write_success(spec0["sample_id"], {"spec": spec0, "record": record0},
                                    operation_counts={"G": 1, "D": 1, "E": 1})
                # A fresh executor models a process restart.  It must reload
                # the verified condition instead of relying on its old map.
                resumed = runner.make_stage_executor("B", source=source, feedback=feedback, run_dir=root)
                resumed(spec1)
            np.testing.assert_array_equal(feedback.seen[-1], condition1)

    def test_b_make_executor_then_run_stage_does_not_precreate_stage_store(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_root = root / "source"; source_root.mkdir()
            condition = np.zeros((1, 1, 1, 1, 2), dtype=np.float32)
            full = np.zeros((1, 1, 3, 1, 2), dtype=np.float32)
            np.save(source_root / "output_full.npy", full)
            mask = np.zeros_like(full, dtype=bool); mask[:, :, 0, :, :] = True
            source = {"raw_root": str(source_root), "decoder_root": str(source_root)}

            class Feedback:
                def __init__(self): self.mask = mask
                def extract_condition(self, value): return np.asarray(value)[:, :, :1, :, :].copy()
                def embed_condition(self, value):
                    carrier = np.zeros_like(full)
                    carrier[:, :, :1, :, :] = np.asarray(value)
                    return carrier
                def step(self, value, step):
                    value = np.asarray(value).copy()
                    encoded = value.copy()
                    carrier = self.embed_condition(value)
                    return {"full_latent": full.copy(), "predicted_latent": full.copy(),
                            "encoded_condition": encoded,
                            "next_condition_fp32": encoded.copy(), "condition_input_fp32": value,
                            "actual": {key: carrier.copy() for key in
                                       ("prepared_condition", "initial_condition", "reference_condition",
                                        "first_condition", "last_condition")}
                                      | {"condition_steps": np.repeat(carrier[None], 30, axis=0)},
                            "evidence": {"operation_counts": {"G": 1, "D": 1, "E": 1},
                                         "prediction_noise_hash": f"noise-{step}"}}

            with mock.patch.object(runner, "_source_condition",
                                   return_value=(condition.copy(), condition.copy())), \
                 mock.patch.object(runner, "_source_sample_root", return_value=source_root):
                executor = runner.make_stage_executor("B", source=source, feedback=Feedback(), run_dir=root)
                self.assertFalse(root.joinpath("stages", "B", "samples").exists())
                result = runner.run_stage("B", root / "stages" / "B", execute=executor,
                                          binding=runner.test_binding("B"),
                                          monitor=runner.StaticMonitor(_safe_snapshot()))

            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["completed_count"], 16)
            self.assertEqual(result["observed_counts"], {"G": 16, "D": 16, "E": 16})

    def test_c_source_uses_b_step0_condition_and_b_step1_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); samples = root / "stages" / "B" / "samples"
            store = runner.Task7SampleStore(samples)
            shape = (1, 1, 1, 1, 2); z1 = np.zeros(shape, dtype=np.float32)
            z2 = np.ones(shape, dtype=np.float32)
            for name in ("baseline_pre", "v0_alpha_00_plus", "v0_alpha_00_minus", "v0_alpha_01_plus",
                         "v0_alpha_01_minus", "v0_alpha_02_plus", "v0_alpha_02_minus"):
                encoded = z1.copy() if name == "baseline_pre" else np.full(shape, len(name), dtype=np.float32)
                spec0 = next(item for item in runner.build_stage_plan("B") if item["sample_id"] == f"B_{name}_step_0")
                record0 = {"encoded_condition": encoded, "evidence": {"operation_counts": {"G": 1, "D": 1, "E": 1}}}
                store.write_success(spec0["sample_id"], {"spec": spec0, "record": record0}, operation_counts={"G": 1, "D": 1, "E": 1})
            spec1 = next(item for item in runner.build_stage_plan("B") if item["sample_id"] == "B_baseline_pre_step_1")
            store.write_success(spec1["sample_id"], {"spec": spec1, "record": {"full_latent": z2,
                "encoded_condition": np.full(shape, 3.0, dtype=np.float32),
                "evidence": {"prediction_noise_hash": "seed1"}}}, operation_counts={"G": 1, "D": 1, "E": 1})
            a_store = runner.Task7SampleStore(root / "stages" / "A" / "samples")
            a_store.write_success("A_fixture", {"record": _a_record()}, operation_counts={"G": 0, "D": 0, "E": 1})
            (root / "stages" / "A" / "run_status.json").parent.mkdir(parents=True, exist_ok=True)
            (root / "stages" / "A" / "run_status.json").write_text(json.dumps({"status": "COMPLETE",
                "completed_samples": ["A_fixture"], "skipped_samples": []}), encoding="utf-8")
            b_ids = [path.name for path in (root / "stages" / "B" / "samples").iterdir() if path.is_dir()]
            (root / "stages" / "B" / "run_status.json").write_text(json.dumps({"status": "COMPLETE",
                "completed_samples": sorted(b_ids), "skipped_samples": []}), encoding="utf-8")
            a_digest = runner._stage_completion_digests(root, "A")
            b_digest = runner._stage_completion_digests(root, "B")
            (root / "analysis_gate.json").write_text(json.dumps({"a_scientific_pass": True, "b_engineering_pass": True,
                "b_repeatability_pass": True, "source_sha256": "source", "code_sha256": "code",
                "a_run_status_sha256": a_digest["run_status_sha256"],
                "a_samples_manifest_sha256": a_digest["samples_manifest_sha256"],
                "b_run_status_sha256": b_digest["run_status_sha256"],
                "b_samples_manifest_sha256": b_digest["samples_manifest_sha256"]}), encoding="utf-8")
            source = {"source_tree_sha256": "source", "task7_code_sha256": "code"}
            runner._prepare_c_source(source, root)
            np.testing.assert_array_equal(source["z1"], z1)
            np.testing.assert_array_equal(source["b_z2"], np.full(shape, 3.0, dtype=np.float32))
            np.testing.assert_array_equal(source["b_z2_full"], z2)
            np.testing.assert_array_equal(source["delta1_directions"]["delta1_00"], np.full(shape, len("v0_alpha_00_plus"), dtype=np.float32))

    def test_success_record_references_must_be_manifest_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            store = runner.Task7SampleStore(Path(temp))
            path = store.write_success("sample", {"array": np.ones((2,), dtype=np.float32)})
            record = json.loads((path / "record.json").read_text(encoding="utf-8"))
            record["tampered"] = {"artifact": "missing.npy", "dtype": "float32", "shape": [1]}
            (path / "record.json").write_text(json.dumps(record), encoding="utf-8")
            status = json.loads((path / "status.json").read_text(encoding="utf-8"))
            status["artifact_sha256"]["record.json"] = runner.sha256_file(path / "record.json")
            (path / "status.json").write_text(json.dumps(status), encoding="utf-8")
            with self.assertRaises(runner.ResumeMismatch):
                store.prepare("sample", resume=True)

    def test_c_gate_rejects_foreign_a_b_completion_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for stage in ("A", "B"):
                stage_root = root / "stages" / stage
                (stage_root / "samples").mkdir(parents=True)
                (stage_root / "run_status.json").write_text(json.dumps({"status": "COMPLETE",
                    "completed_samples": [], "skipped_samples": []}), encoding="utf-8")
            a_digest = runner._stage_completion_digests(root, "A")
            b_digest = runner._stage_completion_digests(root, "B")
            (root / "analysis_gate.json").write_text(json.dumps({
                "a_scientific_pass": True, "b_engineering_pass": True, "b_repeatability_pass": True,
                "source_sha256": "source", "code_sha256": "code",
                "a_run_status_sha256": a_digest["run_status_sha256"],
                "a_samples_manifest_sha256": a_digest["samples_manifest_sha256"],
                "b_run_status_sha256": b_digest["run_status_sha256"],
                "b_samples_manifest_sha256": "foreign"}), encoding="utf-8")
            with self.assertRaises(runner.ResumeMismatch):
                runner._prepare_c_source({"source_tree_sha256": "source", "task7_code_sha256": "code"}, root)

    def test_source_contract_uses_historical_task6_action_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            action = np.arange(160, dtype=np.float32).reshape(16, 10)
            action_path = Path(temp) / "action.json"
            action_path.write_text(json.dumps(action.tolist()), encoding="utf-8")
            stable_action_hash = runner.stable_hash(action)
            legacy_post_vae_hash = runner._task6_array_hash(action)
            self.assertNotEqual(stable_action_hash, legacy_post_vae_hash)
            source = {"metadata": {"action": stable_action_hash, "prompt": "Put the pot to the left of the purple item.",
                                     "state": "bridge_0", "seed": 0}}
            contract = {"prompt": source["metadata"]["prompt"], "group": {"state": "bridge_0", "seed": 0},
                        "action": action.tolist()}
            observed = runner._validate_source_contract(source, contract, action_path)
            np.testing.assert_array_equal(observed, action)
            source["metadata"]["action"] = legacy_post_vae_hash
            with self.assertRaisesRegex(runner.ResumeMismatch, "action evidence"):
                runner._validate_source_contract(source, contract, action_path)

    def test_source_contract_rejects_changed_action_with_stable_hash_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            action = np.arange(160, dtype=np.float32).reshape(16, 10)
            action_path = Path(temp) / "action.json"
            action_path.write_text(json.dumps(action.tolist()), encoding="utf-8")
            source = {"metadata": {"action": runner.stable_hash(action), "prompt": "Put the pot to the left of the purple item.",
                                     "state": "bridge_0", "seed": 0}}
            changed = action.copy(); changed[0, 0] += np.float32(1.0)
            contract = {"prompt": source["metadata"]["prompt"], "group": {"state": "bridge_0", "seed": 0},
                        "action": changed.tolist()}
            with self.assertRaisesRegex(runner.ResumeMismatch, "action evidence"):
                runner._validate_source_contract(source, contract, action_path)


if __name__ == "__main__":
    unittest.main()
