import json
import hashlib
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np


class Task6RunnerTests(unittest.TestCase):
    def public_chain(self, temp, *, execute_error=None, cleanup_error=None):
        import hashlib
        import umi_task6_runtime as runtime_api
        import run_umi_task6_experiment as api
        carrier = np.ones((1, 48, 5, 16, 16), np.float32)
        mask = np.zeros_like(carrier, dtype=bool); mask[:, :, 0] = True
        bank = np.zeros((3,) + carrier.shape, np.float32); bank[:, mask] = 1
        direction_map = runtime_api._derive_frozen_directions_unpinned(bank, mask)
        direction_hashes = {key: runtime_api._array_sha(value) for key, value in direction_map.items()}
        inputs = runtime_api.Task6Inputs(carrier, [0], mask, bank, action=np.zeros((16, 10), np.float32), prompt="pilot", direction_hashes=direction_hashes)
        counts = {"execute": 0, "pilot_execute": 0, "cleanup": 0}; phase = {"name": "smoke"}
        class Runtime:
            provenance = {"source_commit": "2b17a2413bd86b2cf9b03823637108851e4ddf2d"}
            def actual_identity(self): return {"fixture": "public"}
            def execute(self, spec, runtime_inputs, *, scope="full"):
                counts["execute"] += 1
                if phase["name"] == "pilot": counts["pilot_execute"] += 1
                if phase["name"] == "pilot" and execute_error is not None and counts["pilot_execute"] == 1: raise execute_error
                return {"output_full": np.array([counts["execute"]], np.float32)}
            def cleanup(self):
                counts["cleanup"] += 1
                if phase["name"] == "pilot" and cleanup_error is not None: raise cleanup_error
        runtime = Runtime(); direction_path = Path(temp, "directions.npy"); np.save(direction_path, bank)
        asset_path = Path(temp, "asset.bin"); asset_path.write_bytes(b"asset")
        cfg = {"environment": {"name": "cpu"}, "provenance": {"source_commit": "2b17a2413bd86b2cf9b03823637108851e4ddf2d"},
               "asset_hashes": {"asset": hashlib.sha256(asset_path.read_bytes()).hexdigest()}, "asset_paths": {"asset": str(asset_path)},
               "carrier_shape": list(inputs.z0.shape), "carrier_hash": inputs.identity()["z0"], "condition_indexes": [0], "predicted_indexes": [1,2,3,4],
               "mask_shape": list(mask.shape), "mask_hash": inputs.geometry.metadata()["mask_sha256"], "action_shape": [16,10], "action_hash": runtime_api._array_sha(inputs.action),
               "prompt": inputs.prompt, "direction_hashes": direction_hashes, "direction_bank_path": str(direction_path),
               "direction_bank_file_hash": hashlib.sha256(direction_path.read_bytes()).hexdigest(),
               "settings": {"num_steps":30,"guidance":1.0,"shift":10.0,"autocast":False,"tf32":False,"diffusion_cache":False,"batch_size":1},
               "seed_config": {"seed":0,"prepare":0,"sampler":0,"scheduler":0}, "observed_runtime_identity": runtime.actual_identity(), "observed_input_identity": inputs.identity()}
        evidence = Path(temp, "preflight.json"); evidence.write_text(json.dumps(cfg))
        api.execute_task6(api.parse_args(["--phase", "preflight", "--run-dir", temp, "--preflight-json", str(evidence)]), runtime=runtime, inputs=inputs)
        phase["name"] = "smoke"
        safe = lambda: {"gpu_used_gib":0,"gpu_free_gib":100,"gpu_reserved_gib":0,"ram_available_gib":600,"rss_gib":0,"swap_used_gib":0,"disk_free_gib":20}
        api.run_resource_smoke(temp, lifecycle={"pre_load": lambda: None, "load": lambda: (runtime, inputs), "cleanup": lambda: None, "unload": lambda: None}, samplers={key: safe for key in ("gpu", "ram", "disk")})
        api.accept_resource_smoke(temp, hashes=json.loads(Path(temp, "run_status.json").read_text())["hashes"])
        phase["name"] = "pilot"
        return api, runtime, inputs, counts, phase

    def smoke_fixture(self, temp, *, gpu=None, runtime=None, inputs=None, lifecycle_log=None):
        import run_umi_task6_experiment as api
        runtime = runtime or type("Runtime", (), {"actual_identity": lambda self: {"fixture": "runtime"}, "execute": lambda self, spec, inputs, *, scope: {"output_full": np.zeros(8, np.float32)}})()
        inputs = inputs or type("Inputs", (), {"state": "bridge_0", "seed": 0, "identity": lambda self: {"fixture": "inputs"}})()
        binding = api.build_task6_hash_binding(runtime, inputs, api.task6_binding_config(inputs))
        Path(temp, "run_status.json").write_text(json.dumps({"status": "PREFLIGHT_COMPLETE", "phase": "RESOURCE_SMOKE", "hashes": binding}))
        sampler = gpu or (lambda: {"gpu_used_gib": 0, "gpu_free_gib": 100, "ram_available_gib": 600, "disk_free_gib": 20})
        samplers = {key: sampler for key in ("gpu", "ram", "disk")}
        log = lifecycle_log if lifecycle_log is not None else []
        lifecycle = {"pre_load": lambda: log.append("pre_load"), "load": lambda: (log.append("load") or (runtime, inputs)),
                     "cleanup": lambda: log.append("cleanup"), "unload": lambda: log.append("unload")}
        return api, runtime, inputs, samplers, lifecycle, log

    def test_empty_preflight_is_blocked(self):
        import run_umi_task6_experiment as api
        with self.assertRaises(api.BlockedExecution):
            api.execute_task6(api.parse_args([]))

    def test_smoke_stops_at_review_and_state_machine_requires_explicit_phase(self):
        import run_umi_task6_experiment as api
        args = api.parse_args([])
        self.assertEqual(args.phase, "preflight")
        with tempfile.TemporaryDirectory() as temp:
            class Inputs:
                state = "bridge_0"; seed = 0
                def identity(self): return {"fixture": "inputs"}
            class Runtime:
                def __init__(self): self.calls = 0
                def execute(self, spec, inputs, *, scope): self.calls += 1; return {"output_full": np.zeros(8, np.float32)}
                def actual_identity(self): return {"fixture": "runtime"}
            runtime, inputs = Runtime(), Inputs()
            binding = api.build_task6_hash_binding(runtime, inputs, api.task6_binding_config(inputs))
            Path(temp, "run_status.json").write_text(json.dumps({"status": "PREFLIGHT_COMPLETE", "phase": "RESOURCE_SMOKE", "hashes": binding}))
            samplers = {key: (lambda: {"gpu_used_gib": 0, "gpu_free_gib": 100,
                "ram_available_gib": 600, "disk_free_gib": 20}) for key in ("gpu", "ram", "disk")}
            result = api.run_resource_smoke(temp, samplers=samplers,
                                            lifecycle={"pre_load": lambda: None, "load": lambda: (runtime, inputs),
                                                       "cleanup": lambda: None, "unload": lambda: None})
            self.assertEqual(result["status"], "AWAITING_RESOURCE_REVIEW")
            self.assertEqual(result["baseline_calls"], 1)
            self.assertTrue(Path(temp, "run_status.json").is_file())
            self.assertEqual(json.loads(Path(temp, "run_status.json").read_text())["status"], "AWAITING_RESOURCE_REVIEW")

    def test_monitor_failure_is_hard_stop_and_atomic_store_preserves_success(self):
        import run_umi_task6_experiment as api
        self.assertEqual(api.monitor_decision({"monitor_failure": "boom"})["status"], "HARD_STOP")
        with tempfile.TemporaryDirectory() as temp:
            store = api.AtomicSampleStore(temp)
            store.publish("s0", {"x": np.array([1], dtype=np.float32)}, {"code":"c"})
            with self.assertRaises(FileExistsError):
                store.publish("s0", {"x": np.array([2], dtype=np.float32)}, {"code":"c"})
            self.assertTrue(Path(temp, "s0", "MANIFEST.sha256").is_file())

    def test_monitor_flushes_csv_on_failure(self):
        import run_umi_task6_experiment as api
        with tempfile.TemporaryDirectory() as temp:
            monitor = api.ResourceMonitor(temp, gpu_sampler=lambda: (_ for _ in ()).throw(RuntimeError("poll")),
                                          ram_sampler=lambda: {}, disk_sampler=lambda: {})
            monitor.start(); time.sleep(0.03)
            with self.assertRaises(RuntimeError): monitor.stop()
            self.assertTrue(Path(temp, "gpu_samples.csv").is_file())
            self.assertTrue(Path(temp, "disk_samples.csv").is_file())

    def test_pilot_requires_accepted_smoke_decision(self):
        import run_umi_task6_experiment as api
        with tempfile.TemporaryDirectory() as temp:
            runtime = type("Runtime", (), {"execute": lambda self, spec, inputs, *, scope: {"output_full": np.zeros(8, np.float32)}})()
            samplers = {key: (lambda: {"gpu_used_gib": 0, "gpu_free_gib": 100, "ram_available_gib": 600, "disk_free_gib": 20}) for key in ("gpu", "ram", "disk")}
            class Inputs:
                state = "bridge_0"; seed = 0
                def identity(self): return {}
            runtime.inputs = Inputs()
            binding = api.build_task6_hash_binding(runtime, runtime.inputs, api.task6_binding_config(runtime.inputs))
            Path(temp, "run_status.json").write_text(json.dumps({"status": "PREFLIGHT_COMPLETE", "phase": "RESOURCE_SMOKE", "hashes": binding}))
            api.run_resource_smoke(temp, samplers=samplers,
                                   lifecycle={"pre_load": lambda: None, "load": lambda: (runtime, runtime.inputs),
                                              "cleanup": lambda: None, "unload": lambda: None})
            with self.assertRaises(api.BlockedExecution):
                api.authorize_pilot(temp)

    def test_smoke_stage_order(self):
        with tempfile.TemporaryDirectory() as temp:
            api, _, _, samplers, lifecycle, log = self.smoke_fixture(temp)
            result = api.run_resource_smoke(temp, samplers=samplers, lifecycle=lifecycle)
            self.assertEqual(result["status"], "AWAITING_RESOURCE_REVIEW"); self.assertEqual(log, ["pre_load", "load", "cleanup", "unload"]); self.assertEqual(result["baseline_calls"], 1)

    def test_unsafe_preload_no_load(self):
        with tempfile.TemporaryDirectory() as temp:
            log = []; api, _, _, _, lifecycle, _ = self.smoke_fixture(temp, lifecycle_log=log)
            sampler = lambda: {"gpu_used_gib": 2, "gpu_free_gib": 100, "ram_available_gib": 600, "disk_free_gib": 20}
            lifecycle["load"] = lambda: log.append("load")
            result = api.run_resource_smoke(temp, samplers={key: sampler for key in ("gpu", "ram", "disk")}, lifecycle=lifecycle)
            self.assertEqual(result["status"], "RESOURCE_STOP"); self.assertNotIn("load", log)

    def test_load_failure_writes_status(self):
        with tempfile.TemporaryDirectory() as temp:
            api, _, _, samplers, lifecycle, _ = self.smoke_fixture(temp); lifecycle["load"] = lambda: None
            result = api.run_resource_smoke(temp, samplers=samplers, lifecycle=lifecycle)
            self.assertEqual(result["status"], "RESOURCE_STOP"); self.assertTrue(Path(temp, "run_status.json").is_file())

    def test_cleanup_failure_writes_status(self):
        with tempfile.TemporaryDirectory() as temp:
            api, _, _, samplers, lifecycle, _ = self.smoke_fixture(temp); lifecycle["cleanup"] = lambda: (_ for _ in ()).throw(RuntimeError("cleanup"))
            self.assertEqual(api.run_resource_smoke(temp, samplers=samplers, lifecycle=lifecycle)["status"], "RESOURCE_STOP")

    def test_unload_failure_writes_status(self):
        with tempfile.TemporaryDirectory() as temp:
            api, _, _, samplers, lifecycle, _ = self.smoke_fixture(temp); lifecycle["unload"] = lambda: (_ for _ in ()).throw(RuntimeError("unload"))
            self.assertEqual(api.run_resource_smoke(temp, samplers=samplers, lifecycle=lifecycle)["status"], "RESOURCE_STOP")

    def test_public_pilot_requires_real_monitor(self):
        with tempfile.TemporaryDirectory() as temp:
            api, runtime, inputs, _, _, _ = self.smoke_fixture(temp)
            with self.assertRaises(api.BlockedExecution): api.run_pilot(runtime, inputs, temp, monitor=object())

    def test_monitor_joined_before_flush(self):
        import run_umi_task6_experiment as api
        with tempfile.TemporaryDirectory() as temp:
            monitor = api.ResourceMonitor(temp, gpu_sampler=lambda: {}, ram_sampler=lambda: {}, disk_sampler=lambda: {})
            monitor.start(); monitor.stop(); self.assertFalse(monitor._thread.is_alive()); self.assertTrue(Path(temp, "gpu_samples.csv").is_file())

    def test_capture_sample_evaluates_row_and_two_cleanup_growth_stop(self):
        import run_umi_task6_experiment as api
        values = [0]
        def gpu(): values[0] += 3; return {"gpu_used_gib": values[0], "gpu_free_gib": 100}
        ram = lambda: {"ram_available_gib": 600, "rss_gib": 0, "swap_used_gib": 0}
        disk = lambda: {"disk_free_gib": 20}
        with tempfile.TemporaryDirectory() as temp:
            monitor = api.ResourceMonitor(temp, gpu_sampler=gpu, ram_sampler=ram, disk_sampler=disk)
            monitor.start()
            monitor.capture_sample("s0", "post_cleanup", 31)
            monitor.capture_sample("s1", "post_cleanup", 30)
            row = monitor.capture_sample("s2", "post_cleanup", 29, Path(temp))
            self.assertEqual(row["decision_status"], "HARD_STOP")
            monitor.stop()

    def test_synchronous_sampler_failure_writes_terminal_status(self):
        import run_umi_task6_experiment as api
        with tempfile.TemporaryDirectory() as temp:
            runtime = type("Runtime", (), {"actual_identity": lambda self: {"fixture": "r"}, "execute": lambda self, spec, inputs, *, scope: {"output_full": np.zeros(8, np.float32)}})()
            inputs = type("Inputs", (), {"state": "bridge_0", "seed": 0, "identity": lambda self: {"fixture": "i"}})()
            binding = api.build_task6_hash_binding(runtime, inputs, api.task6_binding_config(inputs))
            Path(temp, "run_status.json").write_text(json.dumps({"status": "PREFLIGHT_COMPLETE", "phase": "RESOURCE_SMOKE", "hashes": binding}))
            bad = lambda: (_ for _ in ()).throw(RuntimeError("sampler"))
            good = lambda: {"gpu_used_gib": 0, "gpu_free_gib": 100, "ram_available_gib": 600, "disk_free_gib": 20}
            result = api.run_resource_smoke(temp, lifecycle={"pre_load": lambda: None, "load": lambda: (runtime, inputs), "cleanup": lambda: None, "unload": lambda: None}, samplers={"gpu": bad, "ram": good, "disk": good})
            self.assertEqual(result["status"], "RESOURCE_STOP")

    def test_monitor_latches_first_hard_stop_across_later_normal_samples(self):
        import run_umi_task6_experiment as api
        with tempfile.TemporaryDirectory() as temp:
            current = {"gpu_used_gib": 80.0}
            def gpu():
                return {"gpu_used_gib": current["gpu_used_gib"], "gpu_free_gib": 100.0,
                        "gpu_reserved_gib": 0.0, "ram_available_gib": 600.0,
                        "rss_gib": 0.0, "swap_used_gib": 0.0, "disk_free_gib": 20.0}
            monitor = api.ResourceMonitor(temp, gpu_sampler=gpu, ram_sampler=gpu, disk_sampler=gpu)
            first = monitor.capture_sample("bad", "pre_sample", 2, Path(temp))
            current["gpu_used_gib"] = 0.0
            second = monitor.capture_sample("normal", "pre_sample", 1, Path(temp))
            self.assertEqual(first["decision_status"], "HARD_STOP")
            self.assertEqual(second["decision_status"], "HARD_STOP")
            self.assertEqual(second["reason_code"], first["reason_code"])
            self.assertEqual(monitor.check()["status"], "HARD_STOP")

    def test_stale_smoke_acceptance_is_invalidated_by_new_preflight(self):
        import run_umi_task6_experiment as api
        with tempfile.TemporaryDirectory() as temp:
            api, runtime, inputs, samplers, lifecycle, _ = self.smoke_fixture(temp)
            result = api.run_resource_smoke(temp, samplers=samplers, lifecycle=lifecycle)
            api.accept_resource_smoke(temp, hashes=result["hashes"])
            self.assertTrue(Path(temp, "smoke_acceptance.json").is_file())
            # A new preflight attempt invalidates prior acceptance before
            # validating its new evidence.
            with self.assertRaises(api.BlockedExecution):
                api.execute_task6(api.parse_args(["--phase", "preflight", "--run-dir", temp, "--preflight-json", str(Path(temp, "missing.json"))]), runtime=runtime, inputs=inputs)
            self.assertFalse(Path(temp, "smoke_acceptance.json").exists())

    def test_smoke_peak_reserved_and_worst_disk_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            api, _, _, _, lifecycle, _ = self.smoke_fixture(temp)
            state = {"n": 0}
            def sampler():
                state["n"] += 1
                return {"gpu_used_gib": 0, "gpu_free_gib": 100, "gpu_reserved_gib": 70,
                        "ram_available_gib": 600, "rss_gib": 0, "swap_used_gib": 0,
                        "disk_free_gib": 12 if state["n"] % 2 else 20}
            result = api.run_resource_smoke(temp, samplers={key: sampler for key in ("gpu", "ram", "disk")}, lifecycle=lifecycle)
            self.assertEqual(result["status"], "RESOURCE_STOP")

    def test_smoke_worst_case_includes_background_ram_and_disk_rows(self):
        import run_umi_task6_experiment as api
        with tempfile.TemporaryDirectory() as temp:
            _, _, _, samplers, lifecycle, _ = self.smoke_fixture(temp)
            original = api.ResourceMonitor
            class BackgroundOnlyMonitor:
                def __init__(self, root, **kwargs):
                    self.failure = None; self.last_resources = {"gpu_used_gib": 0.0, "gpu_free_gib": 100.0,
                        "ram_available_gib": 600.0, "rss_gib": 0.0, "swap_used_gib": 0.0, "disk_free_gib": 20.0}
                    self._gpu = [{"gpu_used_gib": 0.0, "gpu_free_gib": 100.0, "gpu_reserved_gib": 0.0,
                                  "gpu_peak_allocated_gib": 0.0, "gpu_peak_nvml_used_gib": 0.0}]
                    self._ram = [{"ram_available_gib": 600.0, "rss_gib": 200.0, "swap_used_gib": 1.0}]
                    self._disk = [{"disk_free_gib": 2.0}]
                    self._sample_rows = []
                def start(self): return self
                def stop(self): return None
            api.ResourceMonitor = BackgroundOnlyMonitor
            try:
                result = api.run_resource_smoke(temp, samplers=samplers, lifecycle=lifecycle)
            finally:
                api.ResourceMonitor = original
            self.assertEqual(result["status"], "RESOURCE_STOP")
            self.assertTrue(result["errors"])
            self.assertEqual(result["last_resource_snapshots"]["background_ram"][0]["swap_used_gib"], 1.0)
            self.assertEqual(result["last_resource_snapshots"]["background_disk"][0]["disk_free_gib"], 2.0)

    def test_smoke_background_disk_hard_limit_is_not_ignored(self):
        import run_umi_task6_experiment as api
        with tempfile.TemporaryDirectory() as temp:
            _, _, _, samplers, lifecycle, _ = self.smoke_fixture(temp)
            original = api.ResourceMonitor
            class BackgroundDiskMonitor:
                failure = None
                last_resources = {"gpu_used_gib": 0.0, "gpu_free_gib": 100.0,
                                  "ram_available_gib": 600.0, "rss_gib": 0.0,
                                  "swap_used_gib": 0.0, "disk_free_gib": 20.0}
                _gpu = []
                _ram = [{"ram_available_gib": 600.0, "rss_gib": 0.0, "swap_used_gib": 0.0}]
                _disk = [{"disk_free_gib": 2.0}]
                _sample_rows = []
                def __init__(self, root, **kwargs): pass
                def start(self): return self
                def stop(self): pass
            api.ResourceMonitor = BackgroundDiskMonitor
            try:
                result = api.run_resource_smoke(temp, samplers=samplers, lifecycle=lifecycle)
            finally:
                api.ResourceMonitor = original
            self.assertEqual(result["status"], "RESOURCE_STOP")
            self.assertTrue(any("disk" in str(item).lower() for item in result["errors"]))

    def test_monitor_start_failure_writes_canonical_smoke_status(self):
        import run_umi_task6_experiment as api
        with tempfile.TemporaryDirectory() as temp:
            api, _, _, samplers, lifecycle, _ = self.smoke_fixture(temp)
            original = api.ResourceMonitor.start
            api.ResourceMonitor.start = lambda self: (_ for _ in ()).throw(RuntimeError("start"))
            try:
                result = api.run_resource_smoke(temp, samplers=samplers, lifecycle=lifecycle)
            finally:
                api.ResourceMonitor.start = original
            self.assertEqual(result["status"], "RESOURCE_STOP")

    def test_public_pilot_execute_failure_cleans_and_records_post_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            api, runtime, inputs, counts, _ = self.public_chain(temp, execute_error=RuntimeError("execute"))
            safe = lambda: {"gpu_used_gib":0,"gpu_free_gib":100,"gpu_reserved_gib":0,"ram_available_gib":600,"rss_gib":0,"swap_used_gib":0,"disk_free_gib":20}
            result = api.run_pilot(runtime, inputs, temp, monitor=api.ResourceMonitor(temp, gpu_sampler=safe, ram_sampler=safe, disk_sampler=safe))
            self.assertIn(result["status"], {"FAILED", "RESOURCE_STOP"}); self.assertEqual(counts["cleanup"], 1)
            self.assertFalse(any(json.loads((sample / "status.json").read_text()).get("status") == "success" for sample in (Path(temp) / "samples").glob("*") if (sample / "status.json").is_file()))
            self.assertTrue(Path(temp, "sample_resource_snapshots.jsonl").is_file())

    def test_public_pilot_cleanup_failure_stops_without_mixed_lists(self):
        with tempfile.TemporaryDirectory() as temp:
            api, runtime, inputs, counts, _ = self.public_chain(temp, cleanup_error=RuntimeError("cleanup"))
            safe = lambda: {"gpu_used_gib":0,"gpu_free_gib":100,"gpu_reserved_gib":0,"ram_available_gib":600,"rss_gib":0,"swap_used_gib":0,"disk_free_gib":20}
            result = api.run_pilot(runtime, inputs, temp, monitor=api.ResourceMonitor(temp, gpu_sampler=safe, ram_sampler=safe, disk_sampler=safe))
            self.assertEqual(result["status"], "RESOURCE_STOP"); self.assertEqual(result["reason_code"], "RESOURCE_CLEANUP_FAILURE")
            self.assertEqual(counts["cleanup"], 1); self.assertFalse(set(result["completed_samples"]) & set(result["failed_samples"]))

    def test_public_pilot_execute_and_cleanup_failures_preserve_both_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            api, runtime, inputs, counts, _ = self.public_chain(temp, execute_error=RuntimeError("execute primary"), cleanup_error=RuntimeError("cleanup secondary"))
            safe = lambda: {"gpu_used_gib":0,"gpu_free_gib":100,"gpu_reserved_gib":0,"ram_available_gib":600,"rss_gib":0,"swap_used_gib":0,"disk_free_gib":20}
            result = api.run_pilot(runtime, inputs, temp, monitor=api.ResourceMonitor(temp, gpu_sampler=safe, ram_sampler=safe, disk_sampler=safe))
            self.assertEqual(result["status"], "RESOURCE_STOP"); self.assertEqual(result["reason_code"], "RESOURCE_CLEANUP_FAILURE"); self.assertEqual(counts["cleanup"], 1)
            self.assertTrue(any(item.get("primary", {}).get("message") == "execute primary" and any(sec.get("message") == "cleanup secondary" for sec in item.get("secondary", [])) for item in result["secondary_errors"]))

    def test_public_pilot_monitor_start_failure_is_canonical_and_no_execute(self):
        with tempfile.TemporaryDirectory() as temp:
            api, runtime, inputs, counts, _ = self.public_chain(temp)
            safe = lambda: {"gpu_used_gib":0,"gpu_free_gib":100,"gpu_reserved_gib":0,"ram_available_gib":600,"rss_gib":0,"swap_used_gib":0,"disk_free_gib":20}
            class BrokenMonitor(api.ResourceMonitor):
                def start(self): raise RuntimeError("monitor start")
            result = api.run_pilot(runtime, inputs, temp, monitor=BrokenMonitor(temp, gpu_sampler=safe, ram_sampler=safe, disk_sampler=safe))
            self.assertEqual(result["status"], "RESOURCE_STOP"); self.assertEqual(result["reason_code"], "MONITOR_FAILURE")
            self.assertEqual(counts["pilot_execute"], 0); self.assertTrue(Path(temp, "MANIFEST.sha256").is_file())

    def test_public_pilot_oom_preserves_primary_classification_over_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            api, runtime, inputs, counts, _ = self.public_chain(temp, execute_error=MemoryError("CUDA out of memory"), cleanup_error=RuntimeError("cleanup"))
            safe = lambda: {"gpu_used_gib":0,"gpu_free_gib":100,"gpu_reserved_gib":0,"ram_available_gib":600,"rss_gib":0,"swap_used_gib":0,"disk_free_gib":20}
            class BrokenStop(api.ResourceMonitor):
                def stop(self):
                    super().stop(); raise RuntimeError("monitor stop")
            result = api.run_pilot(runtime, inputs, temp, monitor=BrokenStop(temp, gpu_sampler=safe, ram_sampler=safe, disk_sampler=safe))
            self.assertEqual(result["status"], "RESOURCE_STOP"); self.assertEqual(result["reason_code"], "CUDA_OOM"); self.assertEqual(counts["cleanup"], 1)
            self.assertTrue(any(item.get("kind") == "monitor_stop" for item in result["secondary_errors"]))
            failure = next(Path(temp, "samples").glob("*/sample.json"))
            self.assertIn("CUDA out of memory", failure.read_text())
            self.assertIn("cleanup", failure.read_text())

    def test_public_pilot_post_cleanup_monitor_failure_is_monitor_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            api, runtime, inputs, counts, _ = self.public_chain(temp)
            safe = lambda: {"gpu_used_gib":0,"gpu_free_gib":100,"gpu_reserved_gib":0,"ram_available_gib":600,"rss_gib":0,"swap_used_gib":0,"disk_free_gib":20}
            class BrokenCapture(api.ResourceMonitor):
                def capture_sample(self, sample_id, phase, remaining, run_dir=None):
                    if phase == "post_cleanup": raise RuntimeError("post sampler")
                    return super().capture_sample(sample_id, phase, remaining, run_dir)
            result = api.run_pilot(runtime, inputs, temp, monitor=BrokenCapture(temp, gpu_sampler=safe, ram_sampler=safe, disk_sampler=safe))
            self.assertEqual(result["status"], "RESOURCE_STOP"); self.assertEqual(result["reason_code"], "MONITOR_FAILURE"); self.assertEqual(counts["pilot_execute"], 1)

    def test_public_pilot_two_success_stop_and_resume_preserves_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            api, runtime, inputs, counts, _ = self.public_chain(temp)
            def disk():
                return {"gpu_used_gib": 0, "gpu_free_gib": 100, "gpu_reserved_gib": 0, "ram_available_gib": 600,
                        "rss_gib": 0, "swap_used_gib": 0, "disk_free_gib": 20 if counts["cleanup"] < 2 else 4}
            safe = lambda: {**disk(), "disk_free_gib": 20}
            first_monitor = api.ResourceMonitor(temp, gpu_sampler=disk, ram_sampler=disk, disk_sampler=disk)
            first = api.run_pilot(runtime, inputs, temp, monitor=first_monitor)
            self.assertEqual(first["status"], "RESOURCE_STOP"); self.assertEqual(counts["pilot_execute"], 2)
            first_samples = {}
            for sample in sorted((Path(temp) / "samples").iterdir()):
                state = json.loads((sample / "status.json").read_text())
                if state.get("status") == "success":
                    first_samples[sample.name] = ([(str(p.relative_to(sample)), hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in sorted(sample.rglob("*")) if p.is_file()], sample.stat().st_mtime_ns)
            self.assertEqual(len(first_samples), 2); self.assertTrue(first.get("smoke_run_id")); self.assertTrue(Path(temp, "MANIFEST.sha256").is_file())
            second = api.run_pilot(runtime, inputs, temp, resume=True,
                                   monitor=api.ResourceMonitor(temp, gpu_sampler=safe, ram_sampler=safe, disk_sampler=safe))
            self.assertEqual(second["status"], "AWAITING_REVIEW"); self.assertEqual(counts["pilot_execute"], 32)
            self.assertEqual(len(second["skipped_samples"]), 2)
            for name, (hashes, mtime) in first_samples.items():
                sample = Path(temp, "samples", name)
                self.assertEqual(mtime, sample.stat().st_mtime_ns)
                self.assertEqual(hashes, [(str(p.relative_to(sample)), hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in sorted(sample.rglob("*")) if p.is_file()])
            runtime.execute = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("partial raw-complete resume must skip generation"))
            third = api.run_pilot(runtime, inputs, temp, resume=True,
                                  monitor=api.ResourceMonitor(temp, gpu_sampler=safe, ram_sampler=safe, disk_sampler=safe))
            self.assertEqual(third, second)
            self.assertEqual(counts["pilot_execute"], 32)

    def test_public_resume_of_complete_raw_run_skips_generation_and_keeps_status(self):
        with tempfile.TemporaryDirectory() as temp:
            api, runtime, inputs, counts, _ = self.public_chain(temp)
            safe = lambda: {"gpu_used_gib": 0, "gpu_free_gib": 100, "gpu_reserved_gib": 0,
                            "ram_available_gib": 600, "rss_gib": 0, "swap_used_gib": 0, "disk_free_gib": 20}
            first = api.run_pilot(runtime, inputs, temp, monitor=api.ResourceMonitor(temp, gpu_sampler=safe, ram_sampler=safe, disk_sampler=safe))
            self.assertEqual(first["status"], "AWAITING_REVIEW")
            telemetry = ("gpu_samples.csv", "ram_samples.csv", "disk_samples.csv",
                         "sample_resource_snapshots.csv", "sample_resource_snapshots.jsonl")
            telemetry_before = {name: ((Path(temp) / name).read_bytes(), (Path(temp) / name).stat().st_mtime_ns)
                                for name in telemetry}
            status_path = Path(temp) / "run_status.json"; before = (status_path.read_bytes(), status_path.stat().st_mtime_ns)
            runtime.execute = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("complete raw resume must skip generation"))
            resumed_monitor = api.ResourceMonitor(temp, gpu_sampler=safe, ram_sampler=safe, disk_sampler=safe)
            resumed_monitor.start()
            resumed = api.run_pilot(runtime, inputs, temp, resume=True, monitor=resumed_monitor)
            self.assertEqual(resumed, json.loads(before[0].decode()))
            self.assertEqual((status_path.read_bytes(), status_path.stat().st_mtime_ns), before)
            self.assertEqual(telemetry_before, {name: ((Path(temp) / name).read_bytes(), (Path(temp) / name).stat().st_mtime_ns)
                                                for name in telemetry})
            self.assertFalse(resumed_monitor._thread.is_alive())


if __name__ == "__main__": unittest.main()
