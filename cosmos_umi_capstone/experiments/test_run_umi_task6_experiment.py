import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np


class Task6RunnerTests(unittest.TestCase):
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


if __name__ == "__main__": unittest.main()
