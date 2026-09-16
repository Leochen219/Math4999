import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np


class Task6RunnerTests(unittest.TestCase):
    def test_empty_preflight_is_blocked(self):
        import run_umi_task6_experiment as api
        with self.assertRaises(api.BlockedExecution):
            api.execute_task6(api.parse_args([]))

    def test_smoke_stops_at_review_and_state_machine_requires_explicit_phase(self):
        import run_umi_task6_experiment as api
        args = api.parse_args([])
        self.assertEqual(args.phase, "preflight")
        with tempfile.TemporaryDirectory() as temp:
            class Runtime:
                def __init__(self): self.calls = 0
                def execute(self, spec, inputs, *, scope): self.calls += 1; return {"output_full": np.zeros(8, np.float32)}
            samplers = {key: (lambda: {"gpu_used_gib": 0, "gpu_free_gib": 100,
                "ram_available_gib": 600, "disk_free_gib": 20}) for key in ("gpu", "ram", "disk")}
            result = api.run_resource_smoke(temp, runtime=Runtime(), inputs=object(), samplers=samplers,
                                            lifecycle={"pre_load": lambda: None, "load": lambda: None,
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
            monitor = api.ResourceMonitor(temp, gpu_sampler=lambda: (_ for _ in ()).throw(RuntimeError("poll")))
            monitor.start(); time.sleep(0.03)
            with self.assertRaises(RuntimeError): monitor.stop()
            self.assertTrue(Path(temp, "gpu_samples.csv").is_file())
            self.assertTrue(Path(temp, "disk_samples.csv").is_file())

    def test_pilot_requires_accepted_smoke_decision(self):
        import run_umi_task6_experiment as api
        with tempfile.TemporaryDirectory() as temp:
            runtime = type("Runtime", (), {"execute": lambda self, spec, inputs, *, scope: {"output_full": np.zeros(8, np.float32)}})()
            samplers = {key: (lambda: {"gpu_used_gib": 0, "gpu_free_gib": 100, "ram_available_gib": 600, "disk_free_gib": 20}) for key in ("gpu", "ram", "disk")}
            api.run_resource_smoke(temp, runtime=runtime, inputs=object(), samplers=samplers,
                                   lifecycle={"pre_load": lambda: None, "load": lambda: None,
                                              "cleanup": lambda: None, "unload": lambda: None})
            with self.assertRaises(api.BlockedExecution):
                api.authorize_pilot(temp)


if __name__ == "__main__": unittest.main()
