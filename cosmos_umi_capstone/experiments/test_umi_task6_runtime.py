import tempfile
import unittest
from pathlib import Path

import numpy as np


class Task6RuntimeTests(unittest.TestCase):
    def setUp(self):
        import umi_task6_runtime as api
        self.api = api

    def fixture(self):
        carrier = np.ones((1, 1, 3, 2, 2), dtype=np.float32)
        mask = np.zeros_like(carrier, dtype=bool); mask[:, :, 0] = True
        bank = np.zeros((3,) + carrier.shape, dtype=np.float32)
        bank[0, :, :, 0] = 1.0
        bank[1, :, :, 0, 0] = np.sqrt(2.0)
        bank[2, :, :, 0, 1] = np.sqrt(2.0)
        return carrier, [0], mask, bank

    def test_preflight_rejects_geometry_before_execute(self):
        carrier, indexes, mask, bank = self.fixture()
        with self.assertRaises(self.api.PreflightError):
            self.api.preflight_task6({"carrier": carrier, "condition_indexes": indexes,
                                      "mask": mask, "direction_bank": bank,
                                      "settings": {"autocast": True}})

    def test_direction_bank_is_frozen_and_extra_direction_rejected(self):
        carrier, indexes, mask, bank = self.fixture()
        with self.assertRaises(ValueError):
            self.api.load_frozen_directions(np.concatenate([bank, bank[:1]]), mask)
        frozen = self.api.load_frozen_directions(bank, mask)
        self.assertEqual(tuple(frozen), ("v0", "v1", "v2", "u01", "u12"))

    def test_runner_executes_exact_plan_and_resume_is_immutable(self):
        carrier, indexes, mask, bank = self.fixture()
        inputs = self.api.Task6Inputs(carrier, indexes, mask, bank,
                                      z_bar=carrier, task4_reference={"fixture": True},
                                      action=np.zeros((16, 10), np.float32), prompt="p")
        class FakeRuntime:
            provenance = {"seed": 0, "diffusion_cache": "off"}
            def __init__(self): self.calls = []
            def actual_identity(self): return {"fixture": "runtime"}
            def execute(self, spec, inputs, *, scope="full"):
                self.calls.append(spec["sample_id"])
                return {"output_full": inputs.for_spec(spec), "spec": dict(spec),
                        "scope": scope, "raw": np.array([len(self.calls)], np.float32)}
        with tempfile.TemporaryDirectory() as temp:
            runtime = FakeRuntime()
            result = self.api.run_task6_group(runtime, inputs, temp)
            self.assertEqual(result["status"], "AWAITING_REVIEW")
            self.assertEqual(result["successful_samples"], 32)
            self.assertEqual(len(runtime.calls), 32)
            self.assertEqual(self.api.run_task6_group(runtime, inputs, temp, resume=True)["successful_samples"], 32)
            self.assertEqual(len(runtime.calls), 32)

    def test_equivalence_requires_four_calls_and_reports_reuse(self):
        class Runtime:
            def __init__(self, equal=True): self.calls = 0; self.equal = equal
            def execute(self, spec, inputs, *, scope="equivalence"):
                self.calls += 1
                return {"raw": np.array([1 if self.equal else self.calls], dtype=np.float32)}
        result = self.api.verify_reference_reuse(Runtime(), None, execute=True)
        self.assertTrue(result["reusable"]); self.assertEqual(result["calls"], 4)

    def test_resource_stop_preserves_completed_samples_and_resume_numbers_incomplete_attempt(self):
        carrier, indexes, mask, bank = self.fixture()
        inputs = self.api.Task6Inputs(carrier, indexes, mask, bank, z_bar=carrier,
                                      action=np.zeros((16, 10), np.float32))
        class Runtime:
            provenance = {"seed": 0}
            def __init__(self): self.calls = []
            def actual_identity(self): return {"fixture": "stop"}
            def execute(self, spec, inputs, *, scope="full"):
                self.calls.append(spec["sample_id"])
                return {"output_full": inputs.for_spec(spec)}
        with tempfile.TemporaryDirectory() as temp:
            runtime = Runtime()
            stopped = self.api.run_task6_group(runtime, inputs, temp, stop_after=2)
            self.assertEqual(stopped["status"], "RESOURCE_STOP")
            resumed = self.api.run_task6_group(runtime, inputs, temp, resume=True)
            self.assertEqual(resumed["successful_samples"], 32)
            self.assertEqual(len(runtime.calls), 32)

    def test_resume_refuses_tampered_success(self):
        carrier, indexes, mask, bank = self.fixture()
        inputs = self.api.Task6Inputs(carrier, indexes, mask, bank, z_bar=carrier)
        class Runtime:
            provenance = {"seed": 0}
            def actual_identity(self): return {"fixture": "tamper"}
            def execute(self, spec, inputs, *, scope="full"): return {"output_full": inputs.for_spec(spec)}
        with tempfile.TemporaryDirectory() as temp:
            runtime = Runtime(); self.api.run_task6_group(runtime, inputs, temp)
            sample = Path(temp, "samples", "bridge_0__seed_0__baseline_pre", "sample.json")
            sample.write_text(sample.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
            with self.assertRaises(ValueError): self.api.run_task6_group(runtime, inputs, temp, resume=True)


if __name__ == "__main__": unittest.main()
