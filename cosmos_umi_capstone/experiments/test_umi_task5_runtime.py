"""CPU contracts for the bounded Task 5 full-generation runner."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class Task5RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import umi_task5_runtime as api
        except ModuleNotFoundError:
            api = None
        cls.api = api

    def setUp(self):
        self.assertIsNotNone(self.api, "Task 5 runtime implementation is missing")

    def inputs(self):
        carrier = np.full((1, 1, 3, 2, 2), 1.25, dtype=np.float32)
        mask = np.zeros_like(carrier, dtype=bool)
        mask[:, :, 0] = True
        bank = np.zeros((3,) + carrier.shape, dtype=np.float32)
        # Deliberately non-orthogonal but non-degenerate frozen directions.
        bank[0, :, :, 0] = 1.0
        bank[1, :, :, 0, 0] = np.sqrt(2.0)
        bank[2, :, :, 0, 1] = np.sqrt(2.0)
        return self.api.Task5Inputs(carrier, [0], mask, bank, z_bar=carrier.copy(),
                                    task4_reference={"run": "fixture", "c_pre": "fixture"})

    def test_inputs_reuse_unmodified_original_directions_and_apply_alpha_times_scale(self):
        inputs = self.inputs()
        self.assertEqual(inputs.direction_for_spec({"kind": "perturbation", "direction_id": "v0"}).tobytes(),
                         inputs.directions["v0"].tobytes())
        spec = {"kind": "perturbation", "direction_id": "v1", "alpha": 1e-3, "sign": 1, "group": "C"}
        target = inputs.for_spec(spec)
        delta = target - inputs.z_bar
        self.assertTrue(np.array_equal(delta[~inputs.geometry.mask], np.zeros_like(delta[~inputs.geometry.mask])))
        self.assertAlmostEqual(float(np.sqrt(np.mean(delta[inputs.geometry.mask].astype(np.float64) ** 2))),
                               1e-3 * inputs.s_z, places=7)
        self.assertEqual(inputs.task5_plan()["formal_call_count"], 32)

    def test_runner_writes_32_immutable_full_records_and_resumes_without_reexecution(self):
        api = self.api

        class Runtime:
            provenance = {"framework": "fixture", "decode": "fixture", "seed": 0}

            def __init__(self):
                self.calls = []

            def actual_identity(self):
                return {"model": "fixture", "noise": "fixed"}

            def execute(self, spec, inputs, *, scope):
                self.calls.append(spec["sample_id"])
                self.assert_scope = scope
                target = inputs.for_spec(spec)
                mask = inputs.geometry.mask
                evidence = api.TensorEvidence()
                evidence.record("weights", api.NativeFixture(np.array([1.0], np.float32), "float32"))
                evidence.record("common_condition", api.NativeFixture(target, "float32"))
                rows, conditions, steps = evidence.rows, [], []
                initial = np.arange(target.size, dtype=np.float32).reshape(target.shape)
                initial[mask] = target[mask]
                for step in range(2):
                    condition = target.copy()
                    conditions.append(condition)
                    for role, value in (("network_condition", condition), ("activation", condition + 1),
                                        ("denoiser_output", condition * 2), ("sampler_velocity", condition * 2),
                                        ("sampler_accumulator", initial), ("sampler_converted", initial),
                                        ("sampler_update", initial)):
                        evidence.record(role, api.NativeFixture(value, "float32"), step=step)
                    steps.append({"step": step, "timestep": float(999 - step)})
                return {"common_input_fp32": target, "condition_steps_fp32": np.stack(conditions),
                        "initial_state": initial, "consumed_initial_state": initial.copy(),
                        "sampler_input_state": initial.copy(), "consumed_initial_mask": mask.copy(),
                        "noise_evidence": {"source": "first_velocity_input", "seed": 0, "prepare_seed": 0, "batch_size": 1},
                        "output_full": initial + target.mean(), "decoded_final": np.zeros((3, 2, 2), np.float32),
                        "tensor_evidence": rows, "steps": steps, "expected_steps": 2,
                        "request_initial_state": {}, "request_final_state": {},
                        "cache": {"requested": False, "installed": False, "initial_empty": True, "final_empty": True},
                        "execution": {"autocast": False, "tf32_matmul": False, "tf32_cudnn": False,
                                      "backend": "fixture", "casts": [], "dispatch_observed": True, "operation_count": 2},
                        "decode_id": "fixture", "scope": scope, "sampler_generator_seeds": [0, 0],
                        "telemetry": {"available": False}}

        with tempfile.TemporaryDirectory() as temp:
            runtime, inputs = Runtime(), self.inputs()
            result = api.run_task5_experiment(runtime, inputs, temp)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["formal_successful"], 32)
            self.assertEqual(len(runtime.calls), 32)
            self.assertTrue(Path(temp, "task5_plan.json").is_file())
            self.assertTrue(Path(temp, "samples", "u12_alpha_02_minus", "output_full.npy").is_file())
            resumed = api.run_task5_experiment(runtime, inputs, temp, resume=True)
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(len(runtime.calls), 32)


if __name__ == "__main__":
    unittest.main()
