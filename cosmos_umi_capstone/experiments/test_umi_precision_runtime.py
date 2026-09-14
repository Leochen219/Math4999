"""Behavioral contracts; fake compute replaces the unavailable local GPU only."""
import copy
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class Native:
    def __init__(self, values, dtype):
        self.values = np.asarray(values, dtype=np.float32).copy()
        self.dtype = dtype

    def detach(self):
        return self

    def float(self):
        return Native(self.values, "float32")

    def cpu(self):
        return self

    def numpy(self):
        return self.values.copy()


class FakeRuntime:
    """One mutable sampler request per call, with observed tensor boundaries."""
    def __init__(self, api, *, fail_full=False, fail_module=False, corrupt=None):
        self.api = api
        self.fail_full, self.fail_module, self.corrupt = fail_full, fail_module, corrupt
        self.calls = []
        self.requests = []
        self.noise = np.arange(12, dtype=np.float32).reshape(1, 1, 3, 2, 2)
        self.provenance = {"framework": "fixture", "weights": "fixed", "seed": 0,
                           "prompt": "fixed", "action": "fixed", "decode": "fixed"}
        self.data_batch = {"image": np.zeros((3, 2, 2), np.uint8), "prompt": "fixed", "action": [1, 2]}
        self.model_state = np.ones(2, np.float32)

    def actual_identity(self):
        return self.api.fingerprint({"data_batch": self.data_batch, "model": self.model_state, "noise": self.noise})

    def execute(self, spec, inputs, *, scope):
        self.calls.append((scope, spec["sample_id"], spec["group"]))
        if spec["group"] == "B" and ((scope == "full" and self.fail_full) or (scope == "module" and self.fail_module)):
            raise self.api.PrecisionCompatibilityError("unsupported FP32 kernel")
        request = {"history": []}
        self.requests.append(request)
        start = copy.deepcopy(request)
        group = spec["group"]
        target = inputs.for_call(group, spec["alpha"], spec["sign"])
        dtype = "bfloat16" if group == "A" else "float32"
        ev = self.api.TensorEvidence()
        ev.record("weights", Native([1], dtype))
        ev.record("common_condition", Native(target, "float32"))
        count = 2 if scope == "full" else 1
        states, conditions = [], []
        noise = self.noise.copy()
        noise[inputs.geometry.mask] = target[inputs.geometry.mask]
        for i in range(count):
            actual = target.copy()
            if self.corrupt == "condition" and i == count - 1:
                actual.flat[0] += 1
            conditions.append(actual)
            ev.record("network_condition", Native(actual, dtype), step=i)
            ev.record("activation", Native(actual + 1, "bfloat16" if self.corrupt == "hidden_cast" else dtype), step=i)
            ev.record("denoiser_output", Native(actual * 2, dtype), step=i)
            if scope == "full":
                ev.record("sampler_velocity", Native(actual * 2, dtype), step=i)
                ev.record("sampler_accumulator", Native(noise, "float32"), step=i)
                ev.record("sampler_converted", Native(noise - actual, "float32"), step=i)
                ev.record("sampler_update", Native(noise + i, "float32"), step=i)
            states.append({"step": i, "timestep": 999 - i})
            request["history"].append(i)
        output = np.broadcast_to(np.mean(target) + self.noise, target.shape).copy()
        record = {"common_input_fp32": target, "condition_steps_fp32": np.stack(conditions),
                  "initial_state": noise, "output_full": output,
                  "consumed_initial_state": noise.copy(), "consumed_initial_mask": inputs.geometry.mask.copy(),
                  "sampler_input_state": noise.copy(),
                  "noise_evidence": {"source": "first_velocity_input", "seed": 0, "prepare_seed": 0, "batch_size": 1},
                  "tensor_evidence": ev.rows, "steps": states, "expected_steps": count,
                  "request_initial_state": start, "request_final_state": copy.deepcopy(request),
                  "cache": {"requested": False, "installed": False, "initial_empty": True, "final_empty": True},
                  "execution": {"autocast": False, "tf32_matmul": False, "tf32_cudnn": False,
                                "backend": "fixture", "casts": ev.casts, "dispatch_observed": True,
                                "operation_count": count},
                  "decode_id": self.provenance["decode"], "scope": scope,
                  "sampler_generator_seeds": [0] * count if scope == "full" else [],
                  "telemetry": {"available": False, "reason": "CPU fixture"}}
        if scope == "full":
            record["decoded_final"] = np.zeros((3, 2, 2), dtype=np.float32)
        if self.corrupt == "noise" and len(self.calls) > 1:
            record["initial_state"][~inputs.geometry.mask] += 1
        if self.corrupt == "state":
            record["request_initial_state"]["history"] = [99]
        if self.corrupt == "autocast":
            record["execution"]["autocast"] = True
        return record


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.api = importlib.import_module("umi_precision_runtime")
        except ModuleNotFoundError:
            cls.api = None

    def setUp(self):
        self.assertIsNotNone(self.api, "precision runtime contract implementation is missing")

    def inputs(self):
        baseline = np.full((1, 1, 3, 2, 2), 1.001, np.float32)
        mask = np.zeros_like(baseline, dtype=bool)
        mask[:, :, 0] = True
        direction = mask.astype(np.float32)
        return self.api.PrecisionInputs(baseline, [0], mask, np.stack([direction, -direction]))

    def test_quantized_ab_exact_and_unquantized_c_preserves_mask(self):
        inputs = self.inputs()
        a = inputs.for_call("A", .0001, 1)
        b = inputs.for_call("B", .0001, 1)
        c = inputs.for_call("C", .0001, 1)
        self.assertEqual(a.dtype, np.float32)
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertTrue(np.all(a == 1))
        self.assertTrue(np.all(c[inputs.geometry.mask] > 1))
        self.assertTrue(np.all(c[~inputs.geometry.mask] == 1))
        self.assertEqual(inputs.for_call("B", 0, 0).tobytes(), inputs.for_call("C", 0, 0).tobytes())
        a[:] = 4
        self.assertTrue(np.all(inputs.z_bar == 1))
        with self.assertRaises(ValueError):
            inputs.z_bar.flat[0] = 5

    def test_native_dtype_survives_fp32_projection(self):
        evidence = self.api.TensorEvidence()
        evidence.record("activation", Native([1.25], "bfloat16"), step=0)
        self.assertEqual(evidence.rows[0]["native_dtype"], "bfloat16")
        self.assertEqual(evidence.rows[0]["projection_dtype"], "float32")

    def test_full_plan_has_exact_group_local_baselines_and_42_calls(self):
        plan = self.api.build_call_plan([.0001, .0003, .001, .003, .01, .03])
        self.assertEqual(len(plan), 42)
        self.assertEqual(len({p["sample_id"] for p in plan}), 42)
        for offset, group in zip((0, 14, 28), "ABC"):
            calls = plan[offset:offset + 14]
            self.assertEqual([c["group"] for c in calls], [group] * 14)
            self.assertEqual([calls[0]["kind"], calls[-1]["kind"]], ["pre", "post"])
            self.assertEqual([c["sign"] for c in calls[1:-1]], [1, -1] * 6)
            self.assertEqual([c["direction_index"] for c in calls[1:-1]], [0] * 12)
        for alphas in ([.1], [.1] * 6, [0, 1, 2, 3, 4, 5]):
            with self.assertRaises(ValueError):
                self.api.build_call_plan(alphas)

    def test_observed_hidden_cast_autocast_state_condition_and_noise_fail(self):
        inputs = self.inputs()
        for corrupt in ("hidden_cast", "autocast", "state", "condition", "noise"):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as temp:
                runtime = FakeRuntime(self.api, corrupt=corrupt)
                result = self.api.run_precision_experiment(runtime, inputs, temp, alphas=[.0001, .0003, .001, .003, .01, .03])
                self.assertNotEqual(result["status"], "complete")
                self.assertEqual(result["formal_successful"], 0)

    def test_full_capture_slicing_pairing_and_strict_resume_no_reruns(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = FakeRuntime(self.api)
            inputs = self.inputs()
            result = self.api.run_precision_experiment(runtime, inputs, temp, alphas=[.0001, .0003, .001, .003, .01, .03])
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["formal_successful"], 42)
            self.assertEqual(result["diagnostic_attempts"], 3)
            self.assertEqual(len(runtime.calls), 45)
            sample = Path(temp) / "samples" / "A_pre"
            selected = np.load(sample / "predicted_latent.npy")
            self.assertEqual(selected.shape, (1, 1, 2, 2, 2))
            self.assertTrue(np.allclose(selected.flatten(), np.arange(4, 12) + 1))
            first_bytes = (sample / "sample.json").read_bytes()
            resumed = self.api.run_precision_experiment(runtime, inputs, temp, alphas=[.0001, .0003, .001, .003, .01, .03], resume=True)
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(len(runtime.calls), 45)
            self.assertEqual(first_bytes, (sample / "sample.json").read_bytes())
            with self.assertRaises(FileExistsError):
                self.api.run_precision_experiment(runtime, inputs, temp, alphas=[.0001, .0003, .001, .003, .01, .03])
            runtime.provenance["action"] = "changed"
            with self.assertRaises(ValueError):
                self.api.run_precision_experiment(runtime, inputs, temp, alphas=[.0001, .0003, .001, .003, .01, .03], resume=True)

    def test_fp32_failure_switches_once_to_step_zero_no_full_formal_or_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = FakeRuntime(self.api, fail_full=True)
            result = self.api.run_precision_experiment(runtime, self.inputs(), temp, alphas=[.0001, .0003, .001, .003, .01, .03])
            self.assertEqual(result["scope"], "module")
            self.assertEqual(result["status"], "module_complete")
            self.assertEqual(result["diagnostic_attempts"], 5)
            self.assertEqual(sum(s == "full" for s, _, _ in runtime.calls), 2)
            self.assertEqual(result["formal_successful"], 42)
            self.assertFalse((Path(temp) / "samples" / "A_pre" / "decoded_final.npy").exists())
            self.assertFalse(result["full_sampling_claim"])
            self.assertFalse(result["image_claim"])
            self.assertIn("unsupported FP32 kernel", json.dumps(result["compatibility_attempts"]))

    def test_general_full_diagnostic_exception_blocks_without_module_fallback(self):
        class GeneralFailure(FakeRuntime):
            def execute(self, spec, inputs, *, scope):
                if scope == "full" and spec["group"] == "B":
                    self.calls.append((scope, spec["sample_id"], spec["group"]))
                    raise RuntimeError("general infrastructure failure")
                return super().execute(spec, inputs, scope=scope)

        with tempfile.TemporaryDirectory() as temp:
            runtime = GeneralFailure(self.api)
            result = self.api.run_precision_experiment(runtime, self.inputs(), temp,
                                                       alphas=[.0001, .0003, .001, .003, .01, .03])
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["scope"], "full")
            self.assertEqual(result["formal_successful"], 0)
            self.assertEqual(sum(scope == "module" for scope, _, _ in runtime.calls), 0)
            self.assertEqual(result["compatibility_attempts"][-1]["error"]["type"], "RuntimeError")

    def test_interrupted_diagnostics_are_bounded_by_six_across_resume(self):
        class AlwaysInterrupted(FakeRuntime):
            def execute(self, spec, inputs, *, scope):
                if spec["kind"] == "diagnostic":
                    self.calls.append((scope, spec["sample_id"], spec["group"]))
                    raise KeyboardInterrupt("simulated interruption")
                return super().execute(spec, inputs, scope=scope)

        with tempfile.TemporaryDirectory() as temp:
            runtime = AlwaysInterrupted(self.api)
            alphas = [.0001, .0003, .001, .003, .01, .03]
            for _ in range(6):
                with self.assertRaises(KeyboardInterrupt):
                    self.api.run_precision_experiment(runtime, self.inputs(), temp, alphas=alphas, resume=Path(temp, "config.json").exists())
            result = self.api.run_precision_experiment(runtime, self.inputs(), temp, alphas=alphas, resume=True)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["diagnostic_attempts"], 6)
            self.assertLessEqual(result["diagnostic_attempts"], 6)
            self.assertEqual(len(runtime.calls), 6)

    def test_failed_fp32_module_is_blocked_without_smaller_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = FakeRuntime(self.api, fail_full=True, fail_module=True)
            result = self.api.run_precision_experiment(runtime, self.inputs(), temp, alphas=[.0001, .0003, .001, .003, .01, .03])
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["formal_successful"], 0)
            self.assertEqual(len(runtime.calls), 4)

    def test_tampered_success_is_rejected_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = FakeRuntime(self.api)
            inputs = self.inputs()
            alphas = [.0001, .0003, .001, .003, .01, .03]
            self.api.run_precision_experiment(runtime, inputs, temp, alphas=alphas)
            file = Path(temp) / "samples" / "A_pre" / "predicted_latent.npy"
            file.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                self.api.run_precision_experiment(runtime, inputs, temp, alphas=alphas, resume=True)
            self.assertEqual(file.read_bytes(), b"corrupt")
            self.assertEqual(len(runtime.calls), 45)

    def test_resume_never_retries_failed_full_compatibility(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = FakeRuntime(self.api, fail_full=True, fail_module=True)
            alphas = [.0001, .0003, .001, .003, .01, .03]
            self.api.run_precision_experiment(runtime, self.inputs(), temp, alphas=alphas)
            runtime.fail_module = False
            result = self.api.run_precision_experiment(runtime, self.inputs(), temp, alphas=alphas, resume=True)
            self.assertEqual(sum(scope == "full" for scope, _, _ in runtime.calls), 2)
            self.assertEqual(result["status"], "blocked")

    def test_full_timestep_schedule_cannot_change_between_groups(self):
        class ChangedSchedule(FakeRuntime):
            def execute(self, spec, inputs, *, scope):
                record = super().execute(spec, inputs, scope=scope)
                if spec["group"] == "B":
                    record["steps"][0]["timestep"] -= 1
                return record
        with tempfile.TemporaryDirectory() as temp:
            result = self.api.run_precision_experiment(ChangedSchedule(self.api), self.inputs(), temp,
                                                       alphas=[.0001, .0003, .001, .003, .01, .03])
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["formal_successful"], 0)

    def test_formal_bc_zero_output_identity_is_checked_after_diagnostics(self):
        class ChangedZero(FakeRuntime):
            def execute(self, spec, inputs, *, scope):
                record = super().execute(spec, inputs, scope=scope)
                if spec["sample_id"] == "C_post":
                    record["output_full"] += 1
                return record
        with tempfile.TemporaryDirectory() as temp:
            result = self.api.run_precision_experiment(ChangedZero(self.api), self.inputs(), temp,
                                                       alphas=[.0001, .0003, .001, .003, .01, .03])
            self.assertEqual(result["status"], "blocked")
            self.assertFalse(result["full_sampling_claim"])

    def test_resume_is_bound_to_actual_runner_source(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "runner.py"
            source.write_text("version one")
            runtime = FakeRuntime(self.api)
            alphas = [.0001, .0003, .001, .003, .01, .03]
            with patch.object(self.api, "__file__", str(source)):
                self.api.run_precision_experiment(runtime, self.inputs(), Path(temp) / "run", alphas=alphas)
                source.write_text("version two")
                with self.assertRaisesRegex(ValueError, "resume"):
                    self.api.run_precision_experiment(runtime, self.inputs(), Path(temp) / "run", alphas=alphas, resume=True)

    def test_interrupted_module_resume_keeps_full_failure_and_diagnostic_accounting(self):
        class Interrupted(FakeRuntime):
            interrupted = False
            def execute(self, spec, inputs, *, scope):
                if scope == "module" and spec["group"] == "B" and not self.interrupted:
                    self.interrupted = True
                    raise KeyboardInterrupt("simulated interruption")
                return super().execute(spec, inputs, scope=scope)
        with tempfile.TemporaryDirectory() as temp:
            runtime = Interrupted(self.api, fail_full=True)
            alphas = [.0001, .0003, .001, .003, .01, .03]
            with self.assertRaises(KeyboardInterrupt):
                self.api.run_precision_experiment(runtime, self.inputs(), temp, alphas=alphas)
            result = self.api.run_precision_experiment(runtime, self.inputs(), temp, alphas=alphas, resume=True)
            self.assertEqual(result["status"], "module_complete")
            self.assertEqual(sum(scope == "full" for scope, _, _ in runtime.calls), 2)
            self.assertEqual(result["diagnostic_attempts"], 6)
            self.assertEqual(sum(a["scope"] == "full" and a["status"] == "fail" for a in result["compatibility_attempts"]), 1)

    def test_failed_evidence_keeps_actual_tensors_and_dtype_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            self.api.run_precision_experiment(FakeRuntime(self.api, corrupt="condition"), self.inputs(), temp,
                                              alphas=[.0001, .0003, .001, .003, .01, .03])
            sample = Path(temp) / "diagnostics" / "full_A_zero"
            payload = json.loads((sample / "sample.json").read_text())
            self.assertIn("partial_capture", payload)
            self.assertTrue(payload["partial_capture"]["tensor_evidence"])
            self.assertTrue(list(sample.glob("*condition_steps_fp32.npy")))
            self.assertIn("EvidenceError", payload["traceback"])

    def test_sampler_only_precision_incompatibility_uses_denoiser_fallback(self):
        class LowPrecisionSampler(FakeRuntime):
            def execute(self, spec, inputs, *, scope):
                result = super().execute(spec, inputs, scope=scope)
                if scope == "full" and spec["group"] == "B":
                    for row in result["tensor_evidence"]:
                        if row["role"] == "sampler_update":
                            row["native_dtype"] = "bfloat16"
                return result
        with tempfile.TemporaryDirectory() as temp:
            runtime = LowPrecisionSampler(self.api)
            result = self.api.run_precision_experiment(runtime, self.inputs(), temp,
                                                       alphas=[.0001, .0003, .001, .003, .01, .03])
            self.assertEqual(result["status"], "module_complete")
            self.assertFalse(result["image_claim"])
            self.assertEqual(sum(scope == "full" for scope, _, _ in runtime.calls), 2)

    def test_actual_data_model_and_noise_resume_mismatches_ignore_caller_lies(self):
        for field in ("image", "action", "prompt", "model", "noise"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                runtime = FakeRuntime(self.api)
                alphas = [.0001, .0003, .001, .003, .01, .03]
                self.api.run_precision_experiment(runtime, self.inputs(), temp, alphas=alphas)
                if field in runtime.data_batch:
                    runtime.data_batch[field] = "actual changed"
                elif field == "model":
                    runtime.model_state[0] += 1
                else:
                    runtime.noise.flat[-1] += 1
                with self.assertRaisesRegex(ValueError, "resume"):
                    self.api.run_precision_experiment(runtime, self.inputs(), temp, alphas=alphas, resume=True)

    def test_runner_recovers_from_real_store_prepublication_interruption(self):
        import umi_precision_storage as storage
        with tempfile.TemporaryDirectory() as temp:
            runtime = FakeRuntime(self.api)
            alphas = [.0001, .0003, .001, .003, .01, .03]
            with patch.object(storage, "publish_directory", side_effect=KeyboardInterrupt("publish interrupted")):
                with self.assertRaises(KeyboardInterrupt):
                    self.api.run_precision_experiment(runtime, self.inputs(), temp, alphas=alphas)
            self.assertFalse((Path(temp) / "diagnostics" / "full_A_zero").exists())
            result = self.api.run_precision_experiment(runtime, self.inputs(), temp, alphas=alphas, resume=True)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["formal_successful"], 42)
            self.assertEqual(result["diagnostic_attempts"], 4)
            self.assertEqual(len(runtime.calls), 46)

    def test_missing_real_sampler_conversion_or_generator_evidence_cannot_pass(self):
        inputs = self.inputs()
        spec = {"sample_id": "B", "group": "B", "alpha": 0., "sign": 0}
        for missing in ("sampler_converted", "generator"):
            with self.subTest(missing=missing):
                record = FakeRuntime(self.api).execute(spec, inputs, scope="full")
                if missing == "generator":
                    record["sampler_generator_seeds"] = [1, 1]
                else:
                    record["tensor_evidence"] = [r for r in record["tensor_evidence"] if r["role"] != missing]
                with self.assertRaises(self.api.EvidenceError):
                    self.api.validate_capture(record, spec, inputs, "full")

    def test_bc_full_fp32_contract_rejects_float64_compute_cast(self):
        inputs = self.inputs()
        spec = {"sample_id": "B", "group": "B", "alpha": 0., "sign": 0}
        record = FakeRuntime(self.api).execute(spec, inputs, scope="full")
        record["execution"]["casts"] = [{"from": "float32", "to": "float64"}]
        with self.assertRaises(self.api.PrecisionCompatibilityError):
            self.api.validate_capture(record, spec, inputs, "full")


if __name__ == "__main__":
    unittest.main()
