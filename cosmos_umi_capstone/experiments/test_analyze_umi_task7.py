"""Pure NumPy contracts for the bounded Task7 numerical API."""
from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

import analyze_umi_task7 as api
import run_umi_task7_experiment as runner


class Task7NumericalTests(unittest.TestCase):
    def test_fp32_difference_float64_reductions_and_signed_zero_bytes(self):
        left = np.array([1.0, 3.0, 0.0], dtype=np.float32)
        right = np.array([0.5, 1.0, 0.0], dtype=np.float32)
        difference = api.fp32_difference(left, right)
        self.assertEqual(difference.dtype, np.dtype(np.float32))
        self.assertTrue(np.allclose(difference, [0.5, 2.0, 0.0]))
        self.assertAlmostEqual(api.rms64(difference), np.sqrt(4.25 / 3.0))
        self.assertAlmostEqual(api.cosine64(difference, difference), 1.0)
        self.assertFalse(api.byte_equal(np.array([0.0], np.float32), np.array([-0.0], np.float32)))

    def test_unmasked_cosine_reduces_full_rank_carrier(self):
        left = np.zeros((1, 1, 2, 2, 2), dtype=np.float32)
        left[..., 0, 0, 0] = 1.0
        right = left.copy()
        self.assertAlmostEqual(api.cosine64(left, right), 1.0)

    def test_input_geometry_uses_actual_lengths_and_exact_exterior(self):
        mask = np.array([True, True, False])
        direction = np.array([np.sqrt(2.0), 0.0, 0.0], dtype=np.float32)
        plus = np.array([0.2, 0.0, 0.0], dtype=np.float32)
        minus = np.array([-0.2, 0.0, 0.0], dtype=np.float32)
        geometry = api.input_geometry(plus, minus, direction, mask)
        self.assertAlmostEqual(geometry["h_plus"], 0.2 / np.sqrt(2.0), places=7)
        self.assertAlmostEqual(geometry["h_minus"], 0.2 / np.sqrt(2.0), places=7)
        self.assertGreaterEqual(geometry["plus_direction_cosine"], 0.99)
        self.assertGreaterEqual(geometry["minus_direction_cosine"], 0.99)
        self.assertLessEqual(geometry["opposite_cosine"], -0.99)
        self.assertTrue(geometry["plus_outside_exact"])
        self.assertTrue(geometry["minus_outside_exact"])

    def test_affine_window_passes_with_three_prespecified_points(self):
        mask = np.array([True, True], dtype=bool)
        direction = np.array([np.sqrt(2.0), 0.0], dtype=np.float32)
        baseline = np.array([4.0, -3.0], dtype=np.float32)
        amplitudes = (0.001, 0.003, 0.01)
        plus_inputs, minus_inputs, plus_outputs, minus_outputs = [], [], [], []
        for alpha in amplitudes:
            h = alpha
            plus_inputs.append((h * direction).astype(np.float32))
            minus_inputs.append((-h * direction).astype(np.float32))
            response = np.array([2.0 * h, h], dtype=np.float32)
            plus_outputs.append((baseline + response).astype(np.float32))
            minus_outputs.append((baseline - response).astype(np.float32))
        result = api.one_direction_window(
            amplitudes,
            direction,
            plus_inputs,
            minus_inputs,
            plus_outputs,
            minus_outputs,
            baseline,
            mask,
            floor=0.0,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["point_count"], 3)
        self.assertAlmostEqual(result["plus_fit"]["slope"], 1.0, places=4)
        self.assertAlmostEqual(result["minus_fit"]["slope"], 1.0, places=4)
        self.assertEqual(result["additivity_status"], "NOT_TESTED")

    def test_non_linear_window_fails_without_dropping_smallest_point(self):
        mask = np.array([True, True], dtype=bool)
        direction = np.array([np.sqrt(2.0), 0.0], dtype=np.float32)
        baseline = np.zeros(2, dtype=np.float32)
        amplitudes = (0.001, 0.003, 0.01)
        plus_inputs = [(a * direction).astype(np.float32) for a in amplitudes]
        minus_inputs = [(-a * direction).astype(np.float32) for a in amplitudes]
        plus_outputs = [np.array([a * a, 0.0], dtype=np.float32) for a in amplitudes]
        minus_outputs = [np.array([-a * a, 0.0], dtype=np.float32) for a in amplitudes]
        result = api.one_direction_window(
            amplitudes, direction, plus_inputs, minus_inputs, plus_outputs, minus_outputs,
            baseline, mask, floor=0.0,
        )
        self.assertEqual(result["point_count"], 3)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("slope" in reason for reason in result["reasons"]))

    def test_high_floor_and_rounding_disappearance_are_not_scientific_passes(self):
        mask = np.array([True, True], dtype=bool)
        direction = np.array([np.sqrt(2.0), 0.0], dtype=np.float32)
        baseline = np.zeros(2, dtype=np.float32)
        amplitudes = (0.001, 0.003, 0.01)
        plus_inputs = [(a * direction).astype(np.float32) for a in amplitudes]
        minus_inputs = [(-a * direction).astype(np.float32) for a in amplitudes]
        plus_outputs = [np.array([5.0, 0.0], dtype=np.float32) for _ in amplitudes]
        minus_outputs = [np.array([-5.0, 0.0], dtype=np.float32) for _ in amplitudes]
        high_floor = api.one_direction_window(
            amplitudes, direction, plus_inputs, minus_inputs, plus_outputs, minus_outputs,
            baseline, mask, floor=1.0,
        )
        self.assertEqual(high_floor["status"], "FAIL")
        self.assertTrue(any("floor" in reason for reason in high_floor["reasons"]))
        disappearing = api.fp32_difference(
            np.array([1.0e8], dtype=np.float32), np.array([1.0e8 + 1.0], dtype=np.float32)
        )
        self.assertEqual(float(disappearing[0]), 0.0)

    def test_propagation_oracle_and_zero_denominators(self):
        mask = np.array([True, True], dtype=bool)
        delta0 = np.array([1.0, 0.0], dtype=np.float32)
        delta1 = np.array([2.0, 1.0], dtype=np.float32)
        delta2 = np.array([1.0, 5.0], dtype=np.float32)
        result = api.propagation_metrics(delta0, delta1, delta2, mask)
        self.assertAlmostEqual(result["A1"], np.sqrt(5.0), places=6)
        self.assertAlmostEqual(result["A2"], np.sqrt(26.0), places=6)
        self.assertAlmostEqual(result["incremental"], np.sqrt(26.0 / 5.0), places=6)
        zero = api.propagation_metrics(np.zeros(2, np.float32), delta1, delta2, mask)
        self.assertIsNone(zero["A1"])
        self.assertIsNone(zero["A2"])
        self.assertIsNone(zero["incremental"])
        self.assertTrue(zero["reasons"])

    def test_current_q_is_denominator_for_adjacent_change(self):
        mask = np.array([True, True], dtype=bool)
        direction = np.array([np.sqrt(2.0), 0.0], dtype=np.float32)
        baseline = np.zeros(2, dtype=np.float32)
        amplitudes = (0.001, 0.003, 0.01)
        plus_inputs, minus_inputs, plus_outputs, minus_outputs = [], [], [], []
        for index, alpha in enumerate(amplitudes):
            h = alpha
            plus_inputs.append((h * direction).astype(np.float32))
            minus_inputs.append((-h * direction).astype(np.float32))
            q = 1.0 if index == 0 else 1.2
            plus_outputs.append(np.array([q * h, 0.0], dtype=np.float32))
            minus_outputs.append(np.array([-q * h, 0.0], dtype=np.float32))
        result = api.one_direction_window(
            amplitudes, direction, plus_inputs, minus_inputs, plus_outputs, minus_outputs,
            baseline, mask, floor=0.0,
        )
        self.assertAlmostEqual(result["secants"][0]["relative_change"], 0.2, places=5)

    def test_asymmetric_actual_lengths_use_their_sum_for_central_q(self):
        mask = np.array([True, True], dtype=bool)
        direction = np.array([np.sqrt(2.0), 0.0], dtype=np.float32)
        baseline = np.zeros(2, dtype=np.float32)
        amplitudes = (0.001, 0.003, 0.01)
        plus_inputs, minus_inputs, plus_outputs, minus_outputs = [], [], [], []
        for alpha in amplitudes:
            plus = (alpha * direction).astype(np.float32)
            minus = (-2.0 * alpha * direction).astype(np.float32)
            plus_inputs.append(plus)
            minus_inputs.append(minus)
            h_plus, h_minus = api.rms64(plus, mask), api.rms64(minus, mask)
            plus_outputs.append(np.array([h_plus, 0.0], dtype=np.float32))
            minus_outputs.append(np.array([-h_minus, 0.0], dtype=np.float32))
        result = api.one_direction_window(
            amplitudes, direction, plus_inputs, minus_inputs, plus_outputs, minus_outputs,
            baseline, mask, floor=0.0,
        )
        self.assertAlmostEqual(result["points"][0]["h_minus"], 2.0 * result["points"][0]["h_plus"], places=7)
        self.assertAlmostEqual(float(result["points"][0]["central_q"][0]), 1.0, places=5)

    def test_fixed_beta_prediction_uses_actual_second_step_ray_and_gate(self):
        mask = np.array([True, True], dtype=bool)
        delta1 = np.array([2.0, 1.0], dtype=np.float32)
        delta2 = np.array([1.0, 5.0], dtype=np.float32)
        ray = delta1 / api.rms64(delta1)
        baseline = np.array([10.0, -4.0], dtype=np.float32)
        plus_input = (0.1 * delta1).astype(np.float32)
        minus_input = (-0.1 * delta1).astype(np.float32)
        # M1 @ ray = (1, 5) * sqrt(2/5), so q * RMS(delta1) = (1, 5).
        q_ray = np.array([1.0, 5.0], dtype=np.float64) * np.sqrt(2.0 / 5.0)
        plus_output = (baseline + (0.1 * api.rms64(delta1) * q_ray)).astype(np.float32)
        minus_output = (baseline - (0.1 * api.rms64(delta1) * q_ray)).astype(np.float32)
        result = api.fixed_beta_prediction(
            actual_delta1=delta1,
            beta_plus_input=plus_input,
            beta_minus_input=minus_input,
            beta_plus_output=plus_output,
            beta_minus_output=minus_output,
            baseline_output=baseline,
            actual_delta2=delta2,
            mask=mask,
            beta=0.1,
            local_window_pass=True,
            response_reliable=True,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertAlmostEqual(result["Eprop"], 0.0, places=5)
        self.assertTrue(np.allclose(result["predicted_delta2"], delta2, atol=2e-6))

    def test_fixed_beta_rejects_tempting_beta_and_unreliable_flags(self):
        mask = np.array([True, True], dtype=bool)
        kwargs = dict(
            actual_delta1=np.array([2.0, 1.0], dtype=np.float32),
            beta_plus_input=np.array([0.2, 0.1], dtype=np.float32),
            beta_minus_input=np.array([-0.2, -0.1], dtype=np.float32),
            beta_plus_output=np.array([1.0, 1.0], dtype=np.float32),
            beta_minus_output=np.array([-1.0, -1.0], dtype=np.float32),
            baseline_output=np.zeros(2, dtype=np.float32),
            actual_delta2=np.array([1.0, 1.0], dtype=np.float32),
            mask=mask,
        )
        with self.assertRaises(api.EngineeringDataError):
            api.fixed_beta_prediction(**kwargs, beta=0.2)
        result = api.fixed_beta_prediction(**kwargs)
        self.assertEqual(result["status"], "UNRELIABLE")

    def test_fixed_beta_zero_evaluation_denominator_is_explicitly_unreliable(self):
        result = api.fixed_beta_prediction(
            actual_delta1=np.array([2.0, 1.0], dtype=np.float32),
            beta_plus_input=np.array([0.2, 0.1], dtype=np.float32),
            beta_minus_input=np.array([-0.2, -0.1], dtype=np.float32),
            beta_plus_output=np.array([1.0, 1.0], dtype=np.float32),
            beta_minus_output=np.array([-1.0, -1.0], dtype=np.float32),
            baseline_output=np.zeros(2, dtype=np.float32),
            actual_delta2=np.zeros(2, dtype=np.float32),
            mask=np.array([True, True]),
            local_window_pass=True,
            response_reliable=True,
        )
        self.assertIsNone(result["Eprop"])
        self.assertFalse(result["reliable"])
        self.assertTrue(any("zero evaluation denominator" in reason for reason in result["reasons"]))

    def test_six_prediction_rows_are_retained(self):
        rows = api.evaluate_fixed_beta_predictions(
            [
                {"ray": np.array([1.0, 0.0], np.float32), "actual_delta2": np.array([1.0, 0.0], np.float32)},
                {"ray": np.array([0.0, 1.0], np.float32), "actual_delta2": np.array([0.0, 1.0], np.float32)},
                {"ray": np.array([1.0, 1.0], np.float32), "actual_delta2": np.array([1.0, 1.0], np.float32)},
                {"ray": np.array([1.0, -1.0], np.float32), "actual_delta2": np.array([1.0, -1.0], np.float32)},
                {"ray": np.array([2.0, 1.0], np.float32), "actual_delta2": np.array([2.0, 1.0], np.float32)},
                {"ray": np.array([-1.0, 2.0], np.float32), "actual_delta2": np.array([-1.0, 2.0], np.float32)},
            ],
            q_by_ray=[
                np.array([1.0, 0.0], np.float32) / api.rms64(np.array([1.0, 0.0], np.float32)),
                np.array([0.0, 1.0], np.float32) / api.rms64(np.array([0.0, 1.0], np.float32)),
                np.array([1.0, 1.0], np.float32) / api.rms64(np.array([1.0, 1.0], np.float32)),
                np.array([1.0, -1.0], np.float32) / api.rms64(np.array([1.0, -1.0], np.float32)),
                np.array([2.0, 1.0], np.float32) / api.rms64(np.array([2.0, 1.0], np.float32)),
                np.array([-1.0, 2.0], np.float32) / api.rms64(np.array([-1.0, 2.0], np.float32)),
            ],
            mask=np.array([True, True]),
            local_window_pass=[True, False, True, True, True, True],
            response_reliable=[True, True, True, False, True, True],
        )
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["ray_index"] for row in rows}, set(range(6)))
        self.assertEqual(rows[0]["status"], "PASS")
        self.assertEqual(rows[1]["status"], "UNRELIABLE")
        self.assertEqual(rows[3]["status"], "UNRELIABLE")

    def test_six_ray_zero_ray_or_zero_evaluation_is_not_reliable_and_flags_are_boolean(self):
        rays = [
            {"ray": np.array([1.0, 0.0], np.float32), "actual_delta2": np.array([1.0, 0.0], np.float32)},
            {"ray": np.array([0.0, 1.0], np.float32), "actual_delta2": np.array([0.0, 1.0], np.float32)},
            {"ray": np.array([1.0, 1.0], np.float32), "actual_delta2": np.array([1.0, 1.0], np.float32)},
            {"ray": np.array([1.0, -1.0], np.float32), "actual_delta2": np.array([1.0, -1.0], np.float32)},
            {"ray": np.array([2.0, 1.0], np.float32), "actual_delta2": np.array([2.0, 1.0], np.float32)},
            {"ray": np.zeros(2, np.float32), "actual_delta2": np.zeros(2, np.float32)},
        ]
        q_values = [np.ones(2, np.float32) for _ in rays]
        with self.assertRaises(api.EngineeringDataError):
            api.evaluate_fixed_beta_predictions(
                rays, q_by_ray=q_values, mask=np.array([True, True]),
                local_window_pass=[True, True, True, True, True, "false"],
                response_reliable=True,
            )
        rays[5]["local_window_pass"] = False
        rows = api.evaluate_fixed_beta_predictions(
            rays, q_by_ray=q_values, mask=np.array([True, True]),
            local_window_pass=True, response_reliable=True,
        )
        self.assertFalse(rows[5]["reliable"])
        self.assertTrue(any("zero ray" in reason for reason in rows[5]["reasons"]))
        self.assertTrue(any("zero evaluation denominator" in reason for reason in rows[5]["reasons"]))

    def test_reliable_is_separate_from_eprop_pass_fail(self):
        result = api.fixed_beta_prediction(
            actual_delta1=np.array([2.0, 1.0], dtype=np.float32),
            beta_plus_input=np.array([0.2, 0.1], dtype=np.float32),
            beta_minus_input=np.array([-0.2, -0.1], dtype=np.float32),
            beta_plus_output=np.array([2.0, 2.0], dtype=np.float32),
            beta_minus_output=np.array([-2.0, -2.0], dtype=np.float32),
            baseline_output=np.zeros(2, dtype=np.float32),
            actual_delta2=np.array([1.0, 5.0], dtype=np.float32),
            mask=np.array([True, True]),
            local_window_pass=True,
            response_reliable=True,
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(result["reliable"])

    def test_invalid_shape_dtype_nonfinite_and_mask_are_engineering_errors(self):
        with self.assertRaises(api.EngineeringDataError):
            api.fp32_difference(np.ones(2, np.float64), np.ones(2, np.float32))
        with self.assertRaises(api.EngineeringDataError):
            api.rms64(np.ones(2, np.float32), mask=np.array([True]))
        with self.assertRaises(api.EngineeringDataError):
            api.rms64(np.array([np.nan], np.float32))
        with self.assertRaises(api.EngineeringDataError):
            api.rms64(np.ones(2, np.float32), mask=np.array([False, False]))


class Task7SavedEvidenceAOnlyTests(unittest.TestCase):
    """Synthetic runner-schema evidence is never presented as model output."""

    @staticmethod
    def _source_fixture():
        names = ("baseline_pre", "v0_alpha_00_plus", "v0_alpha_00_minus",
                 "v0_alpha_01_plus", "v0_alpha_01_minus", "v0_alpha_02_plus",
                 "v0_alpha_02_minus", "baseline_post")
        direction = np.array([np.sqrt(2.0), 0.0], dtype=np.float32)
        baseline = np.array([1.0, -1.0], dtype=np.float32)
        frame = np.zeros((3, 2, 1), dtype=np.float32)
        rows = {}
        for name in names:
            is_baseline = name.startswith("baseline")
            alpha = 0.0 if is_baseline else (0.001 if "00" in name else (0.003 if "01" in name else 0.01))
            sign = 0 if is_baseline else (1 if name.endswith("plus") else -1)
            delta = np.zeros(2, dtype=np.float32) if is_baseline else (np.float32(sign * alpha) * direction)
            rows[name] = {"name": name, "direct": baseline.copy(), "delta_condition": delta,
                          "consumed_condition": (baseline + delta).astype(np.float32),
                          "z0_condition": baseline.copy(), "condition_mask": np.ones(2, bool),
                          "mask": np.ones(2, bool), "raw": Path("raw"), "decoder": Path("decoder"), "frame": frame}
        rows["baseline_post"]["direct"] = (baseline + np.float32(1e-7)).astype(np.float32)
        return {"raw_root": "raw", "decoder_root": "decoder", "source_tree_sha256": "fixture-source",
                "z0_sha256": "z0", "mask_sha256": "mask", "v0_sha256": "v0", "rows": rows,
                "v0_condition": direction, "z0_condition": baseline,
                "condition_mask": np.ones(2, bool), "mask": np.ones(2, bool)}

    @staticmethod
    def _write_a_stage(run: Path, source, *, native_parity=True, native_science_fail=False,
                       bad_fp32_evidence=False, empty_buffers=False):
        stage = run / "stages" / "A"; store = runner.Task7SampleStore(stage / "samples")
        plan = runner.build_stage_plan("A")
        baseline = np.array([1.0, -1.0], dtype=np.float32)
        for spec in plan:
            name, precision = spec["source_name"], spec["precision"]
            row = source["rows"][name]
            if name == "baseline_post":
                output = (baseline + np.float32(1e-7)).astype(np.float32)
            elif name.startswith("baseline"):
                output = baseline.copy()
            else:
                if native_science_fail and precision == "native":
                    output = (baseline + row["delta_condition"] * row["delta_condition"] * np.float32(1000.0)).astype(np.float32)
                else:
                    output = (baseline + np.float32(2.0) * row["delta_condition"]).astype(np.float32)
            if precision == "native" and (native_parity or name != "v0_alpha_00_plus"):
                row["direct"] = output.copy()
            if precision == "native" and not native_parity and name == "v0_alpha_00_plus":
                row["direct"] = (output + np.array([1e-4, 0], np.float32)).astype(np.float32)
            frame = row["frame"]
            encoder_input = np.subtract(np.multiply(frame, np.float32(2.0), dtype=np.float32),
                                        np.float32(1.0), dtype=np.float32)[None, :, None, :, :]
            encoder_evidence = {
                "precision_path": precision, "input_dtype": "float32", "input_shape": list(frame.shape),
                "encoder_input_shape": list(encoder_input.shape), "encoder_input_dtype": "float32",
                "operation_count": 1, "operation_dtypes": {"float32": 1}, "encoder_identity": {"fixture": True},
                "output_dtype": "float32", "inner_input_dtype": "float32", "inner_output_dtype": "float32",
                "state_dtypes": {"parameters": {"p": "float32"},
                                  "buffers": {} if empty_buffers else {"b": "float32"},
                                  "constants": {"c": "float32"}},
                "actual_encoder_input_dtype": "float32", "scaled_latent_dtype": "float32", "actual_output_dtype": "float32",
                "dispatch_observed": True, "autocast_disabled": True, "tf32_disabled": True,
                "cache_cleared_before": True, "cache_cleared_after": True,
            }
            if bad_fp32_evidence and precision == "temporary_fp32":
                encoder_evidence["operation_dtypes"] = {"bfloat16": 1}
            store.write_success(spec["sample_id"], {"spec": spec, "record": {
                "source_name": name, "precision": precision, "encoded_condition": output,
                "encoder": {"evidence": encoder_evidence, "arrays": {
                    "input_rgb": frame, "encoder_input": encoder_input,
                    "actual_encoder_input": encoder_input, "actual_output": output}},
                "evidence": {"operation_counts": {"G": 0, "D": 0, "E": 1}}}},
                operation_counts={"G": 0, "D": 0, "E": 1})
        binding = {"source": "fixture-source", "task7_code_sha256": "fixture-code", "config": "fixture",
                   "model": "fixture", "vae": "fixture", "framework": "fixture", "mask": "mask",
                   "z0": "z0", "v0": "v0", "noise_policy": "fixed-seed-paired"}
        plan_sha256 = __import__("hashlib").sha256(runner.canonical_json(plan).encode("utf-8")).hexdigest()
        (stage / "stage_config.json").write_text(json.dumps({"schema_version": "umi-task7-stage-v2", "stage": "A", "plan": plan,
            "plan_sha256": plan_sha256, "binding": binding}), encoding="utf-8")
        (stage / "run_status.json").write_text(json.dumps({"schema_version": "umi-task7-run-v2", "status": "COMPLETE",
            "stage": "A", "planned_samples": 16, "completed_samples": [item["sample_id"] for item in plan],
            "failed_samples": [], "skipped_samples": [], "completed_count": 16, "failed_count": 0,
            "skipped_count": 0, "formal_counts": {"G": 0, "D": 0, "E": 16}, "binding": binding}), encoding="utf-8")

    def test_a_only_complete_flow_uses_nonzero_saved_floor_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"; root.mkdir(); source = self._source_fixture(); self._write_a_stage(root, source)
            with mock.patch.object(api, "_load_source_evidence", return_value=source):
                result = api.analyze_task7_run(root, stage="A", raw_root="raw", decoder_root="decoder",
                                               output_dir=Path(temp) / "analysis")
            self.assertTrue(result["A"]["a_engineering_pass"])
            self.assertTrue(result["A"]["a_scientific_pass"])
            self.assertGreater(result["A"]["precision"]["native"]["floor"], 0.0)
            self.assertTrue((Path(temp) / "analysis" / "a_geometry.csv").is_file())
            self.assertTrue((Path(temp) / "analysis" / "a_secants.csv").is_file())

    def test_a_native_parity_is_engineering_failure_not_scientific_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"; root.mkdir(); source = self._source_fixture(); self._write_a_stage(root, source, native_parity=False)
            with mock.patch.object(api, "_load_source_evidence", return_value=source):
                result = api.analyze_task7_run(root, stage="A", raw_root="raw", decoder_root="decoder",
                                               output_dir=Path(temp) / "analysis")
            self.assertFalse(result["A"]["a_engineering_pass"])
            self.assertIn("native output differs", " ".join(result["A"]["engineering_reasons"]))

    def test_native_scientific_fail_does_not_block_fp32_scientific_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"; root.mkdir(); source = self._source_fixture()
            self._write_a_stage(root, source, native_science_fail=True)
            with mock.patch.object(api, "_load_source_evidence", return_value=source):
                result = api.analyze_task7_run(root, stage="A", raw_root="raw", decoder_root="decoder",
                                               output_dir=Path(temp) / "analysis")
            self.assertTrue(result["A"]["a_engineering_pass"])
            self.assertEqual(result["A"]["native_scientific_status"], "FAIL")
            self.assertEqual(result["A"]["fp32_scientific_status"], "PASS")
            self.assertTrue(result["A"]["a_scientific_pass"])

    def test_missing_fp32_precision_evidence_is_engineering_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"; root.mkdir(); source = self._source_fixture()
            self._write_a_stage(root, source, bad_fp32_evidence=True)
            with mock.patch.object(api, "_load_source_evidence", return_value=source):
                result = api.analyze_task7_run(root, stage="A", raw_root="raw", decoder_root="decoder",
                                               output_dir=Path(temp) / "analysis")
            self.assertFalse(result["A"]["a_engineering_pass"])
            self.assertTrue(any("operation_dtypes" in reason for reason in result["A"]["engineering_reasons"]))

    def test_empty_registered_buffer_bucket_is_valid_with_schema_and_state_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"; root.mkdir(); source = self._source_fixture()
            self._write_a_stage(root, source, empty_buffers=True)
            with mock.patch.object(api, "_load_source_evidence", return_value=source):
                result = api.analyze_task7_run(root, stage="A", raw_root="raw", decoder_root="decoder",
                                               output_dir=Path(temp) / "analysis")
            self.assertTrue(result["A"]["a_engineering_pass"])
            self.assertTrue(result["A"]["a_scientific_pass"])

    def test_foreign_stage_binding_fails_before_a_science(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"; root.mkdir(); source = self._source_fixture(); self._write_a_stage(root, source)
            config_path = root / "stages" / "A" / "stage_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8")); config["binding"]["source"] = "foreign-source"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with mock.patch.object(api, "_load_source_evidence", return_value=source):
                result = api.analyze_task7_run(root, stage="A", raw_root="raw", decoder_root="decoder",
                                               output_dir=Path(temp) / "analysis")
            self.assertFalse(result["A"]["a_engineering_pass"])
            self.assertTrue(any("binding.source" in reason or "run_status.binding" in reason
                                for reason in result["A"]["engineering_reasons"]))

    def test_a_tampered_artifact_fails_before_science(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"; root.mkdir(); source = self._source_fixture(); self._write_a_stage(root, source)
            sample = root / "stages" / "A" / "samples" / "A_baseline_pre_native"
            artifact = next(sample.glob("*.npy"))
            with artifact.open("ab") as stream:
                stream.write(b"tamper")
            with mock.patch.object(api, "_load_source_evidence", return_value=source):
                result = api.analyze_task7_run(root, stage="A", raw_root="raw", decoder_root="decoder",
                                               output_dir=Path(temp) / "analysis")
            self.assertFalse(result["A"]["a_engineering_pass"])
            self.assertEqual(result["A"]["scientific_status"], "NOT_RUN")


class Task7SavedEvidenceBTests(unittest.TestCase):
    """Synthetic B fixtures exercise only saved-evidence wiring contracts."""

    _NAMES = ("baseline_pre", "v0_alpha_00_plus", "v0_alpha_00_minus",
              "v0_alpha_01_plus", "v0_alpha_01_minus", "v0_alpha_02_plus",
              "v0_alpha_02_minus", "baseline_post")

    @classmethod
    def _record(cls, condition, encoded, noise_hash, *, full_latent=None):
        condition = np.asarray(condition, dtype=np.float32)
        encoded = np.asarray(encoded, dtype=np.float32)
        actual = {key: condition.copy() for key in (
            "prepared_condition", "initial_condition", "reference_condition",
            "first_condition", "last_condition")}
        actual["condition_steps"] = np.stack([condition])
        record = {
            "condition_input_fp32": condition,
            "encoded_condition": encoded,
            "next_condition_fp32": encoded.copy(),
            "actual": actual,
            "evidence": {"operation_counts": {"G": 1, "D": 1, "E": 1},
                         "prediction_noise_hash": noise_hash},
        }
        if full_latent is not None:
            record["full_latent"] = np.asarray(full_latent, dtype=np.float32)
        return record

    @classmethod
    def _fixture(cls, *, tamper_name=None):
        base_input = np.array([1.0, -1.0], dtype=np.float32)
        baseline_step0 = np.array([2.0, -2.0], dtype=np.float32)
        baseline_step1 = np.array([3.0, -3.0], dtype=np.float32)
        mask = np.ones(2, dtype=bool)
        direction = np.array([1.0, 0.0], dtype=np.float32)
        rows, records, a_records = {}, {}, {}
        for name in cls._NAMES:
            if name.startswith("baseline"):
                delta = np.zeros(2, dtype=np.float32)
            else:
                alpha = 0.001 if "00" in name else (0.003 if "01" in name else 0.01)
                sign = 1.0 if name.endswith("plus") else -1.0
                delta = (np.float32(sign * alpha) * direction).astype(np.float32)
            consumed = (base_input + delta).astype(np.float32)
            rows[name] = {"consumed_condition": consumed, "mask": mask,
                          "raw": Path("raw"), "decoder": Path("decoder")}
            step0_encoded = (baseline_step0 + delta).astype(np.float32)
            step1_input = step0_encoded.copy()
            if name == tamper_name:
                step1_input[0] = np.float32(step1_input[0] + 0.25)
            step1_encoded = (baseline_step1 + delta).astype(np.float32)
            records[f"B_{name}_step_0"] = cls._record(
                consumed, step0_encoded, "noise0", full_latent=np.array([7.0, 7.0], np.float32))
            records[f"B_{name}_step_1"] = cls._record(step1_input, step1_encoded, "noise1")
            a_records[f"A_{name}_temporary_fp32"] = {"encoded_condition": step0_encoded}
        binding = {"source": "fixture-source", "task7_code_sha256": "fixture-code",
                   "config": {"stage": "B"}, "model": "fixture", "vae": "fixture",
                   "framework": "fixture", "mask": "fixture-mask", "z0": "fixture-z0",
                   "v0": "fixture-v0", "noise_policy": "fixed-seed-paired"}
        stage = {"run_dir": Path("fixture-run"), "plan": runner.build_stage_plan("B"),
                 "records": records, "config": {"binding": binding},
                 "status": {"binding": binding}}
        source = {"rows": rows, "condition_mask": mask,
                  "source_tree_sha256": "fixture-source", "mask_sha256": "fixture-mask",
                  "z0_sha256": "fixture-z0", "v0_sha256": "fixture-v0"}
        return stage, source, a_records

    def _analyze_fixture(self, *, tamper_name=None):
        stage, source, a_records = self._fixture(tamper_name=tamper_name)
        with mock.patch.object(api, "load_task7_stage", return_value={"records": a_records}), \
             mock.patch.object(api, "_load_runner_array", return_value=np.array([7.0, 7.0], np.float32)):
            return api._b_stage_analysis(stage, source, {}, tensors={})

    def test_each_perturbed_step_one_input_matches_own_step_zero_output(self):
        result = self._analyze_fixture()
        self.assertTrue(result["engineering_pass"])
        self.assertTrue(result["repeatability_pass"])

    def test_perturbed_step_one_input_mismatch_is_engineering_failure(self):
        result = self._analyze_fixture(tamper_name="v0_alpha_00_plus")
        self.assertFalse(result["engineering_pass"])
        self.assertFalse(result["repeatability_pass"])
        self.assertIn(
            "B trajectory step-1 input differs from own step-0 encoded condition: v0_alpha_00_plus",
            result["engineering_reasons"],
        )


class Task7SavedEvidenceCTests(unittest.TestCase):
    """Synthetic C fixtures prove that held-out delta2 comes from B."""

    _NAMES = ("v0_alpha_00_plus", "v0_alpha_00_minus", "v0_alpha_01_plus",
              "v0_alpha_01_minus", "v0_alpha_02_plus", "v0_alpha_02_minus")

    @staticmethod
    def _record(condition, encoded, noise_hash="seed1"):
        condition = np.asarray(condition, dtype=np.float32)
        encoded = np.asarray(encoded, dtype=np.float32)
        actual = {key: condition.copy() for key in (
            "prepared_condition", "initial_condition", "reference_condition",
            "first_condition", "last_condition")}
        actual["condition_steps"] = np.stack([condition])
        return {
            "condition_input_fp32": condition,
            "encoded_condition": encoded,
            "next_condition_fp32": encoded.copy(),
            "actual": actual,
            "evidence": {"operation_counts": {"G": 1, "D": 1, "E": 1},
                         "prediction_noise_hash": noise_hash},
        }

    @classmethod
    def _fixture(cls):
        z1 = np.array([0.0, 0.0], dtype=np.float32)
        z2 = np.array([1.0, 1.0], dtype=np.float32)
        ray = np.array([1.0, 0.0], dtype=np.float32)
        b_delta2 = np.array([1.0, 0.0], dtype=np.float32)
        mask = np.ones(2, dtype=bool)
        b_records = {
            "B_baseline_pre_step_0": {"encoded_condition": z1.copy()},
            "B_baseline_pre_step_1": {"encoded_condition": z2.copy()},
        }
        c_records = {
            "C_baseline_pre": cls._record(z1, z2),
            "C_baseline_post": cls._record(z1, z2),
        }
        for name in cls._NAMES:
            b_records[f"B_{name}_step_0"] = {"encoded_condition": ray.copy()}
            b_records[f"B_{name}_step_1"] = {
                "encoded_condition": (z2 + b_delta2).astype(np.float32)}
            for beta in api.TASK7_BETAS:
                for sign, label in ((1, "plus"), (-1, "minus")):
                    signed_input = z1 + np.float32(sign * beta) * ray
                    # C's beta=.1 response is ten times the held-out B delta2.
                    signed_output = z2 + np.float32(sign * beta * 100.0) * ray
                    sample_id = f"C_delta1_{cls._NAMES.index(name):02d}_beta_{beta:g}_{label}"
                    c_records[sample_id] = cls._record(signed_input, signed_output)
        binding = {"source": "fixture-source", "task7_code_sha256": "fixture-code",
                   "config": {"stage": "C"}, "model": "fixture", "vae": "fixture",
                   "framework": "fixture", "mask": "fixture-mask", "z0": "fixture-z0",
                   "v0": "fixture-v0", "noise_policy": "fixed-seed-paired"}
        stage = {"run_dir": Path("fixture-run"), "records": c_records,
                 "config": {"binding": binding}, "status": {"binding": binding}}
        source = {"rows": {"baseline_pre": {"mask": mask}}, "condition_mask": mask,
                  "source_tree_sha256": "fixture-source", "mask_sha256": "fixture-mask",
                  "z0_sha256": "fixture-z0", "v0_sha256": "fixture-v0"}
        return stage, source, b_records, b_delta2

    def test_c_held_out_delta2_uses_corresponding_b_step_one_not_c_beta_one(self):
        stage, source, b_records, b_delta2 = self._fixture()
        tensors = {}
        with mock.patch.object(api, "load_task7_stage", return_value={"records": b_records}):
            result = api._c_stage_analysis(stage, source, {}, tensors=tensors)
        self.assertEqual(len(result["rows"]), 6)
        np.testing.assert_array_equal(tensors["c_delta1_00_actual_delta2"], b_delta2)
        np.testing.assert_array_equal(tensors["c_delta1_00_actual_delta2"], np.array([1.0, 0.0], np.float32))


if __name__ == "__main__":
    unittest.main()
