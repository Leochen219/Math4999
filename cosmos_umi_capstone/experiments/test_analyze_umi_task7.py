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
        rows = {}
        for name in names:
            is_baseline = name.startswith("baseline")
            alpha = 0.0 if is_baseline else (0.001 if "00" in name else (0.003 if "01" in name else 0.01))
            sign = 0 if is_baseline else (1 if name.endswith("plus") else -1)
            delta = np.zeros(2, dtype=np.float32) if is_baseline else (np.float32(sign * alpha) * direction)
            rows[name] = {"name": name, "direct": baseline.copy(), "delta_condition": delta,
                          "consumed_condition": (baseline + delta).astype(np.float32),
                          "z0_condition": baseline.copy(), "condition_mask": np.ones(2, bool),
                          "mask": np.ones(2, bool), "raw": Path("raw"), "decoder": Path("decoder")}
        rows["baseline_post"]["direct"] = (baseline + np.float32(1e-7)).astype(np.float32)
        return {"raw_root": "raw", "decoder_root": "decoder", "source_tree_sha256": "fixture-source",
                "z0_sha256": "z0", "mask_sha256": "mask", "v0_sha256": "v0", "rows": rows,
                "v0_condition": direction, "z0_condition": baseline,
                "condition_mask": np.ones(2, bool), "mask": np.ones(2, bool)}

    @staticmethod
    def _write_a_stage(run: Path, source, *, native_parity=True):
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
                output = (baseline + np.float32(2.0) * row["delta_condition"]).astype(np.float32)
            if native_parity or name != "v0_alpha_00_plus":
                row["direct"] = output.copy()
            if precision == "native" and not native_parity and name == "v0_alpha_00_plus":
                row["direct"] = (output + np.array([1e-4, 0], np.float32)).astype(np.float32)
            store.write_success(spec["sample_id"], {"spec": spec, "record": {
                "source_name": name, "precision": precision, "encoded_condition": output,
                "encoder": {"schema_version": "fixture"},
                "evidence": {"operation_counts": {"G": 0, "D": 0, "E": 1}}}},
                operation_counts={"G": 0, "D": 0, "E": 1})
        binding = {"source": "fixture-source", "task7_code_sha256": "fixture-code", "config": "fixture",
                   "model": "fixture", "vae": "fixture", "framework": "fixture", "mask": "mask",
                   "z0": "z0", "v0": "v0", "noise_policy": "fixed-seed-paired"}
        (stage / "stage_config.json").write_text(json.dumps({"stage": "A", "plan": plan, "binding": binding}), encoding="utf-8")
        (stage / "run_status.json").write_text(json.dumps({"schema_version": "umi-task7-run-v2", "status": "COMPLETE",
            "stage": "A", "planned_samples": 16, "completed_samples": [item["sample_id"] for item in plan],
            "failed_samples": [], "skipped_samples": [], "completed_count": 16, "failed_count": 0,
            "skipped_count": 0, "formal_counts": {"G": 0, "D": 0, "E": 16}}), encoding="utf-8")

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


if __name__ == "__main__":
    unittest.main()
