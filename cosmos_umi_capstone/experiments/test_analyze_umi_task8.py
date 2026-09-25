from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from task8_frozen.analyze_umi_task8 import (
    AnalysisError,
    analyze_rgb_metrics,
    analyze_task8_outputs,
    analyze_task8_run,
    main,
    condition_latent_metrics,
    cosine_or_none,
    psnr_delta_higher_is_better,
    psnr,
)


class Task8AnalysisTests(unittest.TestCase):
    def test_rgb_metrics_and_zero_error_psnr(self):
        truth = np.zeros((2, 3, 2, 2), np.float32)
        pred = np.ones_like(truth)
        metrics = analyze_rgb_metrics(pred, truth)
        self.assertAlmostEqual(metrics["rmse"], 1.0)
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertEqual(metrics["psnr"], 0.0)
        self.assertEqual(psnr(truth, truth), float("inf"))

    def test_rgb_metrics_rejects_range_and_frame_misalignment(self):
        truth = np.zeros((2, 3, 2, 2), np.float32)
        with self.assertRaises(AnalysisError):
            analyze_rgb_metrics(np.full_like(truth, 1.01), truth)
        with self.assertRaises(AnalysisError):
            analyze_rgb_metrics(np.zeros((1, 3, 2, 2), np.float32), truth)

    def test_condition_metrics_use_rms_and_cosine_with_zero_norm_na(self):
        left = np.asarray([[[[1.0]]]], np.float32)
        right = np.asarray([[[[2.0]]]], np.float32)
        metrics = condition_latent_metrics(left, right)
        self.assertAlmostEqual(metrics["rms"], 1.0)
        self.assertAlmostEqual(metrics["cosine"], 1.0)
        self.assertIsNone(cosine_or_none(np.zeros(3), np.ones(3)))

    def test_delta_feedback_error_is_ar_minus_tf(self):
        truth = np.zeros((1, 3, 1, 1), np.float32)
        result = analyze_rgb_metrics(np.full_like(truth, 0.2), truth)
        baseline = analyze_rgb_metrics(np.full_like(truth, 0.1), truth)
        self.assertAlmostEqual(result["rmse"] - baseline["rmse"], 0.1)

    def test_primary_feedback_delta_uses_last_frame_and_labels_full_sequence_aggregate(self):
        truth = np.zeros((33, 3, 1, 1), np.float32)

        def record(frames, noise):
            return {
                "status": "success",
                "generated_rgb": frames,
                "output_full": frames.copy(),
                "prediction_noise_hash": noise,
                "encoded_condition": np.zeros((1, 1, 1, 1), np.float32),
                "condition_input": np.zeros((1, 1, 1, 1), np.float32),
                "action_hash": "same-action",
                "action_consumption": {"all_steps_match": True, "steps": 30,
                                        "expected_token_hash": "same-packed-action"},
                "precision": {"G": "float32", "D": "float32", "E": "float32"},
            }

        g0 = np.full((16, 3, 1, 1), 0.01, np.float32)
        tf = np.full((16, 3, 1, 1), 0.02, np.float32)
        ar = np.full((16, 3, 1, 1), 0.03, np.float32)
        tf[-1] = 0.10
        ar[-1] = 0.04
        records = {
            "G0_real_x0_seed0": record(g0, "noise-0"),
            "G0_repeat_real_x0_seed0": record(g0.copy(), "noise-0"),
            "TF2_real_x16_seed1": record(tf, "noise-1"),
            "AR2_g0_float_last_fp32_seed1": record(ar, "noise-1"),
        }

        result = analyze_task8_outputs(
            records,
            truth_rgb=truth,
            gt_condition_x16=np.zeros((1, 1, 1, 1), np.float32),
            gt_condition_x32=np.zeros((1, 1, 1, 1), np.float32),
        )
        delta = result["metrics"]["delta_feedback_error"]
        self.assertEqual(delta["primary_scope"], "last_frame")
        self.assertAlmostEqual(delta["rmse"],
                               result["metrics"]["E_AR2"]["rmse"] - result["metrics"]["E_TF2"]["rmse"])
        full = delta["full_16_frame_aggregate"]
        expected_full = analyze_rgb_metrics(ar, truth[17:33])["rmse"] - analyze_rgb_metrics(tf, truth[17:33])["rmse"]
        self.assertAlmostEqual(full["rmse_delta"], expected_full)
        self.assertNotAlmostEqual(delta["rmse"], full["rmse_delta"])
        self.assertAlmostEqual(full["squared_error_decomposition_residual"], 0.0)
        self.assertTrue(full["squared_error_decomposition_sanity"])

    def test_formal_metrics_use_last_frame_and_two_distinct_truth_windows(self):
        truth = (np.arange(33, dtype=np.float32) / 100.0)[:, None, None, None]
        truth = np.broadcast_to(truth, (33, 3, 1, 1)).copy()
        def record(frames, noise, condition):
            return {"status": "success", "generated_rgb": frames, "output_full": frames,
                    "prediction_noise_hash": noise, "encoded_condition": condition,
                    "condition_input": np.zeros((1, 1, 1, 1), np.float32), "action_hash": "a0",
                    "action_consumption": {"all_steps_match": True, "steps": 30,
                                            "expected_token_hash": "action-a"},
                    "precision": {"G": "float32", "D": "float32", "E": "float32"}}
        g0 = np.concatenate([truth[:1], truth[1:17] + 0.01], axis=0)
        tf = truth[17:33] + 0.02
        ar = truth[17:33] + 0.03
        records = {
            "G0_real_x0_seed0": record(g0, "n0", np.zeros((1, 1, 1, 1), np.float32)),
            "G0_repeat_real_x0_seed0": record(g0.copy(), "n0", np.zeros((1, 1, 1, 1), np.float32)),
            "TF2_real_x16_seed1": record(tf, "n1", np.zeros((1, 1, 1, 1), np.float32)),
            "AR2_g0_float_last_fp32_seed1": record(ar, "n1", np.zeros((1, 1, 1, 1), np.float32)),
        }
        result = analyze_task8_outputs(records, truth_rgb=truth,
                                       gt_condition_x16=np.zeros((1, 1, 1, 1), np.float32),
                                       gt_condition_x32=np.zeros((1, 1, 1, 1), np.float32))
        self.assertAlmostEqual(result["metrics"]["E1"]["rmse"], 0.01)
        self.assertAlmostEqual(result["metrics"]["E_TF2"]["rmse"], 0.02)
        self.assertAlmostEqual(result["metrics"]["E_AR2"]["rmse"], 0.03)
        self.assertEqual(result["metrics"]["target_truth_indices"], {"E1": 16, "E_TF2": 32, "E_AR2": 32})
        self.assertAlmostEqual(result["metrics"]["per_frame"]["G0"]["rmse"], 0.01)
        self.assertAlmostEqual(result["metrics"]["per_frame"]["TF2"]["rmse"], 0.02)
        feedback_delta = result["metrics"]["delta_feedback_error"]
        self.assertEqual(feedback_delta["primary_scope"], "last_frame")
        self.assertAlmostEqual(feedback_delta["rmse"], 0.01)
        full_sequence = feedback_delta["full_16_frame_aggregate"]
        self.assertEqual(full_sequence["scope"], "all_16_generated_frames")
        self.assertAlmostEqual(full_sequence["rmse_delta"], 0.01)
        self.assertAlmostEqual(full_sequence["prediction_delta_rmse"], 0.01)
        self.assertTrue(full_sequence["reverse_triangle_sanity"])
        self.assertAlmostEqual(full_sequence["squared_error_decomposition_residual"], 0.0)
        self.assertTrue(result["metrics"]["g0_repeat_exact"])

    def test_condition_evidence_is_required(self):
        truth = np.zeros((33, 3, 1, 1), np.float32)
        record = {"status": "success", "generated_rgb": np.zeros((16, 3, 1, 1), np.float32),
                  "output_full": np.zeros((16, 3, 1, 1), np.float32), "prediction_noise_hash": "n",
                  "action_hash": "a", "action_consumption": {"all_steps_match": True, "steps": 30,
                  "expected_token_hash": "t"}, "precision": {"G": "float32", "D": "float32", "E": "float32"}}
        records = {name: dict(record) for name in ("G0_real_x0_seed0", "G0_repeat_real_x0_seed0",
                  "TF2_real_x16_seed1", "AR2_g0_float_last_fp32_seed1")}
        with self.assertRaises(AnalysisError):
            analyze_task8_outputs(records, truth_rgb=truth,
                                  gt_condition_x16=np.zeros((1, 1, 1, 1), np.float32),
                                  gt_condition_x32=np.zeros((1, 1, 1, 1), np.float32))

    def test_psnr_delta_is_signed_and_inf_safe(self):
        self.assertEqual(psnr_delta_higher_is_better(float("inf"), 0.0), "inf")
        self.assertEqual(psnr_delta_higher_is_better(0.0, float("inf")), "-inf")
        self.assertEqual(psnr_delta_higher_is_better(float("inf"), float("inf")), "N/A")

    def test_run_rows_are_sixteen_per_call_and_exclude_initial_g0_frame(self):
        truth = (np.arange(33, dtype=np.float32) / 100.0)[:, None, None, None]
        truth = np.broadcast_to(truth, (33, 3, 1, 1)).copy()
        frames = {"G0_real_x0_seed0": truth[:17], "G0_repeat_real_x0_seed0": truth[:17],
                  "TF2_real_x16_seed1": truth[17:33], "AR2_g0_float_last_fp32_seed1": truth[17:33]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"; samples = root / "samples"
            for key, values in frames.items():
                sample = samples / key; sample.mkdir(parents=True)
                payload = {"status": "success", "generated_rgb": values.tolist(), "output_full": values.tolist(),
                           "prediction_noise_hash": "n0" if "G0" in key else "n1",
                           "encoded_condition": [[[[0.0]]]], "condition_input": [[[[0.0]]]], "action_hash": "a",
                           "action_consumption": {"all_steps_match": True, "steps": 30, "expected_token_hash": "t"},
                           "precision": {"G": "float32", "D": "float32", "E": "float32"}}
                record_path = sample / "record.json"
                record_path.write_text(json.dumps(payload), encoding="utf-8")
                digest = hashlib.sha256(record_path.read_bytes()).hexdigest()
                (sample / "status.json").write_text(json.dumps({
                    "status": "success", "artifact_sha256": {"record.json": digest}
                }), encoding="utf-8")
            result = analyze_task8_run(root, truth_rgb=truth,
                                       gt_condition_x16=np.zeros((1, 1, 1, 1), np.float32),
                                       gt_condition_x32=np.zeros((1, 1, 1, 1), np.float32),
                                       output_dir=Path(temp) / "out")
            self.assertEqual(len(result["per_frame_rows"]), 48)
            self.assertFalse(any(row["call"] == "G0" and row["truth_index"] == 0 for row in result["per_frame_rows"]))

            tampered = samples / "TF2_real_x16_seed1" / "record.json"
            tampered.write_text(tampered.read_text(encoding="utf-8").replace('"status": "success"', '"status": "tampered"'), encoding="utf-8")
            with self.assertRaises(AnalysisError):
                analyze_task8_run(root, truth_rgb=truth,
                                  gt_condition_x16=np.zeros((1, 1, 1, 1), np.float32),
                                  gt_condition_x32=np.zeros((1, 1, 1, 1), np.float32),
                                  output_dir=Path(temp) / "tampered-out")

    def test_run_rejects_failed_attempt_entry_and_cli_requires_ground_truth_conditions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"; samples = root / "samples"; samples.mkdir(parents=True)
            (samples / "G0_real_x0_seed0.failed-attempt.001").mkdir()
            truth_path = Path(temp) / "truth.npy"
            np.save(truth_path, np.zeros((33, 3, 1, 1), np.float32), allow_pickle=False)
            with self.assertRaises(AnalysisError):
                analyze_task8_run(root, truth_rgb=np.zeros((33, 3, 1, 1), np.float32),
                                  gt_condition_x16=np.zeros((1, 1, 1, 1), np.float32),
                                  gt_condition_x32=np.zeros((1, 1, 1, 1), np.float32),
                                  output_dir=Path(temp) / "out")
            with self.assertRaises(SystemExit):
                main(["--run-dir", str(root), "--truth-rgb", str(truth_path)])


if __name__ == "__main__":
    unittest.main()
