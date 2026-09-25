"""CPU contracts for immutable Task 10 source validation in Task 11 analysis."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_umi_task11_horizon_error import (
    _read_csv_rows,
    _read_npz_first_axis_slice,
    load_record_array,
    validate_metric_reproduction,
    validate_source_identity,
    validate_task8_sample,
    verify_ar_feedback_condition,
    verify_masked_condition_consumption,
    verify_task10_manifest,
)


class SourceIdentityTests(unittest.TestCase):
    def test_locked_task10_identity_requires_all_records_and_seed_schedules(self):
        identity = {"schema": "umi-task10-v1", "record_indices": [1, 2, 3, 6, 7, 14],
                    "seed_pairs": [[0, 1], [2, 3]]}
        status = {"status": "GENERATION_COMPLETE", "completed_samples": 48,
                  "record_indices": [1, 2, 3, 6, 7, 14], "seed_pairs": [[0, 1], [2, 3]]}
        validate_source_identity(identity, status)
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_source_identity({**identity, "record_indices": [1, 2, 3, 6, 7, 15]}, status)
        with self.assertRaisesRegex(ValueError, "complete"):
            validate_source_identity(identity, {**status, "completed_samples": 47})


class SampleArtifactTests(unittest.TestCase):
    def test_task8_sample_hashes_and_mmap_array_descriptor_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            array_path = root / "record_generated_rgb.npy"
            values = np.arange(24, dtype=np.float32).reshape(3, 2, 2, 2) / 24
            np.save(array_path, values, allow_pickle=False)
            metadata = {"generated_rgb": {"artifact": array_path.name, "dtype": "float32",
                                           "shape": list(values.shape)}}
            (root / "record.json").write_text(json.dumps(metadata), encoding="utf-8")
            (root / "status.json").write_text(json.dumps({"status": "success", "artifact_sha256": {
                "record.json": hashlib.sha256((root / "record.json").read_bytes()).hexdigest(),
                array_path.name: hashlib.sha256(array_path.read_bytes()).hexdigest(),
            }}), encoding="utf-8")
            record = validate_task8_sample(root)
            mapped = load_record_array(root, record, "generated_rgb")
            self.assertIsInstance(mapped, np.memmap)
            self.assertEqual(mapped.dtype, np.float32)
            np.testing.assert_array_equal(mapped, values)
            mapped._mmap.close()
            array_path.write_bytes(array_path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "hash"):
                validate_task8_sample(root)


class SourceManifestTests(unittest.TestCase):
    def test_task10_manifest_detects_changed_and_unlisted_source_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "raw.txt"
            data.write_bytes(b"original")
            digest = hashlib.sha256(data.read_bytes()).hexdigest()
            (root / "MANIFEST.sha256").write_text(f"{digest}  raw.txt\n", encoding="ascii")
            self.assertEqual(verify_task10_manifest(root)["file_count"], 1)
            data.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "manifest"):
                verify_task10_manifest(root)

    def test_only_well_formed_detached_review_bundle_sidecar_is_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "raw.txt"
            data.write_bytes(b"original")
            digest = hashlib.sha256(data.read_bytes()).hexdigest()
            (root / "MANIFEST.sha256").write_text(f"{digest}  raw.txt\n", encoding="ascii")
            analysis = root / "analysis"
            analysis.mkdir()
            bundle = analysis / "review_bundle.zip"
            bundle.write_bytes(b"immutable review package")
            sidecar = analysis / "review_bundle.sha256"
            sidecar.write_text(f"{hashlib.sha256(bundle.read_bytes()).hexdigest()}  review_bundle.zip\n",
                               encoding="ascii")
            result = verify_task10_manifest(root)
            self.assertEqual(result["file_count"], 1)
            self.assertEqual(result["excluded_auxiliary_files"],
                             ["analysis/review_bundle.sha256", "analysis/review_bundle.zip"])
            bundle_hash = hashlib.sha256(bundle.read_bytes()).hexdigest()
            (root / "MANIFEST.sha256").write_text(
                f"{digest}  raw.txt\n{bundle_hash}  analysis/review_bundle.zip\n", encoding="ascii")
            manifested_bundle = verify_task10_manifest(root)
            self.assertEqual(manifested_bundle["file_count"], 2)
            self.assertEqual(manifested_bundle["excluded_auxiliary_files"], ["analysis/review_bundle.sha256"])
            sidecar.write_text("not a valid sidecar\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "sidecar"):
                verify_task10_manifest(root)

    def test_unmanifested_review_archive_without_sidecar_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "raw.txt"
            data.write_bytes(b"original")
            digest = hashlib.sha256(data.read_bytes()).hexdigest()
            (root / "MANIFEST.sha256").write_text(f"{digest}  raw.txt\n", encoding="ascii")
            analysis = root / "analysis"
            analysis.mkdir()
            (analysis / "review_bundle.zip").write_bytes(b"review archive")
            with self.assertRaisesRegex(ValueError, "lacks.*sidecar"):
                verify_task10_manifest(root)


class MetricReproductionTests(unittest.TestCase):
    def test_task10_metrics_match_with_declared_float64_tolerance(self):
        baseline = {"g0_latent_rms": 0.25, "tf_rmse": 0.10, "ar_rmse": 0.12,
                    "tf2_latent_rms": 0.31, "ar2_latent_rms": 0.32}
        reproduced = {key: value + (2e-13 if key == "tf_rmse" else 0.0)
                      for key, value in baseline.items()}
        evidence = validate_metric_reproduction(baseline, reproduced)
        self.assertEqual(evidence["status"], "WITHIN_DECLARED_TOLERANCE")
        self.assertEqual(evidence["tolerance"], {"rtol": 1e-10, "atol": 1e-12})
        with self.assertRaisesRegex(ValueError, "reproduction"):
            validate_metric_reproduction(baseline, {**reproduced, "tf_rmse": 0.101})

    def test_task10_csv_schema_requires_exact_historical_latent_metric_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stratum_metrics.csv"
            path.write_text("record_index,seed_pair,g0_latent_rms,tf_rmse,ar_rmse,tf2_latent_rms,ar2_latent_rms\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact 12"):
                _read_csv_rows(path)
            path.write_text("record_index,seed_pair,g0_latent_rms,tf_rmse,ar_rmse,tf_latent_rms,ar2_latent_rms\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "required fields.*tf2_latent_rms"):
                _read_csv_rows(path)


class CompressedNpzSliceTests(unittest.TestCase):
    def test_reads_one_first_axis_frame_with_public_numpy_header_api(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inputs.npz"
            values = np.arange(5 * 3 * 4, dtype=np.float32).reshape(5, 3, 4)
            np.savez_compressed(path, rgb_float32=values)
            for index in (0, 2, 4):
                np.testing.assert_array_equal(
                    _read_npz_first_axis_slice(path, "rgb_float32", index), values[index])


class FeedbackConditionTests(unittest.TestCase):
    def test_wrong_condition_fails_even_if_ar_output_matches_g0(self):
        g0_encoded = np.array([1.0, 2.0], dtype=np.float32)
        ar_condition_input = np.array([0.0, 0.0], dtype=np.float32)
        ar_encoded_output = g0_encoded.copy()
        np.testing.assert_array_equal(ar_encoded_output, g0_encoded)
        with self.assertRaisesRegex(ValueError, "condition input"):
            verify_ar_feedback_condition(g0_encoded, ar_condition_input)

    def test_correct_condition_passes_even_if_ar_output_differs_from_g0(self):
        g0_encoded = np.array([1.0, 2.0], dtype=np.float32)
        ar_condition_input = g0_encoded.copy()
        ar_encoded_output = np.array([9.0, 8.0], dtype=np.float32)
        self.assertFalse(np.array_equal(ar_encoded_output, g0_encoded))
        verify_ar_feedback_condition(g0_encoded, ar_condition_input)


class ConditionCarrierTests(unittest.TestCase):
    def test_only_saved_condition_positions_must_match_input(self):
        condition = np.zeros((1, 2, 1, 1, 1), dtype=np.float32)
        condition[:, 0] = 1.0
        condition[:, 1] = 2.0
        full_mask = np.zeros((1, 2, 3, 1, 1), dtype=bool)
        full_mask[:, :, 0] = True
        carrier = np.broadcast_to(condition, (30, 1, 2, 3, 1, 1)).copy()
        carrier[:, :, :, 1:] = 4.5  # generated region is not supposed to equal the condition input
        verify_masked_condition_consumption(condition, carrier, full_mask, full_mask.copy())
        carrier[0, 0, 0, 0, 0, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "condition positions"):
            verify_masked_condition_consumption(condition, carrier, full_mask, full_mask.copy())

    def test_generation_mask_must_match_saved_truth_full_mask(self):
        condition = np.zeros((1, 1, 1, 1, 1), dtype=np.float32)
        full_mask = np.ones((1, 1, 1, 1, 1), dtype=bool)
        steps = np.zeros((30, 1, 1, 1, 1, 1), dtype=np.float32)
        wrong_mask = np.zeros_like(full_mask)
        with self.assertRaisesRegex(ValueError, "generation mask"):
            verify_masked_condition_consumption(condition, steps, wrong_mask, full_mask)


if __name__ == "__main__":
    unittest.main()
