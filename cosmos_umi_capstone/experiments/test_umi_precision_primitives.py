import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from umi_precision_primitives import (
    build_precision_contract,
    decompose_response_vectors,
    even_symmetry_residual,
    fixed_noise_hash,
    mask_geometry,
    metric_with_reason,
    quantize_bf16_fp32,
    relative_derivative_change,
    slice_predicted_output,
    validate_ab_exact_input,
    validate_bc_zero_input,
)


class PrecisionPrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.shape = (1, 2, 5, 2, 2)
        self.indexes = (0, 2)
        packed = np.zeros((1, 5, 2, 2), dtype=bool)
        packed[:, 0] = True
        packed[:, 2] = True
        self.packed = packed

    def test_mask_geometry_is_runtime_index_driven_and_predictive_slice_is_framewise(self):
        geometry = mask_geometry(self.indexes, self.packed, self.shape)
        self.assertEqual(geometry.temporal_axis, 2)
        self.assertEqual(geometry.condition_indexes, (0, 2))
        self.assertEqual(geometry.predicted_indexes, (1, 3, 4))
        self.assertEqual(geometry.mask.shape, self.shape)
        output = np.arange(np.prod(self.shape), dtype=np.float32).reshape(self.shape)
        sliced, metadata = slice_predicted_output(output, geometry.mask)
        self.assertEqual(sliced.shape, (1, 2, 3, 2, 2))
        np.testing.assert_array_equal(sliced, output[:, :, (1, 3, 4), :, :])
        self.assertEqual(metadata["predicted_indexes"], [1, 3, 4])

    def test_bf16_quantization_is_fp32_value_and_ab_identity_is_quantized(self):
        common = np.zeros(3, dtype=np.float32)
        self.assertEqual(quantize_bf16_fp32(common).dtype, np.float32)
        self.assertTrue(validate_ab_exact_input(common, common.copy())["passed"])
        self.assertFalse(validate_ab_exact_input(common, common + np.float32(1e-3))["passed"])
        contract = build_precision_contract(
            interface_tensors={"A": common, "B": common.copy(), "C": common.copy()},
            network_dtypes={"A": "bfloat16", "B": "float32", "C": "float32"},
        )
        self.assertEqual(contract["common_interface_dtype"], "float32")
        self.assertEqual(contract["network_dtypes"], {"A": "bfloat16", "B": "float32", "C": "float32"})
        self.assertEqual(contract["quantized_input_sha256"], contract["A_B_input_sha256"])
        self.assertTrue(contract["checks"]["A_B"]["passed"])
        self.assertTrue(contract["checks"]["B_C_zero"]["passed"])

    def test_precision_contract_rejects_observed_ab_mismatch(self):
        from umi_precision_primitives import build_precision_contract

        with self.assertRaisesRegex(ValueError, "A/B"):
            build_precision_contract(
                interface_tensors={
                    "A": np.array([1.0], dtype=np.float32),
                    "B": np.array([1.1], dtype=np.float32),
                    "C": np.zeros(1, dtype=np.float32),
                },
                network_dtypes={"A": "bfloat16", "B": "float32", "C": "float32"},
                require_identities=True,
            )

    def test_precision_contract_rejects_observed_bc_nonzero_mismatch(self):
        from umi_precision_primitives import build_precision_contract

        with self.assertRaisesRegex(ValueError, "B/C"):
            build_precision_contract(
                interface_tensors={
                    "A": np.zeros(1, dtype=np.float32),
                    "B": np.zeros(1, dtype=np.float32),
                    "C": np.ones(1, dtype=np.float32),
                },
                network_dtypes={"A": "bfloat16", "B": "float32", "C": "float32"},
                require_identities=True,
            )

    def test_precision_contract_without_actual_interfaces_does_not_claim_identity(self):
        from umi_precision_primitives import build_precision_contract

        contract = build_precision_contract(np.zeros(1, dtype=np.float32))
        self.assertFalse(contract["observed"])
        self.assertIsNone(contract["A_B_exact_input_identity"])
        self.assertIsNone(contract["B_C_zero_input_identity"])

    def test_b_c_zero_input_identity_requires_exact_zero(self):
        zero = np.zeros((2, 3), dtype=np.float32)
        self.assertTrue(validate_bc_zero_input(zero, zero.copy())["passed"])
        nonzero = zero.copy()
        nonzero[0, 0] = np.float32(1e-8)
        self.assertFalse(validate_bc_zero_input(zero, nonzero)["passed"])

    def test_metrics_use_float64_difference_left_denominator_and_na_reason(self):
        left = np.array([2, 4], dtype=np.int16)
        right = np.array([3, 8], dtype=np.int16)
        # RMS([1, 4]) / RMS([2, 4])
        expected = np.sqrt((1.0 + 16.0) / 2.0) / np.sqrt((4.0 + 16.0) / 2.0)
        self.assertAlmostEqual(relative_derivative_change(left, right), expected)
        metric = metric_with_reason("relative_derivative_change", left, right)
        self.assertAlmostEqual(metric["value"], expected)
        undefined = metric_with_reason("relative_derivative_change", np.zeros(2), np.ones(2))
        self.assertIsNone(undefined["value"])
        self.assertIn("left denominator", undefined["reason"])
        self.assertAlmostEqual(even_symmetry_residual(np.array([3]), np.array([1]), np.array([2])), 0.0)
        self.assertAlmostEqual(even_symmetry_residual(np.array([3]), np.array([1]), np.array([1])), 2.0)

    def test_fixed_noise_hash_and_vector_identity_are_reproducible(self):
        noise = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3)
        mask = np.zeros_like(noise, dtype=bool)
        mask[:, :, 0, :] = True
        first = fixed_noise_hash(noise, mask)
        second = fixed_noise_hash(noise.copy(), mask.copy())
        self.assertEqual(first, second)
        vectors = decompose_response_vectors(np.array([5, 7]), np.array([3, 4]), np.array([1, 2]))
        np.testing.assert_array_equal(vectors["r_A_minus_r_C"], np.array([4.0, 5.0]))
        np.testing.assert_array_equal(vectors["reconstructed"], vectors["r_A_minus_r_C"])
        self.assertEqual(vectors["identity_error_rms"], 0.0)


class ReanalysisTests(unittest.TestCase):
    def _fixture(self, root: Path):
        shape = (1, 1, 3, 1, 2)
        mask = np.zeros(shape, dtype=bool)
        mask[:, :, 0] = True
        z0 = np.ones(shape, dtype=np.float32)
        directions = np.zeros((1,) + shape, dtype=np.float32)
        directions[0][mask] = 1.0
        np.save(root / "z0.npy", z0)
        np.save(root / "mask.npy", mask)
        np.save(root / "direction_bank.npy", directions)
        for sample_id, alpha, sign in (("scan_pre", 0.0, 0), ("dir_00_alpha_0p01_minus", 0.01, -1), ("dir_00_alpha_0p01_plus", 0.01, 1)):
            sample = root / "samples" / sample_id
            sample.mkdir(parents=True)
            if sign:
                target = np.zeros(shape, dtype=np.float32)
                target[mask] = alpha * sign
            else:
                target = np.zeros(shape, dtype=np.float32)
            np.save(sample / "target_delta_fp32.npy", target)
            np.save(sample / "carrier_fp32.npy", z0)
            realized_carrier = z0 + target
            np.save(sample / "realized_carrier_fp32.npy", realized_carrier)
            actual_delta = (realized_carrier - z0).astype(np.float32)
            np.save(sample / "actual_delta_fp32.npy", actual_delta)
            np.save(sample / "actual_delta_bf16.npy", (quantize_bf16_fp32(realized_carrier) - quantize_bf16_fp32(z0)).astype(np.float32))
            np.save(sample / "condition_mask.npy", mask)
            np.save(sample / "initial_condition_mask.npy", mask)
            np.save(sample / "network_condition_bf16.npy", np.full(shape, 2.0, dtype=np.float32))
            np.save(sample / "initial_state.npy", np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
            np.save(sample / "final_latent_full.npy", z0 + target)
            np.save(sample / "predicted_latent.npy", (z0 + target)[:, :, 1:])
            record = {
                "sample_id": sample_id,
                "alpha": alpha,
                "sign": sign,
                "direction_index": 0,
                "model_seed": 0,
                "latent_slicing": {
                    "axis": 2,
                    "source_shape": list(shape),
                    "selected_shape": [1, 1, 2, 1, 2],
                    "condition_indexes": [0],
                    "predicted_indexes": [1, 2],
                },
                "network_condition_dtype": "torch.bfloat16",
                "expected_network_condition_hash": "observed-in-run09-schema",
                "condition_step_hashes": ["observed-in-run09-schema"],
            }
            (sample / "sample.json").write_text(json.dumps(record) + "\n", encoding="utf-8")
            (sample / "status.json").write_text(json.dumps({"status": "success"}) + "\n", encoding="utf-8")
        from umi_precision_reanalysis import write_sha256_manifest, reanalyze_old_run
        write_sha256_manifest(root)
        return reanalyze_old_run

    def test_old_data_reanalysis_publishes_new_revision_without_media_metrics(self):
        from umi_precision_reanalysis import reanalyze_old_run
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "run09"
            output = Path(temp) / "revision"
            source.mkdir()
            self._fixture(source)
            before = (source / "MANIFEST.sha256").read_bytes()
            result = reanalyze_old_run(source, output)
            self.assertEqual(result["status"], "COMPLETE")
            self.assertTrue((output / "MANIFEST.sha256").is_file())
            self.assertTrue((output / "metrics.json").is_file())
            self.assertEqual(before, (source / "MANIFEST.sha256").read_bytes())
            self.assertFalse(any(path.suffix.lower() in {".png", ".mp4"} for path in output.rglob("*")))
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("sign_reversal_consistency", metrics)
            self.assertIn("z0.npy", metrics["source_input_hashes"])
            self.assertFalse(metrics["precision_contract"]["observed"])
            self.assertIn("network_condition_observations", metrics["precision_contract"])
            self.assertTrue(metrics["precision_contract"]["network_condition_observations"]["scan_pre"]["observed"])
            self.assertIn("common interface_fp32", metrics["precision_contract"]["cross_group_identity_reason"])
            self.assertIsNone(metrics["vector_decomposition"])
            self.assertIn("common interface_fp32", metrics["vector_decomposition_reason"])

    def test_old_data_reanalysis_accepts_verified_known_analysis_entries_in_root_manifest(self):
        from umi_precision_reanalysis import reanalyze_old_run, sha256_file, write_sha256_manifest

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "run09"
            source.mkdir()
            self._fixture(source)
            attempt = source / "samples" / "scan_pre.attempt01"
            attempt.mkdir()
            (attempt / "partial.log").write_text("interrupted\n", encoding="utf-8")
            write_sha256_manifest(source)
            analysis_manifest = source / "MANIFEST.analysis.sha256"
            review_bundle = source / "review_bundle.zip"
            analysis_manifest.write_text("analysis manifest\n", encoding="utf-8")
            review_bundle.write_bytes(b"bundle")
            root_manifest = source / "MANIFEST.sha256"
            root_manifest.write_text(
                root_manifest.read_text(encoding="ascii")
                + f"{sha256_file(analysis_manifest)}  MANIFEST.analysis.sha256\n"
                + f"{sha256_file(review_bundle)}  review_bundle.zip\n",
                encoding="ascii",
            )
            result = reanalyze_old_run(source, Path(temp) / "revision")
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(analysis_manifest.read_text(encoding="utf-8"), "analysis manifest\n")
            self.assertEqual(review_bundle.read_bytes(), b"bundle")

    def test_old_data_reanalysis_recomputes_missing_delta_artifacts_from_raw_carriers(self):
        from umi_precision_reanalysis import reanalyze_old_run, write_sha256_manifest

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "run09"
            source.mkdir()
            self._fixture(source)
            for sample in (source / "samples").iterdir():
                if sample.is_dir():
                    for name in ("actual_delta_fp32.npy", "actual_delta_bf16.npy"):
                        path = sample / name
                        if path.exists():
                            path.unlink()
            write_sha256_manifest(source)
            result = reanalyze_old_run(source, Path(temp) / "revision")
            self.assertEqual(result["status"], "COMPLETE")
            metrics = json.loads((Path(temp) / "revision" / "metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(metrics["target_delta_recomputed"])

    def test_old_data_reanalysis_refuses_tampered_source_manifest_and_existing_output(self):
        from umi_precision_reanalysis import reanalyze_old_run
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "run09"
            source.mkdir()
            self._fixture(source)
            (source / "z0.npy").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "manifest"):
                reanalyze_old_run(source, Path(temp) / "revision")

    def test_existing_output_is_rejected_when_output_really_exists(self):
        from umi_precision_reanalysis import reanalyze_old_run

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "run09"
            output = Path(temp) / "revision"
            source.mkdir()
            self._fixture(source)
            output.mkdir()
            (output / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "existing revision"):
                reanalyze_old_run(source, output)

    def test_malicious_sample_id_cannot_escape_staging_directory(self):
        from umi_precision_reanalysis import reanalyze_old_run, write_sha256_manifest

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "run09"
            source.mkdir()
            self._fixture(source)
            sample_json = source / "samples" / "scan_pre" / "sample.json"
            record = json.loads(sample_json.read_text(encoding="utf-8"))
            record["sample_id"] = "../../escape"
            sample_json.write_text(json.dumps(record) + "\n", encoding="utf-8")
            write_sha256_manifest(source)
            with self.assertRaisesRegex(ValueError, "sample_id|unsafe"):
                reanalyze_old_run(source, Path(temp) / "revision")
            self.assertFalse((Path(temp) / "escape.npy").exists())

    def test_duplicate_sample_ids_are_rejected(self):
        from umi_precision_reanalysis import reanalyze_old_run, write_sha256_manifest

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "run09"
            source.mkdir()
            self._fixture(source)
            sample_json = source / "samples" / "dir_00_alpha_0p01_plus" / "sample.json"
            record = json.loads(sample_json.read_text(encoding="utf-8"))
            record["sample_id"] = "scan_pre"
            sample_json.write_text(json.dumps(record) + "\n", encoding="utf-8")
            write_sha256_manifest(source)
            with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
                reanalyze_old_run(source, Path(temp) / "revision")

    def test_actual_fp32_delta_must_match_realized_carrier_and_metrics_are_separate(self):
        from umi_precision_reanalysis import reanalyze_old_run, write_sha256_manifest

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "run09"
            output = Path(temp) / "revision"
            source.mkdir()
            self._fixture(source)
            sample = source / "samples" / "dir_00_alpha_0p01_plus"
            np.save(sample / "realized_carrier_fp32.npy", np.load(sample / "carrier_fp32.npy"))
            np.save(sample / "actual_delta_fp32.npy", np.zeros((1, 1, 3, 1, 2), dtype=np.float32))
            np.save(sample / "actual_delta_bf16.npy", np.zeros((1, 1, 3, 1, 2), dtype=np.float32))
            write_sha256_manifest(source)
            result = reanalyze_old_run(source, output)
            rows = json.loads((output / "metrics.json").read_text(encoding="utf-8"))["rows"]
            plus = next(row for row in rows if row["sample_id"] == "dir_00_alpha_0p01_plus")
            self.assertGreater(plus["requested_input_rms_fp32"], 0.0)
            self.assertEqual(plus["realized_input_rms_fp32"], 0.0)
            self.assertEqual(plus["effective_input_rms_bf16"], 0.0)
            self.assertEqual(plus["actual_input_rms"], 0.0)
            self.assertEqual(result["status"], "COMPLETE")

    def test_altered_actual_fp32_delta_is_rejected(self):
        from umi_precision_reanalysis import reanalyze_old_run, write_sha256_manifest

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "run09"
            source.mkdir()
            self._fixture(source)
            sample = source / "samples" / "dir_00_alpha_0p01_plus"
            np.save(sample / "actual_delta_fp32.npy", np.full((1, 1, 3, 1, 2), 7.0, dtype=np.float32))
            write_sha256_manifest(source)
            with self.assertRaisesRegex(ValueError, "actual FP32 delta"):
                reanalyze_old_run(source, Path(temp) / "revision")

    def test_raw_latent_absence_is_a_hard_error(self):
        from umi_precision_reanalysis import reanalyze_old_run, write_sha256_manifest

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "run09"
            source.mkdir()
            self._fixture(source)
            sample = source / "samples" / "scan_pre"
            (sample / "final_latent_full.npy").unlink()
            (sample / "predicted_latent.npy").unlink()
            write_sha256_manifest(source)
            with self.assertRaisesRegex(ValueError, "final_latent_full"):
                reanalyze_old_run(source, Path(temp) / "revision")

    def test_non_carrier_sampler_state_layout_is_a_hard_error(self):
        from umi_precision_reanalysis import reanalyze_old_run, write_sha256_manifest

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "run09"
            source.mkdir()
            self._fixture(source)
            sample = source / "samples" / "scan_pre"
            np.save(sample / "initial_state.npy", np.zeros((2, 2), dtype=np.float32))
            np.save(sample / "initial_condition_mask.npy", np.zeros((2, 2), dtype=bool))
            write_sha256_manifest(source)
            with self.assertRaisesRegex(ValueError, "sampler state|carrier geometry"):
                reanalyze_old_run(source, Path(temp) / "revision")

    def test_source_mutation_after_initial_read_is_caught_by_full_revalidation(self):
        import umi_precision_reanalysis as reanalysis

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "run09"
            source.mkdir()
            self._fixture(source)
            output = Path(temp) / "revision"
            original_loader = reanalysis._load_root_arrays

            def load_then_mutate(root, *args, **kwargs):
                result = original_loader(root, *args, **kwargs)
                (root / "z0.npy").write_bytes(b"mutated-after-initial-manifest")
                return result

            with mock.patch.object(reanalysis, "_load_root_arrays", side_effect=load_then_mutate):
                with self.assertRaisesRegex(ValueError, "manifest"):
                    reanalysis.reanalyze_old_run(source, output)
            self.assertFalse(output.exists())

    def test_publish_fails_if_destination_appears_during_publish(self):
        import umi_precision_reanalysis as reanalysis

        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp) / "stage"
            output = Path(temp) / "revision"
            stage.mkdir()
            (stage / "payload.txt").write_text("staged", encoding="utf-8")
            def race(src, dst):
                Path(dst).mkdir()
                (Path(dst) / "occupied.txt").write_text("racer", encoding="utf-8")
                raise FileExistsError("competing destination")

            with mock.patch.object(reanalysis, "_atomic_rename_noreplace", side_effect=race):
                with self.assertRaisesRegex(FileExistsError, "appeared|existing"):
                    reanalysis._publish_revision(stage, output)
            self.assertTrue(stage.exists())
            self.assertEqual((output / "occupied.txt").read_text(encoding="utf-8"), "racer")
            self.assertFalse((output / "metrics.json").exists())

    def test_publish_rejects_competing_empty_destination_without_partial_output(self):
        import umi_precision_reanalysis as reanalysis

        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp) / "stage"
            output = Path(temp) / "revision"
            stage.mkdir()
            (stage / "payload.txt").write_text("staged", encoding="utf-8")

            def empty_race(src, dst):
                Path(dst).mkdir()
                raise FileExistsError("competing empty destination")

            with mock.patch.object(reanalysis, "_atomic_rename_noreplace", side_effect=empty_race):
                with self.assertRaisesRegex(FileExistsError, "appeared|existing"):
                    reanalysis._publish_revision(stage, output)
            self.assertTrue(stage.exists())
            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])

    def test_manifest_rejects_duplicate_known_generated_entry_before_exclusion(self):
        from umi_precision_reanalysis import sha256_file, validate_original_manifest, write_sha256_manifest

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "run09"
            source.mkdir()
            self._fixture(source)
            generated = source / "MANIFEST.analysis.sha256"
            generated.write_text("analysis\n", encoding="utf-8")
            write_sha256_manifest(source)
            manifest = source / "MANIFEST.sha256"
            manifest.write_text(
                manifest.read_text(encoding="ascii")
                + f"{sha256_file(generated)}  MANIFEST.analysis.sha256\n"
                + f"{sha256_file(generated)}  MANIFEST.analysis.sha256\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validate_original_manifest(source, allow_known_generated_entries=True)

    def test_manifest_does_not_exclude_nested_generated_basename(self):
        from umi_precision_reanalysis import validate_original_manifest, write_sha256_manifest

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "run09"
            source.mkdir()
            self._fixture(source)
            nested = source / "nested"
            nested.mkdir()
            (nested / "review_bundle.zip").write_bytes(b"nested generated-looking file")
            write_sha256_manifest(source)
            manifest = source / "MANIFEST.sha256"
            manifest.write_text(
                "\n".join(line for line in manifest.read_text(encoding="ascii").splitlines()
                            if line.rsplit("  ", 1)[-1] != "nested/review_bundle.zip") + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "inventory|manifest"):
                validate_original_manifest(source)
