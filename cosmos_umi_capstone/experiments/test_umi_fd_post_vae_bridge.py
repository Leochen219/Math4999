import copy
import unittest

import numpy as np

from umi_fd_post_vae_bridge import (
    CachedConditioningData,
    ConditionTokenCapture,
    HookBoundary,
    build_broadcast_condition_mask,
    capture_final_latent,
    construct_delta,
    generate_direction_bank,
    hash_predicted_noisy_region,
    perturbation_metrics,
    predicted_positions,
    sha256_array,
    validate_cache_off,
    validate_condition_tokens,
    validate_decoded_output_array,
    validate_runtime_setup,
)


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.carrier_shape = (1, 2, 5, 2, 2)
        self.indexes = [0, 2]
        self.packed = np.zeros((1, 5, 2, 2), dtype=bool)
        self.packed[:, 0] = True
        self.packed[:, 2] = True

    def test_runtime_indexes_and_packed_mask_agree_and_broadcast(self):
        mask = build_broadcast_condition_mask(self.indexes, self.packed, self.carrier_shape)
        self.assertEqual(mask.shape, self.carrier_shape)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertTrue(np.all(mask[:, :, 0]))
        self.assertTrue(np.all(mask[:, :, 2]))
        self.assertFalse(np.any(mask[:, :, 1]))
        np.testing.assert_array_equal(predicted_positions(mask), ~mask)

    def test_mask_rejects_empty_malformed_and_disagreeing_sources(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            build_broadcast_condition_mask([], self.packed, self.carrier_shape)
        with self.assertRaisesRegex(ValueError, "shape"):
            build_broadcast_condition_mask(self.indexes, np.zeros((5, 2), dtype=bool), self.carrier_shape)
        disagree = self.packed.copy()
        disagree[:, 2] = False
        with self.assertRaisesRegex(ValueError, "agree"):
            build_broadcast_condition_mask(self.indexes, disagree, self.carrier_shape)
        nonfinite = self.packed.astype(np.float32)
        nonfinite[0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            build_broadcast_condition_mask(self.indexes, nonfinite, self.carrier_shape)


class DirectionTests(unittest.TestCase):
    def test_directions_are_reproducible_unique_mask_only_and_unit_rms_without_demeaning(self):
        mask = np.zeros((2, 3, 3), dtype=bool)
        mask[:, :2, :] = True
        first = generate_direction_bank(4, mask.shape, mask=mask, seed=20260912)
        second = generate_direction_bank(4, mask.shape, mask=mask, seed=20260912)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.dtype, np.float32)
        self.assertEqual(len({sha256_array(item) for item in first}), 4)
        for direction in first:
            self.assertAlmostEqual(float(np.sqrt(np.mean(direction[mask].astype(np.float64) ** 2))), 1.0, places=6)
            self.assertTrue(np.all(direction[~mask] == 0))
        self.assertTrue(any(abs(float(item[mask].mean())) > 1e-3 for item in first))


class PerturbationTests(unittest.TestCase):
    def test_plus_minus_use_immutable_baseline_and_report_float_metrics(self):
        z0 = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 2, 2) / 10
        mask = np.zeros_like(z0, dtype=bool)
        mask[:, :, 0] = True
        direction = generate_direction_bank(1, z0.shape, mask=mask, seed=20260912)[0]
        baseline = z0.copy()
        plus = construct_delta(z0, mask, direction, alpha=0.01, sign=1)
        minus = construct_delta(z0, mask, direction, alpha=0.01, sign=-1)
        np.testing.assert_array_equal(z0, baseline)
        np.testing.assert_array_equal(plus.delta[~mask], 0)
        np.testing.assert_array_equal(minus.delta[~mask], 0)
        np.testing.assert_array_equal(plus.latent[mask] - z0[mask], -(minus.latent[mask] - z0[mask]))
        metrics = perturbation_metrics(z0, plus.delta, mask=mask, direction=direction, alpha=0.01)
        self.assertIn("target_rms_fp32", metrics)
        self.assertIn("actual_rms_bf16", metrics)
        self.assertIn("relative_amplitude", metrics)
        self.assertIn("outside_mask_exact", metrics)
        self.assertTrue(metrics["outside_mask_exact"])

    def test_invalid_s_z_is_rejected(self):
        z0 = np.zeros((2, 2), dtype=np.float32)
        mask = np.ones_like(z0, dtype=bool)
        direction = np.ones_like(z0)
        with self.assertRaisesRegex(ValueError, "s_z"):
            construct_delta(z0, mask, direction, alpha=0.1, sign=1)

    def test_fp32_metrics_measure_realized_addition_not_requested_delta(self):
        baseline = np.array([1e8], dtype=np.float32)
        mask = np.ones_like(baseline, dtype=bool)
        direction = np.ones_like(baseline)
        requested = construct_delta(baseline, mask, direction, alpha=1e-8, sign=1)

        metrics = perturbation_metrics(
            baseline,
            requested.delta,
            mask=mask,
            direction=direction,
            alpha=1e-8,
        )

        self.assertEqual(metrics["target_rms_fp32"], 1.0)
        self.assertEqual(metrics["actual_rms_fp32"], 0.0)
        self.assertEqual(metrics["nonzero_ratio"], 0.0)
        self.assertIsNone(metrics["direction_cosine"])


class HashAndCaptureTests(unittest.TestCase):
    def test_predicted_hash_ignores_condition_changes_but_detects_noise_changes(self):
        state = np.arange(20, dtype=np.float32).reshape(1, 2, 5, 2)
        mask = np.zeros_like(state, dtype=bool)
        mask[:, :, 0] = True
        altered_condition = state.copy()
        altered_condition[mask] += 99
        altered_noise = state.copy()
        altered_noise[0, 1, 1, 0] += 1
        self.assertEqual(hash_predicted_noisy_region(state, mask), hash_predicted_noisy_region(altered_condition, mask))
        self.assertNotEqual(hash_predicted_noisy_region(state, mask), hash_predicted_noisy_region(altered_noise, mask))

    def test_final_latent_capture_excludes_condition_positions(self):
        latent = np.arange(1 * 2 * 5 * 2 * 2, dtype=np.float32).reshape(1, 2, 5, 2, 2)
        mask = np.zeros_like(latent, dtype=bool)
        mask[:, :, [0, 2]] = True
        result = capture_final_latent(latent, mask)
        self.assertEqual(result["full"].dtype, np.float32)
        self.assertEqual(result["predicted_only"].shape, (1, 2, 3, 2, 2))
        np.testing.assert_array_equal(result["predicted_only"], latent[:, :, [1, 3, 4]])

    def test_final_latent_capture_rejects_spatially_partial_condition_frames(self):
        latent = np.zeros((1, 1, 2, 2, 2), dtype=np.float32)
        mask = np.zeros_like(latent, dtype=bool)
        mask[0, 0, 0, 0, 0] = True
        with self.assertRaisesRegex(ValueError, "framewise"):
            capture_final_latent(latent, mask)


class HookAndContractTests(unittest.TestCase):
    def test_condition_tokens_persistence_detects_later_step_overwrite(self):
        capture = ConditionTokenCapture(expected=np.array([1, 2], dtype=np.float32), cast_dtype=np.float32)
        capture.record(0, np.array([1, 2], dtype=np.float32))
        capture.record(1, np.array([9, 2], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "conditioned tokens"):
            capture.assert_persistent()

    def test_condition_tokens_persistence_requires_observations(self):
        capture = ConditionTokenCapture(expected=np.array([1, 2], dtype=np.float32), cast_dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "observations"):
            capture.assert_persistent()
        with self.assertRaisesRegex(ValueError, "observations"):
            validate_condition_tokens(np.array([1, 2], dtype=np.float32), [])

    def test_cached_data_clones_once_and_zero_override_injects(self):
        calls = []
        injections = []
        source = {"latent": np.arange(4, dtype=np.float32), "carrier": ["carrier"]}

        def original():
            calls.append(1)
            return copy.deepcopy(source)

        contract = CachedConditioningData(original)
        first = contract.get()
        first["latent"][0] = 99
        second = contract.get()
        self.assertEqual(len(calls), 1)
        self.assertEqual(second["latent"][0], 0)
        explicit = contract.get(delta=np.zeros(4, dtype=np.float32), inject=lambda item, delta: injections.append(delta.copy()) or item)
        self.assertEqual(len(injections), 1)
        np.testing.assert_array_equal(injections[0], 0)
        self.assertEqual(explicit["latent"][0], 0)

    def test_hook_boundary_restores_on_success_and_failure(self):
        target = type("Target", (), {"value": "original"})()
        boundary = HookBoundary(target, "value", lambda: "hooked")
        with boundary:
            self.assertEqual(target.value, "hooked")
        self.assertEqual(target.value, "original")
        with self.assertRaises(RuntimeError):
            with boundary:
                raise RuntimeError("failure")
        self.assertEqual(target.value, "original")


class RuntimeValidationTests(unittest.TestCase):
    def test_cache_off_validator_rejects_requested_or_installed_cache(self):
        validate_cache_off(False, False)
        with self.assertRaisesRegex(ValueError, "cache"):
            validate_cache_off(True, False)
        with self.assertRaisesRegex(ValueError, "cache"):
            validate_cache_off(False, True)

    def test_runtime_validator_requires_official_settings(self):
        good = {"checkpoint_path": "/model", "sampler": "unipc", "precision": "bfloat16", "diffusion_cache_requested": False, "diffusion_cache_installed": False}
        validate_runtime_setup(good, expected_checkpoint="/model")
        bad = dict(good, sampler="ddim")
        with self.assertRaisesRegex(ValueError, "sampler"):
            validate_runtime_setup(bad)

    def test_decoded_float_output_is_validated_and_final_frame_selected(self):
        output = np.linspace(0, 1, 3 * 5 * 2 * 2, dtype=np.float32).reshape(3, 5, 2, 2)
        full, final = validate_decoded_output_array(output, expected_frames=5)
        np.testing.assert_array_equal(full, output)
        np.testing.assert_array_equal(final, output[:, -1])


if __name__ == "__main__":
    unittest.main()
