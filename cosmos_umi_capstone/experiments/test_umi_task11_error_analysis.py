"""CPU contracts for Task 11 true-error geometry and error decomposition."""
from __future__ import annotations

import unittest

import numpy as np

from umi_task11_error_analysis import (
    error_decomposition,
    error_spectrum,
    leave_one_episode_out,
    prediction_error,
    rgb_endpoint_metrics,
)


class PredictionErrorTests(unittest.TestCase):
    def test_error_is_float64_prediction_minus_aligned_truth_inside_mask(self):
        prediction = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        truth = np.array([[[0.5, 1.0], [2.0, 8.0]]], dtype=np.float32)
        mask = np.array([[[True, False], [True, True]]])
        result = prediction_error(prediction, truth, mask=mask)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_array_equal(result, np.array([0.5, 1.0, -4.0], dtype=np.float64))

    def test_prediction_difference_promotes_before_subtraction_for_error_geometry(self):
        tf = np.array([1.0], dtype=np.float32)
        ar = np.array([100_000_000.0], dtype=np.float32)
        truth = np.array([0.0], dtype=np.float32)
        feedback = prediction_error(ar, tf)
        self.assertEqual(feedback[0], 99_999_999.0)
        self.assertEqual((ar - tf)[0], 100_000_000.0)  # FP32 subtraction loses the unit delta.


class ErrorSpectrumTests(unittest.TestCase):
    def test_known_rank_two_spectrum_retains_amplitude_and_truncation_residual(self):
        matrix = np.zeros((4, 12), dtype=np.float64)
        matrix[0, 0] = 3.0
        matrix[1, 1] = 2.0
        result = error_spectrum(matrix)
        self.assertEqual(result["numerical_rank"], 2)
        np.testing.assert_allclose(result["singular_values"][:2], [3.0, 2.0], rtol=0, atol=1e-13)
        self.assertAlmostEqual(result["cumulative_squared_energy"][0], 9.0 / 13.0, places=13)
        self.assertEqual(result["k90"], 2)
        self.assertEqual(result["k95"], 2)
        self.assertAlmostEqual(result["effective_rank"], np.exp(-9 / 13 * np.log(9 / 13) - 4 / 13 * np.log(4 / 13)), places=13)
        self.assertAlmostEqual(result["truncation_relative_frobenius_residual_by_rank"][1], 2.0 / np.sqrt(13.0), places=13)
        self.assertEqual(result["column_count"], 12)

    def test_zero_error_matrix_and_zero_columns_are_reported_as_not_applicable(self):
        result = error_spectrum(np.zeros((5, 12), dtype=np.float64))
        self.assertEqual(result["status"], "ZERO_ERROR_MATRIX")
        self.assertEqual(result["numerical_rank"], 0)
        self.assertIsNone(result["k95"])
        self.assertIsNone(result["effective_rank"])
        self.assertTrue(all(value is None for value in result["truncation_relative_frobenius_residual_by_rank"]))


class LeaveOneEpisodeOutTests(unittest.TestCase):
    def test_both_seed_columns_are_held_out_and_training_mean_excludes_them(self):
        episode_ids = np.repeat(np.arange(6), 2)
        matrix = np.zeros((3, 12), dtype=np.float64)
        for column in range(12):
            matrix[0, column] = float(column + 1)
            matrix[1, column] = float((column + 1) ** 2)
        matrix[:, :2] += 1000.0
        result = leave_one_episode_out(matrix, episode_ids, ranks=(1, 2, 4, 8, 10))
        fold = next(item for item in result["folds"] if item["held_out_episode"] == 0)
        train_columns = [index for index, episode in enumerate(episode_ids) if episode != 0]
        np.testing.assert_array_equal(fold["training_column_indices"], train_columns)
        expected_mean = matrix[:, train_columns].mean(axis=1)
        from umi_task11_error_analysis import array_sha256
        self.assertEqual(fold["training_mean_sha256"], array_sha256(expected_mean))
        first_rank = next(row for row in fold["rank_results"] if row["requested_rank"] == 1)
        largest_rank = next(row for row in fold["rank_results"] if row["requested_rank"] == 10)
        self.assertTrue(largest_rank["rank_capped"])
        self.assertLessEqual(first_rank["effective_rank_uncentered"], fold["training_numerical_rank"])
        self.assertEqual(len(first_rank["seed_rows"]), 2)
        seed = first_rank["seed_rows"][0]
        error = matrix[:, seed["column_index"]]
        basis = np.linalg.svd(matrix[:, train_columns], full_matrices=False)[0][:, :1]
        self.assertAlmostEqual(seed["raw_relative_residual"],
                               np.linalg.norm(error - basis @ (basis.T @ error)) /
                               np.linalg.norm(error), places=12)

    def test_requested_ranks_are_capped_to_numerical_rank_and_centered_fold_uses_train_mean(self):
        episode_ids = np.repeat(np.arange(6), 2)
        matrix = np.zeros((4, 12), dtype=np.float64)
        matrix[0] = np.arange(1, 13, dtype=np.float64)
        matrix[1] = 2.0 * matrix[0]
        matrix[:, :2] += np.array([[5.0], [0.0], [0.0], [0.0]])
        result = leave_one_episode_out(matrix, episode_ids, ranks=(1, 2, 4, 8, 10))
        fold = next(item for item in result["folds"] if item["held_out_episode"] == 0)
        for row in fold["rank_results"]:
            self.assertLessEqual(row["effective_rank_uncentered"], fold["training_numerical_rank"])
            self.assertLessEqual(row["effective_rank_centered"], fold["training_centered_numerical_rank"])
            if row["requested_rank"] in (2, 4, 8, 10):
                self.assertTrue(row["rank_capped"])
        row = next(item for item in fold["rank_results"] if item["requested_rank"] == 1)
        train = [index for index, episode in enumerate(episode_ids) if episode != 0]
        expected_mean = matrix[:, train].mean(axis=1)
        from umi_task11_error_analysis import array_sha256
        self.assertEqual(fold["training_mean_sha256"], array_sha256(expected_mean))
        heldout = matrix[:, 0]
        expected_centered_target = heldout - expected_mean
        self.assertEqual(row["seed_rows"][0]["centered_target_norm"], np.linalg.norm(expected_centered_target))
        centered_train = matrix[:, train] - expected_mean[:, None]
        basis = np.linalg.svd(centered_train, full_matrices=False)[0][:, :1]
        expected_centered_residual = np.linalg.norm(expected_centered_target - basis @ (basis.T @ expected_centered_target)) / np.linalg.norm(expected_centered_target)
        self.assertAlmostEqual(row["seed_rows"][0]["centered_relative_residual"], expected_centered_residual, places=13)
        self.assertAlmostEqual(
            row["seed_rows"][0]["raw_reconstruction_relative_residual"],
            row["seed_rows"][0]["centered_reconstruction_error_norm"] / np.linalg.norm(heldout),
            places=13,
        )

    def test_rank_zero_and_zero_heldout_columns_return_na_residuals(self):
        episode_ids = np.repeat(np.arange(6), 2)
        result = leave_one_episode_out(np.zeros((3, 12)), episode_ids, ranks=(1, 2))
        fold = result["folds"][0]
        self.assertEqual(fold["training_numerical_rank"], 0)
        self.assertEqual(fold["training_k95"], None)
        self.assertIsNone(fold["training_centered_k95"])
        for row in fold["rank_results"]:
            self.assertEqual(row["effective_rank_uncentered"], 0)
            self.assertIsNone(row["seed_rows"][0]["raw_relative_residual"])
            self.assertIsNone(row["seed_rows"][0]["centered_relative_residual"])
            self.assertIsNone(row["seed_rows"][0]["raw_reconstruction_relative_residual"])

    def test_adaptive_training_k95_is_summarized_across_all_six_folds(self):
        episode_ids = np.repeat(np.arange(6), 2)
        matrix = np.zeros((4, 12), dtype=np.float64)
        matrix[0] = [100, 100, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        matrix[1] = [0, 0, 10, 10, 2, 2, 2, 2, 2, 2, 2, 2]
        matrix[2] = [0, 0, 0, 0, 5, 5, 1, 1, 1, 1, 1, 1]
        result = leave_one_episode_out(matrix, episode_ids)
        summaries = result["six_episode_summary_of_episode_means"]
        for kind in ("training_k95", "training_centered_k95"):
            row = next(item for item in summaries if item["rank_kind"] == kind)
            self.assertEqual(row["episode_count"], 6)
            self.assertEqual(len(row["selected_ranks_by_episode"]), 6)
            self.assertGreaterEqual(row["selected_rank_max"], row["selected_rank_min"])


class ErrorDecompositionTests(unittest.TestCase):
    def test_feedback_error_decomposition_identity_and_cosine(self):
        result = error_decomposition(np.array([1.0, 2.0]), np.array([2.0, -1.0]),
                                     combined_error=np.array([3.0, 1.0]))
        self.assertAlmostEqual(result["baseline_mse"], 2.5, places=14)
        self.assertAlmostEqual(result["feedback_change_mse"], 2.5, places=14)
        self.assertAlmostEqual(result["cross_term_2dot_over_n"], 0.0, places=14)
        self.assertAlmostEqual(result["combined_error_mse"], 5.0, places=14)
        self.assertAlmostEqual(result["squared_error_difference"], 2.5, places=14)
        self.assertAlmostEqual(result["identity_residual"], 0.0, places=14)
        self.assertAlmostEqual(result["cosine"], 0.0, places=14)

    def test_identity_residual_compares_decomposition_to_independent_combined_error(self):
        result = error_decomposition(np.array([1.0, 2.0]), np.array([2.0, -1.0]),
                                     combined_error=np.array([3.25, 1.0]))
        self.assertNotEqual(result["identity_residual"], 0.0)
        self.assertAlmostEqual(result["squared_error_difference"], 3.28125, places=14)

    def test_zero_feedback_or_baseline_vector_has_na_cosine(self):
        self.assertIsNone(error_decomposition(np.ones(4), np.zeros(4))["cosine"])
        self.assertIsNone(error_decomposition(np.zeros(4), np.ones(4))["cosine"])


class EndpointMetricTests(unittest.TestCase):
    def test_zero_endpoint_error_has_zero_rmse_and_infinite_psnr(self):
        truth = np.full((3, 8, 8), 0.25, dtype=np.float32)
        result = rgb_endpoint_metrics(truth, truth.copy())
        self.assertEqual(result["rmse"], 0.0)
        self.assertEqual(result["mae"], 0.0)
        self.assertTrue(np.isinf(result["psnr_db"]))


if __name__ == "__main__":
    unittest.main()
