"""Synthetic CPU tests for the standalone Task 11 horizon figure renderer."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from render_umi_task11_horizon_figures import build_figure_source_data, render_figures


SCHEDULES = [[11, 12, 13, 14, 15], [21, 22, 23, 24, 25]]


def _analysis_fixture() -> dict:
    endpoints = []
    for schedule_index, schedule in enumerate(SCHEDULES):
        for horizon in range(1, 6):
            modes = ("G0",) if horizon == 1 else ("TF", "AR")
            for mode in modes:
                if horizon == 1:
                    rgb_rmse = 0.1 + schedule_index * 0.02
                    latent_rms = 0.3 + schedule_index * 0.04
                elif mode == "TF":
                    rgb_rmse = 0.2 * horizon + schedule_index * 0.1
                    latent_rms = 0.5 + horizon * 0.1 + schedule_index * 0.03
                else:
                    rgb_rmse = 0.18 * horizon + schedule_index * 0.12
                    latent_rms = 0.48 + horizon * 0.1 + schedule_index * 0.04
                endpoints.append({
                    "sample_id": f"s{schedule_index}-h{horizon}-{mode}",
                    "schedule_index": schedule_index,
                    "seed": schedule[horizon - 1],
                    "horizon_chunks": horizon,
                    "endpoint_frame_index": horizon * 16,
                    "mode": mode,
                    "rmse": rgb_rmse,
                    "mae": rgb_rmse * 0.8,
                    "psnr": 18.0 + horizon + schedule_index,
                    "psnr_status": "FINITE",
                    "condition_latent_rms": latent_rms,
                    "condition_latent_cosine": 0.9,
                    "action_hash": f"action-{schedule_index}-{horizon}-{mode}",
                    "noise_hash": f"noise-{schedule_index}-{horizon}-{mode}",
                })

    means = []
    for horizon in range(1, 6):
        for mode in (("G0",) if horizon == 1 else ("TF", "AR")):
            selected = [r for r in endpoints if r["horizon_chunks"] == horizon and r["mode"] == mode]
            means.append({
                "horizon_chunks": horizon,
                "mode": mode,
                "seed_count": len(selected),
                "rmse": sum(r["rmse"] for r in selected) / 2,
                "mae": sum(r["mae"] for r in selected) / 2,
                "psnr": sum(r["psnr"] for r in selected) / 2,
                "condition_latent_rms": sum(r["condition_latent_rms"] for r in selected) / 2,
                "condition_latent_cosine": 0.9,
                "finite_psnr_seed_count": 2,
            })

    deltas = []
    for schedule_index in range(2):
        for horizon in range(2, 6):
            tf = next(r for r in endpoints if r["schedule_index"] == schedule_index and r["horizon_chunks"] == horizon and r["mode"] == "TF")
            ar = next(r for r in endpoints if r["schedule_index"] == schedule_index and r["horizon_chunks"] == horizon and r["mode"] == "AR")
            deltas.append({
                "schedule_index": schedule_index,
                "horizon_chunks": horizon,
                "seed": ar["seed"],
                "rgb_rmse_ar_minus_tf": ar["rmse"] - tf["rmse"],
                "condition_latent_rms_ar_minus_tf": ar["condition_latent_rms"] - tf["condition_latent_rms"],
            })

    geometry_rgb = []
    geometry_latent = []
    for schedule_index in range(2):
        for horizon in range(2, 6):
            tf_rgb = next(r for r in endpoints if r["schedule_index"] == schedule_index and r["horizon_chunks"] == horizon and r["mode"] == "TF")["rmse"]
            ar_rgb = next(r for r in endpoints if r["schedule_index"] == schedule_index and r["horizon_chunks"] == horizon and r["mode"] == "AR")["rmse"]
            rgb_change = ar_rgb - tf_rgb
            rgb_cross = 2.0 * tf_rgb * rgb_change
            rgb_change_squared = rgb_change * rgb_change
            rgb_tf_squared = tf_rgb * tf_rgb
            rgb_ar_squared = ar_rgb * ar_rgb
            rgb = {
                "schedule_index": schedule_index,
                "seed": SCHEDULES[schedule_index][horizon - 1],
                "horizon_chunks": horizon,
                "tf_squared_error": rgb_tf_squared,
                "ar_squared_error": rgb_ar_squared,
                "cross_term_2dot_over_n": rgb_cross,
                "feedback_change_squared": rgb_change_squared,
                "squared_error_difference": rgb_ar_squared - rgb_tf_squared,
                "identity_residual": (rgb_ar_squared - rgb_tf_squared) - (rgb_cross + rgb_change_squared),
                "error_feedback_cosine": 0.1,
                "feedback_change_rmse": abs(rgb_change),
            }
            tf_latent = next(r for r in endpoints if r["schedule_index"] == schedule_index and r["horizon_chunks"] == horizon and r["mode"] == "TF")["condition_latent_rms"]
            ar_latent = next(r for r in endpoints if r["schedule_index"] == schedule_index and r["horizon_chunks"] == horizon and r["mode"] == "AR")["condition_latent_rms"]
            latent_change = ar_latent - tf_latent
            latent_cross = 2.0 * tf_latent * latent_change
            latent_change_squared = latent_change * latent_change
            latent_tf_squared = tf_latent * tf_latent
            latent_ar_squared = ar_latent * ar_latent
            latent = {
                "schedule_index": schedule_index,
                "seed": SCHEDULES[schedule_index][horizon - 1],
                "horizon_chunks": horizon,
                "tf_squared_error": latent_tf_squared,
                "ar_squared_error": latent_ar_squared,
                "cross_term_2dot_over_n": latent_cross,
                "feedback_change_squared": latent_change_squared,
                "squared_error_difference": latent_ar_squared - latent_tf_squared,
                "identity_residual": (latent_ar_squared - latent_tf_squared) - (latent_cross + latent_change_squared),
                "error_feedback_cosine": 0.1,
                "feedback_change_rmse": abs(latent_change),
            }
            latent["masked_element_count"] = 4096
            geometry_rgb.append(rgb)
            geometry_latent.append(latent)

    return {
        "status": "PASS",
        "record_index": 15,
        "run_identity_sha256": "a" * 64,
        "seed_schedules": SCHEDULES,
        "interpretation": "two preregistered seeds; means are descriptive, not population confidence intervals",
        "endpoint_metrics_per_seed": endpoints,
        "descriptive_two_seed_means": means,
        "tf_ar_endpoint_error_deltas": deltas,
        "tf_ar_error_geometry_rgb": geometry_rgb,
        "tf_ar_error_geometry_condition_masked_latent": geometry_latent,
        "primary_horizon_axis": "generated chunk index 1 through 5",
        "seconds_axis": "nominally inferred at 5 Hz; source has no actual timestamp feature",
    }


class HorizonFigureSourceTests(unittest.TestCase):
    def test_source_rows_keep_one_g0_per_schedule_and_same_schedule_pairs(self) -> None:
        source = build_figure_source_data(_analysis_fixture(), analysis_sha256="b" * 64)

        endpoints = source["endpoint_rows"]
        self.assertEqual(len(endpoints), 18)
        self.assertEqual(sum(row["horizon_chunks"] == 1 for row in endpoints), 2)
        self.assertEqual(sum(row["horizon_chunks"] == 1 and row["mode"] in {"TF", "AR"} for row in endpoints), 0)
        self.assertEqual(len(source["paired_delta_rows"]), 8)
        self.assertEqual(len(source["decomposition_rows"]["rgb"]), 8)
        self.assertEqual(len(source["decomposition_rows"]["condition_masked_latent"]), 8)
        self.assertEqual(source["scope"]["episode_count"], 1)
        self.assertEqual(source["scope"]["seed_schedule_count"], 2)

        first_delta = next(row for row in source["paired_delta_rows"] if row["schedule_index"] == 0 and row["horizon_chunks"] == 2)
        self.assertAlmostEqual(first_delta["rgb_rmse_ar_minus_tf"], -0.04)
        rgb_geometry = source["decomposition_rows"]["rgb"][0]
        self.assertAlmostEqual(rgb_geometry["squared_error_difference"], rgb_geometry["cross_term_2dot_over_n"] + rgb_geometry["feedback_change_squared"] + rgb_geometry["identity_residual"])

    def test_rejects_missing_mode_and_bad_seed_pairing(self) -> None:
        data = _analysis_fixture()
        data["endpoint_metrics_per_seed"].pop()
        with self.assertRaisesRegex(ValueError, "endpoint rows"):
            build_figure_source_data(data, analysis_sha256="b" * 64)

        data = _analysis_fixture()
        data["tf_ar_endpoint_error_deltas"][0]["seed"] = 999
        with self.assertRaisesRegex(ValueError, "paired delta"):
            build_figure_source_data(data, analysis_sha256="b" * 64)

    def test_rejects_inconsistent_mean_or_error_geometry(self) -> None:
        data = _analysis_fixture()
        data["descriptive_two_seed_means"][1]["rmse"] += 1.0
        with self.assertRaisesRegex(ValueError, "descriptive mean"):
            build_figure_source_data(data, analysis_sha256="b" * 64)

        data = _analysis_fixture()
        data["tf_ar_error_geometry_rgb"][0]["squared_error_difference"] += 0.5
        with self.assertRaisesRegex(ValueError, "error geometry"):
            build_figure_source_data(data, analysis_sha256="b" * 64)

    def test_rejects_geometry_consistent_but_disconnected_from_endpoint_metrics(self) -> None:
        data = _analysis_fixture()
        geometry = data["tf_ar_error_geometry_rgb"][0]
        geometry["tf_squared_error"] += 0.5
        geometry["ar_squared_error"] += 0.5

        with self.assertRaisesRegex(ValueError, "does not match its endpoint RMSE"):
            build_figure_source_data(data, analysis_sha256="b" * 64)

    def test_renderer_writes_static_exports_caption_source_and_manifest(self) -> None:
        data = _analysis_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis_path = root / "task11_horizon_metrics.json"
            analysis_path.write_text(json.dumps(data), encoding="utf-8")
            output_dir = root / "figures"

            result = render_figures(analysis_path, output_dir)

            self.assertEqual(result["status"], "PASS")
            for stem in ("endpoint_error", "paired_rmse_delta", "mse_error_geometry"):
                self.assertTrue((output_dir / f"{stem}.png").is_file())
                self.assertTrue((output_dir / f"{stem}.svg").is_file())
            caption = (output_dir / "figure_caption.md").read_text(encoding="utf-8").lower()
            self.assertIn("one episode", caption)
            self.assertIn("two seed schedules", caption)
            self.assertIn("future-pose", caption)
            self.assertIn("image-condition latent mask", caption)
            self.assertIn("no population", caption)
            self.assertIn("no exponential", caption)
            source = json.loads((output_dir / "figure_source_data.json").read_text(encoding="utf-8"))
            self.assertEqual(source["source_analysis_sha256"], result["source_analysis_sha256"])
            self.assertTrue((output_dir / "figure_manifest.json").is_file())

    def test_refuses_existing_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis_path = root / "task11_horizon_metrics.json"
            analysis_path.write_text(json.dumps(_analysis_fixture()), encoding="utf-8")
            output_dir = root / "figures"
            output_dir.mkdir()
            sentinel = output_dir / "keep.txt"
            sentinel.write_text("untouched", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                render_figures(analysis_path, output_dir)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "untouched")


if __name__ == "__main__":
    unittest.main()
