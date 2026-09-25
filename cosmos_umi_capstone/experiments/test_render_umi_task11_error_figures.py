from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from render_umi_task11_error_figures import create_light_bundle, render_figures


class RenderTask11ErrorFiguresTests(unittest.TestCase):
    def _analysis(self) -> dict:
        ensembles = {}
        for ensemble_index, ensemble in enumerate(("E1", "ETF2", "EAR2")):
            ensembles[ensemble] = {}
            for space_index, space in enumerate(("latent", "rgb")):
                folds = []
                for episode in (1, 2, 3, 6, 7, 14):
                    folds.append({
                        "held_out_episode": episode,
                        "training_k95": 8 + (episode % 2),
                        "training_centered_k95": 7 + (episode % 2),
                        "rank_results": [
                            {
                                "rank_kind": "requested",
                                "requested_rank": rank,
                                "episode_mean": {
                                    "raw_relative_residual": 1.0 - rank / 100 + episode / 10000 + ensemble_index / 100,
                                    "centered_relative_residual": 1.01 - rank / 100 + episode / 10000 + space_index / 100,
                                },
                            }
                            for rank in (1, 2, 4, 8, 10)
                        ],
                    })
                ensembles[ensemble][space] = {
                    "spectrum": {
                        "cumulative_squared_energy": [i / 12 for i in range(1, 13)],
                        "truncation_relative_frobenius_residual_by_rank": [1.0 - i / 13 for i in range(13)],
                    },
                    "leave_one_episode_out": {
                        "folds": folds,
                        "six_episode_summary_of_episode_means": [
                            {
                                "rank_kind": "requested",
                                "requested_rank": rank,
                                "raw_relative_residual": 1.0 - rank / 100 + ensemble_index / 100,
                                "centered_relative_residual": 1.01 - rank / 100 + space_index / 100,
                            }
                            for rank in (1, 2, 4, 8, 10)
                        ] + [{
                            "rank_kind": "training_k95",
                            "requested_rank": None,
                            "raw_relative_residual": 0.5,
                            "centered_relative_residual": 0.6,
                        }],
                    },
                }
        return {
            "schema": "umi-task11-error-analysis-v1",
            "status": "COMPLETE",
            "source": {"manifest_sha256": "a" * 64},
            "definitions": {"loeo": "LOEO definition"},
            "ensembles": ensembles,
        }

    def test_renders_png_svg_and_episode_level_hash_bound_diagnostics_without_matrices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis = root / "analysis.json"
            analysis.write_text(json.dumps(self._analysis()), encoding="utf-8")
            output = root / "figures"

            manifest = render_figures(analysis, output)

            expected = {
                "cumulative_energy.png", "cumulative_energy.svg",
                "truncation_norm_residual.png", "truncation_norm_residual.svg",
                "loeo_episode_residuals.png", "loeo_episode_residuals.svg",
            }
            self.assertEqual(set(manifest["files"]), expected)
            self.assertEqual(manifest["source_analysis_sha256"], hashlib.sha256(analysis.read_bytes()).hexdigest())
            self.assertEqual(manifest["figure_source_data_sha256"], hashlib.sha256((output / "figure_source_data.json").read_bytes()).hexdigest())
            source_data = json.loads((output / "figure_source_data.json").read_text(encoding="utf-8"))
            self.assertEqual(source_data["source_analysis_sha256"], manifest["source_analysis_sha256"])
            points = source_data["loeo_fixed_ranks"]["E1"]["latent"][0]["episode_points"]
            self.assertEqual(len(points), 6)
            self.assertEqual(points[0]["held_out_episode"], 1)
            self.assertIn("six episode points plus descriptive mean", manifest["chart_map"]["loeo_episode_residuals"])
            caption = (output / "figure_caption.md").read_text(encoding="utf-8")
            self.assertIn("E1 is the G0 prediction", caption)
            self.assertIn("requested rank 10 is numerically capped", caption)
            self.assertGreater((output / "cumulative_energy.png").stat().st_size, 1000)
            self.assertGreater((output / "loeo_episode_residuals.svg").stat().st_size, 1000)
            self.assertEqual(json.loads((output / "figure_manifest.json").read_text(encoding="utf-8")), manifest)
            with self.assertRaises(FileExistsError):
                render_figures(analysis, output)

    def test_rejects_noncompleted_or_wrong_schema_input_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis = root / "analysis.json"
            analysis.write_text(json.dumps({"schema": "other", "status": "COMPLETE"}), encoding="utf-8")
            output = root / "figures"
            with self.assertRaises(ValueError):
                render_figures(analysis, output)
            self.assertFalse(output.exists())

    def test_light_bundle_is_manifest_checked_and_omits_large_matrices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = root / "attempt_01"
            attempt.mkdir()
            (root / "run_identity.json").write_text("{}\n", encoding="utf-8")
            (root / "run_status.json").write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")
            source_files = {
                "analysis.json": json.dumps(self._analysis()),
                "episode_column_means.csv": "record_index\n1\n",
                "error_columns.csv": "record_index\n1\n",
                "error_decomposition.csv": "record_index\n1\n",
                "review_report.md": "summary\n",
            }
            for name, content in source_files.items():
                (attempt / name).write_text(content, encoding="utf-8")
            matrix_hashes = {}
            for index in range(6):
                name = f"errors_{index}.npy"
                payload = f"large-matrix-{index}".encode("ascii")
                (attempt / name).write_bytes(payload)
                matrix_hashes[name] = hashlib.sha256(payload).hexdigest()
            manifest_rows = {
                name: hashlib.sha256((attempt / name).read_bytes()).hexdigest()
                for name in source_files
            }
            manifest_rows.update(matrix_hashes)
            (attempt / "MANIFEST.sha256").write_text(
                "".join(f"{digest}  {name}\n" for name, digest in sorted(manifest_rows.items())),
                encoding="ascii",
            )
            analysis = attempt / "analysis.json"
            figure_dir = root / "figures"
            render_figures(analysis, figure_dir)

            bundle = create_light_bundle(attempt, figure_dir)

            self.assertEqual(bundle["source_analysis_json_sha256"], manifest_rows["analysis.json"])
            self.assertEqual(len(bundle["excluded_remote_only_error_matrices"]), 6)
            with ZipFile(figure_dir / "task11_true_error_light_bundle.zip") as archive:
                names = archive.namelist()
                self.assertIn("light_bundle_manifest.json", names)
                self.assertIn("run_status.json", names)
                self.assertIn("figures/cumulative_energy.svg", names)
                self.assertIn("figures/cumulative_energy.png", names)
                self.assertIn("figures/figure_source_data.json", names)
                self.assertIn("figures/figure_caption.md", names)
                self.assertFalse(any(name.endswith(".npy") for name in names))


if __name__ == "__main__":
    unittest.main()
