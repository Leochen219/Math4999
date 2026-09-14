"""Read-only Task 4 provenance guards required before Task 5 launches."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class Task5LauncherTests(unittest.TestCase):
    def test_public_launchers_require_explicit_environment_paths(self):
        import run_umi_precision_experiment as task4
        import run_umi_task5_experiment as task5

        task4_args = task4.parse_args([])
        for field in ("framework_root", "checkpoint_path", "old_run09", "run_root"):
            self.assertIsNone(getattr(task4_args, field), field)

        task5_args = task5.parse_args([])
        for field in ("framework_root", "checkpoint_path", "old_run09", "task4_run", "run_root"):
            self.assertIsNone(getattr(task5_args, field), field)

    def test_task4_fails_closed_when_required_paths_are_omitted(self):
        import run_umi_precision_experiment as task4

        args = task4.parse_args([
            "--framework-root", "/framework", "--checkpoint-path", "/checkpoint",
            "--old-run09", "/old", "--run-root", "/runs",
        ])
        args.old_run09 = None
        with self.assertRaisesRegex(task4.BlockedExecution, "explicit|required"):
            task4.execute_task4(args)

    def test_task4_reference_requires_complete_full_scope_and_c_pre_artifacts(self):
        try:
            import run_umi_task5_experiment as api
        except ModuleNotFoundError:
            self.fail("Task 5 launcher implementation is missing")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "status.json").write_text(json.dumps({"status": "complete", "formal_successful": 42, "scope": "full"}))
            sample = root / "samples" / "C_pre"; sample.mkdir(parents=True)
            for name in ("z_bar", "common_input_fp32", "direction", "mask", "predicted_latent"):
                np.save(sample / f"{name}.npy", np.ones((1, 1, 3, 2, 2), np.float32), allow_pickle=False)
            (sample / "sample.json").write_text(json.dumps({"latent_slicing": {"condition_indexes": [0], "predicted_indexes": [1, 2],
                "source_shape": [1, 1, 3, 2, 2], "selected_shape": [1, 1, 2, 2, 2]}}))
            (sample / "status.json").write_text(json.dumps({"status": "success", "artifact_sha256": {}}))
            reference = api.load_task4_c_reference(root)
            self.assertEqual(reference["z_bar"].shape, (1, 1, 3, 2, 2))
            self.assertEqual(reference["metadata"]["task4_formal_successful"], 42)
            (root / "status.json").write_text(json.dumps({"status": "complete", "formal_successful": 41, "scope": "full"}))
            with self.assertRaises(api.BlockedExecution):
                api.load_task4_c_reference(root)


if __name__ == "__main__":
    unittest.main()
