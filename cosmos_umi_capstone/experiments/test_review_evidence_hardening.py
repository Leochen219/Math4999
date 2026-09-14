import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import analyze_umi_precision_contrast as analyzer


class ReviewEvidenceHardeningTests(unittest.TestCase):
    def _fixture(self, root: Path, *, scope="module"):
        source = root / "source"
        run = root / "run"
        samples = run / "samples"
        source.mkdir(parents=True)
        samples.mkdir(parents=True)
        for name in analyzer._REVIEW_SOURCE_FILES:
            (source / name).write_text(f"source:{name}\n", encoding="utf-8")
        for name in ("config.json", "status.json", "task4_execution.json"):
            (run / name).write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
        sample_dirs = {}
        for sample_id in analyzer._REVIEW_SAMPLE_IDS:
            sample = samples / sample_id
            sample.mkdir()
            sample_dirs[sample_id] = str(sample)
            payload = {"sample_id": sample_id}
            (sample / "sample.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
            shape = (1, 1, 3, 1, 2)
            names = analyzer._REVIEW_REQUIRED_SAMPLE_ARRAY_FILES
            if scope == "full":
                names = (*names, "decoded_final.npy")
            for name in names:
                np.save(sample / name, np.zeros((3, 2, 2), dtype=np.float32) if name == "decoded_final.npy" else np.zeros(shape, dtype=np.float32), allow_pickle=False)
            hashes = {"sample.json": analyzer.sha256_file(sample / "sample.json")}
            hashes.update({name: analyzer.sha256_file(sample / name) for name in names})
            (sample / "status.json").write_text(json.dumps({"status": "success", "artifact_sha256": hashes}) + "\n", encoding="utf-8")
        source_hashes = {str(source / name): analyzer.sha256_file(source / name) for name in analyzer._REVIEW_SOURCE_FILES}
        result = {
            "scope": scope,
            "metadata": {"config": {"source_sha256": source_hashes}, "run_dir": str(run)},
            "review_evidence_sources": {"source_dir": str(source), "run_dir": str(run), "sample_dirs": sample_dirs},
        }
        return result, source, run, sample_dirs

    def test_source_hash_tamper_is_rejected_and_manifest_records_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, source, run, _ = self._fixture(Path(temporary))
            (source / "umi_precision_runtime.py").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                analyzer._write_review_evidence(Path(temporary) / "out", result)
            # Restore the recorded bytes and ensure evidence entries carry both
            # the recorded claim and the copied-byte verification result.
            (source / "umi_precision_runtime.py").write_text("source:umi_precision_runtime.py\n", encoding="utf-8")
            output = Path(temporary) / "out"
            analyzer._write_review_evidence(output, result)
            manifest = json.loads((output / "review_evidence_manifest.json").read_text(encoding="utf-8"))
            entry = next(item for item in manifest["entries"] if item["role"] == "runner_source" and item["path"].endswith("/umi_precision_runtime.py"))
            self.assertEqual(entry["recorded_sha256"], entry["sha256"])
            self.assertTrue(entry["verified"])

    def test_parent_symlink_and_out_of_tree_sample_mapping_are_rejected(self):
        if not hasattr(os, "symlink"):
            self.skipTest("symlink unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            result, source, run, sample_dirs = self._fixture(fixture)
            real_parent = fixture / "real_parent"
            real_parent.mkdir()
            linked = fixture / "linked_parent"
            try:
                os.symlink(real_parent, linked, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"symlink unavailable: {error}")
            result["review_evidence_sources"]["source_dir"] = str(linked / "source")
            with self.assertRaisesRegex(ValueError, "symlink/reparse|evidence"):
                analyzer._write_review_evidence(fixture / "linked_output", result)
            result, source, run, sample_dirs = self._fixture(fixture / "second")
            result["review_evidence_sources"]["sample_dirs"]["A_pre"] = str(fixture / "second" / "outside")
            (fixture / "second" / "outside").mkdir()
            with self.assertRaisesRegex(ValueError, "escapes run samples|mapping"):
                analyzer._write_review_evidence(fixture / "second_out", result)

    def test_missing_sample_or_status_fails_formal_evidence_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            result, source, run, sample_dirs = self._fixture(fixture)
            (run / "samples" / "A_pre" / "status.json").unlink()
            with self.assertRaisesRegex(ValueError, "required review evidence|status"):
                analyzer._write_review_evidence(fixture / "out", result)

    def test_unknown_root_file_is_not_bundled_and_missing_timing_is_na(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "precision_report_zh.md").write_text("report", encoding="utf-8")
            (root / "secret.txt").write_text("must not bundle", encoding="utf-8")
            (root / "unknown").mkdir()
            (root / "unknown" / "nested.txt").write_text("must not bundle", encoding="utf-8")
            analyzer._write_bundle(root)
            import zipfile
            with zipfile.ZipFile(root / "review_bundle.zip") as archive:
                self.assertIn("precision_report_zh.md", archive.namelist())
                self.assertNotIn("secret.txt", archive.namelist())
                self.assertNotIn("unknown/nested.txt", archive.namelist())
            report_result = {"scope": "module", "status": "COMPLETE", "formal_call_count": 42,
                             "expected_formal_call_count": 42, "diagnostic_call_count": 4, "timing": {},
                             "spaces": {"predicted_latent": {"status": "OK"}, "decoded_final_rgb": {"status": "N/A", "reason": "module fallback"}},
                             "blocked_reasons": [], "window_decisions": [], "point_metrics": [], "fit_metrics": [],
                             "baseline_floors": {}, "metadata": {}, "input_identity": {}, "compute_identity": {},
                             "cache_identity": {}, "noise_identity": {}, "vector_identity": {}}
            analyzer._write_report(root, report_result)
            self.assertIn("记录样本累计耗时：`N/A`", (root / "precision_report_zh.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
