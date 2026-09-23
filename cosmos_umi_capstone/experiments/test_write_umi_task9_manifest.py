"""CPU checks for the Task 9 integrity manifest writer."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from write_umi_task9_manifest import write_manifest


class ManifestTests(unittest.TestCase):
    def test_complete_scan_manifest_is_stable_and_excludes_itself(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run_status.json").write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")
            (root / "analysis").mkdir()
            (root / "analysis" / "plot.svg").write_bytes(b"<svg/>")
            manifest = write_manifest(root)
            first = manifest.read_bytes()
            self.assertIn(hashlib.sha256(b"<svg/>").hexdigest().encode(), first)
            self.assertNotIn(b"MANIFEST.sha256", first)
            write_manifest(root)
            self.assertEqual(manifest.read_bytes(), first)

    def test_incomplete_scan_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run_status.json").write_text(json.dumps({"status": "RUNNING"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not complete"):
                write_manifest(root)


if __name__ == "__main__":
    unittest.main()
