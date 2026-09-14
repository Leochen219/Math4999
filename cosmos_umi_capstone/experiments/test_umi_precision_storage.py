"""Real process locking and publication interruption tests."""
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class StorageTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("umi_precision_storage")
        except ModuleNotFoundError:
            self.api = None
        self.assertIsNotNone(self.api, "process-lock and atomic-store implementation missing")

    def test_hard_killed_owner_releases_os_lock_and_concurrent_owner_is_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).parent.resolve()))
            code = "from umi_precision_storage import ProcessLock; import sys,time; lock=ProcessLock(sys.argv[1]); lock.__enter__(); print('locked',flush=True); time.sleep(30)"
            child = subprocess.Popen([sys.executable, "-c", code, str(Path(temp) / "lock")],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            try:
                self.assertEqual(child.stdout.readline().strip(), "locked")
                with self.assertRaises(BlockingIOError):
                    with self.api.ProcessLock(Path(temp) / "lock"):
                        self.fail("concurrent lock owner entered")
                child.kill()
                child.wait(timeout=10)
                with self.api.ProcessLock(Path(temp) / "lock"):
                    pass
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=10)
                child.stdout.close()
                child.stderr.close()

    def test_success_is_published_once_with_complete_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            store = self.api.PrecisionSampleStore(temp)
            store.prepare("A_pre")
            original = self.api.publish_directory
            observed = []
            def inspect_publish(stage, destination):
                state = json.loads((stage / "status.json").read_text())
                self.assertEqual(state["status"], "success")
                self.assertIn("sample.json", state["artifact_sha256"])
                self.assertIn("output.npy", state["artifact_sha256"])
                self.assertFalse(destination.exists())
                observed.append(stage)
                return original(stage, destination)
            with patch.object(self.api, "publish_directory", inspect_publish):
                store.write_success("A_pre", {"identity": "fixed"}, artifacts={"output.npy": np.ones(2)})
            self.assertEqual(len(observed), 1)
            before = (Path(temp) / "A_pre" / "status.json").read_bytes()
            with self.assertRaises(FileExistsError):
                store.write_success("A_pre", {}, artifacts={})
            self.assertEqual(before, (Path(temp) / "A_pre" / "status.json").read_bytes())

    def test_prepublication_crash_is_recoverable_without_broken_success(self):
        with tempfile.TemporaryDirectory() as temp:
            store = self.api.PrecisionSampleStore(temp)
            store.prepare("A_pre")
            with patch.object(self.api, "publish_directory", side_effect=KeyboardInterrupt("prepublish crash")):
                with self.assertRaises(KeyboardInterrupt):
                    store.write_success("A_pre", {"identity": "fixed"}, artifacts={"output.npy": np.ones(2)})
            self.assertFalse((Path(temp) / "A_pre").exists())
            self.assertTrue(list(Path(temp).glob(".A_pre.stage.*")))
            self.assertEqual(store.prepare("A_pre", resume=True), "run")
            store.write_success("A_pre", {"identity": "fixed"}, artifacts={"output.npy": np.ones(2)})
            self.assertEqual(store.prepare("A_pre", resume=True), "skip")
            np.testing.assert_array_equal(store.load_record("A_pre")["output"], [1., 1.])


if __name__ == "__main__":
    unittest.main()
