import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import run_umi_task7_experiment as runner


def _safe_snapshot():
    return {"gpu_used_gib": 0.0, "gpu_free_gib": 100.0, "gpu_reserved_gib": 0.0,
            "gpu_peak_allocated_gib": 0.0, "gpu_peak_nvml_used_gib": 0.0,
            "ram_available_gib": 500.0, "rss_gib": 0.0, "swap_used_gib": 0.0,
            "disk_free_gib": 100.0, "cgroup_memory_limited": False,
            "cgroup_memory_limit_gib": None, "cgroup_memory_current_gib": None,
            "cgroup_memory_free_gib": None}


class Task7RunnerTests(unittest.TestCase):
    def test_cli_exposes_all_task7_bindings(self):
        args = runner.parse_args([
            "--stage", "A", "--run-dir", "run", "--resume", "--raw-root", "raw",
            "--decoder-root", "decoder", "--framework-root", "framework",
            "--checkpoint", "checkpoint", "--vae", "vae", "--action", "action",
            "--video", "video", "--launch-contract", "contract.json",
            "--task5-root", "task5", "--gpu-index", "0",
        ])
        self.assertEqual(args.stage, "A")
        self.assertTrue(args.resume)
        self.assertEqual(args.framework_root, "framework")
        self.assertEqual(args.checkpoint, "checkpoint")
        self.assertEqual(args.gpu_index, 0)

    def test_plans_have_exact_formal_counts_and_fixed_order(self):
        self.assertEqual(len(runner.build_stage_plan("A")), 16)
        self.assertEqual(runner.stage_counts("A"), {"G": 0, "D": 0, "E": 16})
        self.assertEqual(len(runner.build_stage_plan("B")), 16)
        self.assertEqual(runner.stage_counts("B"), {"G": 16, "D": 16, "E": 16})
        plan = runner.build_stage_plan("C")
        self.assertEqual(len(plan), 38)
        self.assertEqual(runner.stage_counts("C"), {"G": 38, "D": 38, "E": 38})
        self.assertEqual(plan[0]["kind"], "baseline_pre")
        self.assertEqual(plan[-1]["kind"], "baseline_post")
        self.assertEqual([row["beta"] for row in plan if row["kind"] == "perturbation"],
                         [0.1, 0.1, 0.2, 0.2, 0.4, 0.4] * 6)

    def test_resource_policy_requires_complete_live_snapshot(self):
        with self.assertRaises(runner.ResourceStop):
            runner.require_resource_snapshot({"gpu_used_gib": 0.0}, phase="stage")
        runner.require_resource_snapshot(_safe_snapshot(), phase="preload")
        snapshot = _safe_snapshot()
        snapshot.update({"gpu_used_gib": 76.0})
        with self.assertRaises(runner.ResourceStop):
            runner.require_resource_snapshot(snapshot, phase="stage")
        snapshot = _safe_snapshot()
        snapshot.update({"cgroup_memory_limited": True, "cgroup_memory_limit_gib": 100,
                         "cgroup_memory_current_gib": 95, "cgroup_memory_free_gib": 5})
        with self.assertRaises(runner.ResourceStop):
            runner.require_resource_snapshot(snapshot, phase="stage")

    def test_store_keeps_failed_attempt_and_verifies_success_on_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            store = runner.Task7SampleStore(Path(temp) / "samples")
            first = store.write_failure("sample", {"error": "oom"}, operation_counts={"G": 1, "D": 0, "E": 0})
            self.assertTrue((first / "status.json").is_file())
            self.assertEqual(store.prepare("sample", resume=True), "run")
            self.assertTrue(any(path.name.startswith("sample.attempt.") for path in store.root.iterdir()))
            store.write_success("sample", {"value": np.ones((2,), dtype=np.float32)}, required_files=("record.json",))
            self.assertEqual(store.prepare("sample", resume=True), "skip")
            (store.root / "sample" / "record.json").write_text("tampered", encoding="utf-8")
            with self.assertRaises(runner.ResumeMismatch):
                store.prepare("sample", resume=True)

    def test_run_stage_uses_observed_failed_attempt_counts_and_never_infers(self):
        with tempfile.TemporaryDirectory() as temp:
            def execute(spec):
                error = RuntimeError("mid-call")
                error.capture = {"operation_counts": {"G": 1, "D": 0, "E": 0}}
                raise error

            result = runner.run_stage("A", temp, execute=execute,
                                      binding=runner.test_binding("A"),
                                      monitor=runner.StaticMonitor(_safe_snapshot()))
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["attempt_counts"], {"G": 1, "D": 0, "E": 0})
            self.assertEqual(result["completed_count"], 0)
            self.assertTrue((Path(temp) / "samples" / "A_baseline_pre_native.attempt.001" / "status.json").is_file())

    def test_run_directory_must_not_be_inside_read_only_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            run = raw / "new-run"
            with self.assertRaises(runner.ResumeMismatch):
                runner.validate_run_roots(run, raw_root=raw, decoder_root=root / "decoder")

    def test_preflight_is_generation_free_and_records_bindings(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            called = []
            result = runner.main([
                "--stage", "preflight", "--run-dir", str(run),
            ], factory=lambda *_a, **_k: called.append(True))
            self.assertEqual(result, 0)
            self.assertEqual(called, [])
            status = json.loads((run / "run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "PREFLIGHT_COMPLETE")
            self.assertFalse(status["generation_started"])


if __name__ == "__main__":
    unittest.main()
