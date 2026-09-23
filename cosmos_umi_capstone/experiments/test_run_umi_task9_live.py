"""CPU-only contracts for the Task 9 production launcher."""
from __future__ import annotations

import json
import hashlib
import inspect
import io
import os
import subprocess
import tempfile
import unittest
import copy
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

try:
    import run_umi_task9_live as launcher
except ModuleNotFoundError:
    launcher = None


def _cli_args(root: Path, *extra: str):
    code_root = root / "task8-runtime"
    framework = root / "framework"
    code_root.mkdir(exist_ok=True)
    framework.mkdir(exist_ok=True)
    checkpoint = root / "checkpoint.pt"
    vae = root / "vae.pt"
    data = root / "preflight_window.npz"
    metadata = root / "task8_preflight.json"
    for path in (checkpoint, vae, data):
        path.write_bytes(b"fixture")
    metadata.write_text(json.dumps({"status": "PREFLIGHT_PASSED_GENERATION_NOT_PERFORMED",
                                    "selected": {"file_path": "scene-a", "episode_id": 32,
                                                 "record_index": 1, "language": "fixture",
                                                 "preflight_npz": {"sha256": hashlib.sha256(data.read_bytes()).hexdigest()}}}),
                        encoding="utf-8")
    return launcher.parse_args([
        "--code-root", str(code_root),
        "--data", str(data),
        "--metadata-json", str(metadata),
        "--framework", str(framework),
        "--checkpoint", str(checkpoint),
        "--vae", str(vae),
        "--run-dir", str(root / "run"),
        "--seed", "0",
        *extra,
    ])


class _FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def current_device():
        return 0

    @staticmethod
    def get_device_name(_index):
        return "CPU contract fake"


class _FakeTorch:
    __version__ = "2.10.0+cu130"
    version = SimpleNamespace(cuda="13.0")
    cuda = _FakeCuda()
    backends = SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        cudnn=SimpleNamespace(allow_tf32=True),
    )


class _Feedback:
    def __init__(self):
        self.z0 = np.full((1, 1, 2, 1, 40), 2.0, dtype=np.float32)
        self.mask = np.zeros_like(self.z0, dtype=bool)
        self.mask[:, :, 0] = True
        self.condition_indexes = (0,)
        self.predicted_indexes = (1,)
        self.temporal_axis = 2
        self.calls = []

    def extract_condition(self, value):
        return np.take(np.asarray(value, dtype=np.float32), self.condition_indexes,
                       axis=self.temporal_axis).copy()

    def step(self, condition_only, seed):
        condition = np.asarray(condition_only, dtype=np.float32).copy()
        self.calls.append((condition.copy(), seed))
        consumed_full = self.z0.copy()
        consumed_full[:, :, 0] = condition[:, :, 0]
        consumed_steps = np.repeat(consumed_full[None, ...], 30, axis=0)
        if hasattr(self, "runtime"):
            packed = SimpleNamespace(action=self.runtime.prepared["action"])
            boundary = inspect.signature(self.runtime.model.denoise).bind_partial(data_batch_packed=packed)
            if boundary.arguments.get("data_batch_packed") is not packed:
                raise AssertionError("precision wrapper cannot inspect data_batch_packed")
            for _ in range(30):
                self.runtime.model.denoise(data_batch_packed=packed)
        marker = np.float32(condition.mean())
        return {
            "predicted_latent": np.full((1, 2), marker, dtype=np.float32),
            "decoded_last_rgb": np.full((3, 2, 2), marker / 10, dtype=np.float32),
            "next_condition_fp32": (condition * np.float32(0.5)).astype(np.float32),
            "prediction_noise_hash": "paired-noise-fixture",
            "actual": {"condition_steps": consumed_steps},
            "evidence": {"operation_counts": {"G": 1, "D": 1, "E": 1},
                         "operation_dtypes": ["float32"]},
            "decoder": {"operation_dtypes": ["float32"], "operation_count": 1},
            "encoder": {"actual_encoder_input_dtype": "float32",
                        "actual_output_dtype": "float32", "operation_count": 1},
        }


class _FakeContext:
    def __init__(self):
        self.feedback = _Feedback()
        self.batch = SimpleNamespace(
            prompt="fixture prompt",
            actions=np.full((2, 16, 10), 0.25, dtype=np.float32),
            raw_actions=np.full((2, 16, 10), 0.5, dtype=np.float32),
        )
        self.runtime = SimpleNamespace(prepared={}, ops=None, model=_FakeModel())
        self.feedback.runtime = self.runtime
        self.contract = {
            "device": "cuda:0", "batch_size": 1, "sampler": "UniPC", "steps": 30,
            "guidance": 1.0, "shift": 10.0, "diffusion_cache": False,
            "autocast": False, "tf32": False, "GDE_precision": "float32",
        }
        self.environment = {"torch": "2.10.0+cu130", "cuda": "13.0", "device": 0}
        self.cleanup_calls = 0

    def cleanup(self):
        self.cleanup_calls += 1


class _FakeMonitor:
    def __init__(self, *args, fail_formal=False, **_kwargs):
        self.run_dir = Path(args[0])
        self.start_calls = 0
        self.stop_calls = 0
        self.checks = []
        self.samples = []
        self.fail_formal = fail_formal

    def start(self):
        self.start_calls += 1
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "monitor.log").write_text("monitor started\n", encoding="utf-8")
        self.check(phase="preload", starting_new_sample=True)
        return self

    def check(self, *, phase, starting_new_sample):
        if self.fail_formal and phase == "formal":
            raise launcher.LiveBlocked("fake hard stop")
        sample = {"phase": phase, "starting_new_sample": starting_new_sample, "disk_free_gib": 10.0}
        self.checks.append(sample)
        self.samples.append(sample)
        return sample

    def stop(self):
        self.stop_calls += 1


class _FakeModel:
    def denoise(self, *, data_batch_packed):
        return data_batch_packed


class _RuntimeAPI:
    class Task8InputAdapter:
        @classmethod
        def from_preflight(cls, npz_path, metadata_path):
            return SimpleNamespace(prompt="fixture prompt", npz_path=npz_path,
                                   metadata_path=metadata_path,
                                   provenance={"scene": "scene-a", "record_index": 1})

    def __init__(self, contract=None, fail_formal=False):
        self.context = _FakeContext()
        if contract is not None:
            self.context.contract = contract
        self.load_calls = []
        self.monitors = []
        self.fail_formal = fail_formal
        self.action_checks = []

    @staticmethod
    def clone_runtime(value):
        return copy.deepcopy(value)

    @staticmethod
    def refresh_official_prepared_action(prepared, action, *, ops=None):
        action = np.asarray(action, dtype=np.float32).copy()
        prepared["action"] = SimpleNamespace(tokens=[action])
        return {"effective_action": action.copy(), "source": "fixture"}

    @staticmethod
    def action_token_hash(tokens):
        return launcher._array_sha256(tokens)

    def validate_action_consumption(self, generation, action, *, chunk_index):
        hashes = list(generation["packed_action_token_hashes"])
        expected = self.action_token_hash(action)
        result = {"chunk_index": chunk_index, "packed_action_token_hashes": hashes,
                  "observed_step_count": len(hashes),
                  "all_steps_match": len(hashes) == 30 and all(item == expected for item in hashes)}
        self.action_checks.append(result)
        if not result["all_steps_match"]:
            raise AssertionError("fixture action was not consumed in all 30 steps")
        return result

    def Task8ResourceMonitor(self, *args, **kwargs):
        monitor = _FakeMonitor(*args, fail_formal=self.fail_formal, **kwargs)
        self.monitors.append(monitor)
        return monitor

    def load_task8_live(self, **kwargs):
        self.load_calls.append(kwargs)
        return self.context


class Task9LauncherTests(unittest.TestCase):
    def test_cli_requires_explicit_paths_and_keeps_release_off_by_default(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launcher.parse_args([])
        with tempfile.TemporaryDirectory() as temporary:
            args = _cli_args(Path(temporary), "--resume", "--smoke")
            self.assertFalse(args.release)
            self.assertTrue(args.resume)
            self.assertTrue(args.smoke)
            self.assertEqual(args.seed, 0)

    def test_release_gate_precedes_runtime_import_or_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = _cli_args(Path(temporary))
            with patch.object(launcher, "_load_task8_api", side_effect=AssertionError("runtime import reached")):
                with self.assertRaisesRegex(launcher.LiveBlocked, "--release"):
                    launcher.execute_task9(args)

    def test_task8_snapshot_manifest_is_verified_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = {}
            for name in ("task8_live.py", "umi_task8_runtime.py", "umi_task7_runtime.py", "umi_task7_encoder.py"):
                contents = f"verified fixture for {name}\n".encode("utf-8")
                (root / name).write_bytes(contents)
                entries[name] = hashlib.sha256(contents).hexdigest()
            manifest = "".join(f"{digest}  {name}\n" for name, digest in entries.items())
            (root / "MANIFEST.sha256").write_text(manifest, encoding="ascii")
            manifest_hash = hashlib.sha256((root / "MANIFEST.sha256").read_bytes()).hexdigest()
            with patch.object(launcher, "TASK8_RUNTIME_MANIFEST_SHA256",
                              manifest_hash):
                evidence = launcher.verify_task8_snapshot(root)
            self.assertEqual(evidence["verified_files"], 4)
            (root / "umi_task7_runtime.py").write_text("tampered\n", encoding="utf-8")
            with patch.object(launcher, "TASK8_RUNTIME_MANIFEST_SHA256",
                              manifest_hash):
                with self.assertRaisesRegex(launcher.LiveBlocked, "manifest"):
                    launcher.verify_task8_snapshot(root)

    def test_framework_commit_is_pinned_and_checkout_must_be_clean(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = [
                subprocess.CompletedProcess([], 0, launcher.TASK8_FRAMEWORK_COMMIT + "\n", ""),
                subprocess.CompletedProcess([], 0, "", ""),
            ]
            with patch.object(launcher.subprocess, "run", side_effect=good):
                self.assertEqual(launcher.verify_framework_checkout(root), launcher.TASK8_FRAMEWORK_COMMIT)
            dirty = [
                subprocess.CompletedProcess([], 0, launcher.TASK8_FRAMEWORK_COMMIT + "\n", ""),
                subprocess.CompletedProcess([], 0, " M file.py\n", ""),
            ]
            with patch.object(launcher.subprocess, "run", side_effect=dirty):
                with self.assertRaisesRegex(launcher.LiveBlocked, "uncommitted"):
                    launcher.verify_framework_checkout(root)

    def test_preflight_metadata_must_bind_the_exact_npz_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = _cli_args(root)
            scene, data_hash, metadata_hash = launcher._preflight_identity(Path(args.data), Path(args.metadata_json))
            self.assertEqual(scene["episode_id"], 32)
            self.assertEqual(data_hash, hashlib.sha256(Path(args.data).read_bytes()).hexdigest())
            self.assertEqual(len(metadata_hash), 64)
            Path(args.data).write_bytes(b"changed")
            with self.assertRaisesRegex(launcher.LiveBlocked, "SHA256"):
                launcher._preflight_identity(Path(args.data), Path(args.metadata_json))

    def test_runtime_contract_and_offline_environment_fail_before_monitor_or_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = _cli_args(root, "--release")
            api = _RuntimeAPI()
            wrong_torch = _FakeTorch()
            wrong_torch.__version__ = "2.9.0+cu128"
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1"}), \
                    patch.object(launcher, "verify_framework_checkout", return_value=launcher.TASK8_FRAMEWORK_COMMIT):
                with self.assertRaisesRegex(launcher.LiveBlocked, r"Torch 2.10.0\+cu130"):
                    launcher.execute_task9(args, runtime_api=api, torch_module=wrong_torch)
            self.assertEqual(api.load_calls, [])
            self.assertEqual(api.monitors, [])

            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "0"}):
                with self.assertRaisesRegex(launcher.LiveBlocked, "HF_HUB_OFFLINE"):
                    launcher.execute_task9(args, runtime_api=api, torch_module=_FakeTorch())
            self.assertEqual(api.load_calls, [])
            self.assertEqual(api.monitors, [])

    def test_full_scan_loads_one_model_and_saves_float_feedback_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = _cli_args(root, "--release")
            api = _RuntimeAPI()
            torch = _FakeTorch()
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1"}), \
                    patch.object(launcher, "TASK9_DISK_RESERVE_BYTES", 0), \
                    patch.object(launcher, "verify_framework_checkout", return_value=launcher.TASK8_FRAMEWORK_COMMIT):
                result = launcher.execute_task9(args, runtime_api=api, torch_module=torch)

            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["completed"], 162)
            self.assertEqual(len(api.load_calls), 1)
            self.assertEqual(len(api.context.feedback.calls), 162)
            self.assertEqual(len(api.action_checks), 162)
            self.assertTrue(all(item["all_steps_match"] for item in api.action_checks))
            self.assertTrue(all(seed == 0 for _, seed in api.context.feedback.calls))
            monitor = api.monitors[0]
            self.assertEqual(monitor.start_calls, 1)
            self.assertEqual(monitor.stop_calls, 1)
            self.assertEqual(len(monitor.checks), 325)
            self.assertEqual(api.context.cleanup_calls, 1)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)

            sample = Path(result["scan_dir"]) / "samples" / "train_00_plus"
            for filename in ("condition.npy", "predicted_latent.npy", "decoded_final_rgb.npy", "next_condition.npy"):
                self.assertEqual(np.load(sample / filename, allow_pickle=False).dtype, np.float32)
            saved = np.load(sample / "next_condition.npy", allow_pickle=False)
            self.assertTrue(np.array_equal(saved, api.context.feedback.calls[1][0] * np.float32(0.5)))
            config = json.loads((Path(result["scan_dir"]) / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(len(config["plan"]), 162)
            self.assertEqual(sum(item["kind"] == "train" for item in config["plan"]), 64)
            self.assertEqual(sum(item["kind"] == "holdout" for item in config["plan"]), 16)
            self.assertEqual(sum(item["kind"] == "calibration" for item in config["plan"]), 80)
            identity = config["identity"]
            self.assertEqual(identity["prompt"], "fixture prompt")
            self.assertEqual(identity["condition_geometry"]["condition_indexes"], [0])
            self.assertEqual(identity["action_chunk0"]["raw_sha256"], launcher._array_sha256(
                api.context.batch.raw_actions[0]))
            self.assertEqual(identity["action_chunk0"]["normalized_sha256"], launcher._array_sha256(
                api.context.batch.actions[0]))
            self.assertEqual(identity["checkpoint"]["sha256"], hashlib.sha256(
                Path(args.checkpoint).read_bytes()).hexdigest())
            self.assertEqual(identity["vae"]["sha256"], hashlib.sha256(
                Path(args.vae).read_bytes()).hexdigest())
            record = json.loads((sample / "record.json").read_text(encoding="utf-8"))
            evidence = record["step_evidence"]
            self.assertEqual(evidence["prompt"], "fixture prompt")
            self.assertEqual(evidence["condition_indexes"], [0])
            self.assertEqual(len(evidence["condition_step_sha256"]), 30)
            self.assertEqual(len(evidence["action_token_evidence"]["packed_action_token_hashes"]), 30)
            self.assertTrue(evidence["action_token_evidence"]["consumption"]["all_steps_match"])
            self.assertEqual(evidence["prediction_noise_hash"], "paired-noise-fixture")
            self.assertEqual(evidence["fp32_precision"]["decoder"]["operation_dtypes"], ["float32"])
            self.assertEqual(evidence["fp32_precision"]["encoder"]["actual_output_dtype"], "float32")
            condition = np.load(sample / "condition.npy", allow_pickle=False)
            self.assertEqual(evidence["actual_consumed_condition_sha256"], launcher._array_sha256(condition))
            self.assertTrue((Path(result["run_dir"]) / "monitor.log").is_file())

    def test_smoke_is_one_baseline_and_gpu_one_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = _cli_args(root, "--release", "--smoke")
            api = _RuntimeAPI()
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1", "HF_HUB_OFFLINE": "1"}), \
                    patch.object(launcher, "TASK9_DISK_RESERVE_BYTES", 0):
                with self.assertRaisesRegex(launcher.LiveBlocked, "CUDA_VISIBLE_DEVICES"):
                    launcher.execute_task9(args, runtime_api=api, torch_module=_FakeTorch())
            self.assertEqual(api.load_calls, [])

            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1"}), \
                    patch.object(launcher, "TASK9_DISK_RESERVE_BYTES", 0), \
                    patch.object(launcher, "verify_framework_checkout", return_value=launcher.TASK8_FRAMEWORK_COMMIT):
                result = launcher.execute_task9(args, runtime_api=api, torch_module=_FakeTorch())
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["completed"], 1)
            self.assertEqual(len(api.load_calls), 1)
            self.assertEqual(len(api.context.feedback.calls), 1)
            self.assertEqual(len(api.action_checks), 1)
            self.assertEqual(len(json.loads((Path(result["scan_dir"]) / "config.json").read_text())["plan"]), 1)
            self.assertTrue((Path(result["run_dir"]) / "monitor.log").is_file())

            args.resume = True
            resumed_api = _RuntimeAPI()
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1"}), \
                    patch.object(launcher, "TASK9_DISK_RESERVE_BYTES", 0), \
                    patch.object(launcher, "verify_framework_checkout", return_value=launcher.TASK8_FRAMEWORK_COMMIT):
                resumed = launcher.execute_task9(args, runtime_api=resumed_api, torch_module=_FakeTorch())
            self.assertEqual(resumed["status"], "COMPLETE")
            self.assertEqual(resumed["completed"], 1)
            self.assertEqual(len(resumed_api.load_calls), 1)
            self.assertEqual(resumed_api.context.feedback.calls, [])
            self.assertEqual(resumed_api.monitors[0].start_calls, 1)

    def test_resource_hard_stop_cleans_up_and_is_not_reported_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = _cli_args(root, "--release")
            api = _RuntimeAPI(fail_formal=True)
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1"}), \
                    patch.object(launcher, "TASK9_DISK_RESERVE_BYTES", 0), \
                    patch.object(launcher, "verify_framework_checkout", return_value=launcher.TASK8_FRAMEWORK_COMMIT):
                with self.assertRaisesRegex(launcher.LiveBlocked, "fake hard stop"):
                    launcher.execute_task9(args, runtime_api=api, torch_module=_FakeTorch())
            self.assertEqual(len(api.load_calls), 1)
            self.assertEqual(api.context.cleanup_calls, 1)
            self.assertEqual(len(api.monitors), 1)
            self.assertEqual(api.monitors[0].stop_calls, 1)
            self.assertEqual(api.context.feedback.calls, [])
            status = json.loads((Path(args.run_dir) / "launcher_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "BLOCKED")
            self.assertTrue(status["generation_started"])


if __name__ == "__main__":
    unittest.main()
