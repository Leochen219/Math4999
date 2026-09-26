"""Stage-gated Task 11 five-chunk run over fixed Bridge record 15.

The preflight input is an immutable CPU bundle. Smoke and formal stages reuse
one frozen Task 8 live context each, and require explicit ``--release``.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .umi_task11_long_horizon import (CHUNK_LENGTH, HORIZON_CHUNKS, RECORD_INDEX,
                                          SEED_SCHEDULES, HorizonTrajectory,
                                          adjacent_ar_change, build_schedule_call_plan,
                                          compare_exact_repeat, compute_endpoint_rgb_metrics,
                                          compute_error_geometry, compute_masked_error_geometry,
                                          evaluate_resource_gate, execute_schedule, file_sha256,
                                          lock_run_identity, masked_latent_metrics,
                                          task8_array_hash, task8_input_for_window,
                                          validate_call_result)
    from .umi_task11_long_horizon_preflight import (EXPECTED_SHARD_SHA256,
                                                     LongHorizonPreflightError,
                                                     load_task11_trajectory,
                                                     run_long_horizon_preflight)
except ImportError:  # direct invocation from experiments/
    from umi_task11_long_horizon import (CHUNK_LENGTH, HORIZON_CHUNKS, RECORD_INDEX,
                                         SEED_SCHEDULES, HorizonTrajectory,
                                         adjacent_ar_change, build_schedule_call_plan,
                                         compare_exact_repeat, compute_endpoint_rgb_metrics,
                                         compute_error_geometry, compute_masked_error_geometry,
                                         evaluate_resource_gate, execute_schedule, file_sha256,
                                         lock_run_identity, masked_latent_metrics,
                                         task8_array_hash, task8_input_for_window,
                                         validate_call_result)
    from umi_task11_long_horizon_preflight import (EXPECTED_SHARD_SHA256,
                                                    LongHorizonPreflightError,
                                                    load_task11_trajectory,
                                                    run_long_horizon_preflight)


EXPECTED_NORMALIZER_SHA256 = "e673c112e9809a2dc3ac4fffd42896c0a3c7538d4808eae35d62059af65cb267"
EXPECTED_PARITY_SHA256 = "e89266d342dd60f6c97bf65ac7fd076c09ceafcfb50fff737d4de3b682bd1442"
EXPECTED_CHECKPOINT_SHA256 = "0a4b762014f9fe3e3e8e13db204a14995d139c7669f58dd786dd6263ca291a9d"
EXPECTED_VAE_SHA256 = "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
EXPECTED_FRAMEWORK_COMMIT = "ffa9c6b60a6b04b2fae337577bc6cbd8a93c39f5"
EXPECTED_DATASET_ROOT = "/root/autodl-tmp/datasets/bridge_v2_subset_20260921"
EXPECTED_PREFLIGHT_INPUT = "/root/autodl-tmp/task11-horizon-preflight-20260925-v1"
EXPECTED_FRAMEWORK_ROOT = "/root/autodl-tmp/cosmos-framework-task6-clean"
EXPECTED_CHECKPOINT_PATH = "/root/cosmos3-edge/cosmos3-edge-model"
EXPECTED_VAE_PATH = "/root/autodl-tmp/cosmos-framework/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"
EXPECTED_NORMALIZER_PATH = "/root/autodl-tmp/cosmos-framework-task6-clean/cosmos_framework/data/generator/action/normalizer_stats/bridge_orig_lerobot_stats.json"
EXPECTED_PARITY_PATH = "/root/autodl-tmp/task8-preflight/official_parity.json"
EXPECTED_TORCH_VERSION = "2.10.0+cu130"
EXPECTED_CUDA_VERSION = "13.0"
TRUE_ENDPOINTS = (0, 16, 32, 48, 64, 80)


def _assert_pinned_model_identity(checkpoint_sha256: str, vae_sha256: str,
                                  framework_commit: str) -> None:
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint does not match the pinned Task 10 model identity")
    if vae_sha256 != EXPECTED_VAE_SHA256:
        raise ValueError("VAE does not match the pinned Task 10 model identity")
    if framework_commit != EXPECTED_FRAMEWORK_COMMIT:
        raise ValueError("framework commit does not match the pinned Task 10 runtime identity")


def _assert_exact_path(label: str, value: str | Path, expected: str) -> None:
    if Path(value).as_posix() != expected:
        raise ValueError(f"{label} path differs from the approved Task 11 location")


def condition_only_mask(full_mask: Any, *, temporal_axis: int,
                        condition_indexes: Sequence[int]) -> np.ndarray:
    mask = np.asarray(full_mask)
    indexes = tuple(int(value) for value in condition_indexes)
    if (mask.dtype != np.bool_ or mask.ndim == 0 or isinstance(temporal_axis, bool)
            or not isinstance(temporal_axis, int) or not -mask.ndim <= temporal_axis < mask.ndim
            or not indexes or len(indexes) != len(set(indexes))):
        raise ValueError("runtime condition geometry is invalid")
    axis = temporal_axis % mask.ndim
    if any(index < 0 or index >= mask.shape[axis] for index in indexes):
        raise ValueError("runtime condition indexes are outside the authoritative full mask")
    packed = np.take(mask, indexes, axis=axis)
    if not np.any(packed):
        raise ValueError("runtime condition-only mask is empty")
    return np.ascontiguousarray(packed)


def validate_cli_contract(args: argparse.Namespace) -> None:
    if args.record_index != RECORD_INDEX:
        raise ValueError("Task 11 is locked to --record-index 15")
    if args.horizon_chunks != HORIZON_CHUNKS:
        raise ValueError("Task 11 is locked to five action chunks")
    if args.seed_schedules != "both":
        raise ValueError("Task 11 formal run requires both locked seed schedules")
    if Path(args.input_run).resolve() == Path(args.run_dir).resolve():
        raise ValueError("--input-run must identify the immutable preflight bundle, separate from --run-dir")
    if args.stage in {"smoke", "formal"} and not args.release:
        raise ValueError(f"{args.stage} model work requires explicit --release")


def validate_smoke_status(status: Mapping[str, Any], expected_identity_sha256: str) -> None:
    if not isinstance(status, Mapping) or status.get("status") != "ENGINEERING_SMOKE_COMPLETE":
        raise ValueError("formal stage requires a complete engineering smoke")
    if status.get("run_identity_sha256") != expected_identity_sha256:
        raise ValueError("engineering smoke identity differs from the locked run identity")
    if status.get("smoke_calls") != 1:
        raise ValueError("engineering smoke must contain exactly one call")
    if status.get("sample_id") != "smoke_G0":
        raise ValueError("engineering smoke sample identity is not smoke_G0")
    size = status.get("measured_sample_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("engineering smoke footprint is missing")
    if status.get("resource_gate_status") != "PASS":
        raise ValueError("engineering smoke resource gates did not pass")


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".csv", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": task8_array_hash(value)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("nonfinite evidence cannot enter JSON")
    return value


def _next_attempt(root: Path, stem: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    ordinal = 1
    while True:
        candidate = root / f"{stem}_{ordinal:03d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            ordinal += 1


def _preflight_paths(input_run: str | Path) -> tuple[Path, Path]:
    root = Path(input_run).resolve()
    return root / "task11_preflight.json", root / "task11_trajectory.npz"


def _load_preflight(input_run: str | Path) -> tuple[dict[str, Any], HorizonTrajectory, Path, Path]:
    report_path, npz_path = _preflight_paths(input_run)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("--input-run must contain a valid immutable Task 11 preflight report") from error
    selected = report.get("selected") if isinstance(report, Mapping) else None
    scan = selected.get("full_shard_scan") if isinstance(selected, Mapping) else None
    camera = selected.get("camera") if isinstance(selected, Mapping) else None
    shard = selected.get("shard") if isinstance(selected, Mapping) else None
    source = report.get("source_identity") if isinstance(report, Mapping) else None
    if (report.get("status") != "PREFLIGHT_PASSED_GENERATION_NOT_PERFORMED"
            or report.get("generation_performed") is not False
            or report.get("record_index") != RECORD_INDEX
            or not isinstance(scan, Mapping) or scan.get("record_count") != 52 or scan.get("crc_checked") is not True
            or not isinstance(camera, Mapping) or camera.get("feature_key") != "steps/observation/image_0"
            or camera.get("all_frames_rgb_256x256") is not True or camera.get("all_frames_nonconstant") is not True
            or not isinstance(shard, Mapping) or shard.get("sha256") != EXPECTED_SHARD_SHA256
            or not isinstance(source, Mapping) or source.get("normalizer_sha256") != EXPECTED_NORMALIZER_SHA256
            or source.get("official_parity_sha256") != EXPECTED_PARITY_SHA256):
        raise ValueError("Task 11 preflight report is missing a locked data/action/source gate")
    trajectory = load_task11_trajectory(npz_path, report_path)
    if (trajectory.rgb.shape != (81, 3, 256, 256)
            or trajectory.states.shape != (81, 7)
            or trajectory.original_actions.shape != (80, 7)
            or trajectory.raw_actions.shape != (5, 16, 10)
            or trajectory.normalized_actions.shape != (5, 16, 10)):
        raise ValueError("Task 11 preflight bundle does not contain 81 real observations and 80 transitions")
    return report, trajectory, report_path, npz_path


def _run_identity(args: argparse.Namespace, report: Mapping[str, Any],
                  report_path: Path, npz_path: Path) -> dict[str, Any]:
    try:
        from task8_frozen.task8_formal import _framework_commit, _sha256_path, _task8_source_paths
        from task8_frozen.umi_task8_runtime import Task8RuntimeConfig
        import umi_task9_bridge_preflight as task9
    except ImportError:  # package import path
        from .task8_frozen.task8_formal import _framework_commit, _sha256_path, _task8_source_paths
        from .task8_frozen.umi_task8_runtime import Task8RuntimeConfig
        from . import umi_task9_bridge_preflight as task9
    _assert_exact_path("dataset", args.dataset_root, EXPECTED_DATASET_ROOT)
    _assert_exact_path("preflight input", args.input_run, EXPECTED_PREFLIGHT_INPUT)
    _assert_exact_path("framework", args.framework_root, EXPECTED_FRAMEWORK_ROOT)
    _assert_exact_path("checkpoint", args.checkpoint, EXPECTED_CHECKPOINT_PATH)
    _assert_exact_path("VAE", args.vae, EXPECTED_VAE_PATH)
    _assert_exact_path("normalizer", args.normalizer_stats, EXPECTED_NORMALIZER_PATH)
    _assert_exact_path("official parity", args.official_parity_report, EXPECTED_PARITY_PATH)
    stats = task9.load_normalizer_stats(args.normalizer_stats)
    parity_path = Path(args.official_parity_report).resolve()
    parity_sha = file_sha256(parity_path)
    if stats.sha256 != EXPECTED_NORMALIZER_SHA256 or parity_sha != EXPECTED_PARITY_SHA256:
        raise ValueError("normalizer/parity artifact does not match the pinned Task 11 inputs")
    framework_commit = _framework_commit(args.framework_root)
    checkpoint_sha256 = _sha256_path(args.checkpoint)
    vae_sha256 = _sha256_path(args.vae)
    _assert_pinned_model_identity(checkpoint_sha256, vae_sha256, framework_commit)
    source_paths = _task8_source_paths() + [Path(__file__).resolve(),
        Path(__file__).with_name("umi_task11_long_horizon.py"),
        Path(__file__).with_name("umi_task11_long_horizon_preflight.py"),
        Path(task9.__file__).resolve()]
    code_hashes = {f"{path.parent.name}/{path.name}": file_sha256(path)
                   for path in sorted(set(source_paths), key=lambda path: str(path))}
    selected = report["selected"]
    source = report["source_identity"]
    return {
        "schema": "umi-task11-five-chunk-v1",
        "record_index": RECORD_INDEX,
        "horizon_chunks": HORIZON_CHUNKS,
        "seed_schedules": [list(row) for row in SEED_SCHEDULES],
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "shard_sha256": selected["shard"]["sha256"],
        "manifest_sha256": source["manifest_sha256"],
        "download_status_sha256": source["download_status_file_sha256"]
            if "download_status_file_sha256" in source else selected["transfer"]["status_file_sha256"],
        "preflight_input_run": str(Path(args.input_run).resolve()),
        "preflight_report_sha256": file_sha256(report_path),
        "preflight_npz_sha256": file_sha256(npz_path),
        "normalizer_path": str(Path(args.normalizer_stats).resolve()),
        "normalizer_sha256": stats.sha256,
        "official_parity_path": str(parity_path), "official_parity_sha256": parity_sha,
        "framework_root": str(Path(args.framework_root).resolve()), "framework_commit": framework_commit,
        "checkpoint_path": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": checkpoint_sha256,
        "vae_path": str(Path(args.vae).resolve()), "vae_sha256": vae_sha256,
        "runtime_config": {**Task8RuntimeConfig().as_dict(), "seed_schedules": [list(row) for row in SEED_SCHEDULES]},
        "official_runtime": {"sampler": "UniPC", "steps": 30, "guidance": 1.0, "shift": 10.0,
                             "precision": "FP32 G/D/E", "cache": False, "autocast": False, "tf32": False,
                             "one_model_runtime_per_process": True},
        "code_sha256": code_hashes,
    }


def _lock_or_check_run_identity(run_dir: Path, identity: Mapping[str, Any], *, create: bool,
                                resume: bool) -> dict[str, Any]:
    if create:
        return lock_run_identity(run_dir, identity, resume=resume)
    path = run_dir / "run_identity.json"
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("smoke/formal stages require the preflight-locked run identity") from error
    if saved != dict(identity):
        raise ValueError("run identity differs from source, model, code, input bundle, or seed schedule")
    return saved


def _verify_preflight_status(run_dir: Path, identity: Mapping[str, Any], input_run: str | Path,
                             report_sha256: str, trajectory_sha256: str) -> dict[str, Any]:
    status_path = Path(run_dir) / "preflight_status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("model stage requires the generation-free preflight receipt") from error
    expected = {"status": "PREFLIGHT_LOCKED_GENERATION_NOT_PERFORMED",
                "run_identity_sha256": _canonical_sha(identity),
                "input_run": str(Path(input_run).resolve()),
                "report_sha256": report_sha256,
                "trajectory_npz_sha256": trajectory_sha256}
    if status != expected:
        raise ValueError("model stage requires an exact generation-free preflight receipt")
    return status


def _verify_runtime() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - live runtime diagnostic
        raise RuntimeError("locked Task 11 smoke requires the existing Torch runtime") from error
    if torch.__version__ != EXPECTED_TORCH_VERSION or torch.version.cuda != EXPECTED_CUDA_VERSION:
        raise RuntimeError(f"Torch/CUDA runtime differs from locked {EXPECTED_TORCH_VERSION}/{EXPECTED_CUDA_VERSION}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("locked Task 11 live run requires the existing CUDA GPU 0 runtime")
    return torch


def _encode_truth_endpoints(context: Any, trajectory: HorizonTrajectory, formal_dir: Path,
                            *, identity: Mapping[str, Any], input_hash: str,
                            resume: bool) -> tuple[dict[int, np.ndarray], np.ndarray, dict[str, Any]]:
    npz_path = formal_dir / "truth_conditions.npz"
    receipt_path = formal_dir / "truth_conditions.json"
    feedback = context.feedback
    mask = np.asarray(feedback.mask, dtype=bool)
    if mask.shape != np.asarray(feedback.z0).shape or not np.any(mask):
        raise ValueError("truth condition mask is not the actual live Task 8 feedback mask")
    try:
        temporal_axis = int(feedback.temporal_axis)
        condition_indexes = tuple(int(index) for index in feedback.condition_indexes)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("live Task 8 feedback is missing condition-axis/index geometry") from error
    selected_mask = condition_only_mask(mask, temporal_axis=temporal_axis,
                                        condition_indexes=condition_indexes)
    if npz_path.exists() or receipt_path.exists():
        if not resume or not (npz_path.is_file() and receipt_path.is_file()):
            raise ValueError("partial truth endpoint artifacts require exact resume; artifacts are immutable")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("trajectory_npz_sha256") != file_sha256(npz_path)
                or receipt.get("run_identity_sha256") != _canonical_sha(identity)
                or receipt.get("input_trajectory_sha256") != input_hash
                or receipt.get("condition_mask_sha256") != task8_array_hash(mask)
                or receipt.get("condition_temporal_axis") != temporal_axis
                or receipt.get("condition_indexes") != list(condition_indexes)):
            raise ValueError("truth endpoint receipt differs from the locked model, input, or actual mask")
        with np.load(npz_path, allow_pickle=False) as data:
            if set(data.files) != {f"condition_x{index}" for index in TRUE_ENDPOINTS} | {"full_condition_mask"}:
                raise ValueError("truth endpoint NPZ is incomplete")
            conditions = {index: np.asarray(data[f"condition_x{index}"]) for index in TRUE_ENDPOINTS}
            stored_mask = np.asarray(data["full_condition_mask"])
        if stored_mask.dtype != np.bool_ or not np.array_equal(stored_mask, mask):
            raise ValueError("saved full condition mask differs from the live authoritative mask")
        for index, value in conditions.items():
            if (value.dtype != np.float32 or not np.isfinite(value).all()
                    or value.shape != selected_mask.shape):
                raise ValueError(f"saved true endpoint x{index} is not finite FP32")
            feedback.embed_condition(value)
        return conditions, mask, receipt
    conditions: dict[int, np.ndarray] = {}
    encoder_rows: dict[str, Any] = {}
    for index in TRUE_ENDPOINTS:
        encoded = context.encoder.encode(np.array(trajectory.rgb[index], copy=True), precision="temporary_fp32")
        arrays = encoded.get("arrays") if isinstance(encoded, Mapping) else None
        value = arrays.get("actual_output") if isinstance(arrays, Mapping) else None
        if not isinstance(value, np.ndarray) or value.dtype != np.float32 or not value.size or not np.isfinite(value).all():
            raise ValueError(f"fresh FP32 encoder output is missing for true frame x{index}")
        conditions[index] = np.ascontiguousarray(value.copy())
        if conditions[index].shape != selected_mask.shape:
            raise ValueError(f"true endpoint x{index} does not match runtime condition geometry")
        feedback.embed_condition(conditions[index])
        encoder_rows[str(index)] = {"encoded_sha256": task8_array_hash(conditions[index]),
                                    "encoder_evidence": _json_safe(encoded.get("evidence", {}))}
        del encoded, arrays, value
    _atomic_npz(npz_path, {**{f"condition_x{index}": conditions[index] for index in TRUE_ENDPOINTS},
                           "full_condition_mask": np.ascontiguousarray(mask, dtype=bool)})
    receipt = {"status": "PASS", "trajectory_npz_sha256": file_sha256(npz_path),
               "run_identity_sha256": _canonical_sha(identity), "input_trajectory_sha256": input_hash,
               "condition_mask_sha256": task8_array_hash(mask), "condition_mask_shape": list(mask.shape),
               "condition_temporal_axis": temporal_axis, "condition_indexes": list(condition_indexes),
               "condition_only_mask_shape": list(selected_mask.shape),
               "true_frame_indexes": list(TRUE_ENDPOINTS), "conditions": encoder_rows,
               "encode_count": len(TRUE_ENDPOINTS), "precision": "fresh temporary FP32 encoder"}
    _atomic_json(receipt_path, receipt)
    return conditions, mask, receipt


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _verify_smoke_artifact(smoke_dir: Path, identity_sha256: str) -> dict[str, Any]:
    status_path = smoke_dir / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    validate_smoke_status(status, identity_sha256)
    try:
        from task8_frozen.run_umi_task8_experiment import Task8SampleStore
    except ImportError:  # pragma: no cover
        from .task8_frozen.run_umi_task8_experiment import Task8SampleStore
    record = Task8SampleStore(smoke_dir / "samples").load_record("smoke_G0")
    try:
        for key in ("output_full", "generated_rgb", "decoded_rgb_full", "decoded_last_rgb",
                    "encoded_condition", "condition_input_fp32", "prediction_noise_hash",
                    "packed_action_token_hashes", "action_consumption", "provenance", "generation",
                    "call_evidence", "action", "action_hash"):
            if key not in record:
                raise ValueError(f"smoke sample lacks required actual-boundary evidence: {key}")
        evidence = record.get("call_evidence")
        if (record.get("sample_id") != "smoke_G0" or record.get("call") != "G0" or record.get("seed") != 0
                or not isinstance(evidence, Mapping) or evidence.get("action_steps") != 30
                or evidence.get("condition_steps") != 30
                or evidence.get("action_hash") != record.get("action_hash")
                or evidence.get("noise_hash") != record.get("prediction_noise_hash")):
            raise ValueError("smoke sample does not verify as the exact G0 call")
        sample_dir = smoke_dir / "samples" / "smoke_G0"
        measured = _tree_bytes(sample_dir)
        if measured != status["measured_sample_bytes"]:
            raise ValueError("smoke sample footprint differs from the measured resource report")
    finally:
        del record
        gc.collect()
    return status


def _attempt_monitor_dir(root: Path) -> Path:
    return _next_attempt(root / "monitor_attempts", "attempt")


def _is_resource_stop(error: BaseException) -> bool:
    return type(error).__name__ in {"ResourceStop", "LiveResourceStop", "ResourceHalt"}


def run_smoke(args: argparse.Namespace, identity: Mapping[str, Any], trajectory: HorizonTrajectory,
              input_npz: Path) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    smoke_dir = run_dir / "smoke"
    status_path = smoke_dir / "run_status.json"
    identity_hash = _canonical_sha(identity)
    if status_path.is_file():
        if not args.resume:
            raise FileExistsError("smoke status already exists; use --resume to verify the immutable success")
        status = _verify_smoke_artifact(smoke_dir, identity_hash)
        return status
    if (smoke_dir / "samples" / "smoke_G0").exists():
        raise ValueError("unreceipted smoke sample exists; preserve and inspect it before continuing")
    _atomic_json(status_path, {"status": "RUNNING", "run_identity_sha256": identity_hash,
                               "smoke_calls": 0, "generation_started": False})
    torch = None
    monitor = None
    monitor_root: Path | None = None
    context = None
    cleanup_error: BaseException | None = None
    generation_started = False
    completed_calls = 0
    resource_stop_type: type[BaseException] | None = None
    try:
        torch = _verify_runtime()
        try:
            from task8_frozen.task8_live import (Task8LiveExecutor, Task8ResourceMonitor,
                                                 load_task8_live)
            from task8_frozen.run_umi_task8_experiment import (ResourceStop, Task8SampleStore,
                                                               evaluate_task8_resources)
        except ImportError:  # pragma: no cover
            from .task8_frozen.task8_live import Task8LiveExecutor, Task8ResourceMonitor, load_task8_live
            from .task8_frozen.run_umi_task8_experiment import (ResourceStop, Task8SampleStore,
                                                                evaluate_task8_resources)
        resource_stop_type = ResourceStop
        adapter = task8_input_for_window(trajectory, trajectory.window(1))
        monitor_root = _attempt_monitor_dir(smoke_dir)
        monitor = Task8ResourceMonitor(monitor_root, gpu_index=0)
        monitor.start()
        preload = monitor.check(phase="preload", starting_new_sample=True)
        evaluate_task8_resources(preload, phase="preload", starting_new_sample=True)
        context = load_task8_live(framework_root=args.framework_root, checkpoint=args.checkpoint,
                                  vae=args.vae, run_dir=smoke_dir / "setup", batch=adapter,
                                  release=True, resume=args.resume)
        true0_result = context.encoder.encode(np.array(trajectory.rgb[0], copy=True), precision="temporary_fp32")
        true0 = np.asarray(true0_result["arrays"]["actual_output"])
        if true0.dtype != np.float32:
            raise ValueError("smoke x0 truth encode is not FP32")
        condition_mask = np.asarray(context.feedback.mask, dtype=bool)
        spec = dict(build_schedule_call_plan(0)[0])
        window = trajectory.window(1)
        context.batch = task8_input_for_window(trajectory, window)
        executor = Task8LiveExecutor(context, resource_monitor=monitor, resource_phase="resource-smoke")
        generation_started = True
        _atomic_json(status_path, {"status": "RUNNING", "run_identity_sha256": identity_hash,
                                   "smoke_calls": 0, "generation_started": True})
        result = executor(spec)
        completed_calls = 1
        evidence = validate_call_result(spec, result,
            expected_action=window.actions[0], expected_condition_input=true0,
            expected_condition_carrier=context.feedback.embed_condition(true0), condition_mask=condition_mask)
        result["condition_rgb"] = np.array(trajectory.rgb[0], copy=True)
        result.update({"sample_id": "smoke_G0", "call": "G0", "horizon": 1, "seed": 0,
                       "smoke": True, "call_evidence": evidence})
        monitor.check(phase="resource-smoke", starting_new_sample=False)
        store = Task8SampleStore(smoke_dir / "samples")
        store.write_success("smoke_G0", result)
        del result, true0_result, true0, executor, context.batch
        gc.collect()
        torch.cuda.empty_cache()
        context.cleanup()
        context = None
        row = monitor._monitor.capture_sample("smoke_G0", "post_cleanup", 20, run_dir=smoke_dir)
        if row.get("decision_status") == "HARD_STOP":
            raise ResourceStop(f"{row.get('reason_code')}: {row.get('reason')}")
        sample_bytes = _tree_bytes(smoke_dir / "samples" / "smoke_G0")
        gate = evaluate_resource_gate(row, phase="post_cleanup", remaining_calls=20, sample_bytes=sample_bytes)
        status = {"status": "ENGINEERING_SMOKE_COMPLETE", "run_identity_sha256": identity_hash,
                  "smoke_calls": 1, "sample_id": "smoke_G0", "measured_sample_bytes": sample_bytes,
                  "remaining_formal_calls": 20, "resource_gate_status": "PASS",
                  "resource_gate": gate, "monitor_attempt": str(monitor_root)}
        validate_smoke_status(status, identity_hash)
        _atomic_json(status_path, status)
        return status
    except BaseException as error:
        _atomic_json(status_path, {"status": "RESOURCE_STOP" if (resource_stop_type is not None
                                   and isinstance(error, resource_stop_type))
                                   or _is_resource_stop(error) else "FAILED",
                                   "run_identity_sha256": identity_hash, "smoke_calls": completed_calls,
                                   "generation_started": generation_started, "error": repr(error)})
        raise
    finally:
        if context is not None:
            try:
                context.cleanup()
            except BaseException as error:
                cleanup_error = error
        gc.collect()
        if torch is not None:
            torch.cuda.empty_cache()
        if monitor is not None:
            try:
                monitor.stop()
            except BaseException as error:
                cleanup_error = cleanup_error or error
        if cleanup_error is not None:
            _atomic_json(status_path, {"status": "CLEANUP_FAILED", "run_identity_sha256": identity_hash,
                                       "error": repr(cleanup_error)})
            raise cleanup_error


def _truth_conditions_from_artifact(npz_path: Path, receipt_path: Path, *,
                                    identity: Mapping[str, Any], input_hash: str,
                                    actual_mask: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray, dict[str, Any]]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("status") != "PASS" or receipt.get("trajectory_npz_sha256") != file_sha256(npz_path)
            or receipt.get("run_identity_sha256") != _canonical_sha(identity)
            or receipt.get("input_trajectory_sha256") != input_hash
            or receipt.get("condition_mask_sha256") != task8_array_hash(actual_mask)):
        raise ValueError("truth endpoint artifact does not match current source/model/mask identity")
    with np.load(npz_path, allow_pickle=False) as data:
        conditions = {index: np.asarray(data[f"condition_x{index}"]) for index in TRUE_ENDPOINTS}
        mask = np.asarray(data["full_condition_mask"])
    if mask.dtype != np.bool_ or not np.array_equal(mask, actual_mask):
        raise ValueError("saved truth mask does not match the full runtime feedback mask")
    if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in conditions.values()):
        raise ValueError("saved true endpoint encodings must remain finite FP32")
    return conditions, mask, receipt


def _verify_formal_store(formal_dir: Path, identity_hash: str,
                         schedule_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        from task8_frozen.run_umi_task8_experiment import Task8SampleStore
    except ImportError:  # pragma: no cover
        from .task8_frozen.run_umi_task8_experiment import Task8SampleStore
    store = Task8SampleStore(formal_dir / "samples")
    plan = tuple(row for index in range(2) for row in build_schedule_call_plan(index))
    if len(plan) != 20 or len({row["sample_id"] for row in plan}) != 20:
        raise ValueError("formal Task 11 plan is not exactly twenty unique calls")
    all_evidence: dict[str, Any] = {}
    success_ids: set[str] = set()
    for schedule_result in schedule_results:
        if schedule_result.get("status") != "COMPLETE" or schedule_result.get("formal_calls") != 10:
            raise ValueError("one Task 11 seed schedule failed its ten-call orchestration gate")
        all_evidence.update(schedule_result.get("call_evidence", {}))
    if set(all_evidence) != {str(row["sample_id"]) for row in plan}:
        raise ValueError("formal call evidence is incomplete or differs from the locked 20-call plan")
    for spec in plan:
        sample_id = str(spec["sample_id"])
        evidence = all_evidence[sample_id]
        if evidence.get("action_steps") != 30 or evidence.get("condition_steps") != 30:
            raise ValueError(f"{sample_id} lacks all-30-step action/condition evidence")
        record = store.load_record(sample_id)
        try:
            for key in ("sample_id", "call", "schedule_index", "seed", "horizon", "mode",
                        "global_action_chunk", "global_condition_frame", "local_action_chunk_index"):
                if record.get(key) != spec.get(key):
                    raise ValueError(f"stored sample metadata differs from locked plan: {sample_id}:{key}")
            if (record.get("action_hash") != evidence.get("action_hash")
                    or record.get("prediction_noise_hash") != evidence.get("noise_hash")
                    or task8_array_hash(record["condition_input_fp32"]) != evidence.get("condition_hash")):
                raise ValueError(f"stored sample evidence differs from the validated call proof: {sample_id}")
            if evidence.get("effective_action_hash") != record.get("provenance", {}).get("action_evidence", {}).get("effective_action_hash"):
                raise ValueError(f"stored effective action hash differs from all-step evidence: {sample_id}")
            success_ids.add(sample_id)
        finally:
            del record
    if success_ids != {str(row["sample_id"]) for row in plan}:
        raise ValueError("not all twenty exact formal sample identities are present")
    for schedule_index in range(2):
        rows = {row["call"]: row for row in build_schedule_call_plan(schedule_index)}
        g0 = store.load_record(rows["G0"]["sample_id"])
        repeat = store.load_record(rows["G0_repeat"]["sample_id"])
        try:
            compare_exact_repeat(g0, repeat)
        finally:
            del g0, repeat
    return {"status": "PASS", "sample_count": len(success_ids), "call_evidence_count": len(all_evidence),
            "all_30_step_action_proofs": True, "all_30_step_condition_proofs": True,
            "exact_repeat_schedules": [0, 1], "run_identity_sha256": identity_hash}


def _mean_numeric(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float)) and row.get(key) is not None]
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64), dtype=np.float64))


def paired_horizon_error_deltas(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        horizon = int(row["horizon_chunks"])
        if horizon == 1:
            continue
        key = (int(row["schedule_index"]), horizon)
        mode = str(row["mode"])
        if mode not in {"TF", "AR"} or mode in grouped.setdefault(key, {}):
            raise ValueError("paired horizon delta rows must have unique TF/AR members")
        grouped[key][mode] = row
    result: list[dict[str, Any]] = []
    for (schedule_index, horizon), modes in sorted(grouped.items()):
        if set(modes) != {"TF", "AR"}:
            raise ValueError("paired horizon delta requires one TF and one AR endpoint")
        tf, ar = modes["TF"], modes["AR"]
        result.append({"schedule_index": schedule_index, "horizon_chunks": horizon,
                       "seed": ar.get("seed"),
                       "rgb_rmse_ar_minus_tf": float(ar["rmse"]) - float(tf["rmse"]),
                       "condition_latent_rms_ar_minus_tf":
                           float(ar["condition_latent_rms"]) - float(tf["condition_latent_rms"])})
    return result


def _write_horizon_analysis(formal_dir: Path, trajectory: HorizonTrajectory,
                            truth_conditions: Mapping[int, np.ndarray], condition_mask: np.ndarray,
                            run_identity: Mapping[str, Any]) -> dict[str, Any]:
    from task8_frozen.run_umi_task8_experiment import Task8SampleStore
    truth_receipt = json.loads((formal_dir / "truth_conditions.json").read_text(encoding="utf-8"))
    with np.load(formal_dir / "truth_conditions.npz", allow_pickle=False) as data:
        if not np.array_equal(np.asarray(data["full_condition_mask"]), condition_mask):
            raise ValueError("metric analysis mask differs from saved live feedback mask")
    if truth_receipt.get("condition_mask_sha256") != task8_array_hash(condition_mask):
        raise ValueError("metric analysis mask differs from the truth endpoint receipt")
    mask_only = condition_only_mask(condition_mask,
        temporal_axis=int(truth_receipt["condition_temporal_axis"]),
        condition_indexes=tuple(truth_receipt["condition_indexes"]))
    if truth_receipt.get("condition_only_mask_shape") != list(mask_only.shape):
        raise ValueError("saved condition-only mask geometry differs from runtime feedback geometry")
    if any(np.asarray(truth_conditions[index]).shape != mask_only.shape for index in TRUE_ENDPOINTS):
        raise ValueError("saved truth endpoints do not align with the runtime condition mask")
    store = Task8SampleStore(formal_dir / "samples")
    endpoint_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    prediction_rgb: dict[tuple[int, int, str], np.ndarray] = {}
    prediction_latent: dict[tuple[int, int, str], np.ndarray] = {}
    for schedule_index in range(2):
        for spec in build_schedule_call_plan(schedule_index):
            if spec["call"] == "G0_repeat":
                continue
            record = store.load_record(str(spec["sample_id"]))
            try:
                horizon = int(spec["horizon"])
                call = str(spec["call"])
                truth_frame = horizon * CHUNK_LENGTH
                truth_rgb = trajectory.rgb[truth_frame]
                generated = np.asarray(record["generated_rgb"])
                if generated.dtype != np.float32 or generated.shape != (3, CHUNK_LENGTH, 256, 256):
                    raise ValueError(f"{spec['sample_id']} output is not the full FP32 16-frame RGB chunk")
                endpoint = generated[:, -1]
                rgb_metrics = compute_endpoint_rgb_metrics(endpoint, truth_rgb)
                encoded = np.asarray(record["encoded_condition"])
                truth_encoded = np.asarray(truth_conditions[truth_frame])
                if encoded.shape != truth_encoded.shape or encoded.dtype != np.float32:
                    raise ValueError(f"{spec['sample_id']} FP32 endpoint encoding differs from true latent geometry")
                latent_metrics = masked_latent_metrics(encoded, truth_encoded, mask_only)
                endpoint_rows.append({"sample_id": spec["sample_id"], "schedule_index": schedule_index,
                    "seed": spec["seed"], "horizon_chunks": horizon, "endpoint_frame_index": truth_frame,
                    "mode": call, **rgb_metrics,
                    "condition_latent_rms": latent_metrics["rms"],
                    "condition_latent_cosine": latent_metrics["cosine"],
                    "action_hash": record["action_hash"], "noise_hash": record["prediction_noise_hash"]})
                prediction_rgb[(schedule_index, horizon, call)] = endpoint.copy()
                prediction_latent[(schedule_index, horizon, call)] = encoded.copy()
                frame_start = (horizon - 1) * CHUNK_LENGTH + 1
                for local in range(CHUNK_LENGTH):
                    metrics = compute_endpoint_rgb_metrics(generated[:, local],
                                                           trajectory.rgb[frame_start + local])
                    frame_rows.append({"sample_id": spec["sample_id"], "schedule_index": schedule_index,
                        "seed": spec["seed"], "horizon_chunks": horizon, "mode": call,
                        "generated_local_frame": local + 1, "source_frame_index": frame_start + local, **metrics})
            finally:
                del record
    geometry_rgb: list[dict[str, Any]] = []
    geometry_latent: list[dict[str, Any]] = []
    adjacent: list[dict[str, Any]] = []
    for schedule_index in range(2):
        for horizon in range(2, HORIZON_CHUNKS + 1):
            truth_rgb = trajectory.rgb[horizon * CHUNK_LENGTH]
            tf_rgb = prediction_rgb[(schedule_index, horizon, "TF")]
            ar_rgb = prediction_rgb[(schedule_index, horizon, "AR")]
            geometry_rgb.append({"schedule_index": schedule_index, "seed": SEED_SCHEDULES[schedule_index][horizon - 1],
                "horizon_chunks": horizon, **compute_error_geometry(tf_rgb, ar_rgb, truth_rgb)})
            truth_latent = truth_conditions[horizon * CHUNK_LENGTH]
            tf_latent = prediction_latent[(schedule_index, horizon, "TF")]
            ar_latent = prediction_latent[(schedule_index, horizon, "AR")]
            geometry_latent.append({"schedule_index": schedule_index,
                "seed": SEED_SCHEDULES[schedule_index][horizon - 1], "horizon_chunks": horizon,
                **compute_masked_error_geometry(tf_latent, ar_latent, truth_latent, mask_only)})
        previous_call = "G0"
        for horizon in range(2, HORIZON_CHUNKS + 1):
            previous_rgb = prediction_rgb[(schedule_index, horizon - 1, previous_call)]
            current_rgb = prediction_rgb[(schedule_index, horizon, "AR")]
            previous_truth = trajectory.rgb[(horizon - 1) * CHUNK_LENGTH]
            current_truth = trajectory.rgb[horizon * CHUNK_LENGTH]
            adjacent.append({"schedule_index": schedule_index, "seed": SEED_SCHEDULES[schedule_index][horizon - 1],
                             "previous_horizon_chunks": horizon - 1, "horizon_chunks": horizon,
                             **adjacent_ar_change(previous_rgb, current_rgb, previous_truth, current_truth)})
            previous_call = "AR"
    descriptive: list[dict[str, Any]] = []
    for horizon in range(1, HORIZON_CHUNKS + 1):
        calls = ("G0",) if horizon == 1 else ("TF", "AR")
        for call in calls:
            selected = [row for row in endpoint_rows if row["horizon_chunks"] == horizon and row["mode"] == call]
            descriptive.append({"horizon_chunks": horizon, "mode": call, "seed_count": len(selected),
                **{key: _mean_numeric(selected, key) for key in
                   ("rmse", "mae", "psnr", "condition_latent_rms", "condition_latent_cosine")},
                "finite_psnr_seed_count": sum(row.get("psnr") is not None for row in selected)})
    endpoint_deltas = paired_horizon_error_deltas(endpoint_rows)
    result = {"status": "PASS", "record_index": RECORD_INDEX,
        "run_identity_sha256": _canonical_sha(run_identity),
        "seed_schedules": [list(schedule) for schedule in SEED_SCHEDULES],
        "interpretation": "two preregistered seeds; means are descriptive, not population confidence intervals",
        "endpoint_metrics_per_seed": endpoint_rows,
        "descriptive_two_seed_means": descriptive,
        "per_frame_metrics": frame_rows,
        "tf_ar_error_geometry_rgb": geometry_rgb,
        "tf_ar_error_geometry_condition_masked_latent": geometry_latent,
        "tf_ar_endpoint_error_deltas": endpoint_deltas,
        "adjacent_ar_endpoint_changes": adjacent,
        "primary_horizon_axis": "generated chunk index 1 through 5",
        "seconds_axis": "nominally inferred at 5 Hz; source has no actual timestamp feature",
    }
    output_root = _next_attempt(formal_dir / "analysis_attempts", "attempt")
    _atomic_json(output_root / "task11_horizon_metrics.json", result)
    endpoint_fields = ("sample_id", "schedule_index", "seed", "horizon_chunks", "endpoint_frame_index", "mode",
                       "rmse", "mae", "psnr", "psnr_status", "condition_latent_rms", "condition_latent_cosine",
                       "action_hash", "noise_hash")
    frame_fields = ("sample_id", "schedule_index", "seed", "horizon_chunks", "mode", "generated_local_frame",
                    "source_frame_index", "rmse", "mae", "psnr", "psnr_status")
    _atomic_csv(output_root / "endpoint_metrics.csv", endpoint_fields, endpoint_rows)
    _atomic_csv(output_root / "per_frame_metrics.csv", frame_fields, frame_rows)
    _atomic_json(output_root / "tf_ar_error_geometry.json", {
        "rgb": geometry_rgb, "condition_masked_latent": geometry_latent, "adjacent_ar": adjacent})
    report = ("# Task 11 five-chunk horizon metrics\n\n"
        "Per-seed endpoints and time-aligned per-frame metrics are retained in CSV. The primary axis is generated "
        "chunk 1–5; nominal seconds are inferred at 5 Hz because the source contains no timestamp feature. Two-seed "
        "means are descriptive only. Both RGB and feedback-condition-masked latent TF/AR error geometry are reported.\n")
    (output_root / "report.md").write_text(report, encoding="utf-8")
    return {"status": "PASS", "output_dir": str(output_root),
            "task11_horizon_metrics_sha256": file_sha256(output_root / "task11_horizon_metrics.json"),
            "endpoint_metrics_sha256": file_sha256(output_root / "endpoint_metrics.csv"),
            "per_frame_metrics_sha256": file_sha256(output_root / "per_frame_metrics.csv"),
            "tf_ar_error_geometry_sha256": file_sha256(output_root / "tf_ar_error_geometry.json"),
            "report_sha256": file_sha256(output_root / "report.md")}


def _verify_truth_receipt(formal_dir: Path, identity: Mapping[str, Any], expected_receipt_sha256: str) -> None:
    npz_path = formal_dir / "truth_conditions.npz"
    receipt_path = formal_dir / "truth_conditions.json"
    if not npz_path.is_file() or not receipt_path.is_file() or file_sha256(receipt_path) != expected_receipt_sha256:
        raise ValueError("completed run truth endpoint receipt is missing or has changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("status") != "PASS" or receipt.get("run_identity_sha256") != _canonical_sha(identity)
            or receipt.get("trajectory_npz_sha256") != file_sha256(npz_path)
            or receipt.get("true_frame_indexes") != list(TRUE_ENDPOINTS)
            or receipt.get("encode_count") != len(TRUE_ENDPOINTS)):
        raise ValueError("completed run truth endpoint receipt does not bind the six real endpoint encodings")
    with np.load(npz_path, allow_pickle=False) as data:
        if set(data.files) != {f"condition_x{index}" for index in TRUE_ENDPOINTS} | {"full_condition_mask"}:
            raise ValueError("completed run truth endpoint NPZ is incomplete")
        mask = np.asarray(data["full_condition_mask"])
        if mask.dtype != np.bool_ or task8_array_hash(mask) != receipt.get("condition_mask_sha256"):
            raise ValueError("completed run truth endpoint mask hash differs from its receipt")
        indexes = tuple(receipt.get("condition_indexes", ()))
        selected_mask = condition_only_mask(mask, temporal_axis=receipt.get("condition_temporal_axis"),
                                            condition_indexes=indexes)
        if receipt.get("condition_only_mask_shape") != list(selected_mask.shape):
            raise ValueError("completed run condition geometry differs from its receipt")
        for index in TRUE_ENDPOINTS:
            value = np.asarray(data[f"condition_x{index}"])
            if value.dtype != np.float32 or value.shape != selected_mask.shape or not np.isfinite(value).all():
                raise ValueError(f"completed run true endpoint x{index} is invalid")


def _verify_horizon_analysis_artifacts(formal_dir: Path, identity: Mapping[str, Any],
                                       metrics: Mapping[str, Any]) -> None:
    if metrics.get("status") != "PASS":
        raise ValueError("completed run horizon analysis did not pass")
    output_root = Path(str(metrics.get("output_dir", ""))).resolve()
    try:
        output_root.relative_to(formal_dir.resolve())
    except ValueError as error:
        raise ValueError("completed run analysis artifacts escape the formal output directory") from error
    expected_files = {
        "task11_horizon_metrics.json": "task11_horizon_metrics_sha256",
        "endpoint_metrics.csv": "endpoint_metrics_sha256",
        "per_frame_metrics.csv": "per_frame_metrics_sha256",
        "tf_ar_error_geometry.json": "tf_ar_error_geometry_sha256",
        "report.md": "report_sha256",
    }
    for name, key in expected_files.items():
        artifact = output_root / name
        if not artifact.is_file() or file_sha256(artifact) != metrics.get(key):
            raise ValueError(f"completed run analysis artifact hash mismatch: {name}")
    summary = json.loads((output_root / "task11_horizon_metrics.json").read_text(encoding="utf-8"))
    if (summary.get("status") != "PASS" or summary.get("run_identity_sha256") != _canonical_sha(identity)
            or len(summary.get("endpoint_metrics_per_seed", [])) != 18
            or len(summary.get("per_frame_metrics", [])) != 288
            or len(summary.get("tf_ar_error_geometry_rgb", [])) != 8
            or len(summary.get("tf_ar_error_geometry_condition_masked_latent", [])) != 8
            or len(summary.get("adjacent_ar_endpoint_changes", [])) != 8
            or len(summary.get("tf_ar_endpoint_error_deltas", [])) != 8):
        raise ValueError("completed run horizon metrics are incomplete or identity-mismatched")


def _verify_completed_run(formal_dir: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    status = json.loads((formal_dir / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE" or status.get("run_identity_sha256") != _canonical_sha(identity):
        raise ValueError("existing formal status is not a complete match to current run identity")
    gates = status.get("completion_gates")
    if (not isinstance(gates, Mapping) or any(gates.get(key) is not True for key in
            ("preflight", "smoke", "truth_endpoint_receipt", "twenty_hash_verified_samples",
             "all_step_action_and_condition_evidence", "exact_repeat_both_schedules",
             "resource_gates", "horizon_analysis"))):
        raise ValueError("completed formal status lacks required data/model/resource/analysis gates")
    expected_ids = [str(row["sample_id"]) for index in range(2) for row in build_schedule_call_plan(index)]
    if (status.get("formal_calls") != 20 or status.get("smoke_calls") != 1
            or status.get("sample_ids") != expected_ids
            or len(status.get("schedule_results", [])) != 2):
        raise ValueError("completed formal status does not contain the exact twenty-call protocol")
    receipt_sha256 = status.get("truth_endpoint_receipt_sha256")
    if not isinstance(receipt_sha256, str):
        raise ValueError("completed formal status omits its truth endpoint receipt hash")
    _verify_truth_receipt(formal_dir, identity, receipt_sha256)
    store_gate = _verify_formal_store(formal_dir, _canonical_sha(identity), status["schedule_results"])
    if store_gate != status.get("sample_store_verification"):
        raise ValueError("completed formal sample verification receipt differs from reloaded samples")
    _verify_horizon_analysis_artifacts(formal_dir, identity, status.get("horizon_analysis", {}))
    latest_resource = formal_dir / "resource_latest.json"
    if not latest_resource.is_file():
        raise ValueError("completed formal resource evidence is missing")
    resource_payload = json.loads(latest_resource.read_text(encoding="utf-8"))
    if (resource_payload.get("capture_ordinal") != 19
            or resource_payload.get("gate", {}).get("status") not in {"OK", "WARNING"}):
        raise ValueError("completed formal resource evidence does not verify the twentieth call")
    return status


def run_formal(args: argparse.Namespace, identity: Mapping[str, Any], trajectory: HorizonTrajectory,
               report: Mapping[str, Any], input_npz: Path) -> dict[str, Any]:
    formal_dir = Path(args.run_dir).resolve() / "formal"
    formal_dir.mkdir(parents=True, exist_ok=True)
    status_path = formal_dir / "run_status.json"
    identity_hash = _canonical_sha(identity)
    prior_status: dict[str, Any] | None = None
    if status_path.is_file():
        prior_status = json.loads(status_path.read_text(encoding="utf-8"))
        if prior_status.get("run_identity_sha256") != identity_hash:
            raise ValueError("formal resume status identity differs from the locked run")
        if prior_status.get("status") == "COMPLETE":
            if not args.resume:
                raise FileExistsError("formal run already completed; --resume is required for immutable verification")
            return _verify_completed_run(formal_dir, identity)
        if not args.resume:
            raise FileExistsError("formal run is incomplete; exact --resume is required")
    smoke = validate_smoke_status(json.loads((Path(args.run_dir) / "smoke" / "run_status.json").read_text(encoding="utf-8")),
                                  identity_hash)
    del smoke
    smoke_status = _verify_smoke_artifact(Path(args.run_dir) / "smoke", identity_hash)
    measured_smoke_bytes = int(smoke_status["measured_sample_bytes"])
    prior_ids = [] if prior_status is None else prior_status.get("sample_ids", [])
    valid_plan_ids = {str(row["sample_id"]) for index in range(2) for row in build_schedule_call_plan(index)}
    if (not isinstance(prior_ids, list) or any(not isinstance(item, str) for item in prior_ids)
            or len(prior_ids) != len(set(prior_ids)) or not set(prior_ids).issubset(valid_plan_ids)):
        raise ValueError("formal resume status contains invalid successful sample identities")
    all_sample_ids: list[str] = list(prior_ids)
    _atomic_json(status_path, {"status": "RUNNING", "run_identity_sha256": identity_hash,
                               "record_index": RECORD_INDEX, "formal_calls_target": 20,
                               "sample_ids": all_sample_ids, "schedules_complete": [],
                               "generation_started": False})
    torch = None
    monitor = None
    context = None
    cleanup_error: BaseException | None = None
    resource_stop_type: type[BaseException] | None = None
    schedule_results: list[dict[str, Any]] = []
    generation_started = False
    try:
        try:
            from task8_frozen.run_umi_task8_experiment import (ResourceStop, Task8SampleStore,
                                                               evaluate_task8_resources)
            from task8_frozen.task8_live import (Task8LiveExecutor, Task8ResourceMonitor,
                                                 load_task8_live)
        except ImportError:  # pragma: no cover - package import path
            from .task8_frozen.run_umi_task8_experiment import (ResourceStop, Task8SampleStore,
                                                                evaluate_task8_resources)
            from .task8_frozen.task8_live import (Task8LiveExecutor, Task8ResourceMonitor,
                                                  load_task8_live)
        resource_stop_type = ResourceStop
        torch = _verify_runtime()
        batch = task8_input_for_window(trajectory, trajectory.window(1))
        monitor = Task8ResourceMonitor(_attempt_monitor_dir(formal_dir), gpu_index=0)
        monitor.start()
        preload = monitor.check(phase="preload", starting_new_sample=True)
        evaluate_task8_resources(preload, phase="preload", starting_new_sample=True,
                                 remaining_samples=20, mean_success_sample_bytes=measured_smoke_bytes)
        context = load_task8_live(framework_root=args.framework_root, checkpoint=args.checkpoint,
                                  vae=args.vae, run_dir=formal_dir / "setup", batch=batch,
                                  release=True, resume=args.resume)
        formal_store = Task8SampleStore(formal_dir / "samples")
        truth_path, truth_receipt = formal_dir / "truth_conditions.npz", formal_dir / "truth_conditions.json"
        conditions, condition_mask, truth_evidence = _encode_truth_endpoints(
            context, trajectory, formal_dir, identity=identity, input_hash=file_sha256(input_npz),
            resume=args.resume)
        executor = Task8LiveExecutor(context, resource_monitor=monitor, resource_phase="formal")
        global_sample_ordinal = 0

        def execute_call(spec: Mapping[str, Any], window: Any) -> Mapping[str, Any]:
            nonlocal generation_started
            context.batch = task8_input_for_window(trajectory, window)
            if spec.get("condition_source") == "g0_float_last_fp32":
                condition_rgb = spec.get("condition_rgb")
                if not isinstance(condition_rgb, np.ndarray) or condition_rgb.dtype != np.float32:
                    raise ValueError("autoregressive model call lacks its prior FP32 RGB endpoint")
                executor._last_g0_frame = np.array(condition_rgb, copy=True)
            generation_started = True
            _atomic_json(status_path, {"status": "RUNNING", "run_identity_sha256": identity_hash,
                "record_index": RECORD_INDEX, "formal_calls_target": 20, "sample_ids": all_sample_ids,
                "schedules_complete": [item["schedule_index"] for item in schedule_results],
                "schedule_results": schedule_results, "generation_started": True})
            return executor(spec)

        def after_sample(sample_id: str, remaining_in_protocol: int) -> None:
            nonlocal global_sample_ordinal
            if sample_id not in all_sample_ids:
                all_sample_ids.append(sample_id)
            _atomic_json(status_path, {"status": "RUNNING", "run_identity_sha256": identity_hash,
                "record_index": RECORD_INDEX, "formal_calls_target": 20,
                "sample_ids": all_sample_ids,
                "schedules_complete": [item["schedule_index"] for item in schedule_results],
                "schedule_results": schedule_results, "generation_started": generation_started})
            gc.collect()
            torch.cuda.empty_cache()
            capture = getattr(monitor._monitor, "capture_sample", None)
            if not callable(capture):
                raise ResourceStop("MONITOR_FAILURE: post-cleanup capture is unavailable")
            row = capture(f"formal_call_{global_sample_ordinal:02d}", "post_cleanup",
                          remaining_in_protocol, run_dir=formal_dir)
            if row.get("decision_status") == "HARD_STOP":
                raise ResourceStop(f"{row.get('reason_code')}: {row.get('reason')}")
            size = row.get("mean_success_sample_bytes")
            if isinstance(size, bool) or not isinstance(size, (int, float)) or size <= 0:
                raise ResourceStop("DISK_FORECAST_UNAVAILABLE: no measured successful sample size")
            gate = evaluate_resource_gate(row, phase="formal", remaining_calls=remaining_in_protocol,
                                          sample_bytes=int(size))
            _atomic_json(formal_dir / "resource_latest.json", {
                "capture_ordinal": global_sample_ordinal, "resource_row": row, "gate": gate})
            monitor.check(phase="formal", starting_new_sample=False)
            global_sample_ordinal += 1

        for schedule_index in range(2):
            snapshot = monitor.check(phase="formal", starting_new_sample=True)
            remaining = 20 - len(all_sample_ids)
            evaluate_task8_resources(snapshot, phase="formal", starting_new_sample=True,
                                     remaining_samples=remaining,
                                     mean_success_sample_bytes=measured_smoke_bytes)
            schedule_result = execute_schedule(trajectory, schedule_index, execute_call,
                truth_conditions=conditions, condition_mask=context.feedback.mask,
                embed_condition=context.feedback.embed_condition, sample_store=formal_store,
                resume=args.resume, future_formal_calls=10 if schedule_index == 0 else 0,
                after_sample=after_sample, capture_results=False)
            if schedule_result.get("status") != "COMPLETE" or schedule_result.get("formal_calls") != 10:
                raise RuntimeError(f"schedule {schedule_index} failed the fixed ten-call plan")
            schedule_results.append(schedule_result)
            all_sample_ids = list(dict.fromkeys(all_sample_ids + schedule_result["sample_ids"]))
            _atomic_json(status_path, {"status": "RUNNING", "run_identity_sha256": identity_hash,
                "record_index": RECORD_INDEX, "formal_calls_target": 20,
                "sample_ids": all_sample_ids, "schedules_complete": [row["schedule_index"] for row in schedule_results],
                "schedule_results": schedule_results, "generation_started": generation_started})
        del executor, batch
        gc.collect()
        torch.cuda.empty_cache()
        context.cleanup()
        context = None
        cleanup_row = monitor._monitor.capture_sample("task11_context_cleanup", "post_cleanup", 0, run_dir=formal_dir)
        if cleanup_row.get("decision_status") == "HARD_STOP":
            raise ResourceStop(f"{cleanup_row.get('reason_code')}: {cleanup_row.get('reason')}")
        cleanup_size = cleanup_row.get("mean_success_sample_bytes")
        if isinstance(cleanup_size, (int, float)) and cleanup_size > 0:
            evaluate_resource_gate(cleanup_row, phase="formal", remaining_calls=0, sample_bytes=int(cleanup_size))
        monitor.check(phase="formal", starting_new_sample=False)
    except BaseException as error:
        _atomic_json(status_path, {"status": "RESOURCE_STOP" if (resource_stop_type is not None
                                   and isinstance(error, resource_stop_type))
                                   or _is_resource_stop(error) else "FAILED",
            "run_identity_sha256": identity_hash, "record_index": RECORD_INDEX,
            "formal_calls_target": 20, "sample_ids": all_sample_ids,
            "schedules_complete": [item["schedule_index"] for item in schedule_results],
            "schedule_results": schedule_results, "generation_started": generation_started,
            "error": repr(error)})
        raise
    finally:
        if context is not None:
            try:
                context.cleanup()
            except BaseException as error:
                cleanup_error = error
        gc.collect()
        if torch is not None:
            torch.cuda.empty_cache()
        if monitor is not None:
            try:
                monitor.stop()
            except BaseException as error:
                cleanup_error = cleanup_error or error
        if cleanup_error is not None:
            _atomic_json(status_path, {"status": "CLEANUP_FAILED", "run_identity_sha256": identity_hash,
                                       "error": repr(cleanup_error)})
            raise cleanup_error
    sample_gate = _verify_formal_store(formal_dir, identity_hash, schedule_results)
    with np.load(truth_path, allow_pickle=False) as data:
        truth_conditions_saved = {index: np.asarray(data[f"condition_x{index}"]) for index in TRUE_ENDPOINTS}
        condition_mask_saved = np.asarray(data["full_condition_mask"])
    metrics = _write_horizon_analysis(formal_dir, trajectory, truth_conditions_saved,
                                      condition_mask_saved, identity)
    gates = {"preflight": True, "smoke": True, "truth_endpoint_receipt": truth_evidence.get("status") == "PASS",
             "twenty_hash_verified_samples": sample_gate.get("sample_count") == 20,
             "all_step_action_and_condition_evidence": sample_gate.get("all_30_step_action_proofs") is True
                and sample_gate.get("all_30_step_condition_proofs") is True,
             "exact_repeat_both_schedules": sample_gate.get("exact_repeat_schedules") == [0, 1],
             "resource_gates": True, "horizon_analysis": metrics.get("status") == "PASS"}
    if not all(gates.values()):
        raise ValueError("Task 11 completion bundle failed one or more scientific/runtime gates")
    completed = {"status": "COMPLETE", "run_identity_sha256": identity_hash,
        "record_index": RECORD_INDEX, "horizon_chunks": HORIZON_CHUNKS,
        "seed_schedules": [list(schedule) for schedule in SEED_SCHEDULES],
        "formal_calls": 20, "smoke_calls": 1, "sample_ids": all_sample_ids,
        "schedule_results": schedule_results, "completion_gates": gates,
        "sample_store_verification": sample_gate, "truth_endpoint_receipt_sha256": file_sha256(truth_receipt),
        "horizon_analysis": metrics, "resource_monitor_attempt": str(monitor.run_dir)}
    _atomic_json(status_path, completed)
    _verify_completed_run(formal_dir, identity)
    return completed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "smoke", "formal"), required=True)
    parser.add_argument("--input-run", required=True,
                        help="immutable Task 11 preflight bundle directory; generated only during preflight stage")
    parser.add_argument("--run-dir", required=True, help="this five_chunk/ run directory")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--framework-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--normalizer-stats", required=True)
    parser.add_argument("--official-parity-report", required=True)
    parser.add_argument("--record-index", type=int, choices=(15,), default=15)
    parser.add_argument("--horizon-chunks", type=int, choices=(5,), default=5)
    parser.add_argument("--seed-schedules", choices=("both",), default="both")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--release", action="store_true", help="root authorization for this exact live stage")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_cli_contract(args)
    if args.stage != "preflight" and not args.release:
        raise ValueError("model generation is unavailable without explicit root release")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("Task 11 requires CUDA_VISIBLE_DEVICES=0 and HF_HUB_OFFLINE=1")
    input_run, run_dir = Path(args.input_run).resolve(), Path(args.run_dir).resolve()
    report_path, npz_path = _preflight_paths(input_run)
    has_report, has_npz = report_path.is_file(), npz_path.is_file()
    if has_report != has_npz:
        raise ValueError("partial preflight input is immutable and cannot be resumed")
    if args.stage == "preflight" and not has_report:
        run_long_horizon_preflight(args.dataset_root, input_run, args.normalizer_stats,
                                   args.official_parity_report)
    report, trajectory, report_path, npz_path = _load_preflight(input_run)
    identity = _run_identity(args, report, report_path, npz_path)
    if args.stage == "preflight":
        saved = _lock_or_check_run_identity(run_dir, identity, create=True, resume=args.resume)
        identity_hash = _canonical_sha(saved)
        status_path = run_dir / "preflight_status.json"
        status = {"status": "PREFLIGHT_LOCKED_GENERATION_NOT_PERFORMED",
                  "run_identity_sha256": identity_hash, "input_run": str(input_run),
                  "report_sha256": file_sha256(report_path), "trajectory_npz_sha256": file_sha256(npz_path)}
        if status_path.exists():
            if not args.resume or json.loads(status_path.read_text(encoding="utf-8")) != status:
                raise FileExistsError("preflight run status is immutable; exact --resume is required")
        else:
            _atomic_json(status_path, status)
        print(json.dumps(status, sort_keys=True))
        return 0
    _lock_or_check_run_identity(run_dir, identity, create=False, resume=args.resume)
    _verify_preflight_status(run_dir, identity, input_run,
                             file_sha256(report_path), file_sha256(npz_path))
    if args.stage == "smoke":
        status = run_smoke(args, identity, trajectory, npz_path)
    else:
        status = run_formal(args, identity, trajectory, report, npz_path)
    print(json.dumps({"status": status["status"], "run_dir": str(run_dir),
                      "run_identity_sha256": _canonical_sha(identity)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
