"""Approved Task 8 formal-call orchestrator.

The module is deliberately a thin, fail-closed seam around one already
reviewed ``Task8LiveContext``/``Task8LiveExecutor``.  It never invokes the
engineering-smoke CLI and never retries a failed formal call.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


class FormalBlocked(RuntimeError):
    """The formal run has not met its explicit release or evidence gates."""


class FormalError(RuntimeError):
    """Formal evidence or interface validation failed; do not retry."""


PREPROCESSING_IDENTITY = {
    "input_shape": [33, 3, 256, 256], "rgb_range": "[0,1]", "resize": "none",
    "normalization": "runtime-approved-256-float-rgb", "action_shape": [2, 16, 10],
}
FORMAL_HASH_KEYS = ("code", "model", "vae", "config", "data", "actions", "preprocessing", "noise")
OFFICIAL_FRAMEWORK_COMMIT = "ffa9c6b60a6b04b2fae337577bc6cbd8a93c39f5"
_SOURCE_EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
TASK8_SOURCE_MODULES = (
    "analyze_umi_task6", "analyze_umi_task8", "run_umi_task6_experiment",
    "run_umi_task6_official", "run_umi_task7_experiment", "run_umi_task8_experiment",
    "task8_formal", "task8_live", "umi_fd_post_vae_bridge", "umi_fd_post_vae_scan",
    "umi_precision_identity", "umi_precision_official", "umi_precision_primitives",
    "umi_precision_reanalysis", "umi_precision_runtime", "umi_precision_storage",
    "umi_task5_decoder", "umi_task5_primitives", "umi_task6_cosmos_loader",
    "umi_task6_decoder", "umi_task6_operational", "umi_task6_primitives",
    "umi_task6_runtime", "umi_task7_encoder", "umi_task7_runtime", "umi_task8_runtime",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": _array_hash(value)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("nonfinite value cannot enter formal JSON")
    return value


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _array_hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode()); digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode()); digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_path(path: str | Path) -> str:
    source = Path(path).resolve()
    if not source.exists():
        raise FormalBlocked(f"binding path is missing: {source}")
    digest = hashlib.sha256()
    if source.is_file():
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    for child in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = child.relative_to(source).as_posix().encode()
        digest.update(relative); digest.update(b"\0")
        with child.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _sha256_source_path(path: str | Path) -> str:
    """Hash stable source/config files while ignoring VCS and interpreter caches."""
    source = Path(path).resolve()
    if not source.exists():
        raise FormalBlocked(f"binding source path is missing: {source}")
    if source.is_file():
        return _sha256_path(source)
    digest = hashlib.sha256()
    files = []
    for child in source.rglob("*"):
        if not child.is_file():
            continue
        relative = child.relative_to(source)
        if any(part in _SOURCE_EXCLUDED_PARTS for part in relative.parts):
            continue
        if child.suffix.lower() in {".pyc", ".pyo"}:
            continue
        files.append((relative.as_posix(), child))
    for relative, child in sorted(files):
        digest.update(relative.encode()); digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_path(child)))
    return digest.hexdigest()


def _framework_commit(framework_root: str | Path) -> str:
    root = Path(framework_root).resolve()
    try:
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip().lower()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise FormalBlocked(f"cannot verify pinned framework revision at {root}") from error
    if revision != OFFICIAL_FRAMEWORK_COMMIT:
        raise FormalBlocked(f"framework commit differs from pinned {OFFICIAL_FRAMEWORK_COMMIT}: {revision}")
    if dirty:
        raise FormalBlocked("framework tracked source has local modifications")
    return revision


def _task8_source_paths(experiments_root: str | Path | None = None) -> list[Path]:
    root = Path(experiments_root).resolve() if experiments_root is not None else Path(__file__).resolve().parent
    paths = [root / f"{module}.py" for module in TASK8_SOURCE_MODULES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FormalBlocked(f"Task 8 transitive source manifest is incomplete: {missing}")
    return paths


def build_binding(*, code_paths: Sequence[str | Path], model_path: str | Path,
                  vae_path: str | Path, input_path: str | Path, actions: Any,
                  metadata: Mapping[str, Any], framework_commit: str,
                  runtime_config: Mapping[str, Any] | None = None,
                  seed_pair: tuple[int, int] = (0, 1)) -> dict[str, str]:
    """Build the exact eight hash identities consumed by ``run_task8``."""
    if not isinstance(metadata, Mapping):
        raise FormalBlocked("preflight metadata must be a mapping for the binding")
    commit = str(framework_commit).lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise FormalBlocked("framework commit must be a full 40-character Git SHA")
    if (len(seed_pair) != 2 or any(not isinstance(seed, int) or seed < 0 for seed in seed_pair)
            or seed_pair[0] == seed_pair[1]):
        raise FormalBlocked("binding requires distinct nonnegative chunk seeds")
    code_digest = _canonical_hash({"paths": [str(Path(path).resolve()) for path in code_paths],
                                   "content": [_sha256_source_path(path) for path in code_paths],
                                   "framework_commit": commit})
    return {
        "code": code_digest,
        "model": _sha256_path(model_path),
        "vae": _sha256_path(vae_path),
        "config": _canonical_hash({"runtime": dict(runtime_config or {}), "metadata": metadata}),
        "data": _sha256_path(input_path),
        "actions": _array_hash(actions),
        "preprocessing": _canonical_hash(PREPROCESSING_IDENTITY),
        "noise": _canonical_hash({"policy": "prediction-region-seeded",
                                   "seeds": [seed_pair[0], seed_pair[0], seed_pair[1], seed_pair[1]],
                                   "pair": "TF2-AR2"}),
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalError(f"invalid JSON evidence: {path}") from error
    if not isinstance(value, Mapping):
        raise FormalError(f"JSON evidence is not an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_smoke_run(smoke_dir: str | Path) -> dict[str, Any]:
    """Accept only a completed, hash-verified engineering smoke artifact."""
    root = Path(smoke_dir).resolve()
    status = _read_json(root / "run_status.json")
    if status.get("status") != "ENGINEERING_SMOKE_COMPLETE":
        raise FormalError(f"smoke is not complete: {status.get('status')!r}")
    measured = status.get("measured_sample_bytes")
    if not isinstance(measured, int) or measured <= 0:
        raise FormalError("smoke lacks a positive measured sample footprint")
    try:
        from .run_umi_task8_experiment import Task8SampleStore
    except ImportError:  # pragma: no cover
        from run_umi_task8_experiment import Task8SampleStore
    store = Task8SampleStore(root / "samples")
    try:
        record = store.load_record("engineering_smoke")
        required = ("output_full", "generated_rgb", "decoded_last_rgb", "encoded_condition",
                    "condition_input_fp32", "prediction_noise_hash", "packed_action_token_hashes")
        if any(key not in record for key in required):
            raise FormalError("smoke record lacks required formal-boundary evidence")
    finally:
        if "record" in locals():
            del record
        gc.collect()
    sample_root = root / "samples" / "engineering_smoke"
    actual_bytes = sum(path.stat().st_size for path in sample_root.rglob("*") if path.is_file())
    if actual_bytes != measured:
        raise FormalError(f"smoke measured bytes disagree with artifacts: {measured} vs {actual_bytes}")
    forecast = status.get("forecast_free_gib_after_four_calls")
    return {"status": status.get("status"), "measured_sample_bytes": measured,
            "status_reported_forecast_free_gib_after_four_calls": forecast,
            "sample_dir": str(sample_root)}


def _require_array(record: Mapping[str, Any], key: str, *, dtype: str | None = None) -> np.ndarray:
    value = record.get(key)
    if not isinstance(value, np.ndarray) or value.size == 0 or not np.all(np.isfinite(value)):
        raise FormalError(f"formal result lacks finite ndarray evidence: {key}")
    if dtype is not None and str(value.dtype) != dtype:
        raise FormalError(f"formal result {key} dtype is {value.dtype}, expected {dtype}")
    return np.ascontiguousarray(value)


class VerifiedFormalExecutor:
    """Bind AR2 to G0's saved decoded frame before invoking live execution."""

    def __init__(self, live_executor: Callable[[Mapping[str, Any]], Mapping[str, Any]], *, formal_dir: str | Path | None = None,
                 monitor: Any | None = None, capture_path: str | Path | None = None):
        self.live_executor = live_executor
        self.monitor = monitor
        self.capture_path = Path(capture_path).resolve() if capture_path is not None else None
        self.g0_decoded_last: np.ndarray | None = None
        self.g0_encoded_condition: np.ndarray | None = None
        self.resource_capture: list[dict[str, Any]] = []
        if formal_dir is not None:
            self._load_saved_g0(Path(formal_dir).resolve())

    def _load_saved_g0(self, formal_dir: Path) -> None:
        sample = formal_dir / "samples" / "G0_real_x0_seed0"
        if not sample.is_dir():
            return
        try:
            from .run_umi_task8_experiment import Task8SampleStore
        except ImportError:  # pragma: no cover
            from run_umi_task8_experiment import Task8SampleStore
        record = Task8SampleStore(formal_dir / "samples").load_record("G0_real_x0_seed0")
        try:
            self.g0_decoded_last = _require_array(record, "decoded_last_rgb", dtype="float32").copy()
            self.g0_encoded_condition = _require_array(record, "encoded_condition", dtype="float32").copy()
        finally:
            del record

    def _sync_and_capture(self, call: str) -> None:
        try:
            import torch
        except ImportError:  # CPU fixture environments do not need CUDA synchronization.
            torch = None
        if torch is not None and torch.cuda.is_available():
            # A synchronization failure can report an asynchronous kernel error
            # or OOM. Let it fail the formal run and persist its status.
            torch.cuda.synchronize()
        if self.monitor is not None:
            snapshot = self.monitor.check(phase="formal", starting_new_sample=False)
            self.resource_capture.append({"call": call, "snapshot": dict(snapshot)})
            if self.capture_path is not None:
                _atomic_json(self.capture_path, {"samples": self.resource_capture})
        gc.collect()

    def __call__(self, spec: Mapping[str, Any]) -> Mapping[str, Any]:
        call = str(spec.get("call"))
        if call == "AR2":
            requested = spec.get("condition_rgb")
            if not isinstance(requested, np.ndarray):
                raise FormalError("AR2 request lacks runner-supplied G0 condition_rgb")
            if self.g0_decoded_last is None:
                raise FormalError("AR2 has no saved G0 decoded frame")
            if requested.dtype != self.g0_decoded_last.dtype or requested.shape != self.g0_decoded_last.shape \
                    or requested.tobytes(order="C") != self.g0_decoded_last.tobytes(order="C"):
                raise FormalError("AR2 condition_rgb differs bytewise from G0 decoded_last_rgb")
            if not hasattr(self.live_executor, "_last_g0_frame"):
                raise FormalError("live executor lacks the explicit G0 feedback-frame seam")
            self.live_executor._last_g0_frame = requested.copy()
        result = self.live_executor(spec)
        if not isinstance(result, Mapping):
            raise FormalError(f"{call} live executor returned a non-mapping result")
        if call == "G0":
            self.g0_decoded_last = _require_array(result, "decoded_last_rgb", dtype="float32").copy()
            self.g0_encoded_condition = _require_array(result, "encoded_condition", dtype="float32").copy()
        self._sync_and_capture(call)
        return result


def _save_gt_conditions(context: Any, batch: Any, formal_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    encoder = getattr(context, "encoder", None)
    feedback = getattr(context, "feedback", None)
    if encoder is None or feedback is None:
        raise FormalError("live context lacks encoder/feedback for independent FP32 ground truth")
    evidence: dict[str, Any] = {"source": "fresh_temporary_fp32_encoder", "setup_bf16_not_reused": True, "conditions": {}}
    outputs: dict[str, np.ndarray] = {}
    for label, index in (("gt_condition_x16", 16), ("gt_condition_x32", 32)):
        result = encoder.encode(np.array(batch.rgb[index], copy=True), precision="temporary_fp32")
        arrays = result.get("arrays") if isinstance(result, Mapping) else None
        output = arrays.get("actual_output") if isinstance(arrays, Mapping) else None
        if not isinstance(output, np.ndarray) or output.dtype != np.float32:
            raise FormalError(f"{label} lacks fresh FP32 encoder output")
        outputs[label] = np.ascontiguousarray(output.copy())
        evidence["conditions"][label] = _json_safe(result.get("evidence", {}))
        del result
    mask = np.asarray(feedback.mask, dtype=bool)
    indexes = np.asarray(feedback.condition_indexes, dtype=np.int64)
    axis = getattr(feedback, "temporal_axis", None)
    if axis is None or mask.ndim == 0:
        raise FormalError("feedback lacks authoritative temporal axis for the condition mask")
    axis = int(axis)
    if axis < 0:
        axis += mask.ndim
    if axis < 0 or axis >= mask.ndim or indexes.size == 0:
        raise FormalError("feedback condition indexes or temporal axis are invalid")
    if np.any(indexes < 0) or np.any(indexes >= mask.shape[axis]):
        raise FormalError("feedback condition index is outside the condition mask")
    condition_mask = np.ascontiguousarray(np.take(mask, indexes, axis=axis), dtype=bool)
    for label, output in outputs.items():
        if output.shape != condition_mask.shape:
            raise FormalError(f"{label} shape {output.shape} does not match condition-only mask {condition_mask.shape}")
    if not np.any(condition_mask):
        raise FormalError("condition-only mask is empty")
    np.savez_compressed(formal_dir / "ground_truth_conditions.npz",
                        gt_condition_x16=outputs["gt_condition_x16"], gt_condition_x32=outputs["gt_condition_x32"],
                        condition_mask=condition_mask, full_condition_mask=mask,
                        condition_indexes=indexes, temporal_axis=np.asarray(axis, dtype=np.int64))
    _atomic_json(formal_dir / "ground_truth_condition_evidence.json", evidence)
    return outputs["gt_condition_x16"], outputs["gt_condition_x32"]


def _write_preliminary_outputs(result: Mapping[str, Any], analysis_dir: Path) -> None:
    metrics = result.get("metrics", {})
    comparison = {"status": result.get("status"), "metrics": metrics,
                  "conditions": result.get("conditions"), "information_availability": result.get("information_availability")}
    _atomic_json(analysis_dir / "comparison.json", comparison)
    rows = []
    for key in ("E1", "E_TF2", "E_AR2"):
        row = metrics.get(key, {})
        rows.append({"metric": key, "rmse": row.get("rmse"), "mae": row.get("mae"), "psnr": row.get("psnr")})
    with (analysis_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["metric", "rmse", "mae", "psnr"]); writer.writeheader(); writer.writerows(rows)
    _atomic_json(analysis_dir / "math.json", metrics.get("delta_feedback_error", {}))
    report = "# Task 8 preliminary formal comparison\n\n"
    report += f"Repeatability status: `{result.get('status')}`.\n\n"
    report += "This is one observed-pose-conditioned trajectory; no threshold-based scientific PASS is asserted.\n\n"
    report += "Per-frame metrics are in `per_frame_metrics.csv`; saved tensors remain under the formal sample store.\n"
    (analysis_dir / "report.md").write_text(report, encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [row["metric"] for row in rows]
        values = [float(row["rmse"]) for row in rows]
        figure, axis = plt.subplots(figsize=(6, 4)); axis.bar(labels, values); axis.set_ylabel("RGB RMSE")
        axis.set_title("Task 8 preliminary formal comparison"); figure.tight_layout()
        figure.savefig(analysis_dir / "comparison.png", dpi=150); figure.savefig(analysis_dir / "comparison.svg")
        plt.close(figure)
    except ImportError:
        (analysis_dir / "plots_unavailable.txt").write_text("matplotlib is unavailable; no plot was generated.\n", encoding="utf-8")


def _write_manifest(root: Path, *, binding: Mapping[str, str]) -> None:
    files = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"):
        files[str(path.relative_to(root).as_posix())] = {"bytes": path.stat().st_size, "sha256": _sha256_path(path)}
    _atomic_json(root / "manifest.json", {"binding": dict(binding), "files": files,
                                            "full_tensors": "retained in formal sample artifacts"})
    bundle = root / "light_review_bundle"; bundle.mkdir(exist_ok=True)
    analysis = root / "analysis"
    for name in ("comparison.json", "comparison.csv", "math.json", "report.md", "comparison.png", "comparison.svg"):
        source = analysis / name
        if source.is_file(): shutil.copy2(source, bundle / name)
    _atomic_json(bundle / "bundle.json", {"contents": sorted(path.name for path in bundle.iterdir()),
                                           "full_tensors": "not copied; see formal sample store"})


def _next_attempt_dir(base: Path) -> Path:
    attempts = base / "attempts"
    index = 1
    while (attempts / f"attempt_{index:03d}").exists():
        index += 1
    return attempts / f"attempt_{index:03d}"


def _validate_formal_identity(root: Path, *, binding: Mapping[str, str], resume: bool,
                              formal_call_plan: Sequence[Mapping[str, Any]]) -> None:
    setup_binding = root / "setup" / "binding.json"
    formal_dir = root / "formal"
    analysis_dir = root / "analysis"
    status_path = root / "task8_formal_status.json"
    if resume:
        if not setup_binding.is_file():
            raise FormalBlocked("resume requires the original immutable setup/binding.json")
        if dict(_read_json(setup_binding)) != dict(binding):
            raise FormalBlocked("resume binding differs from the original run; no files were changed")
        if status_path.is_file():
            prior_orchestration = _read_json(status_path)
            if prior_orchestration.get("binding") != dict(binding):
                raise FormalBlocked("resume orchestration status has a different binding")
        runner_status_path = formal_dir / "run_status.json"
        if runner_status_path.is_file():
            runner_status = _read_json(runner_status_path)
            if runner_status.get("binding") != dict(binding) or runner_status.get("plan") != [dict(row) for row in formal_call_plan]:
                raise FormalBlocked("resume formal run status has a different binding or call plan")
        elif formal_dir.exists() and any(formal_dir.iterdir()):
            raise FormalBlocked("resume formal directory lacks its immutable runner status")
        elif not status_path.is_file():
            raise FormalBlocked("resume has no prior orchestration status to establish run identity")
        return
    if setup_binding.exists() or status_path.exists():
        raise FormalBlocked("formal run identity already exists; choose resume or a new run directory")
    setup_dir = root / "setup"
    if setup_dir.exists() and any(setup_dir.iterdir()):
        raise FormalBlocked(f"setup directory is not empty: {setup_dir}")
    if formal_dir.exists() and any(formal_dir.iterdir()):
        raise FormalBlocked(f"formal directory is not empty: {formal_dir}")
    if analysis_dir.exists() and any(analysis_dir.iterdir()):
        raise FormalBlocked(f"analysis directory is not empty: {analysis_dir}")


def _write_orchestration_status(path: Path, *, status: str, stage: str,
                                binding: Mapping[str, str], error: BaseException | None = None,
                                details: Mapping[str, Any] | None = None) -> None:
    value: dict[str, Any] = {"schema_version": "umi-task8-formal-orchestrator-v1",
                             "status": status, "stage": stage, "binding": dict(binding),
                             "updated_unix_seconds": time.time(), "details": dict(details or {})}
    if error is not None:
        value["error"] = {"type": type(error).__name__, "message": str(error)}
    _atomic_json(path, value)


def _cleanup_formal_resources(context: Any | None, monitor: Any | None) -> list[BaseException]:
    errors: list[BaseException] = []
    if context is not None:
        try:
            context.cleanup()
        except BaseException as error:
            errors.append(error)
    if monitor is not None:
        try:
            monitor.stop()
        except BaseException as error:
            errors.append(error)
    return errors


def _copy_immutable(source: Path, destination: Path) -> None:
    if destination.exists():
        if _sha256_path(source) != _sha256_path(destination):
            raise FormalError(f"refusing to overwrite different immutable evidence: {destination}")
        return
    shutil.copy2(source, destination)


def run_formal(config: Mapping[str, Any], *, release: bool = False,
               context_loader: Callable[..., Any] | None = None,
               monitor_factory: Callable[..., Any] | None = None,
               executor_factory: Callable[..., Any] | None = None) -> dict[str, Any]:
    if not release:
        raise FormalBlocked("formal Task 8 requires main's explicit --release")
    required = ("run_root", "smoke_dir", "inputs_npz", "metadata_json", "framework_root", "checkpoint", "vae")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise FormalBlocked(f"formal configuration is missing: {missing}")
    try:
        from .umi_task8_runtime import Task8InputAdapter, Task8RuntimeConfig
        from .run_umi_task8_experiment import FORMAL_CALL_PLAN, ResourceStop, evaluate_task8_resources, run_task8
        from .task8_live import Task8LiveExecutor, Task8ResourceMonitor, load_task8_live
        from .analyze_umi_task8 import analyze_task8_run
    except ImportError:  # pragma: no cover
        from umi_task8_runtime import Task8InputAdapter, Task8RuntimeConfig
        from run_umi_task8_experiment import FORMAL_CALL_PLAN, ResourceStop, evaluate_task8_resources, run_task8
        from task8_live import Task8LiveExecutor, Task8ResourceMonitor, load_task8_live
        from analyze_umi_task8 import analyze_task8_run
    batch = Task8InputAdapter.from_preflight(config["inputs_npz"], config["metadata_json"])
    metadata = _read_json(Path(config["metadata_json"]))
    smoke = validate_smoke_run(config["smoke_dir"])
    root = Path(config["run_root"]).resolve()
    formal_dir = root / "formal"
    setup_base = root / "setup"
    analysis_base = root / "analysis"
    orchestration_status_path = root / "task8_formal_status.json"
    resume = bool(config.get("resume", False))
    framework_commit = _framework_commit(config["framework_root"])
    code_paths = _task8_source_paths()
    binding = build_binding(code_paths=code_paths, model_path=config["checkpoint"], vae_path=config["vae"],
                            input_path=config["inputs_npz"], actions=batch.actions, metadata=metadata,
                            framework_commit=framework_commit,
                            runtime_config=Task8RuntimeConfig().as_dict())
    _validate_formal_identity(root, binding=binding, resume=resume, formal_call_plan=FORMAL_CALL_PLAN)
    root.mkdir(parents=True, exist_ok=True)
    setup_base.mkdir(parents=True, exist_ok=True)
    formal_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        setup_dir = _next_attempt_dir(setup_base)
        analysis_dir = _next_attempt_dir(analysis_base)
    else:
        setup_dir = setup_base
        analysis_dir = analysis_base
    setup_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        _atomic_json(setup_base / "binding.json", binding)
    _atomic_json(setup_dir / "framework_identity.json", {
        "framework_root": str(Path(config["framework_root"]).resolve()),
        "commit": framework_commit, "working_tree": "clean tracked source",
    })
    _write_orchestration_status(orchestration_status_path, status="RUNNING", stage="resource_gate",
                                binding=binding, details={"resume": resume})
    monitor = None
    context = None
    stage = "resource_gate"
    result: Mapping[str, Any] | None = None
    primary_error: BaseException | None = None
    resource_capture_rows: list[dict[str, Any]] = []
    try:
        monitor = monitor_factory(setup_dir) if monitor_factory is not None else Task8ResourceMonitor(setup_dir, gpu_index=0)
        monitor.start()
        fresh_before_load = monitor.check(phase="formal", starting_new_sample=True)
        initial_gate = evaluate_task8_resources(
            fresh_before_load, phase="formal", starting_new_sample=True,
            remaining_samples=4, mean_success_sample_bytes=int(smoke["measured_sample_bytes"]),
        )
        _atomic_json(setup_dir / "resource_gate.json", {
            "fresh_snapshot_before_load": fresh_before_load, "smoke_measured_sample_bytes": smoke["measured_sample_bytes"],
            "smoke_status_reported_forecast_free_gib_after_four_calls": smoke.get("status_reported_forecast_free_gib_after_four_calls"),
            "fresh_forecast_free_gib_after_four_calls": initial_gate["snapshot"]["forecast_free_gib"],
        })
        stage = "model_load"
        _write_orchestration_status(orchestration_status_path, status="RUNNING", stage=stage, binding=binding)
        loader = context_loader or load_task8_live
        context = loader(framework_root=config["framework_root"], checkpoint=config["checkpoint"], vae=config["vae"],
                         run_dir=setup_dir, batch=batch, release=True, resume=resume)
        # Keep setup/GT artifacts outside the empty directory handed to the
        # atomic runner; publish a copy only after the four-call run starts.
        stage = "ground_truth_encode"
        _write_orchestration_status(orchestration_status_path, status="RUNNING", stage=stage, binding=binding)
        gt16, gt32 = _save_gt_conditions(context, batch, setup_dir)
        live_executor = (executor_factory(context, monitor) if executor_factory is not None
                         else Task8LiveExecutor(context, resource_monitor=monitor, resource_phase="formal"))
        wrapped = VerifiedFormalExecutor(live_executor, formal_dir=formal_dir, monitor=monitor,
                                         capture_path=setup_dir / "formal_resource_capture.json")
        stage = "pre_generation_resource_gate"
        fresh = monitor.check(phase="formal", starting_new_sample=True)
        pre_generation_gate = evaluate_task8_resources(
            fresh, phase="formal", starting_new_sample=True,
            remaining_samples=4, mean_success_sample_bytes=int(smoke["measured_sample_bytes"]),
        )
        _atomic_json(setup_dir / "pre_generation_resource_gate.json", {
            "fresh_snapshot": fresh,
            "smoke_measured_sample_bytes": smoke["measured_sample_bytes"],
            "forecast_free_gib_after_four_calls": pre_generation_gate["snapshot"]["forecast_free_gib"],
        })

        def post_sample_capture(sample_id: str, remaining: int) -> None:
            underlying = getattr(monitor, "_monitor", None)
            capture = getattr(underlying, "capture_sample", None)
            if not callable(capture):
                raise ResourceStop("MONITOR_FAILURE: resource monitor lacks synchronous post-cleanup capture")
            try:
                row = capture(sample_id, "post_cleanup", remaining, run_dir=formal_dir)
                if row.get("decision_status") == "HARD_STOP":
                    raise ResourceStop(f"{row.get('reason_code')}: {row.get('reason')}")
                measured = row.get("mean_success_sample_bytes")
                if isinstance(measured, bool) or not isinstance(measured, (int, float)) or measured <= 0:
                    raise ResourceStop("DISK_FORECAST_UNAVAILABLE: no verified successful sample size after cleanup")
                gate = evaluate_task8_resources(
                    row, phase="formal", starting_new_sample=False,
                    remaining_samples=remaining, mean_success_sample_bytes=int(measured),
                )
                resource_capture_rows.append({"sample": dict(row), "gate": gate})
                # check() observes a latched cleanup-growth/disk stop and applies
                # the same hard limits before run_task8 may start another call.
                monitor.check(phase="formal", starting_new_sample=False)
            except ResourceStop:
                raise
            except BaseException as error:
                raise ResourceStop(f"MONITOR_FAILURE during post-cleanup capture: {error}") from error
            _atomic_json(setup_dir / "post_cleanup_resource_samples.json", {"samples": resource_capture_rows})

        stage = "generation"
        _write_orchestration_status(orchestration_status_path, status="RUNNING", stage=stage, binding=binding)
        result = run_task8(formal_dir, execute_call=wrapped, release=True, resume=resume, binding=binding,
                           resource_snapshot=fresh, resource_monitor=monitor, input_batch=batch,
                           post_sample_callback=post_sample_capture,
                           plan=FORMAL_CALL_PLAN)
    except BaseException as error:
        primary_error = error
    cleanup_errors = _cleanup_formal_resources(context, monitor)
    gc.collect()
    if primary_error is not None:
        for cleanup_error in cleanup_errors:
            add_note = getattr(primary_error, "add_note", None)
            if callable(add_note):
                add_note(f"cleanup also failed: {cleanup_error!r}")
        try:
            _write_orchestration_status(orchestration_status_path, status="FAILED", stage=stage,
                                        binding=binding, error=primary_error,
                                        details={"cleanup_errors": [repr(error) for error in cleanup_errors]})
        except BaseException as status_error:
            add_note = getattr(primary_error, "add_note", None)
            if callable(add_note):
                add_note(f"could not persist orchestration failure status: {status_error!r}")
        raise primary_error
    if cleanup_errors:
        cleanup_failure = RuntimeError("Task 8 formal cleanup failed")
        for cleanup_error in cleanup_errors:
            add_note = getattr(cleanup_failure, "add_note", None)
            if callable(add_note):
                add_note(repr(cleanup_error))
        try:
            _write_orchestration_status(orchestration_status_path, status="FAILED", stage="cleanup",
                                        binding=binding, error=cleanup_failure)
        finally:
            raise cleanup_failure from cleanup_errors[0]

    stage = "analysis"
    _write_orchestration_status(orchestration_status_path, status="RUNNING", stage=stage, binding=binding)
    try:
        _copy_immutable(setup_dir / "ground_truth_conditions.npz", formal_dir / "ground_truth_conditions.npz")
        _copy_immutable(setup_dir / "ground_truth_condition_evidence.json",
                        formal_dir / "ground_truth_condition_evidence.json")
        with np.load(formal_dir / "ground_truth_conditions.npz", allow_pickle=False) as data:
            gt16 = np.array(data["gt_condition_x16"], copy=True)
            gt32 = np.array(data["gt_condition_x32"], copy=True)
            condition_mask = np.array(data["condition_mask"], dtype=bool, copy=True)
        if condition_mask.shape != gt16.shape or condition_mask.shape != gt32.shape:
            raise FormalError("saved condition-only mask does not match GT16/GT32 latent shapes")
        analysis_result = analyze_task8_run(formal_dir, truth_rgb=batch.rgb, gt_condition_x16=gt16,
                                            gt_condition_x32=gt32, condition_mask=condition_mask,
                                            output_dir=analysis_dir)
        _write_preliminary_outputs(analysis_result, analysis_dir)
        _write_manifest(root, binding=binding)
    except BaseException as error:
        try:
            _write_orchestration_status(orchestration_status_path, status="FAILED", stage=stage,
                                        binding=binding, error=error,
                                        details={"formal_status": _read_json(formal_dir / "run_status.json").get("status")
                                                 if (formal_dir / "run_status.json").is_file() else "NOT_STARTED"})
        except BaseException as status_error:
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note(f"could not persist analysis failure status: {status_error!r}")
        raise
    _write_orchestration_status(orchestration_status_path, status="COMPLETE", stage="complete",
                                binding=binding, details={"formal_status": result.get("status") if result else None,
                                                          "analysis_status": analysis_result.get("status")})
    return {"formal": result, "analysis": analysis_result, "binding": binding,
            "formal_dir": str(formal_dir), "analysis_dir": str(analysis_dir)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True); parser.add_argument("--smoke-dir", required=True)
    parser.add_argument("--inputs-npz", required=True); parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--framework-root", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae", required=True); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--release", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_formal(vars(args), release=bool(args.release))
    except (FormalBlocked, FormalError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}, sort_keys=True))
        return 3
    print(json.dumps({"status": "FORMAL_COMPLETE", "formal_dir": result["formal_dir"],
                      "analysis_dir": result["analysis_dir"]}, sort_keys=True))
    return 0


__all__ = ["FormalBlocked", "FormalError", "PREPROCESSING_IDENTITY", "VerifiedFormalExecutor",
           "build_binding", "main", "parse_args", "run_formal", "validate_smoke_run"]


if __name__ == "__main__":
    raise SystemExit(main())
