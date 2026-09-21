"""Operational Task 7 FP32 feedback runner.

Preflight is generation-free.  Smoke and formal stages use one reviewed
Task-6 runtime and the approved Task-7 ``FeedbackRuntime``; the executor seam
is injectable so CPU tests never claim model or GPU evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import numbers
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

try:
    from .umi_task6_primitives import evaluate_resources
    from .run_umi_task6_experiment import ResourceMonitor
    from .umi_precision_storage import ProcessLock, publish_directory
except ImportError:  # direct import from experiments/
    from umi_task6_primitives import evaluate_resources
    from run_umi_task6_experiment import ResourceMonitor
    from umi_precision_storage import ProcessLock, publish_directory


class ResumeMismatch(ValueError):
    """Prior source, configuration, or immutable artifact does not match."""


class ResourceStop(RuntimeError):
    """A resource, monitor, forecast, or cleanup stop is latched."""


class BlockedExecution(RuntimeError):
    """A live stage is missing an approved runtime/authorization seam."""


class SkipSample(Exception):
    """Explicitly skipped formal sample (zero-ray Stage C only)."""

    def __init__(self, reason_code: str, reason: str):
        super().__init__(reason)
        self.reason_code, self.reason = str(reason_code), str(reason)


TASK7_CODE_SOURCES = (
    "run_umi_task7_experiment.py", "umi_task7_runtime.py", "umi_task7_encoder.py",
    "umi_task6_runtime.py", "umi_task6_operational.py", "umi_task6_primitives.py",
    "umi_task6_cosmos_loader.py", "umi_task6_decoder.py", "run_umi_task6_experiment.py",
    "run_umi_task6_official.py", "umi_precision_official.py", "umi_precision_runtime.py",
    "umi_precision_storage.py", "analyze_umi_task7.py",
)
SOURCE_NAMES = ("baseline_pre", "v0_alpha_00_plus", "v0_alpha_00_minus",
                "v0_alpha_01_plus", "v0_alpha_01_minus", "v0_alpha_02_plus",
                "v0_alpha_02_minus", "baseline_post")
REQUIRED_RAW_ARTIFACTS = ("z_bar.npy", "mask.npy", "consumed_input_fp32.npy", "output_full.npy")
REQUIRED_DECODER_ARTIFACTS = ("decoder_input_full_latent.npy", "decoded_final_float32.npy",
                              "direct_condition_latent_float32.npy")
REQUIRED_LIVE_RESOURCE_FIELDS = (
    "gpu_used_gib", "gpu_free_gib", "gpu_reserved_gib", "gpu_peak_allocated_gib",
    "gpu_peak_nvml_used_gib", "ram_available_gib", "rss_gib", "swap_used_gib",
    "disk_free_gib", "cgroup_memory_limited", "cgroup_memory_limit_gib",
    "cgroup_memory_current_gib", "cgroup_memory_free_gib",
)


def _safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {"array": True, "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("nonfinite values are not valid in Task 7 JSON evidence")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii")); digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")); digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(json.dumps(_safe(value), sort_keys=True, indent=2, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResumeMismatch(f"invalid JSON evidence: {path}") from error


def _load_array(path: Path, *, mmap: bool = True) -> np.ndarray:
    mapped = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
    try:
        result = np.array(mapped, copy=True)
    finally:
        closer = getattr(mapped, "_mmap", None)
        if closer is not None:
            closer.close()
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ResumeMismatch(f"nonfinite or empty source array: {path}")
    return result


def _code_digest(root: Path | None = None) -> str:
    root = (root or Path(__file__).resolve().parent).resolve()
    digest, present = hashlib.sha256(), 0
    for name in TASK7_CODE_SOURCES:
        path = root / name
        if path.is_file():
            present += 1
            digest.update(name.encode("utf-8")); digest.update(b"\0"); digest.update(path.read_bytes())
    if present < 3:
        raise ResumeMismatch("Task 7 source binding is incomplete")
    return digest.hexdigest()


def _contained(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def validate_run_roots(run_dir: str | Path, *, raw_root: str | Path | None = None,
                       decoder_root: str | Path | None = None) -> dict[str, str]:
    """Ensure new evidence cannot mutate either historical source root."""
    run = Path(run_dir).resolve()
    if (raw_root is None) != (decoder_root is None):
        raise ResumeMismatch("raw_root and decoder_root must be supplied together")
    roots = {"run": str(run)}
    for label, value in (("raw", raw_root), ("decoder", decoder_root)):
        if value is None:
            continue
        source = Path(value).resolve()
        if source == run or _contained(run, source):
            raise ResumeMismatch(f"new run is contained by read-only {label}_root")
        roots[label] = str(source)
    return roots


def _artifact_status(root: Path, required: tuple[str, ...]) -> dict[str, Any]:
    status_path = root / "status.json"
    if not status_path.is_file():
        raise ResumeMismatch(f"missing source status: {status_path}")
    status = _load_json(status_path)
    if not isinstance(status, Mapping) or status.get("status") != "success":
        raise ResumeMismatch(f"source sample is not successful: {root}")
    hashes = status.get("artifact_sha256")
    if not isinstance(hashes, Mapping) or not hashes:
        raise ResumeMismatch(f"source sample has no artifact hashes: {root}")
    for name in required:
        if not (root / name).is_file() or name not in hashes:
            raise ResumeMismatch(f"source sample missing required artifact {name}: {root}")
    for name, expected in hashes.items():
        if not isinstance(name, str) or Path(name).name != name or not isinstance(expected, str):
            raise ResumeMismatch(f"unsafe source artifact manifest: {root}")
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ResumeMismatch(f"source artifact hash mismatch: {path}")
    return dict(status)


def _source_sample_root(root: Path, name: str) -> Path:
    for candidate in (root / "samples" / f"bridge_0__seed_0__{name}", root / f"bridge_0__seed_0__{name}"):
        if candidate.is_dir():
            return candidate
    raise ResumeMismatch(f"source sample is missing: {name}")


def _decoder_sample_root(root: Path, name: str) -> Path:
    for candidate in (root / f"bridge_0__seed_0__{name}__temporary_fp32",
                      root / "samples" / f"bridge_0__seed_0__{name}__temporary_fp32"):
        if candidate.is_dir():
            return candidate
    raise ResumeMismatch(f"decoder source sample is missing: {name}")


def _metadata_value(payload: Any, key: str) -> Any:
    if isinstance(payload, Mapping):
        if key in payload:
            return payload[key]
        for child_name in ("spec", "group", "metadata", "values", "inputs"):
            if child_name in payload:
                found = _metadata_value(payload[child_name], key)
                if found is not None:
                    return found
    return None


def _tree_digest(*roots: Path) -> str:
    digest = hashlib.sha256()
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8")); digest.update(b"\0")
                digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def validate_task7_sources(raw_root: str | Path, decoder_root: str | Path) -> dict[str, Any]:
    """Validate the eight immutable raw/temporary-FP32 source pairs."""
    raw, decoder = Path(raw_root).resolve(), Path(decoder_root).resolve()
    if not raw.is_dir() or not decoder.is_dir():
        raise ResumeMismatch("raw_root and decoder_root must be existing directories")
    frozen_identity: dict[str, str] | None = None
    frozen_z: np.ndarray | None = None
    frozen_mask: np.ndarray | None = None
    v0_direction: np.ndarray | None = None
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    direction_reference: np.ndarray | None = None
    plan_path = raw / "task6_plan.json"
    if not plan_path.is_file():
        raise ResumeMismatch("raw source task6_plan.json is missing")
    source_plan = _load_json(plan_path)
    plan_inputs = source_plan.get("inputs") if isinstance(source_plan, Mapping) else None
    if not isinstance(plan_inputs, Mapping):
        raise ResumeMismatch("raw task6_plan.json lacks actual inputs provenance")
    for key in ("action", "prompt", "seed", "state", "geometry"):
        if key not in plan_inputs:
            raise ResumeMismatch(f"raw task6_plan.inputs lacks {key}")
    geometry = plan_inputs["geometry"]
    if (not isinstance(geometry, Mapping) or geometry.get("carrier_shape") != [1, 48, 5, 16, 16]
            or geometry.get("condition_indexes") != [0] or geometry.get("predicted_indexes") != [1, 2, 3, 4]):
        raise ResumeMismatch("raw Task 6 geometry does not match the approved UMI carrier")
    if plan_inputs["state"] != "bridge_0" or int(plan_inputs["seed"]) != 0:
        raise ResumeMismatch("raw Task 6 source is not bridge_0/seed0")
    source_provenance: dict[str, Any] = {"prompt": plan_inputs["prompt"], "action": plan_inputs["action"],
                                         "seed": plan_inputs["seed"], "state": plan_inputs["state"], "geometry": geometry}
    source_s_z = plan_inputs.get("s_z")
    for name in SOURCE_NAMES:
        sample, replay = _source_sample_root(raw, name), _decoder_sample_root(decoder, name)
        sample_status = _artifact_status(sample, REQUIRED_RAW_ARTIFACTS)
        replay_status = _artifact_status(replay, REQUIRED_DECODER_ARTIFACTS)
        if sample_status["artifact_sha256"].get("output_full.npy") != replay_status["artifact_sha256"].get("decoder_input_full_latent.npy"):
            raise ResumeMismatch(f"raw output_full and decoder input differ: {name}")
        z0 = _load_array(sample / "z_bar.npy")
        mask = _load_array(sample / "mask.npy").astype(bool, copy=False)
        consumed = _load_array(sample / "consumed_input_fp32.npy")
        if z0.shape != mask.shape or consumed.shape != z0.shape:
            raise ResumeMismatch(f"source carrier/mask shape mismatch: {name}")
        identity = {"z0": _array_hash(z0), "mask": _array_hash(mask)}
        if frozen_identity is None:
            frozen_identity, frozen_z, frozen_mask = identity, z0, mask
        elif identity != frozen_identity:
            raise ResumeMismatch(f"frozen z0/mask differs: {name}")
        delta = np.subtract(consumed, z0, dtype=np.float32)
        if np.any(delta[~mask] != 0):
            raise ResumeMismatch(f"source condition delta changed exterior: {name}")
        frame = _load_array(replay / "decoded_final_float32.npy")
        encoded = _load_array(replay / "direct_condition_latent_float32.npy")
        if frame.dtype != np.float32 or tuple(frame.shape) != (3, 256, 256) or frame.min() < 0 or frame.max() > 1:
            raise ResumeMismatch(f"decoder frame is not float32 RGB CHW [0,1]: {name}")
        temporal_axis = mask.ndim - 3
        condition_indexes = [index for index in range(mask.shape[temporal_axis])
                             if bool(np.all(np.take(mask, index, axis=temporal_axis)))]
        expected_condition_shape = list(mask.shape); expected_condition_shape[temporal_axis] = len(condition_indexes)
        if encoded.dtype != np.float32 or tuple(encoded.shape) != tuple(expected_condition_shape):
            raise ResumeMismatch(f"decoder condition shape does not match mask: {name}")
        sample_json = sample / "sample.json"
        if sample_json.is_file():
            payload = _load_json(sample_json)
            for key in ("seed", "model_seed"):
                value = _metadata_value(payload, key)
                if value is None or canonical_json(value) != canonical_json(0):
                    raise ResumeMismatch(f"source metadata is missing or differs for {key}: {name}")
        else:
            raise ResumeMismatch(f"source provenance record is missing: {sample_json}")
        direction_path = sample / "direction.npy"
        if direction_path.is_file():
            direction = _load_array(direction_path).astype(np.float32, copy=False)
            if direction.shape != z0.shape:
                raise ResumeMismatch(f"source direction shape differs: {name}")
            if name.startswith("baseline"):
                if np.any(direction != 0):
                    raise ResumeMismatch(f"baseline direction is not exactly zero: {name}")
            elif direction_reference is None:
                direction_reference = direction.copy()
            elif not np.array_equal(direction, direction_reference):
                raise ResumeMismatch(f"frozen v0 direction differs: {name}")
            if name == "v0_alpha_00_plus":
                v0_direction = direction.copy()
        elif not name.startswith("baseline"):
            raise ResumeMismatch(f"perturbation direction artifact is missing: {name}")
        if name.startswith("baseline"):
            if not np.array_equal(consumed, z0):
                raise ResumeMismatch(f"baseline consumed input differs from frozen z0: {name}")
        elif v0_direction is not None:
            expected = z0.copy()
            sign = np.float32(1.0 if name.endswith("plus") else -1.0)
            s_z = np.float32(source_s_z) if source_s_z is not None else np.float32(np.sqrt(np.mean(z0[mask].astype(np.float64) ** 2)))
            delta_value = np.multiply(np.float32(sign * _stage_alpha(name) * s_z), v0_direction, dtype=np.float32)
            expected = np.add(expected, delta_value, dtype=np.float32)
            expected[~mask] = z0[~mask]
            if not np.array_equal(consumed, expected):
                raise ResumeMismatch(f"source consumed input is not z0 +/- alpha*s_z*v0: {name}")
        rows.append({"name": name, "raw": str(sample), "decoder": str(replay),
                     "raw_status_sha256": sha256_file(sample / "status.json"),
                     "decoder_record_sha256": sha256_file(replay / "record.json") if (replay / "record.json").is_file() else None,
                     "output_full_sha256": sample_status["artifact_sha256"]["output_full.npy"],
                     "decoder_input_sha256": replay_status["artifact_sha256"]["decoder_input_full_latent.npy"]})
    if frozen_z is None or frozen_mask is None or v0_direction is None:
        raise ResumeMismatch("source v0 direction must be present")
    if not np.any(np.abs(v0_direction[frozen_mask]) > 0) or np.any(v0_direction[~frozen_mask] != 0):
        raise ResumeMismatch("source v0 direction must be nonzero only on the condition mask")
    return {"schema_version": "umi-task7-source-v1", "raw_root": str(raw), "decoder_root": str(decoder),
            "samples": rows, "z0_sha256": frozen_identity["z0"], "mask_sha256": frozen_identity["mask"],
            "v0_sha256": _array_hash(v0_direction), "carrier_shape": list(frozen_z.shape),
            "mask_count": int(frozen_mask.sum()), "metadata": {**metadata, **source_provenance, "s_z": source_s_z},
            "task6_plan_sha256": sha256_file(plan_path),
            "source_tree_sha256": _tree_digest(raw, decoder)}


def _empty_counts() -> dict[str, int]:
    return {"G": 0, "D": 0, "E": 0}


def stage_counts(stage: str) -> dict[str, int]:
    stage = str(stage).upper()
    if stage == "A": return {"G": 0, "D": 0, "E": 16}
    if stage == "B": return {"G": 16, "D": 16, "E": 16}
    if stage == "C": return {"G": 38, "D": 38, "E": 38}
    raise ValueError("stage must be A, B, or C")


def _stage_alpha(name: str) -> float:
    for prefix, value in (("v0_alpha_00", .001), ("v0_alpha_01", .003), ("v0_alpha_02", .01)):
        if name.startswith(prefix):
            return value
    return 0.0


def build_stage_plan(stage: str) -> list[dict[str, Any]]:
    stage = str(stage).upper()
    if stage == "A":
        return [{"sample_id": f"A_{name}_{precision}", "stage": "A", "kind": name,
                 "precision": precision, "alpha": _stage_alpha(name),
                 "sign": 0 if name.startswith("baseline") else (1 if name.endswith("plus") else -1),
                 "source_name": name, "direction_id": None if name.startswith("baseline") else "v0"}
                for name in SOURCE_NAMES for precision in ("native", "temporary_fp32")]
    if stage == "B":
        return [{"sample_id": f"B_{name}_step_{step}", "stage": "B", "kind": name,
                 "trajectory_id": name, "source_name": name, "step_index": step, "seed": step,
                 "alpha": _stage_alpha(name),
                 "sign": 0 if name.startswith("baseline") else (1 if name.endswith("plus") else -1),
                 "direction_id": None if name.startswith("baseline") else "v0"}
                for name in SOURCE_NAMES for step in (0, 1)]
    if stage == "C":
        rows = [{"sample_id": "C_baseline_pre", "stage": "C", "kind": "baseline_pre",
                 "step_index": 2, "runtime_step_index": 1, "seed": 1}]
        for index in range(6):
            for beta in (.1, .2, .4):
                for sign, label in ((1, "plus"), (-1, "minus")):
                    rows.append({"sample_id": f"C_delta1_{index:02d}_beta_{beta:g}_{label}", "stage": "C",
                                 "kind": "perturbation", "direction_id": f"delta1_{index:02d}",
                                 "beta": beta, "sign": sign, "seed": 1, "step_index": 2,
                                 "runtime_step_index": 1})
        rows.append({"sample_id": "C_baseline_post", "stage": "C", "kind": "baseline_post",
                     "step_index": 2, "runtime_step_index": 1, "seed": 1})
        return rows
    raise ValueError("stage must be A, B, or C")


def require_resource_snapshot(snapshot: Mapping[str, Any], *, phase: str,
                              starting_new_sample: bool = False,
                              remaining_samples: int | None = None,
                              measured_success_bytes: int | None = None) -> dict[str, Any]:
    """Use Task 6 policy after proving all live fields are actual observations."""
    if not isinstance(snapshot, Mapping):
        raise ResourceStop("MONITOR_FAILURE: snapshot is not a mapping")
    missing = [key for key in REQUIRED_LIVE_RESOURCE_FIELDS if key not in snapshot]
    if missing:
        raise ResourceStop(f"MONITOR_NO_SNAPSHOT: missing fields {missing}")
    if snapshot.get("monitor_failure") or snapshot.get("monitor_error"):
        raise ResourceStop(f"MONITOR_FAILURE: {snapshot.get('monitor_failure') or snapshot.get('monitor_error')}")
    for key in REQUIRED_LIVE_RESOURCE_FIELDS:
        if key == "cgroup_memory_limited":
            if not isinstance(snapshot[key], bool):
                raise ResourceStop("RESOURCE_SNAPSHOT_NONFINITE: cgroup_memory_limited")
            continue
        value = snapshot[key]
        if value is None:
            if key.startswith("cgroup_memory_") and not snapshot["cgroup_memory_limited"]:
                continue
            raise ResourceStop(f"MONITOR_NO_SNAPSHOT: {key} is missing")
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(float(value)):
            raise ResourceStop(f"RESOURCE_SNAPSHOT_NONFINITE: {key}")
    observed = dict(snapshot)
    if remaining_samples is not None:
        if measured_success_bytes is None and remaining_samples:
            raise ResourceStop("DISK_FORECAST_UNAVAILABLE: no measured successful sample size")
        observed["remaining_samples"] = int(remaining_samples)
        if measured_success_bytes is not None:
            observed["mean_success_sample_bytes"] = int(measured_success_bytes)
    policy_phase = "preload" if phase in {"preflight", "preload"} else ("resource-smoke" if phase in {"smoke", "resource-smoke"} else "pilot")
    decision = evaluate_resources(observed, phase=policy_phase, starting_new_sample=starting_new_sample)
    if decision.get("status") == "HARD_STOP":
        raise ResourceStop(f"{decision.get('reason_code')}: {decision.get('reason')}")
    if measured_success_bytes is not None and remaining_samples is not None:
        forecast = float(observed["disk_free_gib"]) - 1.3 * float(measured_success_bytes) * int(remaining_samples) / 2**30
        observed["forecast_free_gib"] = forecast
        if forecast < 5.0:
            raise ResourceStop("DISK_FORECAST_LOW: forecast completion free space is below 5 GiB")
    if starting_new_sample and float(observed["disk_free_gib"]) < 5.0:
        raise ResourceStop("DISK_FREE_LOW: disk free space is below 5 GiB")
    return {"status": decision.get("status", "OK"), "reason_code": decision.get("reason_code"),
            "reason": decision.get("reason"), "snapshot": observed}


class StaticMonitor:
    """CPU fixture monitor; production uses Task 6 ResourceMonitor."""
    def __init__(self, snapshot: Mapping[str, Any]):
        self.last_resources = dict(snapshot)

    def check(self, **kwargs: Any) -> dict[str, Any]:
        return require_resource_snapshot(self.last_resources, phase=kwargs.get("phase", "stage"),
                                         starting_new_sample=bool(kwargs.get("starting_new_sample", False)))


def evaluate_task7_resources(snapshot: Mapping[str, Any], *, phase: str = "stage",
                            starting_new_sample: bool = False) -> dict[str, Any]:
    try:
        return require_resource_snapshot(snapshot, phase=phase, starting_new_sample=starting_new_sample)
    except ResourceStop as error:
        code, _, reason = str(error).partition(": ")
        return {"status": "HARD_STOP", "reason_code": code, "reason": reason,
                "snapshot": dict(snapshot) if isinstance(snapshot, Mapping) else {}}


class Task7SampleStore:
    """Atomic publication, immutable success, and preserved failed attempts."""
    def __init__(self, root: str | Path):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_id(sample_id: str) -> str:
        if not sample_id or any(part in sample_id for part in ("/", "\\", "..")):
            raise ValueError("unsafe Task 7 sample identifier")
        return sample_id

    def _path(self, sample_id: str) -> Path:
        return self.root / self._safe_id(sample_id)

    def _verify(self, path: Path, required_files: tuple[str, ...] = ("record.json",)) -> dict[str, Any]:
        status = _load_json(path / "status.json")
        if not isinstance(status, Mapping) or status.get("status") != "success":
            raise ResumeMismatch(f"sample is not successful: {path}")
        hashes = status.get("artifact_sha256")
        if not isinstance(hashes, Mapping) or any(name not in hashes for name in required_files):
            raise ResumeMismatch(f"sample manifest is incomplete: {path}")
        for name, expected in hashes.items():
            if not (path / name).is_file() or sha256_file(path / name) != expected:
                raise ResumeMismatch(f"sample artifact hash mismatch: {path / name}")
        return dict(status)

    def prepare(self, sample_id: str, *, resume: bool = False,
                required_files: tuple[str, ...] = ("record.json",)) -> str:
        path = self._path(sample_id)
        if not path.exists():
            return "run"
        status = _load_json(path / "status.json") if (path / "status.json").is_file() else {}
        if status.get("status") == "success":
            self._verify(path, required_files)
            if not resume:
                raise FileExistsError(path)
            return "skip"
        if not resume:
            raise FileExistsError(path)
        index = 1
        while (self.root / f"{sample_id}.attempt.{index:03d}").exists():
            index += 1
        path.rename(self.root / f"{sample_id}.attempt.{index:03d}")
        return "run"

    def _publish(self, sample_id: str, payload: Any, *, status: str,
                 operation_counts: Mapping[str, Any] | None = None,
                 required_files: tuple[str, ...] = ("record.json",),
                 destination_id: str | None = None) -> Path:
        destination = self._path(destination_id or sample_id)
        if destination.exists():
            raise FileExistsError(destination)
        stage = Path(tempfile.mkdtemp(prefix=f".{sample_id}.stage.", dir=str(self.root)))
        arrays: dict[str, np.ndarray] = {}

        def encode(value: Any, name: str) -> Any:
            if isinstance(value, np.ndarray):
                filename = name.replace("/", "_").replace("\\", "_") + ".npy"
                if filename in arrays:
                    raise ValueError("duplicate Task 7 array artifact")
                arrays[filename] = np.array(value, copy=True)
                return {"artifact": filename, "dtype": str(value.dtype), "shape": list(value.shape)}
            if isinstance(value, Mapping):
                return {str(key): encode(item, f"{name}_{key}") for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(item, f"{name}_{index}") for index, item in enumerate(value)]
            return _safe(value)

        try:
            _atomic_json(stage / "record.json", encode(payload, "record"))
            for filename, array in arrays.items():
                with (stage / filename).open("wb") as stream:
                    np.save(stream, array, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
            hashes = {path.name: sha256_file(path) for path in sorted(stage.iterdir()) if path.is_file()}
            if any(name not in hashes for name in required_files):
                raise ValueError("required Task 7 artifact was not serialized")
            _atomic_json(stage / "status.json", {"status": status, "required_artifacts": list(required_files),
                                                  "artifact_sha256": hashes,
                                                  "operation_counts": dict(operation_counts or {})})
            publish_directory(stage, destination)
            return destination
        except BaseException:
            if stage.exists():
                index = 1
                while (self.root / f"{sample_id}.failed-attempt.{index:03d}").exists():
                    index += 1
                stage.rename(self.root / f"{sample_id}.failed-attempt.{index:03d}")
            raise

    def write_success(self, sample_id: str, payload: Any, *, operation_counts: Mapping[str, Any] | None = None,
                      required_files: tuple[str, ...] = ("record.json",)) -> Path:
        return self._publish(sample_id, payload, status="success", operation_counts=operation_counts, required_files=required_files)

    def write_failure(self, sample_id: str, payload: Any, *, operation_counts: Mapping[str, Any] | None = None) -> Path:
        index = 1
        while (self.root / f"{sample_id}.attempt.{index:03d}").exists():
            index += 1
        return self._publish(sample_id, payload, status="failed", operation_counts=operation_counts,
                             destination_id=f"{sample_id}.attempt.{index:03d}")

    def load_record(self, sample_id: str) -> dict[str, Any]:
        path = self._path(sample_id); status = self._verify(path)
        def decode(value: Any) -> Any:
            if isinstance(value, Mapping) and set(value) >= {"artifact", "dtype", "shape"}:
                return _load_array(path / str(value["artifact"]), mmap=False)
            if isinstance(value, Mapping):
                return {str(key): decode(item) for key, item in value.items()}
            if isinstance(value, list):
                return [decode(item) for item in value]
            return value
        record = decode(_load_json(path / "record.json")); record["_status"] = status
        return record


def _validate_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    required = {"source", "task7_code_sha256", "config", "model", "vae", "framework", "mask", "z0", "v0", "noise_policy"}
    if not isinstance(binding, Mapping):
        raise ResumeMismatch("Task 7 binding must be a mapping")
    missing = sorted(required - set(binding))
    if missing:
        raise ResumeMismatch(f"Task 7 binding is incomplete: {missing}")
    if any(not binding[key] for key in required if key != "config"):
        raise ResumeMismatch("Task 7 binding contains an empty identity")
    return _safe(dict(binding))


def test_binding(stage: str) -> dict[str, Any]:
    return {"source": "fixture-source", "task7_code_sha256": "fixture-task7-code",
            "config": {"stage": stage}, "model": "fixture-model", "vae": "fixture-vae",
            "framework": "fixture-framework", "mask": "fixture-mask", "z0": "fixture-z0",
            "v0": "fixture-v0", "noise_policy": "fixed-seed-paired"}


def _binding_config(stage: str, plan: list[dict[str, Any]], binding: Mapping[str, Any]) -> dict[str, Any]:
    checked = _validate_binding(binding)
    return {"schema_version": "umi-task7-stage-v2", "stage": stage, "plan": plan,
            "plan_sha256": hashlib.sha256(canonical_json(plan).encode()).hexdigest(), "binding": checked}


def _capture_counts(value: Any) -> dict[str, int]:
    evidence = value.get("evidence") if isinstance(value, Mapping) else None
    counts = evidence.get("operation_counts") if isinstance(evidence, Mapping) else None
    if not isinstance(counts, Mapping) and isinstance(value, Mapping):
        counts = value.get("operation_counts")
    if not isinstance(counts, Mapping) or any(key not in counts for key in ("G", "D", "E")):
        raise ValueError("actual operation_counts G/D/E are required; no inference is allowed")
    result = {}
    for key in ("G", "D", "E"):
        item = counts[key]
        if isinstance(item, bool) or not isinstance(item, numbers.Integral) or int(item) < 0:
            raise ValueError(f"operation_counts[{key}] is invalid")
        result[key] = int(item)
    return result


def _add_counts(left: dict[str, int], right: Mapping[str, Any]) -> None:
    for key in ("G", "D", "E"):
        left[key] += int(right.get(key, 0))


def _error_counts(error: BaseException) -> dict[str, int]:
    capture = getattr(error, "capture", None)
    if not isinstance(capture, Mapping):
        return _empty_counts()
    try:
        return _capture_counts(capture)
    except ValueError:
        return _empty_counts()


def _measured_success_bytes(root: Path) -> int | None:
    sizes: list[int] = []
    # A formal stage lives at run/stages/{A,B,C}; smoke lives beside stages.
    search_roots = [root, root.parent, root.parent.parent]
    seen: set[Path] = set()
    for base in search_roots:
        if not base.is_dir():
            continue
        for status_path in base.glob("**/samples/*/status.json"):
            sample = status_path.parent
            if sample in seen:
                continue
            seen.add(sample)
            try:
                status = _load_json(status_path)
                hashes = status.get("artifact_sha256", {})
                if status.get("status") == "success" and all((sample / name).is_file() and sha256_file(sample / name) == digest
                                                               for name, digest in hashes.items()):
                    sizes.append(sum(path.stat().st_size for path in sample.rglob("*") if path.is_file()))
            except (OSError, ValueError, KeyError):
                continue
    return int(sum(sizes) / len(sizes)) if sizes else None
def run_stage(stage: str, run_dir: str | Path, *, execute: Callable[[Mapping[str, Any]], Any],
              binding: Mapping[str, Any], resume: bool = False, monitor: Any | None = None,
              allowed_skips: bool = False) -> dict[str, Any]:
    """Run exactly one stage with a process lock and strict evidence resume."""
    stage = str(stage).upper()
    if stage not in {"A", "B", "C"}:
        raise ValueError("run_stage accepts only formal A/B/C")
    if not callable(execute):
        raise BlockedExecution("formal stage requires an injected executor")
    if monitor is None:
        raise BlockedExecution("formal stage requires ResourceMonitor or an injected monitor")
    root = Path(run_dir).resolve(); root.mkdir(parents=True, exist_ok=True)
    plan, config = build_stage_plan(stage), None
    config = _binding_config(stage, plan, binding)
    config_path, status_path = root / "stage_config.json", root / "run_status.json"
    with ProcessLock(root / ".runner.lock"):
        if config_path.is_file():
            if canonical_json(_load_json(config_path)) != canonical_json(config):
                raise ResumeMismatch("stage source/code/config binding differs")
        elif any(path.name != ".runner.lock" for path in root.iterdir()) and not resume:
            raise ResumeMismatch("run directory is nonempty without Task 7 stage config")
        else:
            _atomic_json(config_path, config)
        store = Task7SampleStore(root / "samples")
        completed: list[str] = []; failed: list[str] = []; skipped: list[str] = []
        attempt_counts, completed_counts = _empty_counts(), _empty_counts()
        failed_attempts: list[dict[str, Any]] = []
        if status_path.is_file() and resume:
            prior = _load_json(status_path)
            if canonical_json(prior.get("binding")) != canonical_json(config["binding"]):
                raise ResumeMismatch("run status binding differs")
            if prior.get("status") in {"FAILED", "RESOURCE_STOP"}:
                # OOM/monitor failures are evidence, not an automatic retry.
                # A human must start a new run or explicitly repair the source.
                for sample_id in prior.get("completed_samples", []):
                    store._verify(store._path(sample_id))
                return prior
            if prior.get("status") == "COMPLETE":
                planned_ids = {spec["sample_id"] for spec in plan}
                completed_ids, skipped_ids = set(prior.get("completed_samples", [])), set(prior.get("skipped_samples", []))
                if completed_ids | skipped_ids != planned_ids or completed_ids & skipped_ids or prior.get("failed_samples"):
                    raise ResumeMismatch("complete status sample-id set is inconsistent")
                recomputed_attempts, recomputed_completed = _empty_counts(), _empty_counts()
                for spec in plan:
                    path = store._path(spec["sample_id"]); store._verify(path)
                    record = _load_json(path / "record.json")
                    counts = _load_json(path / "status.json").get("operation_counts")
                    if not isinstance(counts, Mapping):
                        raise ResumeMismatch("complete sample lacks observed operation counts")
                    _add_counts(recomputed_attempts, counts)
                    if spec["sample_id"] in completed_ids:
                        _add_counts(recomputed_completed, counts)
                    elif record.get("status") != "SKIPPED_C":
                        raise ResumeMismatch("skipped sample lacks explicit SKIPPED_C evidence")
                if prior.get("attempt_counts") != recomputed_attempts or prior.get("completed_counts") != recomputed_completed:
                    raise ResumeMismatch("complete status counters do not match verified artifacts")
                return prior

        def write_status(state: str, **extra: Any) -> dict[str, Any]:
            payload = {"schema_version": "umi-task7-run-v2", "status": state, "stage": stage,
                       "planned_samples": len(plan), "completed_samples": list(completed),
                       "failed_samples": list(failed), "skipped_samples": list(skipped),
                       "completed_count": len(completed), "failed_count": len(failed),
                       "skipped_count": len(skipped), "attempt_counts": dict(attempt_counts),
                       "observed_counts": dict(attempt_counts), "completed_counts": dict(completed_counts),
                       "formal_counts": stage_counts(stage), "failed_attempts": list(failed_attempts),
                       "binding": config["binding"], "last_resources": getattr(monitor, "last_resources", {}) or {}, **extra}
            _atomic_json(status_path, payload)
            return payload

        write_status("RUNNING")
        for ordinal, spec in enumerate(plan):
            try:
                decision = monitor.check(phase="stage", starting_new_sample=True,
                                         remaining_samples=len(plan) - ordinal) if callable(getattr(monitor, "check", None)) else None
            except BaseException as monitor_error:
                return write_status("RESOURCE_STOP", reason_code=type(monitor_error).__name__, reason=str(monitor_error))
            if not isinstance(decision, Mapping) or decision.get("status") == "HARD_STOP":
                return write_status("RESOURCE_STOP", reason_code=(decision or {}).get("reason_code", "MONITOR_FAILURE"),
                                    reason=(decision or {}).get("reason", "invalid monitor decision"))
            if isinstance(monitor, ResourceMonitor):
                measured = _measured_success_bytes(root)
                try:
                    require_resource_snapshot(monitor.last_resources, phase="stage", starting_new_sample=True,
                                               remaining_samples=len(plan) - ordinal, measured_success_bytes=measured)
                except BaseException as monitor_error:
                    return write_status("RESOURCE_STOP", reason_code=type(monitor_error).__name__, reason=str(monitor_error))
            disposition = store.prepare(spec["sample_id"], resume=resume)
            if disposition == "skip":
                prior_record = store.load_record(spec["sample_id"])
                counts = prior_record.get("_status", {}).get("operation_counts")
                if not isinstance(counts, Mapping):
                    raise ResumeMismatch("successful sample lacks observed operation counts")
                _add_counts(attempt_counts, counts); _add_counts(completed_counts, counts)
                completed.append(spec["sample_id"]); continue
            try:
                value = execute(dict(spec))
                if isinstance(value, SkipSample):
                    if not allowed_skips: raise RuntimeError("skip is not allowed for this stage")
                    store.write_success(spec["sample_id"], {"spec": spec, "status": "SKIPPED_C",
                        "reason_code": value.reason_code, "reason": value.reason}, operation_counts=_empty_counts())
                    skipped.append(spec["sample_id"]); continue
                if not isinstance(value, Mapping):
                    raise ValueError("stage executor must return a mapping record")
                counts = _capture_counts(value)
                _add_counts(attempt_counts, counts); _add_counts(completed_counts, counts)
                store.write_success(spec["sample_id"], {"spec": spec, "record": value}, operation_counts=counts)
                completed.append(spec["sample_id"]); write_status("RUNNING")
            except SkipSample as skip:
                if not allowed_skips: raise
                store.write_success(spec["sample_id"], {"spec": spec, "status": "SKIPPED_C",
                    "reason_code": skip.reason_code, "reason": skip.reason}, operation_counts=_empty_counts())
                skipped.append(spec["sample_id"]); write_status("RUNNING")
            except BaseException as error:
                counts = _error_counts(error); _add_counts(attempt_counts, counts)
                failed.append(spec["sample_id"]); failed_attempts.append({"sample_id": spec["sample_id"],
                    "type": type(error).__name__, "message": str(error), "operation_counts": counts})
                try:
                    store.write_failure(spec["sample_id"], {"spec": spec, "error": str(error),
                        "capture": getattr(error, "capture", {"error": str(error)})}, operation_counts=counts)
                except BaseException as serialize_error:
                    failed_attempts[-1]["serialization_error"] = repr(serialize_error)
                return write_status("FAILED", reason_code=type(error).__name__, reason=str(error))
        expected = stage_counts(stage)
        if stage == "C" and skipped:
            expected = {key: max(0, value - len(skipped)) for key, value in expected.items()}
        if any(completed_counts[key] != expected[key] for key in expected):
            return write_status("FAILED", reason_code="COUNT_MISMATCH",
                                reason=f"observed successful counts {completed_counts} != expected {expected}")
        return write_status("COMPLETE", reason_code=None, reason=None)


def _source_condition(source: Mapping[str, Any], name: str) -> tuple[np.ndarray, np.ndarray]:
    root = _source_sample_root(Path(source["raw_root"]), name)
    return _load_array(root / "z_bar.npy"), _load_array(root / "consumed_input_fp32.npy")


def _source_frame(source: Mapping[str, Any], name: str) -> np.ndarray:
    root = _decoder_sample_root(Path(source["decoder_root"]), name)
    return _load_array(root / "decoded_final_float32.npy").astype(np.float32, copy=False)


def _prepare_c_source(source: dict[str, Any], run_root: Path) -> None:
    """Load six independent B step-1 rays and the shared B baseline z1."""
    gate_path = run_root / "analysis_gate.json"
    if not gate_path.is_file():
        raise BlockedExecution("C requires analysis_gate.json from the independent analyzer")
    source["analysis_gate"] = _load_json(gate_path)
    b_samples = run_root / "stages" / "B" / "samples"
    if not b_samples.is_dir():
        raise BlockedExecution("C requires completed Stage B samples")
    b_store = Task7SampleStore(b_samples)
    baseline = b_store.load_record("B_baseline_pre_step_1")
    baseline_record = baseline.get("record", baseline)
    z1 = baseline_record.get("full_latent")
    if not isinstance(z1, np.ndarray):
        raise ResumeMismatch("B baseline step-1 record lacks full_latent")
    directions: dict[str, np.ndarray] = {}
    names = ("v0_alpha_00_plus", "v0_alpha_00_minus", "v0_alpha_01_plus", "v0_alpha_01_minus",
             "v0_alpha_02_plus", "v0_alpha_02_minus")
    for index, name in enumerate(names):
        record = b_store.load_record(f"B_{name}_step_1").get("record", {})
        full = record.get("full_latent")
        if not isinstance(full, np.ndarray) or full.shape != z1.shape:
            raise ResumeMismatch(f"B step-1 full_latent is missing or malformed: {name}")
        directions[f"delta1_{index:02d}"] = np.subtract(full, z1, dtype=np.float32)
    source["z1"], source["delta1_directions"] = np.array(z1, copy=True), directions


def make_stage_executor(stage: str, *, source: Mapping[str, Any], feedback: Any,
                        run_dir: str | Path | None = None) -> Callable[[Mapping[str, Any]], Any]:
    """Bind source samples to the approved feedback adapter.

    Stage A encodes each frozen decoder frame twice and checks native parity.
    Stage B feeds each trajectory's own observed encoder output into step 1.
    Stage C consumes an explicit analyzer gate and independently normalizes
    actual B ``delta1`` rays; zero rays are skipped with evidence.
    """
    stage = str(stage).upper()
    if stage not in {"A", "B", "C"}:
        raise ValueError("stage must be A/B/C")
    trajectories: dict[str, Mapping[str, Any]] = {}
    c_state: dict[str, Any] = {}

    def execute(spec: Mapping[str, Any]) -> Any:
        if stage == "A":
            encoder = getattr(feedback, "encoder", None)
            if encoder is None or not callable(getattr(encoder, "encode", None)):
                raise BlockedExecution("Stage A requires the reviewed FeedbackEncoder")
            result = encoder.encode(_source_frame(source, spec["source_name"]), precision=spec["precision"])
            arrays = result.get("arrays", {}) if isinstance(result, Mapping) else {}
            output = arrays.get("actual_output")
            if output is None:
                raise ValueError("Stage A encoder did not expose actual output")
            historical = _load_array(_decoder_sample_root(Path(source["decoder_root"]), spec["source_name"]) / "direct_condition_latent_float32.npy")
            if spec["precision"] == "native" and not np.array_equal(np.asarray(output), historical):
                raise ValueError("Stage A native output differs from historical direct condition")
            return {"source_name": spec["source_name"], "precision": spec["precision"],
                    "encoded_condition": np.array(output, copy=True), "encoder": result,
                    "evidence": {"operation_counts": {"G": 0, "D": 0, "E": 1}}}
        if stage == "B":
            trajectory = str(spec["trajectory_id"])
            if int(spec["step_index"]) == 0:
                _, consumed = _source_condition(source, spec["source_name"])
                condition = feedback.extract_condition(consumed)
            else:
                prior = trajectories.get(trajectory)
                if not isinstance(prior, Mapping) or "encoded_condition" not in prior:
                    raise ResumeMismatch(f"B step 1 has no verified own step 0 feedback: {trajectory}")
                condition = prior["encoded_condition"]
            record = feedback.step(condition, int(spec["step_index"]))
            if int(spec["step_index"]) == 0:
                expected = _load_array(_source_sample_root(Path(source["raw_root"]), spec["source_name"]) / "output_full.npy")
                actual = record.get("full_latent")
                if actual is None and isinstance(record.get("generation"), Mapping):
                    actual = record["generation"].get("output_full")
                if actual is None or not np.array_equal(np.asarray(actual), expected):
                    raise ValueError("B step 0 G output differs from historical output_full")
            trajectories[trajectory] = {"encoded_condition": np.array(record["encoded_condition"], copy=True), "record": record}
            return record
        if not c_state:
            gate = source.get("analysis_gate")
            if not isinstance(gate, Mapping):
                raise BlockedExecution("C requires an explicit analyzer gate")
            required = ("a_scientific_pass", "b_engineering_pass", "b_repeatability_pass", "source_sha256", "code_sha256")
            if any(not gate.get(key) for key in required):
                raise BlockedExecution("C analyzer gate is incomplete or not approved")
            if gate["source_sha256"] != source.get("source_tree_sha256") or gate["code_sha256"] != source.get("task7_code_sha256"):
                raise ResumeMismatch("C analyzer gate is not bound to this source/code")
            c_state["gate"] = gate
            z1 = source.get("z1")
            if z1 is None:
                raise BlockedExecution("C shared baseline z1 is unavailable")
            c_state["z1"] = np.asarray(z1, dtype=np.float32)
            c_state["directions"] = source.get("delta1_directions", {})
        if str(spec["kind"]).startswith("baseline"):
            return feedback.step(feedback.extract_condition(c_state["z1"]), 1)
        direction = c_state["directions"].get(spec["direction_id"]) if isinstance(c_state["directions"], Mapping) else None
        if direction is None:
            raise SkipSample("SKIPPED_C_ZERO_RAY", f"{spec['direction_id']} has no delta1 ray")
        direction = np.asarray(direction, dtype=np.float32)
        rms = float(np.sqrt(np.mean(direction.astype(np.float64) ** 2)))
        if not np.isfinite(rms) or rms == 0:
            raise SkipSample("SKIPPED_C_ZERO_RAY", f"{spec['direction_id']} has zero delta1 RMS")
        target = c_state["z1"].copy(); mask = np.asarray(feedback.mask, dtype=bool)
        target[mask] += np.float32(spec["sign"] * spec["beta"] * rms) * direction[mask]
        return feedback.step(feedback.extract_condition(target), 1)

    return execute


def _load_contract(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        raise BlockedExecution("--launch-contract is required for live Task 7 stages")
    contract = _load_json(Path(path))
    if not isinstance(contract, Mapping) or not isinstance(contract.get("code_bundle_sha256"), str) or len(contract["code_bundle_sha256"]) != 64:
        raise BlockedExecution("new launch contract must contain actual code_bundle_sha256")
    return dict(contract)


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.run_dir).resolve() if args.run_dir else (Path.cwd() / "umi_task7_preflight").resolve()
    root.mkdir(parents=True, exist_ok=True)
    validate_run_roots(root, raw_root=args.raw_root, decoder_root=args.decoder_root)
    bindings: dict[str, Any] = {"generation_started": False, "raw_root": args.raw_root, "decoder_root": args.decoder_root,
                                "framework_root": args.framework_root, "checkpoint": args.checkpoint, "vae": args.vae,
                                "action": args.action, "video": args.video, "task5_root": args.task5_root,
                                "gpu_index": args.gpu_index, "task7_code_sha256": _code_digest()}
    if args.raw_root or args.decoder_root:
        if not args.raw_root or not args.decoder_root:
            raise BlockedExecution("preflight requires both raw and decoder roots")
        bindings["source"] = validate_task7_sources(args.raw_root, args.decoder_root)
    if args.launch_contract:
        contract = _load_contract(args.launch_contract)
        try:
            from .umi_task6_operational import _validate_static_contract, checkpoint_content_identity, verify_pinned_bridge_assets
        except ImportError:
            from umi_task6_operational import _validate_static_contract, checkpoint_content_identity, verify_pinned_bridge_assets
        _validate_static_contract(contract)
        if args.action and args.video:
            bindings["assets"] = verify_pinned_bridge_assets(args.action, args.video,
                                                              expected_hashes=contract.get("bridge_asset_hashes"))
        if args.checkpoint:
            observed_checkpoint = checkpoint_content_identity(args.checkpoint)
            expected_checkpoint = contract.get("checkpoint_identity", {}).get("sha256")
            if expected_checkpoint and observed_checkpoint != expected_checkpoint:
                raise ResumeMismatch("checkpoint identity differs from launch contract")
            bindings["checkpoint_sha256"] = observed_checkpoint
        if args.vae:
            observed_vae = sha256_file(args.vae)
            if contract.get("vae_sha256") and observed_vae != contract["vae_sha256"]:
                raise ResumeMismatch("VAE identity differs from launch contract")
            bindings["vae_sha256"] = observed_vae
        bindings["launch_contract_sha256"] = sha256_file(args.launch_contract)
        bindings["code_bundle_sha256"] = contract["code_bundle_sha256"]
    else:
        bindings["launch_contract_sha256"], bindings["code_bundle_sha256"] = None, None
    status = {"schema_version": "umi-task7-run-v2", "status": "PREFLIGHT_COMPLETE", "phase": "PREFLIGHT",
              "generation_started": False, "bindings": bindings, "observed_counts": _empty_counts(),
              "completed_samples": [], "failed_samples": [], "skipped_samples": []}
    _atomic_json(root / "run_status.json", status)
    return status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "smoke", "A", "B", "C"), default="preflight")
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--raw-root")
    parser.add_argument("--decoder-root")
    parser.add_argument("--framework-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--vae")
    parser.add_argument("--action")
    parser.add_argument("--video")
    parser.add_argument("--launch-contract")
    parser.add_argument("--task5-root")
    parser.add_argument("--gpu-index", type=int, default=0)
    return parser.parse_args(argv)


def _default_factory(args: argparse.Namespace, source: Mapping[str, Any]) -> Any:
    if not all((args.framework_root, args.checkpoint, args.vae, args.action, args.video, args.task5_root)):
        raise BlockedExecution("live stages require framework/checkpoint/vae/action/video/task5-root")
    try:
        from .umi_task6_operational import OfficialRuntimeFactory, extract_task5_directions, verify_pinned_bridge_assets
        from .umi_task6_cosmos_loader import load_task6_cosmos_runtime
    except ImportError:
        from umi_task6_operational import OfficialRuntimeFactory, extract_task5_directions, verify_pinned_bridge_assets
        from umi_task6_cosmos_loader import load_task6_cosmos_runtime
    contract = source["launch_contract"]
    task5_contract = contract.get("task5", {})
    task5 = extract_task5_directions(args.task5_root,
        expected_manifest_sha256=task5_contract.get("manifest_sha256"),
        expected_plan_sha256=task5_contract.get("plan_sha256"),
        expected_direction_file_hashes=task5_contract.get("direction_file_sha256"),
        expected_direction_hashes=task5_contract.get("direction_sha256"))
    assets = verify_pinned_bridge_assets(args.action, args.video, expected_hashes=contract.get("bridge_asset_hashes"))
    return OfficialRuntimeFactory(loader=load_task6_cosmos_runtime, framework_root=args.framework_root,
        checkpoint=args.checkpoint, vae=args.vae, contract=contract, direction_bank=task5["bank"],
        action=contract["action"], prompt=contract["prompt"], video=args.video, phase=args.stage,
        run_dir=args.run_dir, resume=args.resume)


def _record_live_failure(root: Path, error: BaseException, *, generation_started: bool = False) -> None:
    _atomic_json(root / "run_status.json", {"schema_version": "umi-task7-run-v2",
        "status": "RESOURCE_STOP" if isinstance(error, ResourceStop) else "FAILED",
        "generation_started": bool(generation_started), "reason_code": type(error).__name__, "reason": str(error)})


def main(argv: list[str] | None = None, *, factory: Any | None = None,
         monitor: Any | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "preflight":
        try:
            _preflight(args)
            return 0
        except BaseException as error:
            if args.run_dir:
                root = Path(args.run_dir).resolve(); root.mkdir(parents=True, exist_ok=True)
                _atomic_json(root / "run_status.json", {"schema_version": "umi-task7-run-v2",
                    "status": "RESOURCE_STOP" if isinstance(error, ResourceStop) else "FAILED",
                    "generation_started": False, "reason_code": type(error).__name__, "reason": str(error)})
            raise
    if not args.run_dir:
        raise BlockedExecution("--run-dir is required for smoke/A/B/C")
    root = Path(args.run_dir).resolve(); validate_run_roots(root, raw_root=args.raw_root, decoder_root=args.decoder_root)
    try:
        source = validate_task7_sources(args.raw_root, args.decoder_root) if args.raw_root and args.decoder_root else {}
        contract = _load_contract(args.launch_contract)
    except BaseException as error:
        _record_live_failure(root, error, generation_started=False)
        raise
    source.update({"task7_code_sha256": _code_digest(), "launch_contract": contract})
    if monitor is None:
        try:
            from .run_umi_task6_official import resource_samplers
        except ImportError:
            from run_umi_task6_official import resource_samplers
        samplers = resource_samplers(str(root), gpu_index=args.gpu_index)
        monitor = ResourceMonitor(root / "monitor", gpu_sampler=samplers["gpu"], ram_sampler=samplers["ram"], disk_sampler=samplers["disk"])
        monitor.start()
    if not callable(getattr(monitor, "check", None)):
        raise BlockedExecution("live stages require Task 6 ResourceMonitor")
    try:
        preload = monitor.check(phase="preload", starting_new_sample=False)
        if not isinstance(preload, Mapping) or preload.get("status") == "HARD_STOP":
            raise ResourceStop(str(preload))
        # Re-check with the strict Task 7 field-completeness wrapper.  The
        # Task-6 evaluator's defaults are intentionally not trusted here.
        require_resource_snapshot(getattr(monitor, "last_resources", {}), phase="preload")
    except BaseException:
        if args.run_dir:
            _atomic_json(root / "run_status.json", {"schema_version": "umi-task7-run-v2", "status": "RESOURCE_STOP",
                "generation_started": False, "reason_code": "PRELOAD_RESOURCE_STOP"})
        raise
    try:
        if factory is None:
            factory = _default_factory(args, source)
        built = factory.build() if hasattr(factory, "build") else (factory(args, source) if callable(factory) else factory)
    except BaseException as error:
        _record_live_failure(root, error, generation_started=False)
        raise
    if not isinstance(built, (tuple, list)) or len(built) < 3:
        raise BlockedExecution("factory must return (Task6RuntimeAdapter, inputs, encoder)")
    adapter, inputs, encoder = built[0], built[1], built[2]
    try:
        from .umi_task7_runtime import FeedbackRuntime
    except ImportError:
        from umi_task7_runtime import FeedbackRuntime
    official = getattr(adapter, "runtime", None)
    if official is None:
        raise BlockedExecution("factory result lacks underlying OfficialPrecisionRuntime")
    try:
        source_z0, _ = _source_condition(source, "baseline_pre")
        source_mask = _load_array(_source_sample_root(Path(source["raw_root"]), "baseline_pre") / "mask.npy").astype(bool, copy=False)
        source_v0 = _load_array(_source_sample_root(Path(source["raw_root"]), "v0_alpha_00_plus") / "direction.npy").astype(np.float32, copy=False)
        if (_array_hash(source_z0) != _array_hash(inputs.z0) or _array_hash(source_mask) != _array_hash(inputs.geometry.mask)
                or _array_hash(source_v0) != _array_hash(inputs.directions["v0"])):
            raise ResumeMismatch("factory frozen z0/mask/v0 differs from immutable source artifacts")
    except BaseException as error:
        _record_live_failure(root, error, generation_started=False)
        raise
    try:
        from .umi_task7_encoder import FeedbackEncoder
    except ImportError:
        from umi_task7_encoder import FeedbackEncoder
    if not isinstance(encoder, FeedbackEncoder):
        encoder = FeedbackEncoder(getattr(encoder, "encoder", encoder), device="cuda")
    try:
        feedback = FeedbackRuntime(official, encoder=encoder, z0=source_z0, mask=source_mask,
                                   condition_indexes=inputs.geometry.condition_indexes, v0=source_v0, seed=0)
    except BaseException as error:
        _record_live_failure(root, error, generation_started=False)
        raise
    source.update({"z0": _array_hash(source_z0), "mask": _array_hash(source_mask), "v0": _array_hash(source_v0)})
    try:
        if args.stage == "smoke":
            require_resource_snapshot(getattr(monitor, "last_resources", {}), phase="smoke")
            _, consumed = _source_condition(source, "baseline_pre")
            record = feedback.step(feedback.extract_condition(consumed), 0)
            store = Task7SampleStore(root / "smoke" / "samples")
            store.write_success("smoke_baseline", {"record": record, "engineering": True}, operation_counts=_capture_counts(record))
            _atomic_json(root / "smoke" / "run_status.json", {"status": "SMOKE_COMPLETE", "engineering": True,
                "formal_counts": _empty_counts(), "observed_counts": _capture_counts(record)})
            return 0
        if args.stage == "C":
            _prepare_c_source(source, root)
        binding = {"source": source.get("source_tree_sha256", "source-bound"), "task7_code_sha256": source["task7_code_sha256"],
                   "config": {"stage": args.stage, "plan": build_stage_plan(args.stage)},
                   "model": contract.get("checkpoint_identity", {}).get("sha256", ""), "vae": contract.get("vae_sha256", ""),
                   "framework": contract.get("framework_commit", ""), "mask": source["mask"], "z0": source["z0"],
                   "v0": source["v0"], "noise_policy": "prediction-region-seed-paired"}
        executor = make_stage_executor(args.stage, source=source, feedback=feedback, run_dir=root)
        result = run_stage(args.stage, root / "stages" / args.stage, execute=executor, binding=binding,
                           resume=args.resume, monitor=monitor, allowed_skips=args.stage == "C")
        return 0 if result.get("status") == "COMPLETE" else 1
    except BaseException as error:
        _record_live_failure(root, error, generation_started=True)
        raise
    finally:
        if callable(getattr(monitor, "stop", None)):
            try: monitor.stop()
            except BaseException: pass
        if callable(getattr(factory, "unload", None)):
            factory.unload()


__all__ = ["BlockedExecution", "ResourceStop", "ResumeMismatch", "SkipSample", "StaticMonitor",
           "Task7SampleStore", "build_stage_plan", "canonical_json", "evaluate_task7_resources",
           "main", "make_stage_executor", "parse_args", "require_resource_snapshot", "run_stage",
           "sha256_file", "stage_counts", "test_binding", "validate_run_roots",
           "validate_task7_sources"]


if __name__ == "__main__":
    raise SystemExit(main())
