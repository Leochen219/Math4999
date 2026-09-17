"""Pure, deterministic contracts for the Task 6 cross-context experiment.

This module deliberately has no framework, OpenCV, or CUDA imports. Asset
loading and model execution belong to the later runtime; these helpers only
validate and describe values that cross those boundaries. The cgroup sampler is
the sole read-only operating-system probe used by resource policy evaluation.
"""
from __future__ import annotations

import hashlib
import json
import numbers
import os
from pathlib import Path
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


SOURCE_COMMIT = "2b17a2413bd86b2cf9b03823637108851e4ddf2d"
ALPHAS = (0.001, 0.003, 0.01)
DIRECTION_IDS = ("v0", "v1", "v2", "u01", "u12")
STATE_IDS = ("umi_reference", "bridge_0", "bridge_384")
SEEDS = (0, 1)
TERMINAL_STATUSES = frozenset({
    "AWAITING_RESOURCE_REVIEW", "AWAITING_REVIEW", "COMPLETE", "FAILED",
    "SKIPPED", "RESOURCE_STOP", "INTERRUPTED",
})
# These root-relative control/monitor files are intentionally mutable after a
# generation run (decoder monitoring may append to them or update status).
# They are excluded from the raw immutable evidence manifest everywhere.
RAW_MUTABLE_FILES = frozenset({
    "run_status.json", "invocation_history.jsonl", "gpu_samples.csv",
    "ram_samples.csv", "disk_samples.csv", "sample_resource_snapshots.csv",
    "sample_resource_snapshots.jsonl",
})


@dataclass(frozen=True)
class StateAsset:
    state: str
    source_commit: str
    video_path: str | None
    action_path: str
    source_kind: str

    @property
    def input_video(self) -> str | None:
        return self.video_path

    @property
    def action_asset(self) -> str:
        return self.action_path


# Paths are repository-relative paths in the pinned cosmos-dependencies tree.
# Reference data intentionally names the Task 5 first-frame/action provenance,
# rather than pretending it is one of the paired Bridge assets.
STATE_CATALOG = MappingProxyType({
    "umi_reference": StateAsset("umi_reference", SOURCE_COMMIT,
                                 "task5/first_frame.png", "task5/action_chunk_0.json", "task5_reuse"),
    "bridge_0": StateAsset("bridge_0", SOURCE_COMMIT,
                            "inputs/action/bridge_20260501_0.mp4", "inputs/action/bridge_20260501_0.json", "official_paired_asset"),
    "bridge_384": StateAsset("bridge_384", SOURCE_COMMIT,
                              "inputs/action/bridge_20260501_384.mp4", "inputs/action/bridge_20260501_384.json", "official_paired_asset"),
})
STATE_ASSETS = STATE_CATALOG


@dataclass(frozen=True)
class ExperimentGroup:
    state: str
    seed: int

    def __post_init__(self) -> None:
        if self.state not in STATE_IDS or isinstance(self.seed, bool) or not isinstance(self.seed, numbers.Integral) or int(self.seed) not in SEEDS:
            raise ValueError("Task 6 groups require one of the exact integer seeds 0 or 1")

    @property
    def group_id(self) -> str:
        return f"{self.state}__seed_{self.seed}"


GROUPS = tuple(ExperimentGroup(state, seed) for state in STATE_IDS for seed in SEEDS)


def pilot_groups() -> tuple[ExperimentGroup, ...]:
    """Return the only group authorized for the first pilot."""
    return (ExperimentGroup("bridge_0", 0),)


select_pilot_groups = pilot_groups


def _require_group(state: str, seed: int) -> ExperimentGroup:
    if not isinstance(state, str) or isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError("state and seed identify an approved Task 6 group")
    try:
        group = ExperimentGroup(state, int(seed))
    except (TypeError, ValueError) as error:
        raise ValueError("state and seed identify an approved Task 6 group") from error
    if group not in GROUPS:
        raise ValueError(f"unapproved Task 6 group: {state!r}, seed {seed!r}")
    return group


def build_generation_plan(state: str = "bridge_0", seed: int = 0) -> list[dict[str, Any]]:
    """Build one group-local plan: baseline, 30 signed calls, baseline."""
    group = _require_group(state, seed)
    result: list[dict[str, Any]] = [{
        "sample_id": f"{group.state}__seed_{group.seed}__baseline_pre",
        "state": group.state, "seed": group.seed, "model_seed": group.seed,
        "kind": "baseline", "alpha": 0.0, "sign": 0, "direction_id": None,
    }]
    for direction_id in DIRECTION_IDS:
        for ordinal, alpha in enumerate(ALPHAS):
            for sign, label in ((1, "plus"), (-1, "minus")):
                result.append({
                    "sample_id": f"{group.state}__seed_{group.seed}__{direction_id}_alpha_{ordinal:02d}_{label}",
                    "state": group.state, "seed": group.seed, "model_seed": group.seed,
                    "kind": "perturbation", "alpha": alpha, "sign": sign,
                    "direction_id": direction_id,
                })
    result.append({
        "sample_id": f"{group.state}__seed_{group.seed}__baseline_post",
        "state": group.state, "seed": group.seed, "model_seed": group.seed,
        "kind": "baseline", "alpha": 0.0, "sign": 0, "direction_id": None,
    })
    if len(result) != 32 or len({row["sample_id"] for row in result}) != 32:
        raise AssertionError("Task 6 generation plan must contain exactly 32 unique calls")
    return result


build_task6_call_plan = build_generation_plan


def build_decoder_replay_plan(state: str = "bridge_0", seed: int = 0) -> list[dict[str, Any]]:
    """Build 16 serial decoder calls for eight v0-selected latent records."""
    group = _require_group(state, seed)
    logical: list[dict[str, Any]] = [{
        "sample_id": f"{group.state}__seed_{group.seed}__baseline_pre", "kind": "baseline",
        "alpha": 0.0, "sign": 0, "direction_id": None,
    }]
    for ordinal, alpha in enumerate(ALPHAS):
        for sign, label in ((1, "plus"), (-1, "minus")):
            logical.append({"sample_id": f"{group.state}__seed_{group.seed}__v0_alpha_{ordinal:02d}_{label}",
                            "kind": "perturbation", "alpha": alpha, "sign": sign, "direction_id": "v0"})
    logical.append({"sample_id": f"{group.state}__seed_{group.seed}__baseline_post", "kind": "baseline",
                    "alpha": 0.0, "sign": 0, "direction_id": None})
    result = []
    for item in logical:
        for decode_precision in ("native_bf16", "temporary_fp32"):
            result.append({**item, "state": group.state, "seed": group.seed,
                           "decode_precision": decode_precision,
                           "replay_id": f"{item['sample_id']}__{decode_precision}"})
    if len(result) != 16:
        raise AssertionError("Task 6 decoder replay plan must contain exactly 16 calls")
    return result


def preprocess_frame(frame: Any, *, size: int | tuple[int, int] = 256, resize_backend=None) -> np.ndarray:
    """Apply official geometry with a lazy, bindable resize backend.

    Scale is capped at one (never upscale), rounded as ``int(scale*dim+0.5)``;
    padding is right/bottom only. The official backend is Torchvision bicubic
    antialiasing, imported only at call time. A NumPy nearest fallback is
    explicitly not pixel-equivalent to that backend.
    """
    source = np.asarray(frame)
    if source.ndim != 3 or source.shape[0] < 1 or source.shape[1] < 1 or source.shape[2] < 1:
        raise ValueError("frame must be a nonempty HWC array")
    if not np.issubdtype(source.dtype, np.number) or not np.all(np.isfinite(source)):
        raise ValueError("frame must contain finite numeric values")
    if isinstance(size, numbers.Integral):
        target_h = target_w = int(size)
    else:
        if len(size) != 2:
            raise ValueError("size must be a positive integer or (height, width)")
        target_h, target_w = (int(size[0]), int(size[1]))
    if target_h < 1 or target_w < 1:
        raise ValueError("size must be positive")
    height, width = source.shape[:2]
    scale = min(target_w / width, target_h / height, 1.0)
    resized_h = max(1, int(scale * height + 0.5))
    resized_w = max(1, int(scale * width + 0.5))
    if (resized_h, resized_w) == (height, width):
        resized = source.copy()
    elif resize_backend is not None:
        resized = np.asarray(resize_backend(source, resized_w, resized_h))
        if resized.shape != (resized_h, resized_w, source.shape[2]):
            raise ValueError("resize backend returned the wrong HWC shape")
        if resized.dtype != source.dtype:
            resized = resized.astype(source.dtype)
    else:
        try:
            import torch
            from torchvision.transforms.functional import InterpolationMode, resize as tv_resize
            tensor = torch.from_numpy(np.ascontiguousarray(source)).permute(2, 0, 1)
            resized_tensor = tv_resize(tensor, [resized_h, resized_w], interpolation=InterpolationMode.BICUBIC, antialias=True)
            resized = resized_tensor.permute(1, 2, 0).cpu().numpy().astype(source.dtype, copy=False)
        except Exception as error:
            raise RuntimeError("official Torchvision resize unavailable or failed; bind resize_backend") from error
    pad_h = target_h - resized_h
    pad_w = target_w - resized_w
    if pad_h < 0 or pad_w < 0:
        raise AssertionError("aspect resize exceeded target")
    mode = "edge" if pad_h >= resized_h or pad_w >= resized_w else "reflect"
    output = np.pad(resized, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    return np.ascontiguousarray(output, dtype=source.dtype)


# Descriptive aliases used by callers that distinguish frame zero explicitly.
preprocess_frame0 = preprocess_frame
preprocess_frame_zero = preprocess_frame


def parse_action(action: Any) -> np.ndarray:
    """Validate an action and return an independent contiguous FP32 array."""
    try:
        raw = np.asarray(action)
        if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
            raise ValueError("action must be finite numeric values with shape (16, 10)")
        array = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("action must be finite numeric values with shape (16, 10)") from error
    if array.shape != (16, 10) or not np.all(np.isfinite(array)):
        raise ValueError("action must be finite numeric values with shape (16, 10)")
    return np.ascontiguousarray(array.copy())


validate_action = parse_action


def stable_hash(value: Any) -> str:
    """Hash canonical dtype/shape/bytes for arrays, or canonical JSON values."""
    if isinstance(value, np.ndarray) or hasattr(value, "__array__"):
        array = np.ascontiguousarray(np.asarray(value))
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("ascii")); digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")); digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        return digest.hexdigest()
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


hash_action = stable_hash
hash_array = stable_hash


@dataclass(frozen=True)
class ResourceSnapshot:
    """Optional-field snapshot convenient for tests and monitor adapters."""
    gpu_used_gib: float = 0.0
    gpu_free_gib: float = 1e300
    gpu_reserved_gib: float = 0.0
    gpu_peak_allocated_gib: float = 0.0
    gpu_peak_nvml_used_gib: float = 0.0
    ram_available_gib: float = 1e300
    rss_gib: float = 0.0
    swap_used_gib: float = 0.0
    disk_free_gib: float = 1e300
    forecast_free_gib: float = 1e300
    gpu_cleanup_growth_gib: float = 0.0
    ram_cleanup_growth_gib: float = 0.0
    consecutive_growth_samples: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sample_cgroup_memory(root: str | os.PathLike[str] = "/sys/fs/cgroup") -> dict[str, Any]:
    """Read cgroup v2 memory limit/usage, returning explicit failures."""
    directory = Path(root)
    limit_path, current_path = directory / "memory.max", directory / "memory.current"
    if not limit_path.exists() and not current_path.exists():
        return {}
    try:
        limit_text = limit_path.read_text(encoding="ascii").strip()
        current_bytes = int(current_path.read_text(encoding="ascii").strip(), 10)
        if current_bytes < 0:
            raise ValueError("memory.current must be nonnegative")
        if limit_text == "max":
            return {"cgroup_memory_limited": False,
                    "cgroup_memory_limit_gib": None,
                    "cgroup_memory_current_gib": current_bytes / 2**30,
                    "cgroup_memory_free_gib": None}
        limit_bytes = int(limit_text, 10)
        if limit_bytes < 0 or current_bytes > limit_bytes:
            raise ValueError("cgroup memory values are out of range")
        return {"cgroup_memory_limited": True,
                "cgroup_memory_limit_gib": limit_bytes / 2**30,
                "cgroup_memory_current_gib": current_bytes / 2**30,
                "cgroup_memory_free_gib": (limit_bytes - current_bytes) / 2**30}
    except (OSError, TypeError, ValueError) as error:
        return {"monitor_failure": f"cgroup v2 memory sampler failed: {error}"}


def _number(snapshot: Mapping[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in snapshot and snapshot[name] is not None:
            if isinstance(snapshot[name], bool):
                return float("nan")
            try:
                return float(snapshot[name])
            except (TypeError, ValueError):
                return float("nan")
    return default


def evaluate_resources(snapshot: Mapping[str, Any] | ResourceSnapshot, *, phase: str = "pilot", starting_new_sample: bool = False) -> dict[str, Any]:
    """Evaluate all Task 6 resource gates without side effects.

    Values are GiB except swap (GiB) and the integer consecutive-growth count.
    Hard-stop codes are deterministic and always dominate warning codes.
    """
    if isinstance(snapshot, ResourceSnapshot):
        snapshot = snapshot.as_dict()
    if not isinstance(snapshot, Mapping):
        raise TypeError("resource snapshot must be a mapping")
    hard: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []
    gpu_used = _number(snapshot, "gpu_used_gib", "nvml_used_gib")
    gpu_free = _number(snapshot, "gpu_free_gib", "nvml_free_gib", default=1e300)
    gpu_reserved = _number(snapshot, "gpu_reserved_gib", "torch_reserved_gib")
    ram_available = _number(snapshot, "ram_available_gib", "available_ram_gib", default=1e300)
    rss = _number(snapshot, "rss_gib", "process_rss_gib")
    swap = _number(snapshot, "swap_used_gib", "swap_gib")
    disk = _number(snapshot, "disk_free_gib", "free_disk_gib", default=1e300)
    forecast = _number(snapshot, "forecast_free_gib", "forecast_completion_free_gib", default=1e300)
    consecutive_raw = _number(snapshot, "consecutive_growth_samples", default=0)
    gpu_consecutive_raw = _number(snapshot, "gpu_consecutive_growth_samples", default=consecutive_raw)
    ram_consecutive_raw = _number(snapshot, "ram_consecutive_growth_samples", default=consecutive_raw)
    consecutive = int(consecutive_raw) if np.isfinite(consecutive_raw) and consecutive_raw.is_integer() and consecutive_raw >= 0 else float("nan")
    gpu_consecutive = int(gpu_consecutive_raw) if np.isfinite(gpu_consecutive_raw) and gpu_consecutive_raw.is_integer() and gpu_consecutive_raw >= 0 else float("nan")
    ram_consecutive = int(ram_consecutive_raw) if np.isfinite(ram_consecutive_raw) and ram_consecutive_raw.is_integer() and ram_consecutive_raw >= 0 else float("nan")
    gpu_growth = _number(snapshot, "gpu_cleanup_growth_gib", "gpu_cleanup_baseline_growth_gib")
    ram_growth = _number(snapshot, "ram_cleanup_growth_gib", "ram_cleanup_baseline_growth_gib")
    cgroup_limited = snapshot.get("cgroup_memory_limited", False)
    if not isinstance(cgroup_limited, bool):
        cgroup_limited = None
    cgroup_limit = _number(snapshot, "cgroup_memory_limit_gib", default=1e300)
    cgroup_current = _number(snapshot, "cgroup_memory_current_gib", default=0.0)
    cgroup_free = _number(snapshot, "cgroup_memory_free_gib", default=1e300)
    cgroup_invalid = False
    if cgroup_limited is True:
        cgroup_invalid = any(name not in snapshot for name in (
            "cgroup_memory_limit_gib", "cgroup_memory_current_gib", "cgroup_memory_free_gib"))
        cgroup_invalid = cgroup_invalid or any(
            not np.isfinite(value) or value < 0 for value in (cgroup_limit, cgroup_current, cgroup_free))
        cgroup_invalid = cgroup_invalid or cgroup_current > cgroup_limit
        cgroup_invalid = cgroup_invalid or not np.isclose(cgroup_free, cgroup_limit - cgroup_current, rtol=0, atol=1e-6)

    numeric_values = (gpu_used, gpu_free, gpu_reserved, ram_available, rss, swap, disk, forecast,
                      gpu_growth, ram_growth)
    if any(not np.isfinite(value) for value in numeric_values) or not np.isfinite(consecutive) or not np.isfinite(gpu_consecutive) or not np.isfinite(ram_consecutive):
        hard.append(("RESOURCE_SNAPSHOT_NONFINITE", "resource snapshot contains nonfinite or malformed numeric values"))
    if bool(snapshot.get("cuda_oom", False)) or bool(snapshot.get("cuda_out_of_memory", False)):
        hard.append(("CUDA_OOM", "CUDA reported an out-of-memory failure"))
    if snapshot.get("monitor_failure") or snapshot.get("monitor_error"):
        hard.append(("MONITOR_FAILURE", "resource monitor reported a failure"))
    if cgroup_limited is None or cgroup_invalid:
        hard.append(("RESOURCE_SNAPSHOT_NONFINITE", "cgroup memory snapshot is malformed"))
    mean_bytes = _number(snapshot, "mean_success_sample_bytes", default=0.0)
    remaining_samples = _number(snapshot, "remaining_samples", default=0.0)
    raw_mean = snapshot.get("mean_success_sample_bytes")
    raw_remaining = snapshot.get("remaining_samples")
    valid_mean = "mean_success_sample_bytes" not in snapshot or (not isinstance(raw_mean, bool) and isinstance(raw_mean, numbers.Real) and np.isfinite(mean_bytes) and mean_bytes >= 0)
    valid_remaining = "remaining_samples" not in snapshot or (not isinstance(raw_remaining, bool) and isinstance(raw_remaining, numbers.Integral) and int(raw_remaining) >= 0)
    if not valid_mean or not valid_remaining:
        hard.append(("RESOURCE_SNAPSHOT_NONFINITE", "disk forecast inputs are malformed"))
    elif ("disk_free_gib" in snapshot or "free_disk_gib" in snapshot) and (mean_bytes or remaining_samples):
        forecast = disk - (mean_bytes * remaining_samples * 1.3) / (1024 ** 3)
    matrix_estimate = _number(snapshot, "remaining_matrix_estimate_gib", default=0.0)
    full_matrix = bool(snapshot.get("full_matrix_launch", False)) or phase == "full-matrix"
    raw_matrix_estimate = snapshot.get("remaining_matrix_estimate_gib")
    valid_matrix_estimate = ("remaining_matrix_estimate_gib" in snapshot and
                             not isinstance(raw_matrix_estimate, bool) and
                             isinstance(raw_matrix_estimate, numbers.Real) and
                             np.isfinite(matrix_estimate) and matrix_estimate >= 0)
    if full_matrix and (not valid_matrix_estimate or disk <= 1.3 * matrix_estimate + 5.0):
        hard.append(("DISK_FULL_MATRIX_INSUFFICIENT", "free disk must exceed 1.3 times remaining matrix estimate plus 5 GiB"))

    if phase in ("start", "startup", "preload") and gpu_used > 1:
        hard.append(("GPU_START_USED_HIGH", "GPU start used memory exceeds 1 GiB"))
    if gpu_used > 75: hard.append(("GPU_USED_HIGH", "NVML GPU used memory exceeds 75 GiB"))
    if gpu_free < 20: hard.append(("GPU_FREE_LOW", "NVML GPU free memory is below 20 GiB"))
    if gpu_reserved > 65: hard.append(("GPU_RESERVED_HIGH", "PyTorch reserved memory exceeds 65 GiB"))
    if phase == "resource-smoke" and _number(snapshot, "gpu_peak_allocated_gib") > 35: hard.append(("GPU_SMOKE_PEAK_ALLOCATED_HIGH", "GPU smoke peak allocated exceeds 35 GiB"))
    if phase == "resource-smoke" and _number(snapshot, "gpu_peak_nvml_used_gib") > 45: hard.append(("GPU_SMOKE_PEAK_USED_HIGH", "GPU smoke peak NVML used exceeds 45 GiB"))
    if phase in ("start", "startup", "preload") and ram_available < 500:
        hard.append(("RAM_START_AVAILABLE_LOW", "RAM start availability is below 500 GiB"))
    if ram_available < 300: hard.append(("RAM_AVAILABLE_LOW", "available RAM is below 300 GiB"))
    if cgroup_limited is True and cgroup_free < 10:
        hard.append(("CGROUP_MEMORY_HEADROOM_CRITICAL", "cgroup memory headroom is below 10 GiB"))
    if rss > 160: hard.append(("RAM_RSS_HIGH", "process RSS exceeds 160 GiB"))
    if swap > 0: hard.append(("SWAP_IN_USE", "swap is in use"))
    if phase in ("start", "startup", "preload") and disk < 10:
        hard.append(("DISK_START_FREE_LOW", "disk start free space is below 10 GiB"))
    if starting_new_sample and disk < 5: hard.append(("DISK_FREE_LOW", "disk free space is below 5 GiB"))
    if gpu_consecutive >= 2 and gpu_growth > 2: hard.append(("GPU_CLEANUP_GROWTH", "GPU cleanup-baseline growth exceeds 2 GiB twice consecutively"))
    if ram_consecutive >= 2 and ram_growth > 10: hard.append(("RAM_CLEANUP_GROWTH", "RAM cleanup-baseline growth exceeds 10 GiB twice consecutively"))

    if gpu_used > 60: warnings.append(("GPU_USED_WARNING", "GPU used memory exceeds 60 GiB"))
    if gpu_free < 35: warnings.append(("GPU_FREE_WARNING", "GPU free memory is below 35 GiB"))
    if ram_available < 400: warnings.append(("RAM_AVAILABLE_WARNING", "available RAM is below 400 GiB"))
    if cgroup_limited is True and cgroup_free < 20:
        warnings.append(("CGROUP_MEMORY_HEADROOM_LOW", "cgroup memory headroom is below 20 GiB"))
    if rss > 100: warnings.append(("RAM_RSS_WARNING", "process RSS exceeds 100 GiB"))
    if disk < 8: warnings.append(("DISK_FREE_WARNING", "disk free space is below 8 GiB"))
    if forecast < 6: warnings.append(("DISK_FORECAST_WARNING", "forecast completion free space is below 6 GiB"))
    if hard:
        status, selected = "HARD_STOP", hard[0]
    elif warnings:
        status, selected = "WARNING", warnings[0]
    else:
        status, selected = "OK", (None, None)
    return {"status": status, "reason_code": selected[0], "reason": selected[1],
            "hard_stop_reasons": [{"code": c, "reason": r} for c, r in hard],
            "warning_reasons": [{"code": c, "reason": r} for c, r in warnings]}


evaluate_resource_policy = evaluate_resources


def build_run_status(status: str, *, reason_code: str | None = None, reason: str | None = None,
                     completed: Any = (), failed: Any = (), skipped: Any = (),
                     resource_snapshots: Mapping[str, Any] | None = None,
                     hashes: Mapping[str, str] | None = None, **extra: Any) -> dict[str, Any]:
    """Build a stable terminal status payload suitable for canonical JSON."""
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"status is not terminal: {status!r}")
    if not isinstance(hashes, Mapping) or set(hashes) != {"code", "model", "config", "direction", "input", "noise"}:
        raise ValueError("run status requires exactly the six canonical nonempty hash fields")
    if any(not isinstance(value, str) or not value for value in hashes.values()):
        raise ValueError("run status hash fields must be nonempty strings")
    payload: dict[str, Any] = {
        "status": str(status), "reason_code": reason_code, "reason": reason,
        "completed_samples": sorted(str(x) for x in completed),
        "failed_samples": sorted(str(x) for x in failed),
        "skipped_samples": sorted(str(x) for x in skipped),
        "last_resource_snapshots": dict(resource_snapshots or {}),
        "hashes": {str(key): str(value) for key, value in sorted((hashes or {}).items())},
    }
    payload.update(extra)
    # Ensure unsupported NaN values cannot enter run_status.json.
    json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return payload


canonical_run_status = build_run_status


__all__ = ["ALPHAS", "DIRECTION_IDS", "GROUPS", "SEEDS", "SOURCE_COMMIT", "STATE_ASSETS",
           "STATE_CATALOG", "STATE_IDS", "ExperimentGroup", "ResourceSnapshot", "StateAsset", "build_decoder_replay_plan",
           "build_generation_plan", "build_run_status", "build_task6_call_plan", "evaluate_resources",
           "evaluate_resource_policy", "hash_action", "hash_array", "parse_action", "pilot_groups",
           "preprocess_frame", "preprocess_frame0", "preprocess_frame_zero", "select_pilot_groups",
           "stable_hash", "validate_action", "canonical_run_status", "TERMINAL_STATUSES",
           "sample_cgroup_memory"]
