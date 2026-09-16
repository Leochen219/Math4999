"""Official-runtime UMI post-VAE scan runner.

The module deliberately keeps Cosmos and Torch behind lazy runtime seams.  The
call-plan, gate, persistence, and resume contract can therefore be exercised
on a CPU machine, while ``OfficialPostVaeRuntimeAdapter`` connects the same
contract to the installed official pipeline on the execution host.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

try:  # package import when launched as experiments.umi_fd_post_vae_scan
    from .umi_fd_post_vae_bridge import (
        CachedConditioningData,
        HookBoundary,
        build_setup_overrides_kwargs,
        build_broadcast_condition_mask,
        capture_final_latent,
        construct_delta,
        generate_direction_bank,
        hash_predicted_noisy_region,
        sha256_array,
        validate_cache_off,
        validate_runtime_setup,
    )
except ImportError:  # direct ``python experiments/umi_fd_post_vae_scan.py``
    from umi_fd_post_vae_bridge import (
        CachedConditioningData,
        HookBoundary,
        build_setup_overrides_kwargs,
        build_broadcast_condition_mask,
        capture_final_latent,
        construct_delta,
        generate_direction_bank,
        hash_predicted_noisy_region,
        sha256_array,
        validate_cache_off,
        validate_runtime_setup,
    )


ALPHAS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
DIRECTION_COUNT = 4
DIRECTION_SEED = 20260912
MODEL_SEED = 0
EXPECTED_CALL_COUNT = 54
EXPECTED_STAGE_A_COUNT = 4
EXPECTED_SCAN_COUNT = 50
NUM_STEPS = 30
SAMPLER = "unipc"
PRECISION = "bfloat16"
PARALLELISM_PRESET = "latency"
FRAMEWORK_COMMIT = "ffa9c6b60a6b04b2fae337577bc6cbd8a93c39f5"
COMPATIBILITY_FIELDS = (
    "framework_sha256",
    "model_sha256",
    "vae_sha256",
    "code_sha256",
    "bridge_sha256",
    "config_sha256",
    "input_sha256",
    "action_sha256",
    "direction_sha256",
    "z0_sha256",
    "mask_sha256",
    "noise_policy_sha256",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    files = [item for item in root.rglob("*") if item.is_file() and ".git" not in item.parts and "__pycache__" not in item.parts]
    for item in sorted(files, key=lambda value: value.relative_to(root).as_posix()):
        relative = item.relative_to(root).as_posix()
        content_hash = sha256_file(item)
        digest.update(relative.encode("utf-8")); digest.update(b"\0"); digest.update(content_hash.encode("ascii")); digest.update(b"\n")
    return digest.hexdigest()


def sha256_json_value(value: Any) -> str:
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _alpha_label(alpha: float) -> str:
    return format(float(alpha), ".10g").replace(".", "p").replace("-", "m")


def build_call_plan(
    *,
    alphas: Iterable[float] = ALPHAS,
    direction_count: int = DIRECTION_COUNT,
    direction_seed: int = DIRECTION_SEED,
    model_seed: int = MODEL_SEED,
) -> list[dict[str, Any]]:
    """Build the approved A/B/C/D + 50-call order exactly once."""

    values = tuple(float(alpha) for alpha in alphas)
    if values != tuple(ALPHAS):
        raise ValueError("the approved alpha grid is fixed")
    if int(direction_count) != DIRECTION_COUNT or int(direction_seed) != DIRECTION_SEED:
        raise ValueError("the approved direction count and seed are fixed")
    if int(model_seed) != MODEL_SEED:
        raise ValueError("model seed is fixed to 0")
    plan: list[dict[str, Any]] = []
    for sample_id, injection in (("A", "normal"), ("B", "explicit_zero"), ("C", "direction_plus"), ("D", "unchanged")):
        plan.append({
            "sample_id": sample_id, "phase": "stage_a", "kind": "control",
            "injection": injection, "direction_index": 0 if sample_id == "C" else None,
            "alpha": 0.01 if sample_id == "C" else 0.0, "sign": 1 if sample_id == "C" else 0,
            "model_seed": MODEL_SEED,
        })
    plan.append({"sample_id": "scan_pre", "phase": "scan", "kind": "baseline", "injection": "unchanged", "direction_index": None, "alpha": 0.0, "sign": 0, "model_seed": MODEL_SEED})
    for direction_index in range(DIRECTION_COUNT):
        for alpha in values:
            for sign, label in ((-1, "minus"), (1, "plus")):
                plan.append({
                    "sample_id": f"dir_{direction_index:02d}_alpha_{_alpha_label(alpha)}_{label}",
                    "phase": "scan", "kind": "perturbation", "injection": "direction",
                    "direction_index": direction_index, "alpha": alpha, "sign": sign,
                    "model_seed": MODEL_SEED,
                })
    plan.append({"sample_id": "scan_post", "phase": "scan", "kind": "baseline", "injection": "unchanged", "direction_index": None, "alpha": 0.0, "sign": 0, "model_seed": MODEL_SEED})
    validate_call_plan(plan)
    return plan


def validate_call_plan(plan: Iterable[Mapping[str, Any]]) -> None:
    calls = list(plan)
    if len(calls) != EXPECTED_CALL_COUNT:
        raise ValueError(f"call plan must contain exactly {EXPECTED_CALL_COUNT} calls")
    if [call.get("sample_id") for call in calls[:4]] != ["A", "B", "C", "D"]:
        raise ValueError("Stage A order must be A, B, C, D")
    if calls[4].get("sample_id") != "scan_pre" or calls[-1].get("sample_id") != "scan_post":
        raise ValueError("scan controls must surround the 48 perturbations")
    expected: list[tuple[int, float, int]] = []
    for direction in range(DIRECTION_COUNT):
        for alpha in ALPHAS:
            expected.extend(((direction, alpha, -1), (direction, alpha, 1)))
    got = [(int(call["direction_index"]), float(call["alpha"]), int(call["sign"])) for call in calls[5:-1]]
    if got != expected:
        raise ValueError("scan order must be direction-major, alpha-major, minus-then-plus")


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: tuple[str, ...] = ()
    checked: tuple[str, ...] = ()


class RuntimeCaptureError(RuntimeError):
    """Runtime failure carrying the detached boundary evidence captured so far."""

    def __init__(self, message: str, capture: Mapping[str, Any]):
        super().__init__(message)
        self.capture = dict(capture)


def _equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    try:
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if left_array.dtype != right_array.dtype or left_array.shape != right_array.shape or left_array.size == 0:
            return False
        if not (np.all(np.isfinite(left_array)) and np.all(np.isfinite(right_array))):
            return False
        return left_array.tobytes(order="C") == right_array.tobytes(order="C")
    except (TypeError, ValueError):
        return left == right


def _record_value(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def evaluate_stage_a_gate(records: Mapping[str, Mapping[str, Any]]) -> GateResult:
    """Apply the exact Stage A gate and return all diagnostics, never partial pass."""

    failures: list[str] = []
    checked: list[str] = []
    missing = [name for name in "ABCD" if name not in records]
    if missing:
        return GateResult(False, tuple(f"missing Stage A record {name}" for name in missing), ())
    a, b, c, d = (records[name] for name in "ABCD")
    def exact_pair(left: Mapping[str, Any], right: Mapping[str, Any], label: str, fields: tuple[str, ...]) -> None:
        for field_name in fields:
            checked.append(f"{label}:{field_name}")
            left_value = _record_value(left, field_name)
            right_value = _record_value(right, field_name)
            if left_value is None or right_value is None or not _equal(left_value, right_value):
                failures.append(f"{label} {field_name} mismatch")

    exact_pair(a, b, "A/B", ("carrier_fp32", "network_condition_bf16", "predicted_latent", "decoded_float"))
    exact_pair(a, d, "A/D", ("predicted_latent", "decoded_float"))
    a_noise = _record_value(a, "initial_noise_hash", "predicted_noise_hash")
    d_noise = _record_value(d, "initial_noise_hash", "predicted_noise_hash")
    checked.append("A/D:initial_noise_hash")
    if a_noise is None or d_noise is None or a_noise != d_noise:
        failures.append("A/D initial_noise_hash mismatch")
    for field_name in ("carrier_fp32", "initial_state", "initial_condition_mask", "network_condition_bf16", "final_latent_full", "predicted_latent", "decoded_float", "decoded_final"):
        array = _record_array(c, field_name)
        if array is None or array.size == 0 or not np.all(np.isfinite(array)):
            failures.append(f"C {field_name} is empty or non-finite")
    if not _initial_layout_matches(c):
        failures.append("C initial sampler state/mask layout does not map to carrier")
    mask = _record_value(c, "condition_mask", "mask")
    delta = _record_value(c, "actual_delta_fp32", "realized_delta_fp32")
    delta_bf16 = _record_value(c, "actual_delta_bf16", "realized_delta_bf16")
    if mask is None or delta is None:
        failures.append("C missing actual delta or condition mask")
    else:
        mask_raw = np.asarray(mask)
        if mask_raw.size == 0 or not np.all(np.isfinite(mask_raw)):
            failures.append("C condition mask is empty or non-finite")
        mask_array = mask_raw.astype(bool)
        delta_array = np.asarray(delta)
        if delta_array.shape != mask_array.shape or delta_array.size == 0 or not np.all(np.isfinite(delta_array)):
            failures.append("C actual delta/mask shape mismatch")
        else:
            if not np.any(delta_array[mask_array] != 0):
                failures.append("C actual delta is zero inside mask")
            if np.any(delta_array[~mask_array] != 0):
                failures.append("C actual delta is nonzero outside mask")
    if delta_bf16 is None or np.asarray(delta_bf16).shape != np.asarray(mask).shape or np.asarray(delta_bf16).size == 0 or not np.all(np.isfinite(np.asarray(delta_bf16))) or not np.any(np.asarray(delta_bf16) != 0):
        failures.append("C BF16 perturbation was erased")
    a_noise = _record_value(a, "initial_noise_hash", "predicted_noise_hash")
    c_noise = _record_value(c, "initial_noise_hash", "predicted_noise_hash")
    if a_noise is None or c_noise is None or a_noise != c_noise:
        failures.append("C predicted noise hash differs from A")
    observations = _record_value(c, "condition_step_hashes", "denoise_condition_hashes")
    if observations is None or len(list(observations)) < 2:
        failures.append("C is missing first/final condition observations")
    else:
        observations = list(observations)
        first_hash = _record_value(c, "first_denoise_hash", "first_condition_hash")
        final_hash = _record_value(c, "final_denoise_hash", "final_condition_hash")
        expected_hash = _record_value(c, "expected_network_condition_hash", "network_condition_hash")
        dtype_name = str(_record_value(c, "network_condition_dtype") or "")
        observed_network = _record_value(c, "network_condition_bf16")
        if first_hash is None or final_hash is None or expected_hash is None:
            failures.append("C is missing explicit first/final condition hashes")
        elif first_hash != observations[0] or final_hash != observations[-1]:
            failures.append("C first/final condition hashes do not match all-step evidence")
        if expected_hash is not None and first_hash != expected_hash:
            failures.append("C network condition hash differs from expected perturbed condition")
        if expected_hash is not None and any(item != expected_hash for item in observations):
            failures.append("C denoise condition hash differs from expected perturbed condition")
        if not _network_condition_matches(c):
            failures.append("C observed BF16 network condition hash is not reproducible")
        if "bfloat16" not in dtype_name.lower() and "bf16" not in dtype_name.lower():
            failures.append("C network condition was not observed in BF16")
        if observed_network is None or np.asarray(observed_network).size == 0 or not np.all(np.isfinite(np.asarray(observed_network))):
            failures.append("C network condition evidence is empty or non-finite")
        lifecycle = _record_value(c, "text_kv_lifecycle")
        if lifecycle is None or len(list(lifecycle)) < 2 or not any(isinstance(item, Mapping) and any(item.get(key) not in (None, [], ()) for key in ("before", "after", "initialized")) for item in lifecycle):
            failures.append("C text-KV request-local lifecycle was not proven")
        cfg_semantics = _record_value(c, "cfg_branch_semantics")
        branch_observations = _record_value(c, "cfg_branch_observations")
        if not _cfg_observations_match(c):
            failures.append("C CFG branch semantics were not proven at the denoise seam")
        if isinstance(cfg_semantics, Mapping) and float(cfg_semantics.get("guidance", 1.0)) != 1.0 and int(cfg_semantics.get("unconditional_calls", 0)) < 1:
            failures.append("C unconditional CFG branch was not observed")
        step_evidence = _record_value(c, "denoise_step_evidence")
        expected_steps = int(c.get("expected_denoise_steps", NUM_STEPS))
        if expected_steps != NUM_STEPS:
            failures.append(f"C denoise step count is not configured NUM_STEPS={NUM_STEPS}")
        expected_multiplicity = int(cfg_semantics.get("expected_multiplicity", 1)) if isinstance(cfg_semantics, Mapping) else 1
        if not isinstance(step_evidence, (list, tuple)) or len(step_evidence) != expected_steps * expected_multiplicity:
            failures.append("C denoise step index/timestep coverage is missing")
        else:
            try:
                valid_timesteps = all(np.isfinite(float(item.get("timestep"))) for item in step_evidence if isinstance(item, Mapping)) and all(isinstance(item, Mapping) for item in step_evidence)
            except (TypeError, ValueError):
                valid_timesteps = False
            if not valid_timesteps:
                failures.append("C denoise step evidence has a missing/non-finite timestep")
            else:
                try:
                    contiguous = [int(item.get("step_index", -1)) for item in step_evidence] == list(range(len(step_evidence)))
                except (TypeError, ValueError):
                    contiguous = False
                if not contiguous:
                    failures.append("C denoise step evidence indexes are not contiguous")
        if len(set(observations)) != 1:
            failures.append("C condition changed across denoise steps")
    for field_name, aliases, metadata_name, image in (("predicted_latent_sliced", ("predicted_latent",), "latent_slicing", False), ("decoded_final_frame", ("decoded_final",), "image_slicing", True)):
        value = _record_value(c, *aliases)
        value_array = _record_array(c, *aliases)
        valid = _valid_slice_metadata(c, value_array, c.get(metadata_name), image=image)
        if not valid:
            failures.append(f"C {field_name} evidence is missing")
    if not _predicted_latent_matches(c):
        failures.append("C predicted latent does not match final latent slicing")
    if not _decoded_final_matches(c):
        failures.append("C decoded final does not match decoded frame slicing")
    return GateResult(not failures, tuple(failures), tuple(checked))


stage_a_gate = evaluate_stage_a_gate


def run_stage_a_and_scan(
    execute_call: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    plan: Iterable[Mapping[str, Any]] | None = None,
    stage_a_only: bool = False,
    after_stage_a: Callable[[Mapping[str, Mapping[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    """Execute Stage A, stop before scan on any gate failure, then scan."""

    calls = list(plan or build_call_plan())
    validate_call_plan(calls)
    records: dict[str, dict[str, Any]] = {}
    for call in calls[:4]:
        try:
            records[call["sample_id"]] = dict(execute_call(call))
        except Exception as error:
            records[call["sample_id"]] = {"status": "fail", "error": {"type": type(error).__name__, "message": str(error)}}
        if call["sample_id"] == "A" and after_stage_a is not None:
            after_stage_a(records)
    gate = evaluate_stage_a_gate(records)
    failed_stage_a = [name for name, record in records.items() if record.get("status") == "fail"]
    if failed_stage_a:
        gate = GateResult(False, tuple(list(gate.failures) + [f"Stage A call {name} failed" for name in failed_stage_a]), gate.checked)
    result: dict[str, Any] = {"stage_a": records, "gate": gate, "scan_calls": []}
    if stage_a_only or not gate.passed:
        result["status"] = "STAGE_A_ONLY" if stage_a_only and gate.passed else "FAIL"
        result["skipped_scan_count"] = EXPECTED_SCAN_COUNT
        return result
    for call in calls[4:]:
        try:
            scan_record = dict(execute_call(call))
        except Exception as error:
            scan_record = {"status": "fail", "error": {"type": type(error).__name__, "message": str(error)}}
        if scan_record.get("status", "success") == "success":
            evidence = validate_sample_evidence(scan_record, stage_a=False)
            if not evidence.passed:
                scan_record = dict(scan_record)
                scan_record["status"] = "fail"
                scan_record["error"] = {"type": "EvidenceValidationError", "message": "; ".join(evidence.failures)}
        result["scan_calls"].append(scan_record)
    result["status"] = "COMPLETE" if all(record.get("status", "success") == "success" for record in result["scan_calls"]) else "FAIL"
    return result


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float") and hasattr(value, "cpu"):
        value = value.float().cpu().numpy()
    return np.asarray(value)


def _numpy_capture(value: Any) -> Any:
    """Detach official tensor-like values before persistence or JSON routing."""
    if isinstance(value, Mapping):
        return {key: _numpy_capture(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_numpy_capture(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_numpy_capture(item) for item in value)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except (TypeError, RuntimeError, ValueError):
            converted = value.float() if hasattr(value, "float") else None
            if converted is None:
                raise TypeError("tensor-like value cannot be converted to NumPy")
            if hasattr(converted, "cpu"):
                converted = converted.cpu()
            if not hasattr(converted, "numpy"):
                raise TypeError("float tensor-like value cannot be converted to NumPy")
            value = converted.numpy()
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    return value


def _normalize_official_mask(value: Any, carrier_shape: tuple[int, ...]) -> np.ndarray:
    """Normalize OmniMoTModel's packed mask while preserving exact layout evidence."""
    raw = np.asarray(_array(value), dtype=bool)
    expected_size = int(np.prod(carrier_shape, dtype=np.int64))
    if raw.ndim == 1:
        if raw.size != expected_size:
            raise ValueError(f"flattened official condition mask has {raw.size} values; expected {expected_size}")
        return np.ascontiguousarray(raw.reshape(carrier_shape), dtype=bool)
    if raw.size == expected_size and raw.shape != carrier_shape:
        return np.ascontiguousarray(raw.reshape(carrier_shape), dtype=bool)
    return raw


def _normalize_condition_reference(value: Any, carrier_shape: tuple[int, ...]) -> np.ndarray:
    """Preserve the official float condition reference; it is not a mask."""
    raw = np.asarray(_numpy_capture(value))
    if raw.dtype.kind not in "fiu" or raw.dtype == np.dtype(bool) or raw.size == 0 or not np.all(np.isfinite(raw)):
        raise ValueError("official condition reference must be a finite, non-empty numeric array")
    expected_size = int(np.prod(carrier_shape, dtype=np.int64))
    if raw.ndim == 1 and raw.size == expected_size:
        raw = raw.reshape(carrier_shape)
    elif raw.size == expected_size and raw.shape != carrier_shape:
        raw = raw.reshape(carrier_shape)
    return np.ascontiguousarray(raw.copy())


def _record_array(record: Mapping[str, Any], *names: str) -> np.ndarray | None:
    value = _record_value(record, *names)
    if value is None:
        return None
    try:
        return np.asarray(_numpy_capture(value))
    except (TypeError, ValueError):
        return None


def _valid_slice_metadata(record: Mapping[str, Any], value: np.ndarray | None, metadata: Any, *, image: bool) -> bool:
    if value is None or value.size == 0 or not np.all(np.isfinite(value)) or not isinstance(metadata, Mapping) or not metadata:
        return False
    if image:
        frame_index = metadata.get("frame_index")
        source = _record_array(record, "decoded_float")
        return isinstance(frame_index, (int, np.integer)) and source is not None and source.ndim >= 2 and 0 <= int(frame_index) < int(source.shape[1])
    indexes = metadata.get("predicted_indexes") or metadata.get("indexes")
    if not isinstance(indexes, (list, tuple, np.ndarray)) or not len(indexes):
        return False
    try:
        axis = int(metadata.get("axis", 0))
        source_shape = tuple(int(dim) for dim in metadata.get("source_shape", value.shape))
        return 0 <= axis < len(source_shape) and all(0 <= int(index) < source_shape[axis] for index in indexes)
    except (TypeError, ValueError):
        return False


def _initial_layout_matches(record: Mapping[str, Any]) -> bool:
    carrier = _record_array(record, "carrier_fp32")
    condition_mask = _record_array(record, "condition_mask")
    initial_state = _record_array(record, "initial_state")
    initial_mask = _record_array(record, "initial_condition_mask")
    if any(array is None or array.size == 0 or not np.all(np.isfinite(array)) for array in (carrier, condition_mask, initial_state, initial_mask)):
        return False
    if condition_mask.shape != carrier.shape or initial_state.shape != initial_mask.shape:
        return False
    if initial_state.size != carrier.size or initial_mask.size != carrier.size:
        return False
    return np.array_equal(initial_mask.reshape(-1), condition_mask.reshape(-1))


def _network_condition_matches(record: Mapping[str, Any]) -> bool:
    network = _record_array(record, "network_condition_bf16")
    expected = record.get("expected_network_condition_hash")
    hashes = record.get("condition_step_hashes", record.get("denoise_condition_hashes"))
    if network is None or network.size == 0 or not np.all(np.isfinite(network)) or not expected or not isinstance(hashes, (list, tuple)) or not hashes:
        return False
    observed = sha256_array(network)
    return observed == expected and all(item == observed for item in hashes)


def _predicted_latent_matches(record: Mapping[str, Any]) -> bool:
    full = _record_array(record, "final_latent_full")
    predicted = _record_array(record, "predicted_latent")
    condition_mask = _record_array(record, "condition_mask", "mask")
    metadata = record.get("latent_slicing")
    if full is None or predicted is None or condition_mask is None or not isinstance(metadata, Mapping):
        return False
    try:
        if condition_mask.shape != full.shape or full.ndim < 3:
            return False
        axis = full.ndim - 3
        if int(metadata.get("axis", -1)) != axis:
            return False
        if tuple(int(dim) for dim in metadata.get("source_shape", full.shape)) != full.shape:
            return False
        mask = np.asarray(condition_mask, dtype=bool)
        reduce_axes = tuple(index for index in range(mask.ndim) if index != axis)
        frame_any = np.any(mask, axis=reduce_axes)
        frame_all = np.all(mask, axis=reduce_axes)
        if not np.array_equal(frame_any, frame_all):
            return False
        expected_indexes = [int(index) for index in np.flatnonzero(~frame_any)]
        supplied = metadata.get("predicted_indexes")
        if not isinstance(supplied, (list, tuple, np.ndarray)) or [int(index) for index in supplied] != expected_indexes:
            return False
        expected = np.take(full, expected_indexes, axis=axis)
    except (TypeError, ValueError, IndexError, OverflowError):
        return False
    if "selected_shape" in metadata and tuple(int(dim) for dim in metadata["selected_shape"]) != predicted.shape:
        return False
    return _equal(expected, predicted)


def _decoded_final_matches(record: Mapping[str, Any]) -> bool:
    full = _record_array(record, "decoded_float")
    final = _record_array(record, "decoded_final")
    metadata = record.get("image_slicing")
    if full is None or final is None or not isinstance(metadata, Mapping):
        return False
    try:
        frame_index = int(metadata["frame_index"])
        axis = int(metadata.get("axis", 1))
        if full.ndim < 2 or axis != 1 or frame_index != int(full.shape[1]) - 1:
            return False
        if tuple(int(dim) for dim in metadata.get("source_shape", full.shape)) != full.shape:
            return False
        expected = np.take(full, frame_index, axis=axis)
    except (TypeError, ValueError, IndexError, KeyError):
        return False
    if "selected_shape" in metadata and tuple(int(dim) for dim in metadata["selected_shape"]) != final.shape:
        return False
    return _equal(expected, final)


def _cfg_observations_match(record: Mapping[str, Any]) -> bool:
    semantics = record.get("cfg_branch_semantics")
    observations = record.get("cfg_branch_observations")
    if not isinstance(semantics, Mapping) or not isinstance(observations, (list, tuple)):
        return False
    evidence = record.get("denoise_step_evidence")
    try:
        expected_steps = int(record.get("expected_denoise_steps", NUM_STEPS))
    except (TypeError, ValueError):
        return False
    if expected_steps != NUM_STEPS or not isinstance(evidence, (list, tuple)) or len(evidence) != NUM_STEPS:
        return False
    conditional = [item for item in observations if isinstance(item, Mapping) and item.get("branch") == "conditional" and item.get("used") is True]
    unconditional = [item for item in observations if isinstance(item, Mapping) and item.get("branch") == "unconditional" and item.get("used") is True]
    try:
        calls = int(record["cfg_branch_calls"])
        expected_conditional = int(semantics["conditional_calls"])
        expected_unconditional = int(semantics.get("unconditional_calls", 0))
        guidance = float(semantics["guidance"])
    except (KeyError, TypeError, ValueError):
        return False
    if calls != len(observations) or expected_conditional != len(conditional) or expected_unconditional != len(unconditional):
        return False
    if guidance == 1.0:
        if calls != NUM_STEPS or expected_conditional != NUM_STEPS or expected_unconditional != 0 or len(observations) != NUM_STEPS or unconditional:
            return False
        for observation, denoise in zip(observations, evidence):
            if not isinstance(denoise, Mapping) or observation.get("step_index") != denoise.get("step_index"):
                return False
            if "timestep" in observation and observation.get("timestep") != denoise.get("timestep"):
                return False
        return True
    return len(conditional) >= 1 and len(unconditional) >= 1


def validate_sample_evidence(record: Mapping[str, Any], *, stage_a: bool = False) -> GateResult:
    """Fail closed on incomplete per-call evidence before publishing success."""
    failures: list[str] = []
    required_arrays = ("carrier_fp32", "condition_mask", "initial_state", "initial_condition_mask", "final_latent_full", "predicted_latent", "decoded_float")
    for name in required_arrays:
        array = _record_array(record, name)
        if array is None or array.size == 0 or not np.all(np.isfinite(array)):
            failures.append(f"missing/invalid {name}")
    if not _initial_layout_matches(record):
        failures.append("invalid initial sampler state/mask layout")
    final_array = _record_array(record, "decoded_final")
    if final_array is None or final_array.size == 0 or not np.all(np.isfinite(final_array)):
        failures.append("missing decoded final frame")
    if record.get("initial_noise_hash", record.get("predicted_noise_hash")) is None:
        failures.append("missing initial noise hash")
    hashes = record.get("condition_step_hashes", record.get("denoise_condition_hashes"))
    evidence = record.get("denoise_step_evidence")
    if not isinstance(hashes, (list, tuple)) or not hashes or not isinstance(evidence, (list, tuple)) or len(hashes) != len(evidence):
        failures.append("missing denoise evidence")
    if not isinstance(record.get("latent_slicing"), Mapping) or not isinstance(record.get("image_slicing"), Mapping):
        failures.append("missing latent/image slicing metadata")
    if not _valid_slice_metadata(record, _record_array(record, "predicted_latent"), record.get("latent_slicing"), image=False):
        failures.append("invalid latent slicing metadata")
    if not _valid_slice_metadata(record, _record_array(record, "decoded_final"), record.get("image_slicing"), image=True):
        failures.append("invalid image slicing metadata")
    if record.get("prepare_error") or record.get("latent_slicing_error") or record.get("condition_error"):
        failures.append("runtime evidence contains a capture error")
    if record.get("missing_network_condition_tokens") or record.get("cache_lifecycle_valid") is False:
        failures.append("runtime cache/condition evidence failed")
    expected_hash = record.get("expected_network_condition_hash")
    dtype_name = str(record.get("network_condition_dtype", ""))
    network_array = _record_array(record, "network_condition_bf16")
    if not expected_hash or network_array is None or network_array.size == 0 or not np.all(np.isfinite(network_array)) or ("bfloat16" not in dtype_name.lower() and "bf16" not in dtype_name.lower()):
        failures.append("missing expected BF16 network evidence")
    if not isinstance(record.get("text_kv_lifecycle"), (list, tuple)) or not any(
        isinstance(item, Mapping) and any(item.get(key) not in (None, [], ()) for key in ("before", "after", "initialized"))
        for item in record.get("text_kv_lifecycle", ())
    ):
        failures.append("missing request-local KV lifecycle evidence")
    semantics = record.get("cfg_branch_semantics")
    if not _cfg_observations_match(record):
        failures.append("missing CFG branch observations")
    if not _network_condition_matches(record):
        failures.append("observed BF16 network condition hash is not reproducible")
    if not _predicted_latent_matches(record):
        failures.append("predicted latent does not match final latent slicing")
    if not _decoded_final_matches(record):
        failures.append("decoded final does not match decoded frame slicing")
    if hashes and evidence and isinstance(evidence, (list, tuple)):
        try:
            expected_steps = int(record.get("expected_denoise_steps", NUM_STEPS))
            if expected_steps != NUM_STEPS:
                failures.append(f"denoise coverage is not configured NUM_STEPS={NUM_STEPS}")
            expected_calls = expected_steps * int(semantics.get("expected_multiplicity", 1)) if isinstance(semantics, Mapping) else expected_steps
            if len(evidence) != expected_calls:
                failures.append(f"denoise coverage {len(evidence)} != expected {expected_calls}")
            if [int(item.get("step_index", -1)) for item in evidence] != list(range(len(evidence))):
                failures.append("denoise indexes are not contiguous")
            if any(not np.isfinite(float(item.get("timestep"))) for item in evidence):
                failures.append("denoise timestep evidence is invalid")
            if any(item != expected_hash for item in hashes):
                failures.append("denoise condition hash differs from expected network condition")
            if record.get("first_denoise_hash") != hashes[0] or record.get("final_denoise_hash") != hashes[-1]:
                failures.append("first/final denoise hashes do not match indexed evidence")
        except (TypeError, ValueError, AttributeError):
            failures.append("denoise index/timestep evidence is invalid")
    return GateResult(not failures, tuple(failures), ())


def _clone_runtime(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return [_clone_runtime(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_runtime(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_runtime(item) for key, item in value.items()}
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    return copy.deepcopy(value)


def _assign_add(value: Any, delta: np.ndarray) -> Any:
    if hasattr(value, "detach") and hasattr(value, "clone"):
        try:
            import torch  # noqa: PLC0415
            tensor = value.detach().clone()
            tensor_delta = torch.as_tensor(delta, device=tensor.device, dtype=tensor.dtype)
            tensor.add_(tensor_delta.reshape(tensor.shape))
            return tensor
        except (ImportError, RuntimeError, ValueError):
            pass
    array = np.asarray(value).copy()
    array += np.asarray(delta, dtype=array.dtype).reshape(array.shape)
    return array


def _bf16_delta(baseline: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    base = _bf16_values(np.asarray(baseline, dtype=np.float32))
    pert = _bf16_values(np.asarray(perturbed, dtype=np.float32))
    return (pert - base).astype(np.float32)


def _bf16_values(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32).copy()
    bits = source.view(np.uint32)
    rounding = ((bits >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((bits + rounding) & np.uint32(0xFFFF0000)).view(np.float32)


def _cache_has_values(value: Any) -> bool:
    """Return whether an official cache contains tensors, not just empty slots."""
    if value is None:
        return False
    if isinstance(value, Mapping):
        return any(_cache_has_values(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(item is not None and _cache_has_values(item) for item in value)
    if hasattr(value, "numel"):
        try:
            return int(value.numel()) > 0
        except (TypeError, ValueError, RuntimeError):
            return True
    return True


def _select_condition_values(tokens: Any, mask: Any) -> np.ndarray:
    """Select packed condition tokens while retaining the official layout."""
    token_array = _array(tokens)
    mask_array = np.asarray(_array(mask), dtype=bool)
    try:
        return token_array[np.broadcast_to(mask_array, token_array.shape)]
    except ValueError:
        flat_mask = mask_array.reshape(-1)
        if token_array.ndim >= 2 and flat_mask.size and token_array.shape[0] % flat_mask.size == 0:
            tokens_by_frame = token_array.reshape(flat_mask.size, -1, *token_array.shape[1:])
            return tokens_by_frame[flat_mask].reshape(-1)
        if token_array.size % flat_mask.size == 0:
            return token_array.reshape(flat_mask.size, -1)[flat_mask].reshape(-1)
        raise ValueError(f"packed condition mask shape {mask_array.shape} cannot select tokens {token_array.shape}")


def _extract_carrier(data: Any) -> np.ndarray:
    if hasattr(data, "x0_tokens_vision"):
        items = getattr(data, "x0_tokens_vision")
        if not items:
            raise ValueError("GenerationDataClean has no vision carrier")
        return _array(items[0]).astype(np.float32, copy=True)
    if isinstance(data, dict) and "x0_tokens_vision" in data:
        items = data["x0_tokens_vision"]
        return _array(items[0]).astype(np.float32, copy=True)
    raise ValueError("official condition result has no x0_tokens_vision carrier")


def _inject_carrier(data: Any, delta: np.ndarray) -> Any:
    result = _clone_runtime(data)
    if hasattr(result, "x0_tokens_vision"):
        items = list(result.x0_tokens_vision)
        items[0] = _assign_add(items[0], delta)
        result.x0_tokens_vision = items
        return result
    if isinstance(result, dict) and "x0_tokens_vision" in result:
        result["x0_tokens_vision"] = list(result["x0_tokens_vision"])
        result["x0_tokens_vision"][0] = _assign_add(result["x0_tokens_vision"][0], delta)
        return result
    raise ValueError("official condition result has no injectable x0_tokens_vision carrier")


def _capture_network_condition_before_denoise(capture: dict[str, Any], args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
    """Capture the packed condition exactly as supplied to the network.

    This must run before the official denoise call.  Some official sampler
    paths reuse the packed object and may mutate its token storage after the
    forward; observing it only after return would falsely report a mid-step
    condition overwrite that was not present at the network boundary.
    """
    step_identity = len(capture["denoise_step_evidence"])
    capture["cfg_branch_calls"] += 1
    capture["cfg_branch_observations"].append({"branch": "conditional", "memory_identity": None, "used": True, "step_index": step_identity})
    tokens = None
    observed_dtype = None
    timestep_value = None
    for value in tuple(args) + tuple(kwargs.values()):
        vision_data = getattr(value, "vision", None)
        candidate = getattr(vision_data, "tokens", None)
        if candidate is not None:
            timestep_value = getattr(vision_data, "timesteps", None)
            try:
                timestep_value = float(_array(timestep_value).reshape(-1)[0])
            except (AttributeError, IndexError, TypeError, ValueError):
                timestep_value = None
            capture["denoise_step_evidence"].append({"step_index": step_identity, "timestep": timestep_value})
            tokens = candidate[0] if isinstance(candidate, (list, tuple)) else candidate
            observed_dtype = str(getattr(tokens, "dtype", ""))
            condition_tokens_mask = getattr(vision_data, "condition_mask", None)
            if condition_tokens_mask is not None:
                mask_value = condition_tokens_mask[0] if isinstance(condition_tokens_mask, (list, tuple)) else condition_tokens_mask
                try:
                    tokens = _select_condition_values(tokens, mask_value)
                except ValueError as error:
                    capture["condition_token_selection_error"] = str(error)
            break
    if tokens is None:
        capture["missing_network_condition_tokens"] = True
    else:
        # Copy now, before the official call can mutate a request-local pack.
        capture["denoise_step_hashes"].append(sha256_array(_array(tokens).astype(np.float32)))
        capture["network_condition_bf16"] = _array(tokens).astype(np.float32, copy=True)
        capture["network_condition_dtype"] = observed_dtype or str(getattr(tokens, "dtype", ""))


def _reimpose_packed_condition(capture: dict[str, Any], tokens: Any, mask: Any) -> None:
    """Restore only condition positions from the immutable packed reference.

    UniPC evolves the whole packed latent state between solver calls.  Its
    condition slots can therefore acquire BF16 roundoff during corrector
    steps even when their velocity is zero.  Re-impose the runtime condition
    at the network boundary while leaving every predicted position untouched.
    """
    reference = capture.get("packed_condition_reference")
    if reference is None:
        return
    token_array = _array(tokens)
    reference_array = _bf16_values(_array(reference))
    if reference_array.shape != token_array.shape:
        if reference_array.ndim == token_array.ndim + 1 and reference_array.shape[0] == 1:
            reference_array = reference_array[0]
        elif reference_array.size == token_array.size:
            reference_array = reference_array.reshape(token_array.shape)
        else:
            raise ValueError(f"packed condition reference shape {reference_array.shape} cannot restore tokens {token_array.shape}")
    mask_array = np.asarray(_array(mask), dtype=bool)
    try:
        full_mask = np.broadcast_to(mask_array, token_array.shape)
    except ValueError as error:
        raise ValueError(f"packed condition mask shape {mask_array.shape} cannot restore tokens {token_array.shape}") from error
    if not np.any(full_mask):
        raise ValueError("packed condition mask is empty at denoise boundary")
    if hasattr(tokens, "detach") and hasattr(tokens, "copy_"):
        try:
            import torch  # noqa: PLC0415
            expected = torch.as_tensor(reference_array, device=tokens.device, dtype=tokens.dtype)
            selector = torch.as_tensor(full_mask, device=tokens.device, dtype=torch.bool)
            tokens[selector] = expected[selector]
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"failed to restore torch condition tokens: {error}") from error
    else:
        token_array[full_mask] = reference_array[full_mask]


class OfficialPostVaeRuntimeAdapter:
    """Small seam around the official pipeline; heavy imports occur only on use."""

    def __init__(self, get_data_and_condition: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self.conditioning = CachedConditioningData(get_data_and_condition, *args, **kwargs)
        self.model: Any | None = None
        self.runtime_setup: dict[str, Any] = {}

    @classmethod
    def from_pipeline(cls, pipeline: Any, *args: Any, **kwargs: Any) -> "OfficialPostVaeRuntimeAdapter":
        model = getattr(pipeline, "model", pipeline)
        callback = getattr(model, "get_data_and_condition", None)
        if callback is None or not callable(callback):
            raise AttributeError("official pipeline.model must expose get_data_and_condition")
        result = cls(callback, *args, **kwargs)
        result.pipeline = pipeline
        result.model = model
        result._baseline_data = None
        result._last_capture: dict[str, Any] = {}
        result.sample_settings: dict[str, Any] = {}
        result.sample_args: Any | None = None
        result._condition_mask: np.ndarray | None = None
        result._condition_indexes: list[int] = []
        result._baseline_carrier: np.ndarray | None = None
        return result

    @classmethod
    def load_official(cls, factory: Callable[..., Any], *, checkpoint_path: str | Path, output_dir: str | Path, **kwargs: Any) -> "OfficialPostVaeRuntimeAdapter":
        setup_kwargs = build_setup_overrides_kwargs(checkpoint_path=checkpoint_path, output_dir=output_dir, **kwargs)
        setup_kwargs.update({"sampler": SAMPLER, "num_steps": NUM_STEPS, "precision": PRECISION, "batch_size": 1, "seed": MODEL_SEED, "action_chunk_index": 0, "gpu_index": 0})
        if setup_kwargs.get("diffusion_cache", False):
            raise ValueError("official post-VAE scan requires diffusion_cache=False")
        pipeline = factory(**setup_kwargs)
        adapter = cls.from_pipeline(pipeline)
        adapter.runtime_setup = dict(setup_kwargs)
        installed = bool(getattr(pipeline, "_diffusion_cache_installed", getattr(pipeline, "diffusion_cache_installed", False)))
        adapter.runtime_setup["diffusion_cache_requested"] = False
        adapter.runtime_setup["diffusion_cache_installed"] = installed
        validate_cache_off(False, installed)
        if hasattr(pipeline, "eval"):
            pipeline.eval()
        return adapter

    def condition_for_call(self, call_id: str, *, delta: Any | None = None, inject: Callable[[Any, Any], Any] | None = None) -> Any:
        if call_id == "A" and self.conditioning.captured:
            raise RuntimeError("Stage A call A is the sole encoder call")
        return self.conditioning.get(delta=delta, inject=inject)

    def prime_baseline(
        self,
        data_batch: Any,
        *,
        condition_indexes: Iterable[int],
        expected_carrier: Any | None = None,
        expected_mask: Any | None = None,
        expected_direction_bank: Any | None = None,
    ) -> Any:
        """Restore the complete cached GenerationDataClean before resumed B/C/D."""
        if self.model is None or not hasattr(self.model, "get_data_and_condition"):
            raise RuntimeError("cannot restore baseline without model-owned get_data_and_condition")
        indexes = [int(index) for index in condition_indexes]
        operation = lambda: self.model.get_data_and_condition(
            data_batch,
            vision_condition_indexes=[indexes],
            retain_raw_state_vision=True,
        )
        result = self.run_with_boundaries(operation)
        fresh_carrier = _extract_carrier(result)
        if expected_carrier is not None and not _equal(fresh_carrier, _array(expected_carrier).astype(np.float32)):
            raise ValueError("saved z0 does not match deterministic resumed encode")
        if expected_mask is not None:
            fresh_mask = np.asarray(_array(expected_mask), dtype=bool)
            if fresh_mask.size == 0 or fresh_mask.shape != fresh_carrier.shape:
                raise ValueError("saved mask does not match deterministic resumed carrier")
            fresh_full_mask = build_broadcast_condition_mask(indexes, fresh_mask, fresh_carrier.shape)
            if not _equal(fresh_full_mask, fresh_mask):
                raise ValueError("saved mask does not match deterministic resumed preparation")
            self._condition_mask = fresh_mask.copy()
        if expected_direction_bank is not None:
            expected = generate_directions_for_mask(self._condition_mask, count=np.asarray(expected_direction_bank).shape[0])
            if not _equal(expected, np.asarray(expected_direction_bank)):
                raise ValueError("saved direction bank does not match deterministic resumed mask")
        self._baseline_data = _clone_runtime(result)
        self.conditioning._baseline = _clone_runtime(result)
        self.conditioning._captured = True
        self._baseline_carrier = fresh_carrier.copy()
        self._condition_indexes = indexes
        return _clone_runtime(result)

    @property
    def z0(self) -> Any:
        """Return a fresh immutable-carrier clone for a runtime call."""
        baseline = self.conditioning.baseline
        return copy.deepcopy(baseline)

    def run_with_boundaries(self, operation: Callable[[], Any], hooks: Iterable[HookBoundary] = ()) -> Any:
        """Run one call under inference mode and restore every hook in finally."""
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            return operation()
        self._reset_request_local_state()
        with torch.inference_mode():
            from contextlib import ExitStack
            with ExitStack() as stack:
                for hook in hooks:
                    stack.enter_context(hook)
                return operation()

    def _reset_request_local_state(self) -> None:
        model = self.model
        if model is None:
            return
        for name in ("_sampler_state", "_text_kv_cache", "text_kv_cache"):
            value = getattr(model, name, None)
            if hasattr(value, "clear"):
                value.clear()

    def _assert_cache_lifecycle(self) -> None:
        """Assert actual Wan tokenizer cache policy after the request/decode."""
        candidates = [self.model]
        for tokenizer in (getattr(self.model, "tokenizer_vision_gen", None), getattr(self.model, "tokenizer_vision", None)):
            candidates.append(tokenizer)
            candidates.append(getattr(tokenizer, "model", None) if tokenizer is not None else None)
            try:
                candidates.append(tokenizer.active_decoder if tokenizer is not None else None)
            except (AttributeError, RuntimeError):
                pass
        for candidate in candidates:
            if candidate is None:
                continue
            if bool(getattr(candidate, "_keep_decoder_cache", False)):
                raise ValueError("decoder cache keep policy is enabled")
            decoder_cache = getattr(candidate, "_dec_cache", None)
            if _cache_has_values(decoder_cache):
                raise ValueError("decoder cache remains populated after decode")
            encoder_cache = getattr(candidate, "_enc_cache", None)
            if _cache_has_values(encoder_cache):
                raise ValueError("encoder cache escaped its request-local encode")

    def _replace_method(self, name: str, wrapper: Callable[..., Any]) -> tuple[Any, bool]:
        if self.model is None or not hasattr(self.model, name):
            return None, False
        original = getattr(self.model, name)
        setattr(self.model, name, wrapper)
        return original, True

    def execute_call(self, spec: Mapping[str, Any], *, data_batch: Any = None, sample_dir: str | Path | None = None, direction: Any | None = None) -> dict[str, Any]:
        """Execute one official model call while wrapping true model-owned boundaries."""
        if self.model is None:
            raise RuntimeError("adapter has no official model")
        capture: dict[str, Any] = {"denoise_step_hashes": [], "denoise_step_evidence": [], "text_kv_lifecycle": [], "cfg_branch_calls": 0, "cfg_branch_observations": []}
        originals: dict[str, Any] = {}
        spec_id = str(spec.get("sample_id", ""))
        baseline = self._baseline_data
        requested_delta: np.ndarray | None = None

        def _checkpoint_capture() -> None:
            if sample_dir is None:
                return
            destination = Path(sample_dir)
            destination.mkdir(parents=True, exist_ok=True)
            for name in ("carrier_fp32", "condition_mask", "initial_state", "initial_condition_mask", "packed_condition_reference"):
                value = capture.get(name)
                if value is None:
                    continue
                array = _record_array({name: value}, name)
                if array is not None and array.size:
                    np.save(destination / f"{name}.npy", array, allow_pickle=False)

        def get_data_wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal baseline, requested_delta
            if spec_id == "A":
                result = originals["get_data_and_condition"](*args, **kwargs)
                baseline = _clone_runtime(result)
                self._baseline_data = _clone_runtime(result)
                capture["carrier_fp32"] = _extract_carrier(result)
                capture["realized_carrier_fp32"] = capture["carrier_fp32"].copy()
                self._baseline_carrier = capture["carrier_fp32"].copy()
                _checkpoint_capture()
                # Wan's encoder cache is request-local.  Check it at the true
                # encode return boundary, before any sampler work can hide a leak.
                self._assert_cache_lifecycle()
                return result
            if baseline is None:
                raise RuntimeError("Stage A carrier must be captured before a non-A call")
            result = _clone_runtime(baseline)
            if spec.get("injection") in {"direction", "direction_plus"}:
                if direction is None:
                    direction_array = np.zeros_like(capture.get("carrier_fp32", _extract_carrier(result)))
                    direction_array[np.asarray(capture["condition_mask"], dtype=bool)] = 1.0
                else:
                    direction_array = _array(direction).astype(np.float32, copy=False)
                base_array = capture.get("carrier_fp32", _extract_carrier(result))
                mask_array = capture.get("condition_mask", self._condition_mask)
                if mask_array is None:
                    raise RuntimeError("runtime packed condition mask has not yet been captured")
                capture["condition_mask"] = np.asarray(mask_array, dtype=bool)
                built = construct_delta(base_array, mask_array, direction_array, alpha=float(spec.get("alpha", 0.01)), sign=int(spec.get("sign", 1)))
                requested_delta = built.delta
                capture["target_delta_fp32"] = built.delta.copy()
                capture["target_alpha"] = float(spec.get("alpha", 0.0))
                capture["target_epsilon"] = float(built.target_rms)
                capture["target_rms"] = float(built.target_rms)
                capture["s_z"] = float(built.s_z)
                result = _inject_carrier(result, built.delta)
            elif spec.get("injection") == "explicit_zero":
                requested_delta = np.zeros_like(capture.get("carrier_fp32", _extract_carrier(result)), dtype=np.float32)
                result = _inject_carrier(result, requested_delta)
            capture.setdefault("carrier_fp32", _extract_carrier(baseline))
            if "condition_mask" not in capture and self._condition_mask is None:
                raise RuntimeError("runtime packed condition mask has not yet been captured")
            capture.setdefault("condition_mask", self._condition_mask)
            capture["realized_carrier_fp32"] = _extract_carrier(result)
            return result

        def prepare_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = originals["_prepare_inference_data"](*args, **kwargs)
            try:
                sequence_plans = result[0] if isinstance(result, tuple) else result.get("sequence_plans")
                packed_masks = result[6] if isinstance(result, tuple) else result.get("condition_mask")
                if sequence_plans is None or packed_masks is None:
                    raise ValueError("official preparation did not return sequence plans and packed vision masks")
                plan = sequence_plans[0]
                indexes = getattr(plan, "condition_frame_indexes_vision", None)
                if indexes is None:
                    raise ValueError("sequence plan did not expose vision condition indexes")
                packed_mask = packed_masks[0] if isinstance(packed_masks, (list, tuple)) else packed_masks
                carrier_shape = capture["carrier_fp32"].shape
                packed_mask_raw = np.asarray(_array(packed_mask), dtype=bool)
                capture["packed_condition_mask_raw"] = packed_mask_raw.copy()
                packed_mask = _normalize_official_mask(packed_mask, carrier_shape)
                packed_mask_array = np.asarray(packed_mask, dtype=bool)
                if len(carrier_shape) == 4 and packed_mask_array.ndim == 3 and packed_mask_array.shape[0] == carrier_shape[1]:
                    # Official packed masks are [T,1,1], while clean latent
                    # carriers are [C,T,H,W]. Expand only the packed mask's
                    # singleton spatial axes; never infer a source mask from
                    # GenerationDataClean.
                    packed_mask = np.broadcast_to(packed_mask_array, (carrier_shape[1], carrier_shape[2], carrier_shape[3])).copy()
                elif len(carrier_shape) == 4 and packed_mask_array.ndim == 4 and packed_mask_array.shape[0] == 1:
                    packed_mask = np.broadcast_to(packed_mask_array[0], (carrier_shape[1], carrier_shape[2], carrier_shape[3])).copy()
                capture["condition_mask"] = build_broadcast_condition_mask(indexes, packed_mask, capture["carrier_fp32"].shape)
                capture["packed_condition_mask"] = np.asarray(packed_mask, dtype=bool).copy()
                capture["condition_indexes"] = [int(index) for index in indexes]
                if self._condition_mask is not None and not np.array_equal(self._condition_mask, capture["condition_mask"]):
                    raise ValueError("runtime packed condition mask changed across requests")
                self._condition_mask = capture["condition_mask"].copy()
                self._condition_indexes = list(capture["condition_indexes"])
                initial = result[4][0] if isinstance(result, tuple) else result["initial_noise"][0]
                mask = result[6][0] if isinstance(result, tuple) else result["condition_mask"][0]
                references = result[5] if isinstance(result, tuple) else result.get("condition_reference")
                if references:
                    reference_value = references[0] if isinstance(references, (list, tuple)) else references
                    capture["packed_condition_reference"] = _normalize_condition_reference(reference_value, carrier_shape)
                    capture["packed_condition_mask"] = np.asarray(packed_mask, dtype=bool)
                initial_array, mask_array = _array(initial).astype(np.float32), np.asarray(_array(mask), dtype=bool)
                if mask_array.shape != initial_array.shape:
                    if mask_array.size == initial_array.size:
                        mask_array = mask_array.reshape(initial_array.shape)
                    else:
                        full_mask = np.asarray(capture["condition_mask"], dtype=bool)
                        if full_mask.size != initial_array.size:
                            raise ValueError("initial sampler state and runtime condition mask have incompatible sizes")
                        mask_array = full_mask.reshape(initial_array.shape)
                capture["initial_state"] = initial_array.copy()
                capture["initial_condition_mask"] = mask_array.copy()
                capture["initial_noise_hash"] = hash_predicted_noisy_region(initial_array, mask_array)
                _checkpoint_capture()
            except (IndexError, KeyError, TypeError, ValueError) as error:
                capture["prepare_error"] = str(error)
            return result

        def denoise_wrapper(*args: Any, **kwargs: Any) -> Any:
            memory = args[2] if len(args) > 2 else kwargs.get("memory")
            memory_cache = getattr(memory, "_text_kv_cache", None)
            before = memory_cache if memory_cache is not None else getattr(self.model, "text_kv_cache", None)
            if isinstance(before, Mapping):
                capture["text_kv_lifecycle"].append({"before": tuple(sorted(before))})
            elif isinstance(before, (list, tuple)):
                capture["text_kv_lifecycle"].append({"before": [id(item) for item in before], "initialized": [bool(getattr(item, "is_initialized", False)) for item in before]})
            else:
                capture["text_kv_lifecycle"].append({"before": None})
            # Capture the network-bound condition before the official call.  The
            # sampler may mutate the packed request in-place after the forward,
            # so post-call inspection is not evidence of what the network saw.
            for value in tuple(args) + tuple(kwargs.values()):
                vision_data = getattr(value, "vision", None)
                candidate = getattr(vision_data, "tokens", None)
                if candidate is not None:
                    tokens = candidate[0] if isinstance(candidate, (list, tuple)) else candidate
                    condition_tokens_mask = getattr(vision_data, "condition_mask", None)
                    if condition_tokens_mask is not None:
                        mask_value = condition_tokens_mask[0] if isinstance(condition_tokens_mask, (list, tuple)) else condition_tokens_mask
                        _reimpose_packed_condition(capture, tokens, mask_value)
                    break
            _capture_network_condition_before_denoise(capture, args, kwargs)
            capture["cfg_branch_observations"][-1]["memory_identity"] = id(memory) if memory is not None else None
            result = originals["denoise"](*args, **kwargs)
            after = memory_cache if memory_cache is not None else getattr(self.model, "text_kv_cache", None)
            if isinstance(after, Mapping):
                capture["text_kv_lifecycle"].append({"after": tuple(sorted(after))})
            elif isinstance(after, (list, tuple)):
                capture["text_kv_lifecycle"].append({"after": [id(item) for item in after], "initialized": [bool(getattr(item, "is_initialized", False)) for item in after]})
            return result

        def generate_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = originals["generate_samples_from_batch"](*args, **kwargs)
            vision = result.get("vision") if isinstance(result, dict) else None
            if isinstance(vision, (list, tuple)) and vision:
                full = _array(vision[0]).astype(np.float32, copy=True)
                capture["final_latent_full"] = full
                try:
                    selected = capture_final_latent(full, capture["condition_mask"])
                    capture["predicted_latent"] = selected["predicted_only"]
                    capture["latent_slicing"] = {
                        key: value.tolist() for key, value in selected.items() if "indexes" in key
                    } | {"axis": int(full.ndim - 3), "source_shape": list(full.shape), "selected_shape": list(selected["predicted_only"].shape)}
                except (KeyError, ValueError) as error:
                    capture["latent_slicing_error"] = str(error)
            return result

        decode_name = "decode_vision" if hasattr(self.model, "decode_vision") else "decode"
        def decode_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = originals[decode_name](*args, **kwargs)
            # The official pipeline's local decode_vision closure calls
            # model.decode; this is the actual VAE boundary.  Check decoder
            # state immediately after it returns, before finalization masks it.
            self._assert_cache_lifecycle()
            decoded = _array(result).astype(np.float32, copy=True)
            if decode_name == "decode":
                decoded = np.clip((1.0 + decoded) / 2.0, 0.0, 1.0).astype(np.float32)
            capture["decoded_float"] = decoded[0] if decoded.ndim == 5 and decoded.shape[0] == 1 else decoded
            if capture["decoded_float"].ndim >= 4 and capture["decoded_float"].shape[1] > 0:
                frame_index = int(capture["decoded_float"].shape[1] - 1)
                capture["decoded_final"] = np.take(capture["decoded_float"], frame_index, axis=1)
                capture["image_slicing"] = {"frame_index": frame_index, "axis": 1, "source_shape": list(capture["decoded_float"].shape), "selected_shape": list(capture["decoded_final"].shape), "source": "decode_vision"}
            return result

        try:
            for name, wrapper in (("get_data_and_condition", get_data_wrapper), ("_prepare_inference_data", prepare_wrapper), ("denoise", denoise_wrapper), ("generate_samples_from_batch", generate_wrapper), (decode_name, decode_wrapper)):
                original, installed = self._replace_method(name, wrapper)
                if installed:
                    originals[name] = original
            if "get_data_and_condition" not in originals:
                raise RuntimeError("official model boundary get_data_and_condition is unavailable")
            generate = originals.get("generate_samples_from_batch")
            if generate is None:
                raise RuntimeError("official model boundary generate_samples_from_batch is unavailable")
            generation_kwargs = {
                "seed": [MODEL_SEED], "num_steps": NUM_STEPS, "guidance": float(self.sample_settings.get("guidance", 1.0)),
                "shift": float(self.sample_settings.get("shift", 10.0)), "sampler": self.sample_settings.get("sampler"),
                "guidance_interval": self.sample_settings.get("guidance_interval"), "has_negative_prompt": bool(self.sample_settings.get("has_negative_prompt", False)),
                "skip_text_tokens_for_cfg": bool(self.sample_settings.get("skip_text_tokens_for_cfg", False)), "normalize_cfg": bool(self.sample_settings.get("normalize_cfg", False)),
                "use_batched_cfg": bool(self.sample_settings.get("use_batched_cfg", False)),
            }
            if self.sample_args is not None and hasattr(self.pipeline, "generate_batch"):
                if sample_dir is not None and hasattr(self.sample_args, "output_dir"):
                    self.sample_args.output_dir = Path(sample_dir)
                operation = lambda: self.pipeline.generate_batch([self.sample_args], data_batch, save_outputs=True)
            else:
                operation = lambda: getattr(self.model, "generate_samples_from_batch")(data_batch, **generation_kwargs)
            operation_error: BaseException | None = None
            try:
                result = self.run_with_boundaries(operation)
            except BaseException as error:
                operation_error = error
                result = None
            if "decoded_float" not in capture and isinstance(result, dict) and result.get("decoded") is not None:
                capture["decoded_float"] = _array(result["decoded"]).astype(np.float32)
        finally:
            for name, original in originals.items():
                setattr(self.model, name, original)
            self._reset_request_local_state()
            capture["text_kv_cleared"] = not any(bool(getattr(self.model, name, None)) for name in ("text_kv_cache", "_text_kv_cache") if hasattr(self.model, name))
            try:
                self._assert_cache_lifecycle()
                capture["cache_lifecycle_valid"] = True
            except ValueError as error:
                capture["cache_lifecycle_valid"] = False
                capture["cache_error"] = str(error)
        if requested_delta is not None:
            capture["actual_delta_fp32"] = ((_array(capture["realized_carrier_fp32"]) - _array(capture["carrier_fp32"]))).astype(np.float32)
            capture["actual_delta_bf16"] = _bf16_delta(_array(capture["carrier_fp32"]), _array(capture["realized_carrier_fp32"]))
        else:
            capture.setdefault("target_delta_fp32", np.zeros_like(_array(capture.get("carrier_fp32", capture.get("realized_carrier_fp32"))), dtype=np.float32))
            capture.setdefault("target_alpha", float(spec.get("alpha", 0.0)))
            capture.setdefault("target_epsilon", 0.0)
            capture.setdefault("target_rms", 0.0)
        if capture.get("realized_carrier_fp32") is not None and capture.get("condition_mask") is not None:
            expected_values = capture.get("packed_condition_reference")
            expected_mask = capture.get("packed_condition_mask")
            if expected_values is not None and expected_mask is not None:
                expected_values = _bf16_values(_array(expected_values))
                capture["expected_network_condition_hash"] = sha256_array(_select_condition_values(expected_values, expected_mask))
            else:
                # A packed reference is the authoritative expected condition
                # order.  The carrier fallback is BF16-rounded, never FP32,
                # and is retained only for minimal official-shaped stubs.
                expected_values = _bf16_values(_array(capture["realized_carrier_fp32"]))
                expected_mask = np.asarray(capture["condition_mask"], dtype=bool)
                capture["expected_network_condition_hash"] = sha256_array(expected_values[expected_mask])
        capture["condition_step_hashes"] = capture.pop("denoise_step_hashes")
        if capture.get("condition_mask") is not None:
            self._condition_mask = np.asarray(capture["condition_mask"], dtype=bool).copy()
        if capture.get("missing_network_condition_tokens"):
            capture["status"] = "fail"
            capture["condition_error"] = "actual packed BF16 condition tokens were not observed"
        capture["predicted_latent_sliced"] = bool(capture.get("predicted_latent") is not None and np.asarray(capture["predicted_latent"]).size and np.all(np.isfinite(capture["predicted_latent"])))
        capture["decoded_final_frame"] = bool(capture.get("decoded_final") is not None and np.asarray(capture["decoded_final"]).size and np.all(np.isfinite(capture["decoded_final"])))
        capture["status"] = "success" if capture.get("cache_lifecycle_valid", True) and not capture.get("condition_error") else "fail"
        guidance = float(self.sample_settings.get("guidance", 1.0))
        expected_steps = int(self.sample_settings.get("num_steps", getattr(self.sample_args, "num_steps", NUM_STEPS)))
        capture["expected_denoise_steps"] = expected_steps
        capture["cfg_branch_semantics"] = {"guidance": guidance, "conditional_calls": int(capture["cfg_branch_calls"]), "unconditional_calls": 0 if guidance == 1.0 else int(capture["cfg_branch_calls"]), "expected_multiplicity": 1 if guidance == 1.0 else 2, "source": "official denoise memory seam"}
        capture["cfg_branch_observations"] = list(capture["cfg_branch_observations"])
        capture["first_denoise_hash"] = capture["condition_step_hashes"][0] if capture["condition_step_hashes"] else None
        capture["final_denoise_hash"] = capture["condition_step_hashes"][-1] if capture["condition_step_hashes"] else None
        capture["condition_capture"] = {"encode_call_count": 1 if spec_id == "A" else 0, "denoise_step_hashes": capture["condition_step_hashes"], "denoise_step_evidence": capture["denoise_step_evidence"], "first_denoise_hash": capture["first_denoise_hash"], "final_denoise_hash": capture["final_denoise_hash"], "expected_network_condition_hash": capture.get("expected_network_condition_hash"), "network_condition_dtype": capture.get("network_condition_dtype", ""), "text_kv_cleared": capture["text_kv_cleared"], "cfg_branch_semantics": capture["cfg_branch_semantics"], "cfg_branch_observations": capture["cfg_branch_observations"]}
        if operation_error is not None:
            raise RuntimeCaptureError(str(operation_error), {key: _numpy_capture(value) for key, value in capture.items()}) from operation_error
        return capture


RuntimeAdapter = OfficialPostVaeRuntimeAdapter


def local_vae_experiment_overrides(vae_path: str | Path) -> list[str]:
    """Return Hydra overrides that force the approved local VAE path."""
    return [
        f"model.config.tokenizer.vae_path={json.dumps(str(Path(vae_path).resolve()))}",
        "model.config.tokenizer.bucket_name=\"\"",
        "model.config.tokenizer.object_store_credential_path_pretrained=\"\"",
    ]


def assert_resume_compatible(requested: Mapping[str, Any], existing: Mapping[str, Any]) -> None:
    missing = [key for key in COMPATIBILITY_FIELDS if key not in requested or key not in existing]
    changed = [key for key in sorted(set(requested) | set(existing)) if requested.get(key) != existing.get(key)]
    if missing or changed:
        fields = sorted(set(missing + changed))
        raise ValueError(f"resume compatibility mismatch in fields: {', '.join(fields)}")


validate_resume_compatibility = assert_resume_compatible
validate_resume_compatible = assert_resume_compatible


def validate_runtime_metadata(metadata: Mapping[str, Any], *, expected_setup: Mapping[str, Any] | None = None) -> None:
    """Reject a runtime that silently changed any approved sampling setting."""
    setup = metadata.get("runtime_setup", metadata)
    required = {"checkpoint_path", "sampler", "precision", "diffusion_cache_requested", "diffusion_cache_installed"}
    missing = required.difference(setup)
    if missing:
        raise ValueError(f"runtime setup is missing fields: {sorted(missing)}")
    validate_runtime_setup(dict(setup))
    if expected_setup is not None:
        changed = sorted(key for key in set(expected_setup) | set(setup) if expected_setup.get(key) != setup.get(key))
        if changed:
            raise ValueError(f"runtime setup mismatch in fields: {', '.join(changed)}")
    sample = metadata.get("runtime_sample")
    if sample is not None:
        if int(sample.get("seed", MODEL_SEED)) != MODEL_SEED or int(sample.get("num_steps", NUM_STEPS)) != NUM_STEPS:
            raise ValueError("runtime sample settings differ from the approved experiment")


def build_run_specs(*, direction_count: int = DIRECTION_COUNT, alphas: Iterable[float] = ALPHAS, model_seed: int = MODEL_SEED) -> list[dict[str, Any]]:
    """Compatibility spelling for callers that used the historical scan runner."""
    return build_call_plan(alphas=alphas, direction_count=direction_count, model_seed=model_seed)


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        temp.write_text(json.dumps(_numpy_capture(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        with temp.open("rb+") as stream:
            os.fsync(stream.fileno())
        temp.replace(destination)
    finally:
        if temp.exists():
            temp.unlink()


def append_invocation_history(root: str | Path, invocation: Mapping[str, Any]) -> Path:
    """Append one process invocation without losing the stage-A/resume audit trail."""
    destination = Path(root) / "invocation_history.json"
    history: list[Any] = []
    if destination.is_file():
        try:
            loaded = json.loads(destination.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                history = loaded
        except (OSError, json.JSONDecodeError):
            history = []
    history.append(dict(invocation))
    write_json(destination, history)
    return destination


def write_sha256_manifest(root: str | Path) -> Path:
    root_path = Path(root)
    manifest = root_path / "MANIFEST.sha256"
    lines = []
    for path in sorted(item for item in root_path.rglob("*") if item.is_file() and item != manifest):
        lines.append(f"{sha256_file(path)}  {path.relative_to(root_path).as_posix()}")
    manifest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")
    return manifest


def classify_scan_status(*, total: int, successful: int, failed: int = 0, gate_passed: bool = True) -> str:
    if not gate_passed or int(failed) or int(successful) != int(total):
        return "FAIL"
    return "COMPLETE"


class GpuMonitor:
    """Best-effort single-GPU sampler with deterministic shutdown."""
    def __init__(self, path: str | Path, gpu_index: int = 0, interval: float = 0.5) -> None:
        self.path, self.gpu_index, self.interval = Path(path), int(gpu_index), float(interval)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.current_sample: str | None = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("epoch_s,iso_utc,sample_id,gpu_index,memory_used_mib,utilization_pct\n", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, name="umi-gpu-monitor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.run(["nvidia-smi", "-i", str(self.gpu_index), "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=max(1.0, self.interval))
                values = [part.strip() for part in output.stdout.strip().split(",")]
                if len(values) >= 3:
                    now = time.time()
                    with self.path.open("a", encoding="utf-8") as stream:
                        stream.write(f"{now},{datetime.fromtimestamp(now, timezone.utc).isoformat()},{self.current_sample or ''},{values[0]},{values[1]},{values[2]}\n")
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval * 3.0))


def _fsync_file(path: Path) -> None:
    with path.open("rb+") as stream:
        os.fsync(stream.fileno())


class SampleStore:
    """Atomic sample-state store that never replaces an existing success."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.active_attempt_dirs: dict[str, Path] = {}

    def _path(self, sample_id: str) -> Path:
        return self.root / str(sample_id)

    def prepare(self, sample_id: str, *, resume: bool = False, required_files: Iterable[str] = ()) -> str:
        path = self._path(sample_id)
        status = path / "status.json"
        complete = False
        status_candidates = [status, path / "sample_metrics.json"]
        for status_candidate in status_candidates:
            if not status_candidate.is_file():
                continue
            try:
                value = json.loads(status_candidate.read_text(encoding="utf-8"))
                stored_required = value.get("required_artifacts", [])
                required = list(required_files) + [name for name in stored_required if name not in required_files]
                artifact_hashes = value.get("artifact_sha256", {})
                complete = value.get("status") == "success" and all((path / name).is_file() for name in required)
                if complete and artifact_hashes:
                    complete = all(sha256_file(path / name) == expected for name, expected in artifact_hashes.items() if (path / name).is_file())
            except (OSError, json.JSONDecodeError):
                complete = False
            if complete:
                break
        if complete:
            if not resume:
                raise FileExistsError(f"successful sample already exists: {path}")
            return "skip"
        if path.exists():
            if not resume:
                raise FileExistsError(f"sample directory is not empty: {path}")
            if any(path.iterdir()):
                index = 1
                while True:
                    archive = self.root / f"{sample_id}.attempt{index:02d}"
                    if not archive.exists():
                        path.rename(archive)
                        break
                    index += 1
            else:
                path.rmdir()
        path.mkdir(parents=True, exist_ok=False)
        running_path = path / "status.json"
        running_path.write_text(json.dumps({"status": "running", "sample_id": str(sample_id), "started_utc": utc_now()}, indent=2) + "\n", encoding="utf-8")
        _fsync_file(running_path)
        self.active_attempt_dirs[str(sample_id)] = path
        return "run"

    def active_path(self, sample_id: str) -> Path:
        return self.active_attempt_dirs.get(str(sample_id), self._path(sample_id))

    def _atomic_payload(self, sample_id: str, payload: Mapping[str, Any], status_value: str, artifacts: Mapping[str, Any] | None = None) -> Path:
        destination = self._path(sample_id)
        started_utc: str | None = None
        running_status = destination / "status.json"
        if running_status.is_file():
            try:
                candidate = json.loads(running_status.read_text(encoding="utf-8")).get("started_utc")
                if isinstance(candidate, str) and candidate:
                    started_utc = candidate
            except (OSError, json.JSONDecodeError):
                pass
        if destination.exists() and (destination / "status.json").is_file():
            try:
                if json.loads((destination / "status.json").read_text(encoding="utf-8")).get("status") == "success":
                    raise FileExistsError(f"refusing to overwrite successful sample: {destination}")
            except json.JSONDecodeError:
                pass
        temp = Path(tempfile.mkdtemp(prefix=f".{sample_id}.attempt.", dir=self.root))
        try:
            auto_artifacts: dict[str, Any] = {}
            def json_safe(value: Any, name: str) -> Any:
                value = _numpy_capture(value)
                if isinstance(value, np.ndarray):
                    artifact_name = f"{name}.npy"
                    auto_artifacts[artifact_name] = value
                    return {"artifact": artifact_name, "dtype": str(value.dtype), "shape": list(value.shape)}
                if isinstance(value, Mapping):
                    return {str(key): json_safe(item, f"{name}_{key}") for key, item in value.items()}
                if isinstance(value, (list, tuple)):
                    return [json_safe(item, f"{name}_{index}") for index, item in enumerate(value)]
                return value
            safe_payload = json_safe(payload, "payload")
            (temp / "status.json").write_text(json.dumps({"status": "running", "started_utc": started_utc or utc_now()}, indent=2) + "\n", encoding="utf-8")
            (temp / "sample.json").write_text(json.dumps(safe_payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            for relative_name, value in {**auto_artifacts, **(artifacts or {})}.items():
                value = _numpy_capture(value)
                artifact_path = temp / relative_name
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(value, np.ndarray):
                    with artifact_path.open("wb") as stream:
                        np.save(stream, value, allow_pickle=False)
                elif isinstance(value, (bytes, bytearray, memoryview)):
                    artifact_path.write_bytes(bytes(value))
                elif isinstance(value, (str, Path)) and Path(value).is_file():
                    shutil.copy2(value, artifact_path)
                else:
                    raise TypeError(f"artifact {relative_name!r} must be a NumPy array, bytes, or source path")
            artifact_names = sorted(str(name) for name in {**auto_artifacts, **(artifacts or {})})
            artifact_hashes = {name: sha256_file(temp / name) for name in artifact_names if (temp / name).is_file()}
            final_status = {"status": status_value, "finished_utc": utc_now(), "required_artifacts": artifact_names, "artifact_sha256": artifact_hashes}
            if started_utc is not None:
                final_status["started_utc"] = started_utc
            (temp / "status.json").write_text(json.dumps(final_status, indent=2) + "\n", encoding="utf-8")
            _fsync_file(temp / "status.json")
            _fsync_file(temp / "sample.json")
            if destination.exists():
                if any(destination.iterdir()):
                    index = 1
                    while True:
                        archive = self.root / f"{sample_id}.attempt{index:02d}"
                        if not archive.exists():
                            destination.rename(archive)
                            break
                        index += 1
                else:
                    destination.rmdir()
            temp.rename(destination)
            return destination
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise

    def write_success(self, sample_id: str, payload: Mapping[str, Any], *, artifacts: Mapping[str, Any] | None = None) -> Path:
        return self._atomic_payload(sample_id, payload, "success", artifacts)

    def write_failure(self, sample_id: str, payload: Mapping[str, Any], *, artifacts: Mapping[str, Any] | None = None) -> Path:
        return self._atomic_payload(sample_id, payload, "fail", artifacts)

    def load_record(self, sample_id: str) -> dict[str, Any]:
        path = self._path(sample_id)
        record = json.loads((path / "sample.json").read_text(encoding="utf-8"))
        for artifact in path.rglob("*.npy"):
            record[artifact.stem] = np.load(artifact, allow_pickle=False)
        return record


def prepare_sample_dir_for_run(sample_dir: str | Path, *, resume: bool, required_files: Iterable[str], successful_validator: Callable[[Path], None] | None = None) -> str:
    store = SampleStore(Path(sample_dir).parent)
    result = store.prepare(Path(sample_dir).name, resume=resume, required_files=required_files)
    if result == "skip" and successful_validator is not None:
        successful_validator(Path(sample_dir))
    return result


def generate_directions_for_mask(mask: Any, *, count: int = DIRECTION_COUNT, seed: int = DIRECTION_SEED) -> np.ndarray:
    return generate_direction_bank(count, np.asarray(mask).shape, mask=mask, seed=seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage-a-only", action="store_true")
    parser.add_argument("--framework-root")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--vae-path")
    parser.add_argument("--input-path")
    parser.add_argument("--action-path")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--action-chunk-index", type=int, default=0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--direction-seed", type=int, default=DIRECTION_SEED)
    parser.add_argument("--model-seed", type=int, default=MODEL_SEED)
    parser.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS))
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--sampler", default=SAMPLER)
    parser.add_argument("--precision", default=PRECISION)
    parser.add_argument("--parallelism-preset", default=PARALLELISM_PRESET)
    parser.add_argument("--diffusion-cache", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args(argv)
    if args.model_seed != MODEL_SEED:
        parser.error("--model-seed is fixed to 0")
    if args.action_chunk_index != 0 or args.gpu_index != 0:
        parser.error("action chunk index and GPU index are fixed to 0")
    if args.num_steps != NUM_STEPS or str(args.sampler).lower() != SAMPLER:
        parser.error("sampler and step count are fixed to UniPC with 30 steps")
    if "bfloat16" not in str(args.precision).lower() or args.parallelism_preset != PARALLELISM_PRESET or args.batch_size != 1:
        parser.error("precision, parallelism, and batch size are fixed by the approved experiment")
    if args.diffusion_cache:
        parser.error("diffusion cache must be disabled")
    build_call_plan(alphas=args.alphas, direction_seed=args.direction_seed)
    return args


def load_official_runtime(args: argparse.Namespace, run_dir: Path) -> OfficialPostVaeRuntimeAdapter:
    """Construct the installed official OmniInference and bind its model-owned API."""
    framework_root = Path(args.framework_root).resolve()
    if str(framework_root) not in os.sys.path:
        os.sys.path.insert(0, str(framework_root))
    from cosmos_framework.inference.args import OmniSetupOverrides  # type: ignore[import-not-found]
    from cosmos_framework.inference.common.init import init_output_dir, init_script  # type: ignore[import-not-found]
    setup_kwargs = build_setup_overrides_kwargs(
        checkpoint_path=Path(args.checkpoint_path), output_dir=run_dir,
        parallelism_preset=args.parallelism_preset, guardrails=False, diffusion_cache=False,
    )
    # Task 4 may request one uniform eager backend so A/B/C do not compare a
    # compiled BF16 closure against a cloned FP32 module.  Ordinary scan
    # callers keep the official default because they do not carry this flag.
    if hasattr(args, "use_torch_compile"):
        setup_kwargs["use_torch_compile"] = bool(args.use_torch_compile)
    # The VAE is a scientific input, not merely provenance.  Point the
    # official tokenizer config at the caller-supplied local artifact before
    # model construction so encode/decode actually consume this file.
    setup_kwargs["experiment_overrides"] = local_vae_experiment_overrides(args.vae_path)
    setup_overrides = OmniSetupOverrides(**setup_kwargs)
    setup_args = setup_overrides.build_setup()
    if str(setup_args.sampler).lower() != SAMPLER or int(getattr(setup_args, "num_steps", NUM_STEPS)) != NUM_STEPS:
        raise ValueError("official runtime did not resolve the fixed sampler configuration")
    # The standalone runner is not launched by the framework's usual script
    # entrypoint.  Initialize the official logger/distributed singleton first;
    # otherwise init_output_dir formats FLAGS with a missing job_name.
    init_script(training=False)
    init_output_dir(run_dir, resume=bool(args.resume))
    pipeline = setup_args.get_inference_cls().create(setup_args)
    adapter = OfficialPostVaeRuntimeAdapter.from_pipeline(pipeline)
    adapter.runtime_setup = {
        "checkpoint_path": str(Path(setup_args.checkpoint_path).resolve()),
        "vae_path": str(Path(args.vae_path).resolve()),
        "sampler": str(setup_args.sampler),
        "precision": str(getattr(adapter.model.config, "precision", args.precision)),
        "diffusion_cache_requested": bool(getattr(setup_args, "diffusion_cache", False)),
        "diffusion_cache_installed": bool(getattr(adapter.model, "_diffusion_cache_installed", False)),
        "use_torch_compile": bool(getattr(setup_args, "use_torch_compile", True)),
    }
    validate_runtime_metadata({"runtime_setup": adapter.runtime_setup})
    if hasattr(adapter.model, "eval"):
        adapter.model.eval()
    return adapter


def resolved_sample_fps(args: argparse.Namespace) -> int:
    """Use official default 20, while Bridge contracts explicitly pin 5."""
    return int(getattr(args, "fps", 20))


def load_official_data_batch(adapter: OfficialPostVaeRuntimeAdapter, args: argparse.Namespace, run_dir: Path) -> tuple[Any, Any]:
    """Build the official sample args/data batch from the CLI input/action/prompt."""
    if not args.input_path or not args.action_path:
        raise ValueError("--input-path and --action-path are required for official execution")
    from cosmos_framework.inference.args import OmniSampleOverrides  # type: ignore[import-not-found]
    from cosmos_framework.inference.inference import get_sample_data  # type: ignore[import-not-found]
    sample = {
        "name": "umi_fd_post_vae_scan", "model_mode": "forward_dynamics", "domain_name": "umi",
        "view_point": "ego_view", "fps": resolved_sample_fps(args), "image_size": 256, "action_chunk_size": 16,
        "prompt": args.prompt, "vision_path": str(Path(args.input_path).resolve()),
        "action_path": str(Path(args.action_path).resolve()), "seed": MODEL_SEED, "guidance": 1.0, "shift": 10.0,
    }
    overrides = OmniSampleOverrides.model_validate(sample)
    overrides.output_dir = run_dir / "inputs"
    overrides.download(run_dir / "inputs")
    sample_args = overrides.build_sample(model_config=adapter.pipeline.model_config)
    adapter.sample_args = sample_args
    adapter.sample_settings = {
        "guidance": float(getattr(sample_args, "guidance", 1.0)), "shift": float(getattr(sample_args, "shift", 10.0)),
        "guidance_interval": getattr(sample_args, "guidance_interval", None), "has_negative_prompt": bool(getattr(sample_args, "has_negative_prompt", False)),
        "skip_text_tokens_for_cfg": bool(getattr(sample_args, "skip_text_tokens_for_cfg", False)), "normalize_cfg": bool(getattr(sample_args, "normalize_cfg", False)),
        "use_batched_cfg": bool(getattr(sample_args, "use_batched_cfg", False)), "sampler": getattr(adapter.model, "fixed_step_sampler", None),
    }
    if adapter.sample_settings["guidance"] != 1.0 or adapter.sample_settings["shift"] != 10.0:
        raise ValueError("resolved official sample settings must be guidance=1.0 and shift=10.0")
    return get_sample_data(sample_args, adapter.model, device="cuda"), sample_args


def _compatibility_for_args(args: argparse.Namespace, run_dir: Path | None = None) -> dict[str, Any]:
    def path_hash(value: str | None) -> str:
        if value and Path(value).is_file():
            return sha256_file(value)
        if value and Path(value).is_dir():
            return sha256_tree(value)
        raise FileNotFoundError(value or "<missing path>")
    framework_root = Path(args.framework_root).resolve()
    values = {
        "framework_sha256": sha256_tree(framework_root),
        "model_sha256": path_hash(args.checkpoint_path), "vae_sha256": path_hash(args.vae_path),
        "code_sha256": sha256_file(Path(__file__)), "bridge_sha256": sha256_file(Path(__file__).with_name("umi_fd_post_vae_bridge.py")), "config_sha256": sha256_json_value({key: value for key, value in vars(args).items() if key not in {"resume", "stage_a_only", "run_dir"}}),
        "input_sha256": path_hash(args.input_path), "action_sha256": path_hash(args.action_path),
        "direction_sha256": sha256_json_value({"seed": args.direction_seed, "count": DIRECTION_COUNT, "alphas": list(ALPHAS)}),
        "z0_sha256": "pending-stage-a", "mask_sha256": "pending-stage-a", "noise_policy_sha256": sha256_json_value({"seed": MODEL_SEED, "predicted_region_only": True}),
    }
    if run_dir is not None:
        for field, filename in (("z0_sha256", "z0.npy"), ("mask_sha256", "mask.npy"), ("direction_sha256", "direction_bank.npy")):
            path = run_dir / filename
            if path.is_file():
                values[field] = sha256_file(path)
    return values


def _validate_required_paths(args: argparse.Namespace) -> None:
    required = {"framework-root": args.framework_root, "checkpoint-path": args.checkpoint_path, "vae-path": args.vae_path, "input-path": args.input_path, "action-path": args.action_path}
    missing = [name for name, value in required.items() if not value or not Path(value).exists()]
    if missing:
        raise FileNotFoundError("required runtime inputs are missing: " + ", ".join(missing))


def run_experiment(args: argparse.Namespace, *, execute_call: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None, runtime_factory: Callable[[argparse.Namespace, Path], Any] | None = None) -> dict[str, Any]:
    """Run all approved calls through the official model-owned runtime seams."""
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    preexisting_run = any(run_dir.iterdir())
    config_path = run_dir / "config.json"
    try:
        _validate_required_paths(args)
        compatibility = _compatibility_for_args(args, run_dir)
        if args.resume:
            if not config_path.is_file():
                raise FileNotFoundError(f"resume run is missing config.json: {run_dir}")
            assert_resume_compatible(compatibility, json.loads(config_path.read_text(encoding="utf-8")))
        elif preexisting_run:
            raise FileExistsError(f"run directory is not empty; pass --resume explicitly: {run_dir}")
    except BaseException as error:
        if preexisting_run:
            raise
        write_json(run_dir / "status.json", {"status": "FAIL", "phase": "input-validation", "finished_utc": utc_now(), "error": {"type": type(error).__name__, "message": str(error)}})
        write_sha256_manifest(run_dir)
        raise
    write_json(run_dir / "status.json", {"status": "RUNNING", "phase": "setup", "started_utc": utc_now()})
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    plan = build_call_plan(alphas=args.alphas, direction_seed=args.direction_seed)
    def provenance_payload() -> dict[str, Any]:
        # The detached sample provenance is deliberately the same finalized
        # identity object as the root provenance.  This makes a sample unable
        # to quietly drift to a different runner/input/action binding.
        return {
            **dict(compatibility),
            "framework_root": str(Path(args.framework_root).resolve()),
            "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
            "vae_path": str(Path(args.vae_path).resolve()),
            "input_path": str(Path(args.input_path).resolve()),
            "action_path": str(Path(args.action_path).resolve()),
            "runner_sha256": compatibility["code_sha256"],
            "checkpoint_sha256": compatibility["model_sha256"],
            "runtime": "official OmniInference model-owned boundaries",
        }
    write_json(config_path, compatibility)
    write_json(run_dir / "provenance.json", provenance_payload())
    invocation = {"argv": list(os.sys.argv), "started_utc": utc_now(), "runner_sha256": sha256_file(Path(__file__))}
    write_json(run_dir / "invocation.json", invocation)
    append_invocation_history(run_dir, invocation)
    write_json(run_dir / "call_plan.json", plan)
    gpu_csv = run_dir / "gpu_samples.csv"
    if not gpu_csv.exists():
        gpu_csv.write_text("epoch_s,iso_utc,sample_id,gpu_index,memory_used_mib,utilization_pct\n", encoding="utf-8")
    if execute_call is None:
        try:
            runtime = runtime_factory(args, run_dir) if runtime_factory is not None else load_official_runtime(args, run_dir)
        except Exception as error:
            write_json(run_dir / "status.json", {"status": "FAIL", "phase": "setup", "finished_utc": utc_now(), "error": {"type": type(error).__name__, "message": str(error)}})
            write_sha256_manifest(run_dir)
            raise
        if callable(runtime) and not isinstance(runtime, OfficialPostVaeRuntimeAdapter):
            execute_call = runtime
        else:
            direction_bank: np.ndarray | None = np.load(run_dir / "direction_bank.npy", allow_pickle=False) if (args.resume and (run_dir / "direction_bank.npy").is_file()) else None
            try:
                data_batch, _sample_args = load_official_data_batch(runtime, args, run_dir)
                if args.resume and (samples_dir / "A" / "sample.json").is_file() and isinstance(runtime, OfficialPostVaeRuntimeAdapter):
                    saved_a = SampleStore(samples_dir).load_record("A")
                    saved_indexes = saved_a.get("condition_indexes")
                    saved_mask = saved_a.get("condition_mask")
                    saved_carrier = saved_a.get("carrier_fp32")
                    if saved_indexes is None or saved_mask is None or saved_carrier is None:
                        raise ValueError("resume A sample is missing baseline artifacts for restoration")
                    runtime.prime_baseline(
                        _clone_runtime(data_batch), condition_indexes=saved_indexes,
                        expected_carrier=saved_carrier, expected_mask=saved_mask,
                        expected_direction_bank=direction_bank,
                    )
            except BaseException as error:
                write_json(run_dir / "status.json", {"status": "FAIL", "phase": "data-load-or-resume", "finished_utc": utc_now(), "error": {"type": type(error).__name__, "message": str(error)}})
                write_sha256_manifest(run_dir)
                raise
            def execute_call(spec: Mapping[str, Any], sample_dir: Path | None = None) -> Mapping[str, Any]:
                nonlocal direction_bank
                batch = _clone_runtime(data_batch)
                direction = None
                if spec.get("direction_index") is not None and direction_bank is not None:
                    direction = direction_bank[int(spec["direction_index"])]
                record = runtime.execute_call(spec, data_batch=batch, sample_dir=sample_dir or (samples_dir / str(spec["sample_id"])), direction=direction)
                if direction_bank is None and record.get("condition_mask") is not None:
                    direction_bank = generate_directions_for_mask(record["condition_mask"])
                    np.save(run_dir / "direction_bank.npy", direction_bank, allow_pickle=False)
                    np.save(run_dir / "mask.npy", np.asarray(record["condition_mask"], dtype=bool), allow_pickle=False)
                    np.save(run_dir / "z0.npy", np.asarray(record.get("carrier_fp32")), allow_pickle=False)
                return record
    store = SampleStore(samples_dir)
    def finalize_after_stage_a(records: Mapping[str, Mapping[str, Any]]) -> None:
        first = records.get("A", {})
        carrier = _record_array(first, "carrier_fp32")
        mask = _record_array(first, "condition_mask")
        if carrier is None or mask is None or carrier.size == 0 or mask.shape != carrier.shape:
            return
        np.save(run_dir / "z0.npy", carrier.astype(np.float32), allow_pickle=False)
        np.save(run_dir / "mask.npy", mask.astype(bool), allow_pickle=False)
        direction_path = run_dir / "direction_bank.npy"
        if not direction_path.is_file():
            np.save(direction_path, generate_directions_for_mask(mask), allow_pickle=False)
        compatibility["z0_sha256"] = sha256_file(run_dir / "z0.npy")
        compatibility["mask_sha256"] = sha256_file(run_dir / "mask.npy")
        compatibility["direction_sha256"] = sha256_file(direction_path)
        write_json(run_dir / "config.json", compatibility)
        write_json(run_dir / "provenance.json", provenance_payload())
    monitor = GpuMonitor(run_dir / "gpu_samples.csv", args.gpu_index)
    monitor.start()
    def persisted_executor(spec: Mapping[str, Any]) -> Mapping[str, Any]:
        sample_id = str(spec["sample_id"])
        disposition = store.prepare(sample_id, resume=bool(args.resume), required_files=("sample.json",))
        if disposition == "skip":
            return store.load_record(sample_id)
        try:
            monitor.current_sample = sample_id
            started = time.perf_counter()
            record: dict[str, Any] = {}
            try:
                record = dict(execute_call(spec, sample_dir=store.active_path(sample_id)))
            except TypeError as error:
                if "sample_dir" not in str(error):
                    raise
                record = dict(execute_call(spec))
            status = str(record.get("status", "success"))
            if sample_id == "A" and status == "success":
                finalize_after_stage_a({"A": record})
            record.update({"sample_id": sample_id, "spec": dict(spec), "elapsed_seconds": time.perf_counter() - started, "provenance": provenance_payload(), "compatibility": dict(compatibility)})
            if status == "success":
                evidence = validate_sample_evidence(record, stage_a=str(spec.get("phase")) == "stage_a")
                if not evidence.passed:
                    status = "fail"
                    record["status"] = status
                    record["error"] = {"type": "EvidenceValidationError", "message": "; ".join(evidence.failures)}
            artifacts = {f"{key}.npy": value for key, value in record.items() if isinstance(value, np.ndarray)}
            store.write_success(sample_id, {key: value for key, value in record.items() if not isinstance(value, np.ndarray)}, artifacts=artifacts) if status == "success" else store.write_failure(sample_id, {key: value for key, value in record.items() if not isinstance(value, np.ndarray)}, artifacts=artifacts)
            return record
        except Exception as error:
            record = dict(record)
            partial_capture = getattr(error, "capture", None)
            if isinstance(partial_capture, Mapping):
                record.update({key: _numpy_capture(value) for key, value in partial_capture.items() if key not in record})
            record.update({"status": "fail", "sample_id": sample_id, "spec": dict(spec), "elapsed_seconds": time.perf_counter() - started if "started" in locals() else 0.0, "provenance": provenance_payload(), "compatibility": dict(compatibility), "error": {"type": type(error).__name__, "message": str(error)}})
            artifacts = {f"{key}.npy": value for key, value in record.items() if isinstance(value, np.ndarray)}
            record = {key: value for key, value in record.items() if not isinstance(value, np.ndarray)}
            store.write_failure(sample_id, record, artifacts=artifacts)
            return {**record, **{Path(name).stem: value for name, value in artifacts.items()}}
        finally:
            monitor.current_sample = None
    try:
        result = run_stage_a_and_scan(persisted_executor, plan=plan, stage_a_only=args.stage_a_only, after_stage_a=finalize_after_stage_a)
    except BaseException as error:
        monitor.stop()
        write_json(run_dir / "status.json", {"status": "FAIL", "phase": "execution", "finished_utc": utc_now(), "error": {"type": type(error).__name__, "message": str(error)}})
        write_sha256_manifest(run_dir)
        raise
    finally:
        monitor.stop()
    if result.get("stage_a"):
        first = result["stage_a"].get("A", {})
        if first.get("carrier_fp32") is not None:
            np.save(run_dir / "z0.npy", np.asarray(first["carrier_fp32"]), allow_pickle=False)
        if first.get("condition_mask") is not None:
            np.save(run_dir / "mask.npy", np.asarray(first["condition_mask"], dtype=bool), allow_pickle=False)
    compatibility["z0_sha256"] = sha256_file(run_dir / "z0.npy") if (run_dir / "z0.npy").is_file() else compatibility["z0_sha256"]
    compatibility["mask_sha256"] = sha256_file(run_dir / "mask.npy") if (run_dir / "mask.npy").is_file() else compatibility["mask_sha256"]
    compatibility["direction_sha256"] = sha256_file(run_dir / "direction_bank.npy") if (run_dir / "direction_bank.npy").is_file() else compatibility["direction_sha256"]
    write_json(run_dir / "config.json", compatibility)
    status = {"status": result["status"], "finished_utc": utc_now(), "completed": len(result.get("scan_calls", [])) + 4, "total": EXPECTED_CALL_COUNT, "gate": {"passed": result["gate"].passed, "failures": list(result["gate"].failures)}}
    write_json(run_dir / "status.json", status)
    write_sha256_manifest(run_dir)
    return result


__all__ = [
    "ALPHAS", "COMPATIBILITY_FIELDS", "DIRECTION_COUNT", "DIRECTION_SEED", "EXPECTED_CALL_COUNT", "EXPECTED_SCAN_COUNT", "FRAMEWORK_COMMIT", "GateResult", "GpuMonitor", "MODEL_SEED", "OfficialPostVaeRuntimeAdapter", "RuntimeAdapter", "RuntimeCaptureError", "SampleStore", "assert_resume_compatible", "build_call_plan", "build_run_specs", "capture_final_latent", "classify_scan_status", "evaluate_stage_a_gate", "generate_directions_for_mask", "hash_predicted_noisy_region", "parse_args", "prepare_sample_dir_for_run", "run_experiment", "run_stage_a_and_scan", "sha256_array", "sha256_file", "sha256_json_value", "sha256_tree", "stage_a_gate", "validate_call_plan", "validate_resume_compatible", "validate_resume_compatibility", "validate_runtime_metadata", "validate_sample_evidence", "write_json", "write_sha256_manifest",
]


if __name__ == "__main__":
    parsed = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    run_experiment(parsed)
