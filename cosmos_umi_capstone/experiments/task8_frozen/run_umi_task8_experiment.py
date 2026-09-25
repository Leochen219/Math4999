"""Gated Task 8 runner.

This file is intentionally generation-free until a caller supplies
``release=True`` and an injected executor.  The production launcher can bind
that executor to one already-loaded official runtime after main's review;
tests use a tiny CPU executor and therefore make no model/GPU claim.
"""
from __future__ import annotations

import hashlib
import gc
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from .umi_task8_runtime import Task8InputAdapter, Task8RuntimeConfig
    from .umi_task6_primitives import evaluate_resources
except ImportError:
    from umi_task8_runtime import Task8InputAdapter, Task8RuntimeConfig
    from umi_task6_primitives import evaluate_resources


class ResumeMismatch(ValueError):
    """Existing run evidence is not bound to this exact invocation."""


class ResourceStop(RuntimeError):
    """Resource or cleanup policy has latched a hard stop."""


class BlockedExecution(RuntimeError):
    """Live generation was attempted without the explicit release gate."""


REQUIRED_BINDING_KEYS = (
    "code", "model", "vae", "config", "data", "actions", "preprocessing", "noise",
)
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"array": True, "dtype": str(value.dtype), "shape": list(value.shape),
                "sha256": _array_hash(value)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("nonfinite values cannot enter JSON evidence")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _array_hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256(); digest.update(str(array.dtype).encode()); digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode()); digest.update(b"\0")
    digest.update(array.tobytes(order="C")); return digest.hexdigest()


def _validate_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(binding, Mapping):
        raise BlockedExecution("Task 8 release binding must be a mapping")
    missing = [key for key in REQUIRED_BINDING_KEYS if key not in binding]
    if missing:
        raise BlockedExecution(f"Task 8 release binding is missing hashes: {missing}")
    normalized: dict[str, Any] = {}
    for key in REQUIRED_BINDING_KEYS:
        value = binding[key]
        digest = value.get("sha256") if isinstance(value, Mapping) else value
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise BlockedExecution(f"Task 8 binding {key!r} must be a SHA256 identity")
        normalized[key] = value
    return dict(binding)


def _required_array(result: Mapping[str, Any], names: Sequence[str], *, sample_id: str,
                    label: str) -> np.ndarray:
    for name in names:
        if name in result:
            value = result[name]
            if not isinstance(value, np.ndarray):
                raise ResumeMismatch(f"{sample_id} {label} must be a saved ndarray")
            array = np.ascontiguousarray(value)
            if array.size == 0 or not np.all(np.isfinite(array)):
                raise ResumeMismatch(f"{sample_id} {label} must be finite and non-empty")
            return array
    raise ResumeMismatch(f"{sample_id} lacks a saved {label}")


def _noise_hash(result: Mapping[str, Any], *, sample_id: str) -> str:
    for name in ("prediction_noise_hash", "noise_hash", "initial_noise_hash"):
        value = result.get(name)
        if isinstance(value, str) and SHA256_RE.fullmatch(value) is not None:
            return value
    raise ResumeMismatch(f"{sample_id} lacks a SHA256 prediction-noise identity")


def _action_consumption(result: Mapping[str, Any], *, sample_id: str) -> tuple[str, ...]:
    hashes = result.get("packed_action_token_hashes")
    consumption = result.get("action_consumption")
    if not isinstance(hashes, (list, tuple)) or len(hashes) != 30:
        raise ResumeMismatch(f"{sample_id} lacks exactly 30 packed action-token hashes")
    if any(not isinstance(value, str) or SHA256_RE.fullmatch(value) is None for value in hashes):
        raise ResumeMismatch(f"{sample_id} contains an invalid packed action-token hash")
    if not isinstance(consumption, Mapping) or consumption.get("all_steps_match") is not True \
            or int(consumption.get("steps", 0)) != 30:
        raise ResumeMismatch(f"{sample_id} lacks all-step action-consumption evidence")
    consumed = consumption.get("consumed_token_hashes")
    if not isinstance(consumed, (list, tuple)) or tuple(consumed) != tuple(hashes):
        raise ResumeMismatch(f"{sample_id} action-consumption readback differs from packed hashes")
    expected = consumption.get("expected_token_hash")
    if not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
        raise ResumeMismatch(f"{sample_id} lacks expected action-token hash")
    if any(value != expected for value in hashes):
        raise ResumeMismatch(f"{sample_id} consumed action tokens differ from the expected token hash")
    return tuple(hashes)


FORMAL_CALL_PLAN: tuple[dict[str, Any], ...] = (
    {"sample_id": "G0_real_x0_seed0", "call": "G0", "condition_source": "real_x0", "action_source": "a0", "chunk_index": 0, "seed": 0},
    {"sample_id": "G0_repeat_real_x0_seed0", "call": "G0_repeat", "condition_source": "real_x0", "action_source": "a0", "chunk_index": 0, "seed": 0},
    {"sample_id": "TF2_real_x16_seed1", "call": "TF2", "condition_source": "real_x16", "action_source": "a1", "chunk_index": 1, "seed": 1},
    {"sample_id": "AR2_g0_float_last_fp32_seed1", "call": "AR2", "condition_source": "g0_float_last_fp32", "action_source": "a1", "chunk_index": 1, "seed": 1},
)

FORMAL_SAMPLE_IDS = tuple(row["sample_id"] for row in FORMAL_CALL_PLAN)


ENGINEERING_SMOKE_PLAN: tuple[dict[str, Any], ...] = (
    {"sample_id": "engineering_smoke", "call": "SMOKE", "condition_source": "real_x0", "action_source": "a0", "chunk_index": 0, "seed": 0},
)


def build_formal_call_plan() -> list[dict[str, Any]]:
    return [dict(row) for row in FORMAL_CALL_PLAN]


def _required_resources(snapshot: Mapping[str, Any]) -> None:
    required = ("gpu_used_gib", "gpu_free_gib", "gpu_reserved_gib", "gpu_peak_allocated_gib",
                "gpu_peak_nvml_used_gib", "ram_available_gib", "rss_gib", "swap_used_gib",
                "disk_free_gib", "cgroup_memory_limited", "cgroup_memory_limit_gib",
                "cgroup_memory_current_gib", "cgroup_memory_free_gib")
    missing = [key for key in required if key not in snapshot]
    if missing:
        raise ResourceStop(f"MONITOR_NO_SNAPSHOT: missing fields {missing}")
    if snapshot.get("monitor_failure") or snapshot.get("monitor_error"):
        raise ResourceStop(f"MONITOR_FAILURE: {snapshot.get('monitor_failure') or snapshot.get('monitor_error')}")
    if not isinstance(snapshot["cgroup_memory_limited"], bool):
        raise ResourceStop("RESOURCE_SNAPSHOT_NONFINITE: cgroup_memory_limited")
    for key in required:
        if key == "cgroup_memory_limited" or (key.startswith("cgroup_memory_") and not snapshot[key] if key == "cgroup_memory_limited" else False):
            continue
        value = snapshot[key]
        if value is None and key.startswith("cgroup_memory_") and not snapshot["cgroup_memory_limited"]:
            continue
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            raise ResourceStop(f"RESOURCE_SNAPSHOT_NONFINITE: {key}")


def evaluate_task8_resources(snapshot: Mapping[str, Any], *, phase: str = "formal",
                             starting_new_sample: bool = False,
                             remaining_samples: int | None = None,
                             mean_success_sample_bytes: int | None = None) -> dict[str, Any]:
    """Apply Task 7 GPU/RAM/cgroup/swap policy plus the immutable 5 GiB reserve."""
    if not isinstance(snapshot, Mapping):
        raise ResourceStop("MONITOR_FAILURE: snapshot is not a mapping")
    _required_resources(snapshot)
    observed = dict(snapshot)
    if remaining_samples is not None:
        if mean_success_sample_bytes is None:
            raise ResourceStop("DISK_FORECAST_UNAVAILABLE: no measured successful sample size")
        observed["remaining_samples"] = int(remaining_samples)
        observed["mean_success_sample_bytes"] = int(mean_success_sample_bytes)
    # Task 8 does not inherit Task 6's 10 GiB startup-disk gate.  The only
    # Task-8 disk invariant is the explicit 5 GiB reserve plus the 1.3x
    # completion forecast below.  The old GPU<=1 GiB preload check remains
    # limited to the model-preload phase.
    policy_phase = "resource-smoke" if phase in {"smoke", "resource-smoke"} else "pilot"
    decision = evaluate_resources(observed, phase=policy_phase, starting_new_sample=starting_new_sample)
    if decision.get("status") == "HARD_STOP":
        raise ResourceStop(f"{decision.get('reason_code')}: {decision.get('reason')}")
    disk = float(observed["disk_free_gib"])
    if phase == "preload" and float(observed["gpu_used_gib"]) > 1.0:
        raise ResourceStop("GPU_START_USED_HIGH: model preload requires GPU used memory <= 1 GiB")
    if starting_new_sample and disk < 5.0:
        raise ResourceStop("DISK_FREE_LOW: disk free space is below 5 GiB")
    if remaining_samples is not None and mean_success_sample_bytes is not None:
        forecast = disk - 1.3 * int(mean_success_sample_bytes) * int(remaining_samples) / 2**30
        observed["forecast_free_gib"] = forecast
        if forecast < 5.0:
            raise ResourceStop("DISK_FORECAST_LOW: forecast completion free space is below 5 GiB")
    return {"status": decision.get("status", "OK"), "reason_code": decision.get("reason_code"),
            "reason": decision.get("reason"), "snapshot": observed}


class Task8SampleStore:
    """Atomic sample publication with immutable successes and failed attempts."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_id(sample_id: str) -> str:
        if not sample_id or "/" in sample_id or "\\" in sample_id or ".." in sample_id:
            raise ValueError("unsafe Task 8 sample identifier")
        return sample_id

    def _path(self, sample_id: str) -> Path:
        return self.root / self._safe_id(sample_id)

    def prepare(self, sample_id: str, *, resume: bool = False) -> str:
        path = self._path(sample_id)
        if not path.exists():
            return "run"
        status_path = path / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        if status.get("status") == "success":
            self._verify(path)
            if not resume:
                raise FileExistsError(path)
            return "skip"
        if not resume:
            raise FileExistsError(path)
        index = 1
        while (self.root / f"{sample_id}.attempt.{index:03d}").exists(): index += 1
        path.rename(self.root / f"{sample_id}.attempt.{index:03d}")
        return "run"

    def _verify(self, path: Path) -> None:
        status = json.loads((path / "status.json").read_text(encoding="utf-8"))
        if not isinstance(status, Mapping) or status.get("status") != "success":
            raise ResumeMismatch(f"sample is not a published success: {path}")
        hashes = status.get("artifact_sha256")
        if not isinstance(hashes, Mapping) or "record.json" not in hashes:
            raise ResumeMismatch(f"sample artifact manifest is missing: {path}")
        for name, digest in hashes.items():
            relative = Path(str(name))
            if relative.is_absolute() or relative.name != str(name):
                raise ResumeMismatch(f"unsafe sample artifact path: {name!r}")
            artifact = path / name
            if not artifact.is_file() or _file_hash(artifact) != digest:
                raise ResumeMismatch(f"sample artifact hash mismatch: {artifact}")

    def write_success(self, sample_id: str, payload: Mapping[str, Any]) -> Path:
        destination = self._path(sample_id)
        if destination.exists():
            raise FileExistsError(destination)
        stage = Path(tempfile.mkdtemp(prefix=f".{sample_id}.", dir=str(self.root)))
        try:
            encoded, arrays = _encode_arrays(payload, "record")
            _atomic_json(stage / "record.json", encoded)
            for filename, array in arrays.items():
                with (stage / filename).open("wb") as stream:
                    np.save(stream, array, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
            hashes = {path.name: _file_hash(path) for path in stage.iterdir() if path.is_file()}
            _atomic_json(stage / "status.json", {"status": "success", "artifact_sha256": hashes})
            hashes["status.json"] = _file_hash(stage / "status.json")
            os.replace(stage, destination)
            return destination
        except BaseException:
            if stage.exists():
                failed = self.root / f"{sample_id}.failed-attempt.001"
                index = 1
                while failed.exists(): index += 1; failed = self.root / f"{sample_id}.failed-attempt.{index:03d}"
                stage.rename(failed)
            raise

    def load_record(self, sample_id: str) -> dict[str, Any]:
        path = self._path(sample_id); self._verify(path)
        return _decode_arrays(json.loads((path / "record.json").read_text(encoding="utf-8")), path)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _encode_arrays(value: Any, prefix: str) -> tuple[Any, dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    def encode(item: Any, name: str) -> Any:
        if isinstance(item, np.ndarray):
            filename = name.replace("/", "_") + ".npy"; arrays[filename] = np.array(item, copy=True)
            return {"artifact": filename, "dtype": str(item.dtype), "shape": list(item.shape)}
        if isinstance(item, Mapping): return {str(k): encode(v, f"{name}_{k}") for k, v in item.items()}
        if isinstance(item, (list, tuple)): return [encode(v, f"{name}_{i}") for i, v in enumerate(item)]
        return _json_safe(item)
    return encode(value, prefix), arrays


def _decode_arrays(value: Any, root: Path) -> Any:
    if isinstance(value, Mapping) and "artifact" in value:
        return np.load(root / str(value["artifact"]), allow_pickle=False)
    if isinstance(value, Mapping): return {k: _decode_arrays(v, root) for k, v in value.items()}
    if isinstance(value, list): return [_decode_arrays(v, root) for v in value]
    return value


def _result_identity(result: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and retain only small identities after an output is published."""
    sample_id = str(spec["sample_id"])
    full = _required_array(result, ("output_full",), sample_id=sample_id, label="full latent output")
    decoded = _required_array(result, ("generated_rgb", "decoded_generated_rgb", "decoded_frames"),
                               sample_id=sample_id, label="decoded floating output")
    condition = _required_array(result, ("condition_input_fp32", "condition_input", "condition_fp32"),
                                sample_id=sample_id, label="FP32 consumed condition")
    encoded_condition = _required_array(result, ("encoded_condition",), sample_id=sample_id,
                                        label="FP32 output encoded condition")
    action = _required_array(result, ("action",), sample_id=sample_id, label="requested action")
    action_hash = result.get("action_hash")
    if not isinstance(action_hash, str) or SHA256_RE.fullmatch(action_hash) is None:
        raise ResumeMismatch(f"{sample_id} lacks a SHA256 requested-action identity")
    if action_hash != _array_hash(action):
        raise ResumeMismatch(f"{sample_id} requested-action hash does not match the saved action array")
    noise = _noise_hash(result, sample_id=sample_id)
    token_hashes = _action_consumption(result, sample_id=sample_id)
    if not isinstance(result.get("precision"), Mapping):
        raise ResumeMismatch(f"{sample_id} lacks runtime precision evidence")
    decoded_last = None
    if "decoded_last_rgb" in result:
        decoded_last = _required_array(result, ("decoded_last_rgb",), sample_id=sample_id,
                                       label="decoded last RGB frame")
    condition_rgb = None
    if "condition_rgb" in result:
        condition_rgb = _required_array(result, ("condition_rgb",), sample_id=sample_id,
                                        label="condition source RGB")
    identity = {
        "full_latent_hash": _array_hash(full), "decoded_hash": _array_hash(decoded),
        "condition_hash": _array_hash(condition), "encoded_condition_hash": _array_hash(encoded_condition),
        "action_hash": action_hash,
        "action_array_hash": _array_hash(action), "noise_hash": noise,
        "token_hashes": token_hashes,
    }
    if decoded_last is not None:
        identity["decoded_last_hash"] = _array_hash(decoded_last)
    if condition_rgb is not None:
        identity["condition_rgb_hash"] = _array_hash(condition_rgb)
    return identity


def _validate_formal_result(result: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed when a formal call does not preserve its required evidence."""
    if not isinstance(result, Mapping):
        raise ResumeMismatch(f"{spec['sample_id']} executor result is not a mapping")
    identity = _result_identity(result, spec)
    source = spec.get("condition_source")
    provenance = result.get("provenance")
    if isinstance(provenance, Mapping) and provenance.get("condition_source") not in (None, source):
        raise ResumeMismatch(f"{spec['sample_id']} condition-source provenance differs from the call plan")
    if str(spec.get("call")) == "G0" or str(spec.get("call")) == "G0_repeat":
        if "decoded_last_hash" not in identity:
            raise ResumeMismatch(f"{spec['sample_id']} lacks G0 decoded-last-frame evidence")
    if str(spec.get("call")) == "AR2":
        if source != "g0_float_last_fp32" or "condition_rgb_hash" not in identity:
            raise ResumeMismatch("AR2 lacks its G0 FP32 output-frame source evidence")
    return identity


def _compare_repeat_identity(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    fields = ("full_latent_hash", "decoded_hash", "condition_hash", "encoded_condition_hash", "action_hash",
              "action_array_hash", "noise_hash")
    if any(reference.get(field) != candidate.get(field) for field in fields):
        raise ResumeMismatch("exact G0 repeat condition/action/noise/latent/decoded evidence differs")
    if tuple(reference.get("token_hashes", ())) != tuple(candidate.get("token_hashes", ())):
        raise ResumeMismatch("exact G0 repeat action-token evidence differs")


def _validate_second_chunk_pair(identities: Mapping[str, Mapping[str, Any]],
                                call_ids: Mapping[str, str] | None = None) -> None:
    ids = call_ids or {row["call"]: row["sample_id"] for row in FORMAL_CALL_PLAN}
    tf = identities.get(ids["TF2"])
    ar = identities.get(ids["AR2"])
    if tf is None or ar is None:
        return
    for field in ("noise_hash", "action_hash", "action_array_hash"):
        if tf.get(field) != ar.get(field):
            raise ResumeMismatch(f"TF2 and AR2 {field} evidence differs")
    if tuple(tf.get("token_hashes", ())) != tuple(ar.get("token_hashes", ())):
        raise ResumeMismatch("TF2 and AR2 consumed action-token hashes differ")


def _status_payload(*, status: str, binding: Mapping[str, Any], plan: Sequence[Mapping[str, Any]],
                    completed: Sequence[str], skipped: Sequence[str], sample_paths: Mapping[str, Path],
                    generation_started: bool, current_sample: str | None = None,
                    error: BaseException | None = None, resource_snapshot: Mapping[str, Any] | None = None,
                    previous_status: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "umi-task8-run-v2", "status": status,
        "binding": _json_safe(binding), "plan": _json_safe(list(plan)),
        "completed_samples": list(completed), "skipped_samples": list(skipped),
        "sample_paths": {key: str(path) for key, path in sorted(sample_paths.items())},
        "generation_started": bool(generation_started),
        "current_sample": current_sample, "resource_snapshot": _json_safe(dict(resource_snapshot or {})),
    }
    if previous_status is not None:
        payload["previous_status"] = previous_status
    if error is not None:
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
    return payload


def _monitor_snapshot(observed: Any) -> dict[str, Any]:
    if not isinstance(observed, Mapping):
        raise ResourceStop("MONITOR_FAILURE: monitor did not return a mapping")
    if observed.get("status") == "HARD_STOP" or observed.get("hard_stop") is True:
        raise ResourceStop(f"MONITOR_HARD_STOP: {observed.get('reason') or observed.get('reason_code') or 'unspecified'}")
    candidate = observed.get("snapshot") if "snapshot" in observed else observed
    if not isinstance(candidate, Mapping):
        raise ResourceStop("MONITOR_FAILURE: monitor snapshot is not a mapping")
    return dict(candidate)


def run_task8(run_dir: str | Path, *, execute_call: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
              release: bool = False, resume: bool = False, binding: Mapping[str, Any] | None = None,
              resource_snapshot: Mapping[str, Any] | None = None,
              resource_monitor: Any | None = None,
              input_batch: Task8InputAdapter | None = None,
              post_sample_callback: Callable[[str, int], Any] | None = None,
              plan: Sequence[Mapping[str, Any]] | None = None,
              allow_custom_plan: bool = False) -> dict[str, Any]:
    """Run only the four approved formal calls through an injected executor."""
    if not release:
        raise BlockedExecution("Task 8 formal generation requires main's explicit release")
    if execute_call is None or not callable(execute_call):
        raise BlockedExecution("Task 8 formal generation requires an injected single-runtime executor")
    root = Path(run_dir).resolve(); root.mkdir(parents=True, exist_ok=True)
    plan_rows = [dict(row) for row in (plan or FORMAL_CALL_PLAN)]
    if plan_rows != [dict(row) for row in FORMAL_CALL_PLAN]:
        if not allow_custom_plan:
            raise ResumeMismatch("Task 8 requires the immutable four-call formal plan")
        expected = [dict(row) for row in FORMAL_CALL_PLAN]
        if len(plan_rows) != 4 or [row.get("call") for row in plan_rows] != [row["call"] for row in expected]:
            raise ResumeMismatch("custom plan must preserve four-call order")
        for row, original in zip(plan_rows, expected):
            if any(row.get(key) != original[key] for key in ("condition_source", "action_source", "chunk_index")):
                raise ResumeMismatch("custom plan changes condition or action routing")
        first, second = plan_rows[0].get("seed"), plan_rows[2].get("seed")
        if (not isinstance(first, int) or not isinstance(second, int) or first < 0 or second < 0
                or first == second or plan_rows[1].get("seed") != first or plan_rows[3].get("seed") != second):
            raise ResumeMismatch("custom plan does not pair chunk seeds")
    call_ids = {str(row["call"]): str(row["sample_id"]) for row in plan_rows}
    if len(set(call_ids.values())) != 4 or any(not value or "/" in value or "\\" in value or ".." in value
                                          for value in call_ids.values()):
        raise ResumeMismatch("custom plan has unsafe or duplicate sample IDs")
    if binding is None:
        raise BlockedExecution("Task 8 release requires immutable code/model/config/data binding")
    binding_value = _validate_binding(binding)
    status_path = root / "run_status.json"
    prior: Mapping[str, Any] | None = None
    if status_path.is_file():
        try:
            loaded_status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ResumeMismatch("existing Task 8 run_status.json is invalid") from error
        if not isinstance(loaded_status, Mapping):
            raise ResumeMismatch("existing Task 8 run_status.json is not an object")
        prior = loaded_status
        if not resume:
            raise FileExistsError(root)
        if prior.get("binding") != _json_safe(binding_value) or prior.get("plan") != _json_safe(plan_rows):
            raise ResumeMismatch("resume binding or formal call plan differs")
    elif any(root.iterdir()):
        raise ResumeMismatch("new Task 8 run directory is non-empty")

    completed: list[str] = []
    skipped: list[str] = []
    sample_paths: dict[str, Path] = {}
    identities: dict[str, dict[str, Any]] = {}
    current_sample: str | None = None
    generation_started = bool(prior and prior.get("generation_started"))
    g0_condition_frame: np.ndarray | None = None
    status_kwargs = {"binding": binding_value, "plan": plan_rows, "completed": completed,
                     "skipped": skipped, "sample_paths": sample_paths,
                     "generation_started": generation_started,
                     "previous_status": prior.get("status") if prior else None}
    _atomic_json(status_path, _status_payload(status="RUNNING", **status_kwargs))
    try:
        if resource_snapshot is None and resource_monitor is None:
            raise ResourceStop("MONITOR_NO_SNAPSHOT: no resource snapshot was supplied")
        if resource_monitor is not None:
            resource_snapshot = _monitor_snapshot(resource_monitor.check(phase="formal", starting_new_sample=True))
        assert resource_snapshot is not None
        evaluate_task8_resources(resource_snapshot, phase="formal", starting_new_sample=True)
        store = Task8SampleStore(root / "samples")
        for ordinal, spec in enumerate(plan_rows):
            sample_id = str(spec["sample_id"]); current_sample = sample_id
            if resource_monitor is not None:
                observed_snapshot = _monitor_snapshot(resource_monitor.check(phase="formal", starting_new_sample=True))
                evaluate_task8_resources(observed_snapshot, phase="formal", starting_new_sample=True)
                resource_snapshot = observed_snapshot
            state = _status_payload(status="RUNNING", **{**status_kwargs, "current_sample": sample_id,
                                                           "generation_started": generation_started,
                                                           "resource_snapshot": resource_snapshot})
            _atomic_json(status_path, state)
            if store.prepare(sample_id, resume=resume) == "skip":
                loaded = store.load_record(sample_id)
                identity = _validate_formal_result(loaded, spec)
                if spec.get("call") == "G0":
                    g0_condition_frame = np.array(loaded["decoded_last_rgb"], copy=True)
                if spec.get("call") == "G0_repeat":
                    _compare_repeat_identity(identities[call_ids["G0"]], identity)
                if spec.get("call") == "AR2":
                    if identity.get("condition_hash") != identities[call_ids["G0"]].get("encoded_condition_hash"):
                        raise ResumeMismatch("AR2 FP32 condition input differs from G0's encoded output frame")
                identities[sample_id] = identity; sample_paths[sample_id] = store._path(sample_id)
                completed.append(sample_id); skipped.append(sample_id)
                del loaded
                current_sample = None
                _atomic_json(status_path, _status_payload(status="RUNNING", **{**status_kwargs,
                                                                                   "current_sample": None,
                                                                                   "resource_snapshot": resource_snapshot}))
                gc.collect()
                if post_sample_callback is not None:
                    post_sample_callback(sample_id, len(plan_rows) - ordinal - 1)
                continue
            request = dict(spec); request["ordinal"] = ordinal; request["runtime_contract"] = Task8RuntimeConfig().as_dict()
            if input_batch is not None:
                request["action"] = np.array(input_batch.actions[int(spec["chunk_index"])], copy=True)
                request["action_hash"] = _array_hash(request["action"])
                if spec.get("condition_source") == "real_x0":
                    request["condition_rgb"] = np.array(input_batch.rgb[0], copy=True)
                elif spec.get("condition_source") == "real_x16":
                    request["condition_rgb"] = np.array(input_batch.rgb[16], copy=True)
            if spec.get("condition_source") == "g0_float_last_fp32":
                if g0_condition_frame is None:
                    raise ResumeMismatch("AR2 requested before a saved G0 decoded FP32 output frame")
                request["condition_rgb"] = np.array(g0_condition_frame, copy=True)
            generation_started = True
            _atomic_json(status_path, _status_payload(status="RUNNING", **{**status_kwargs,
                                                                               "current_sample": sample_id,
                                                                               "generation_started": generation_started,
                                                                               "resource_snapshot": resource_snapshot}))
            result = execute_call(request)
            if not isinstance(result, Mapping):
                raise TypeError("Task 8 executor must return a mapping")
            result = dict(result)
            # The request frame is only source provenance; the executor must
            # still return its own FP32 encoded condition input evidence.
            if "condition_rgb" in request and "condition_rgb" not in result:
                result["condition_rgb"] = np.array(request["condition_rgb"], copy=True)
            result.update({"sample_id": sample_id, "call": spec.get("call"), "status": "success"})
            identity = _validate_formal_result(result, spec)
            if spec.get("call") == "G0":
                g0_condition_frame = np.array(result["decoded_last_rgb"], copy=True)
            if spec.get("call") == "G0_repeat":
                _compare_repeat_identity(identities[call_ids["G0"]], identity)
            if spec.get("call") == "AR2":
                if identity.get("condition_hash") != identities[call_ids["G0"]].get("encoded_condition_hash"):
                    raise ResumeMismatch("AR2 FP32 condition input differs from G0's encoded output frame")
            destination = store.write_success(sample_id, result)
            identities[sample_id] = identity; sample_paths[sample_id] = destination
            completed.append(sample_id)
            del result
            current_sample = None
            _atomic_json(status_path, _status_payload(status="RUNNING", **{**status_kwargs,
                                                                               "current_sample": None,
                                                                               "generation_started": generation_started,
                                                                               "resource_snapshot": resource_snapshot}))
            gc.collect()
            if post_sample_callback is not None:
                post_sample_callback(sample_id, len(plan_rows) - ordinal - 1)
        _validate_second_chunk_pair(identities, call_ids)
        final = _status_payload(status="COMPLETE", **{**status_kwargs, "current_sample": None,
                                                        "generation_started": generation_started,
                                                        "resource_snapshot": resource_snapshot})
        final.update({"formal_calls": len(plan_rows), "samples": sorted(sample_paths),
                      "generation_started": True})
        _atomic_json(status_path, final)
        return final
    except BaseException as error:
        failed = [current_sample] if current_sample is not None else []
        terminal_status = "RESOURCE_STOP" if isinstance(error, ResourceStop) else "FAILED"
        failure = _status_payload(status=terminal_status, **{**status_kwargs, "current_sample": current_sample,
                                                        "generation_started": generation_started,
                                                        "resource_snapshot": resource_snapshot,
                                                        "error": error})
        failure["failed_samples"] = failed
        try:
            _atomic_json(status_path, failure)
        except BaseException:
            pass
        raise
    finally:
        if g0_condition_frame is not None:
            del g0_condition_frame


__all__ = ["BlockedExecution", "ENGINEERING_SMOKE_PLAN", "FORMAL_CALL_PLAN", "ResourceStop",
           "ResumeMismatch", "Task8SampleStore", "build_formal_call_plan", "evaluate_task8_resources", "run_task8"]
