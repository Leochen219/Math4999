"""CPU-testable Task 6 runtime adapters and the atomic group runner.

The actual Cosmos implementation is injected by :class:`Task6RuntimeAdapter`.
This module owns the fixed inputs, provenance gates, call accounting and
durable publication; it never imports torch or starts model execution.
"""
from __future__ import annotations

import hashlib
import json
import numbers
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from .umi_fd_post_vae_bridge import sha256_array
    from .umi_precision_primitives import canonical_json, mask_geometry
    from .umi_precision_runtime import projection
    from .umi_precision_storage import ProcessLock, PrecisionSampleStore
    from .umi_task5_primitives import freeze_task5_directions
    from .umi_task6_primitives import (ALPHAS, DIRECTION_IDS, build_generation_plan,
        build_run_status, evaluate_resources, stable_hash, STATE_IDS, SEEDS)
except ImportError:
    from umi_fd_post_vae_bridge import sha256_array
    from umi_precision_primitives import canonical_json, mask_geometry
    from umi_precision_runtime import projection
    from umi_precision_storage import ProcessLock, PrecisionSampleStore
    from umi_task5_primitives import freeze_task5_directions
    from umi_task6_primitives import ALPHAS, DIRECTION_IDS, build_generation_plan, build_run_status, evaluate_resources, stable_hash, STATE_IDS, SEEDS


class PreflightError(ValueError):
    """Generation-free configuration or provenance mismatch."""


class ResourceStop(RuntimeError):
    """A resource gate stopped a run; completed samples remain valid."""


def _sha(value: Any) -> str:
    return stable_hash(value)


def _array_sha(value: Any) -> str:
    # Task 5 direction manifests use its established array hash format.
    return sha256_array(np.asarray(value))


def load_frozen_directions(direction_bank: Any, mask: Any, *, expected_hashes: Mapping[str, str] | None = None) -> dict[str, np.ndarray]:
    """Load exactly the three Task 5 vectors and derive its two frozen combos."""
    if isinstance(direction_bank, (str, os.PathLike)):
        path = Path(direction_bank)
        if not path.is_file():
            raise ValueError(f"direction bank is missing: {path}")
        direction_bank = np.load(path, allow_pickle=False)
        if isinstance(direction_bank, np.lib.npyio.NpzFile):
            keys = set(direction_bank.files)
            if keys != {"bank"}:
                raise ValueError("direction archive must contain exactly 'bank'")
            direction_bank = direction_bank["bank"]
    if isinstance(direction_bank, Mapping):
        if set(direction_bank) != {"v0", "v1", "v2"}:
            raise ValueError("direction mapping must contain exactly v0, v1, v2")
        direction_bank = np.stack([direction_bank[key] for key in ("v0", "v1", "v2")])
    bank = np.asarray(direction_bank)
    mask_array = np.asarray(mask, dtype=bool)
    if bank.ndim != mask_array.ndim + 1 or bank.shape[0] != 3 or tuple(bank.shape[1:]) != tuple(mask_array.shape):
        raise ValueError("Task 5 direction bank must contain exactly three arrays matching the runtime mask")
    frozen = freeze_task5_directions(bank, mask_array)
    result = {key: np.ascontiguousarray(frozen["directions"][key], dtype=np.float32) for key in DIRECTION_IDS}
    if expected_hashes is not None:
        if set(expected_hashes) != set(DIRECTION_IDS):
            raise ValueError("direction hash manifest must contain exactly the five frozen direction ids")
        for key, value in result.items():
            if _array_sha(value) != str(expected_hashes[key]):
                raise ValueError(f"direction hash mismatch: {key}")
    return result


class Task6Inputs:
    """Immutable state/action/prompt and runtime carrier geometry for one group."""
    def __init__(self, carrier: Any, condition_indexes: Any, packed_mask: Any, direction_bank: Any,
                 *, z_bar: Any | None = None, task4_reference: Mapping[str, Any] | None = None,
                 action: Any | None = None, prompt: str = "", state: str = "bridge_0", seed: int = 0,
                 direction_hashes: Mapping[str, str] | None = None):
        self.z0 = projection(carrier)
        self.geometry = mask_geometry(condition_indexes, packed_mask, self.z0.shape)
        self.z_bar = projection(self.z0 if z_bar is None else z_bar)
        if self.z_bar.shape != self.z0.shape:
            raise ValueError("Task 6 z_bar must match carrier shape")
        self.directions = load_frozen_directions(direction_bank, self.geometry.mask, expected_hashes=direction_hashes)
        self.direction_ids = tuple(DIRECTION_IDS)
        self.direction_bank_hash = _array_sha(np.stack([self.directions[key] for key in ("v0", "v1", "v2")]))
        self.action = None if action is None else np.ascontiguousarray(np.asarray(action, dtype=np.float32).copy())
        if self.action is not None and self.action.shape != (16, 10):
            raise ValueError("Task 6 action must have shape (16, 10)")
        self.prompt = str(prompt)
        if state not in STATE_IDS or isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or int(seed) not in SEEDS:
            raise ValueError("Task 6 inputs require an approved state and seed 0 or 1")
        self.state = str(state); self.seed = int(seed)
        self.task4_reference = json.loads(canonical_json(dict(task4_reference or {})))
        for value in (self.z0, self.z_bar, *self.directions.values()):
            value.setflags(write=False)

    def for_spec(self, spec: Mapping[str, Any]) -> np.ndarray:
        if str(spec.get("state", self.state)) != self.state or int(spec.get("seed", self.seed)) != self.seed:
            raise ValueError("sample spec is not bound to this Task 6 group")
        if spec.get("kind") == "baseline":
            if float(spec.get("alpha", 0.0)) != 0.0 or int(spec.get("sign", 0)) != 0:
                raise ValueError("baseline must have zero alpha and sign")
            return self.z_bar.copy()
        if spec.get("kind") != "perturbation" or spec.get("direction_id") not in self.directions:
            raise ValueError("unknown frozen Task 6 direction")
        alpha = float(spec.get("alpha")); sign = int(spec.get("sign", 0))
        if alpha not in ALPHAS or sign not in (-1, 1):
            raise ValueError("Task 6 amplitude/sign is outside the fixed plan")
        result = self.z_bar.astype(np.float64) + sign * alpha * self.directions[spec["direction_id"]].astype(np.float64)
        result = result.astype(np.float32)
        if not np.array_equal(result[~self.geometry.mask], self.z_bar[~self.geometry.mask]):
            raise ValueError("condition perturbation escaped runtime mask")
        return result

    def identity(self) -> dict[str, Any]:
        return {"z0": _sha(self.z0), "z_bar": _sha(self.z_bar), "geometry": self.geometry.metadata(),
                "directions": {key: _array_sha(value) for key, value in self.directions.items()},
                "direction_bank": self.direction_bank_hash, "action": None if self.action is None else _sha(self.action),
                "prompt": self.prompt, "state": self.state, "seed": self.seed, "task4_reference": self.task4_reference}


class Task6RuntimeAdapter:
    """Narrow adapter allowing one resident validated Task 5 runtime to be reused."""
    def __init__(self, runtime: Any, inputs: Task6Inputs):
        self.runtime = runtime
        self.inputs = inputs

    @property
    def provenance(self):
        return getattr(self.runtime, "provenance", {})

    @property
    def model_seed(self):
        return getattr(self.runtime, "model_seed", self.inputs.seed)

    def actual_identity(self):
        method = getattr(self.runtime, "actual_identity", None)
        return method() if callable(method) else {"runtime": type(self.runtime).__name__}

    def execute(self, spec: Mapping[str, Any], inputs: Task6Inputs | None = None, *, scope: str = "full"):
        target = inputs or self.inputs
        request = dict(spec)
        # The validated Task 5 seam names its single FP32 execution path C;
        # Task 6 state/seed are carried alongside it, never used to branch the
        # denoiser implementation.
        request.setdefault("group", "C")
        request.setdefault("model_seed", request.get("seed", self.inputs.seed))
        return self.runtime.execute(request, target, scope=scope)


Task6Runtime = Task6RuntimeAdapter


def preflight_task6(config: Mapping[str, Any], *, strict: bool = False) -> dict[str, Any]:
    """Validate generation-free evidence; no model/runtime calls are made."""
    if not isinstance(config, Mapping):
        raise PreflightError("preflight configuration must be a mapping")
    failures: list[str] = []
    required = ("environment", "provenance", "asset_hashes", "carrier_shape", "condition_indexes",
                "predicted_indexes", "mask_shape", "action_shape", "direction_hashes", "settings", "seed_config")
    if strict:
        failures.extend(key for key in required if key not in config)
    shape = config.get("carrier_shape")
    if shape is not None and list(shape) != [1, 48, 5, 16, 16]: failures.append("carrier_shape")
    if "condition_indexes" in config and list(config["condition_indexes"]) != [0]: failures.append("condition_indexes")
    if "predicted_indexes" in config and list(config["predicted_indexes"]) != [1, 2, 3, 4]: failures.append("predicted_indexes")
    if "mask_shape" in config and shape is not None and list(config["mask_shape"]) != list(shape): failures.append("mask_shape")
    if "action_shape" in config and list(config["action_shape"]) != [16, 10]: failures.append("action_shape")
    if "action" in config:
        try:
            if np.asarray(config["action"]).shape != (16, 10): failures.append("action")
        except Exception: failures.append("action")
    directions = config.get("direction_hashes")
    if directions is not None and set(directions) != set(DIRECTION_IDS): failures.append("direction_hashes")
    if directions is not None:
        for key, value in directions.items():
            if not isinstance(value, str) or len(value) != 64:
                failures.append(f"direction_hashes.{key}")
    settings = config.get("settings", {})
    expected_settings = {"num_steps": 30, "guidance": 1.0, "shift": 10.0, "autocast": False, "tf32": False, "diffusion_cache": False, "batch_size": 1}
    for key, expected in expected_settings.items():
        if key not in settings and strict: failures.append(f"settings.{key}")
        elif key in settings and settings[key] != expected: failures.append(f"settings.{key}")
    seed = config.get("seed_config", {})
    if strict and (not isinstance(seed, Mapping) or any(key not in seed for key in ("seed", "prepare", "sampler", "scheduler"))):
        failures.append("seed_config")
    if seed and (seed.get("prepare") != seed.get("seed") or seed.get("sampler") != seed.get("seed") or seed.get("scheduler") != seed.get("seed")):
        failures.append("seed_config")
    for key in ("environment", "provenance", "asset_hashes"):
        value = config.get(key)
        if value is not None and (not isinstance(value, Mapping) or (strict and not value)): failures.append(key)
    # Environment and asset identities are intentionally separate.  A label
    # such as ``task5_reuse`` is provenance for umi_reference, not a substitute
    # for the pinned Bridge video/action hashes.
    assets = config.get("asset_hashes")
    if isinstance(assets, Mapping):
        for key, value in assets.items():
            if not isinstance(value, str) or len(value) != 64: failures.append(f"asset_hashes.{key}")
    provenance = config.get("provenance")
    if isinstance(provenance, Mapping) and provenance.get("source_commit") is not None:
        if str(provenance.get("source_commit")) != "2b17a2413bd86b2cf9b03823637108851e4ddf2d":
            failures.append("provenance.source_commit")
    if failures:
        raise PreflightError("Task 6 preflight failed: " + ", ".join(dict.fromkeys(failures)))
    return {"status": "PASS", "generation_started": False, "checked": sorted(set(config.keys()))}


def verify_reference_reuse(runtime: Any, inputs: Any, *, execute: bool = True) -> dict[str, Any]:
    """Perform exactly four engineering-equivalence calls for reference reuse."""
    specs = [
        {"sample_id": "baseline_pre", "kind": "baseline", "alpha": 0.0, "sign": 0, "direction_id": None},
        {"sample_id": "baseline_post", "kind": "baseline", "alpha": 0.0, "sign": 0, "direction_id": None},
        {"sample_id": "v0_alpha_00_plus", "kind": "perturbation", "alpha": 0.001, "sign": 1, "direction_id": "v0"},
        {"sample_id": "v0_alpha_00_minus", "kind": "perturbation", "alpha": 0.001, "sign": -1, "direction_id": "v0"},
    ]
    records = []
    for spec in specs:
        if not execute: raise ValueError("reference reuse verification requires execution of all four calls")
        records.append(runtime.execute(spec, inputs, scope="equivalence"))
    def bits(record):
        raw = record.get("raw_outputs", record.get("raw")) if isinstance(record, Mapping) else record
        if isinstance(raw, Mapping): return tuple((str(k), np.asarray(v).dtype.str, np.asarray(v).shape, np.asarray(v).tobytes()) for k, v in sorted(raw.items()))
        array = np.asarray(raw)
        return (array.dtype.str, array.shape, array.tobytes())
    reusable = all(bits(records[index]) == bits(records[0]) for index in range(1, 4))
    return {"reusable": reusable, "calls": 4, "required_calls": [x["sample_id"] for x in specs],
            "reason": None if reusable else "engineering-equivalence mismatch; all 32 calls must be rerun"}


def _status_hashes(runtime: Any, inputs: Task6Inputs, config: Mapping[str, Any]) -> dict[str, str]:
    actual = runtime.actual_identity() if hasattr(runtime, "actual_identity") else {}
    return {"code": _sha(type(runtime).__name__), "model": _sha(actual), "config": _sha(config),
            "direction": _sha(inputs.identity()["directions"]), "input": _sha(inputs.identity()),
            "noise": _sha({"seed": inputs.seed, "strategy": "runtime_prepare_noise"})}


def _write_manifest(root: Path) -> None:
    """Write the root raw-evidence manifest without self-reference."""
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"MANIFEST.sha256", ".runner.lock"}:
            lines.append(f"{_file_sha(path)}  {path.relative_to(root).as_posix()}")
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def run_task6_group(runtime: Any, inputs: Task6Inputs, run_dir: str | Path, *, resume: bool = False,
                    resource_sampler: Any | None = None, stop_after: int | None = None) -> dict[str, Any]:
    """Execute one exact 32-call group serially with immutable samples."""
    root = Path(run_dir); root.mkdir(parents=True, exist_ok=True)
    group = (inputs.state, inputs.seed)
    plan = build_generation_plan(*group)
    config = {"schema_version": "umi-task6-v1", "group": {"state": inputs.state, "seed": inputs.seed},
              "plan": plan, "inputs": inputs.identity(), "provenance": getattr(runtime, "provenance", {}),
              "actual_runtime": runtime.actual_identity() if hasattr(runtime, "actual_identity") else {}}
    identity = _sha(config)
    plan_path = root / "task6_plan.json"
    if plan_path.exists():
        if not resume: raise FileExistsError("Task 6 run exists; explicit resume is required")
        if json.loads(plan_path.read_text(encoding="utf-8")) != json.loads(canonical_json(config)): raise ValueError("strict Task 6 plan identity mismatch")
    else:
        plan_path.write_text(canonical_json(config) + "\n", encoding="utf-8")
    history = root / "invocation_history.jsonl"
    samples = PrecisionSampleStore(root / "samples")
    completed: list[str] = []; failed: list[str] = []; skipped: list[str] = []
    status_path = root / "run_status.json"
    hashes = _status_hashes(runtime, inputs, config)
    def write_status(status, reason_code=None, reason=None):
        payload = build_run_status(status, reason_code=reason_code, reason=reason, completed=completed,
                                   failed=failed, skipped=skipped, hashes=hashes,
                                   group={"state": inputs.state, "seed": inputs.seed}, successful_samples=len(completed))
        status_path.write_text(canonical_json(payload) + "\n", encoding="utf-8"); return payload
    with ProcessLock(root / ".runner.lock"):
        # Successful samples are immutable and must match this run's strict
        # identity before any resume call is skipped.
        if (root / "samples").exists():
            for existing in (root / "samples").iterdir():
                if not existing.is_dir() or ".attempt." in existing.name: continue
                status_file = existing / "status.json"
                if status_file.is_file():
                    state = json.loads(status_file.read_text(encoding="utf-8"))
                    if state.get("status") == "success":
                        record_file = existing / "sample.json"
                        if not record_file.is_file(): raise ValueError("successful Task 6 sample is missing sample.json")
                        record = json.loads(record_file.read_text(encoding="utf-8"))
                        if record.get("identity") != identity: raise ValueError("strict Task 6 resume identity mismatch")
        for ordinal, spec in enumerate(plan):
            # Give incomplete prior attempts deterministic, reviewable names.
            sample_path = root / "samples" / spec["sample_id"]
            if resume and sample_path.is_dir() and (sample_path / "status.json").is_file():
                existing_state = json.loads((sample_path / "status.json").read_text(encoding="utf-8"))
                if existing_state.get("status") != "success":
                    attempt_no = 1
                    while (sample_path.parent / f"{sample_path.name}.attempt.{attempt_no}").exists(): attempt_no += 1
                    sample_path.rename(sample_path.parent / f"{sample_path.name}.attempt.{attempt_no}")
            disposition = samples.prepare(spec["sample_id"], resume=resume, required_files=("sample.json",))
            if disposition == "skip":
                skipped.append(spec["sample_id"]); completed.append(spec["sample_id"]); continue
            if stop_after is not None and len(completed) >= stop_after:
                payload = write_status("RESOURCE_STOP", "RESOURCE_STOP", "injected resource stop"); _write_manifest(root); return payload
            if resource_sampler is not None:
                decision = resource_sampler() if callable(resource_sampler) else resource_sampler
                if decision.get("status") == "HARD_STOP":
                    payload = write_status("RESOURCE_STOP", decision.get("reason_code"), decision.get("reason")); _write_manifest(root); return payload
            with history.open("a", encoding="utf-8") as stream:
                stream.write(canonical_json({"sample_id": spec["sample_id"], "identity": identity, "started_at_unix": time.time()}) + "\n")
            try:
                record = dict(runtime.execute(spec, inputs, scope="full"))
                if "output_full" not in record: raise ValueError("runtime did not return output_full")
                record.update({"status": "success", "identity": identity, "spec": spec})
                arrays = {key + ".npy": value for key, value in record.items() if isinstance(value, np.ndarray)}
                samples.write_success(spec["sample_id"], {key: value for key, value in record.items() if not isinstance(value, np.ndarray)}, artifacts=arrays)
                completed.append(spec["sample_id"])
            except KeyboardInterrupt:
                payload = write_status("INTERRUPTED", "SIGNAL_INTERRUPTED", "interrupt received"); _write_manifest(root); return payload
            except Exception as error:
                failed.append(spec["sample_id"])
                try:
                    samples.write_failure(spec["sample_id"], {"status": "fail", "identity": identity,
                        "spec": spec, "error": {"type": type(error).__name__, "message": str(error)}})
                except Exception:
                    # The terminal status remains authoritative even if a
                    # failure record cannot be published after a hard stop.
                    pass
                payload = write_status("FAILED", type(error).__name__, str(error)); _write_manifest(root); return payload
        if len(completed) != 32:
            payload = write_status("FAILED", "INCOMPLETE_GROUP", "group did not produce exactly 32 successes"); _write_manifest(root); return payload
        payload = write_status("AWAITING_REVIEW", "PILOT_COMPLETE", "32 generation successes; analysis pending")
        payload["analysis_pending"] = True
        status_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        _write_manifest(root)
        return payload


validate_task6_preflight = preflight_task6
load_task5_direction_bank = load_frozen_directions
run_task6_experiment = run_task6_group
verify_umi_reference_reuse = verify_reference_reuse

__all__ = ["PreflightError", "ResourceStop", "Task6Inputs", "Task6Runtime", "Task6RuntimeAdapter",
           "load_frozen_directions", "load_task5_direction_bank", "preflight_task6", "run_task6_experiment",
           "run_task6_group", "validate_task6_preflight", "verify_reference_reuse", "verify_umi_reference_reuse"]
