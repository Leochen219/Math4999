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
import gc
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from .umi_fd_post_vae_bridge import construct_delta, sha256_array
    from .umi_precision_primitives import canonical_json, mask_geometry
    from .umi_precision_runtime import projection
    from .umi_precision_storage import ProcessLock, PrecisionSampleStore
    from .umi_task5_primitives import freeze_task5_directions, rms64
    from .umi_task6_primitives import (ALPHAS, DIRECTION_IDS, build_generation_plan,
        build_run_status, evaluate_resources, stable_hash, STATE_IDS, SEEDS, parse_action)
except ImportError:
    from umi_fd_post_vae_bridge import construct_delta, sha256_array
    from umi_precision_primitives import canonical_json, mask_geometry
    from umi_precision_runtime import projection
    from umi_precision_storage import ProcessLock, PrecisionSampleStore
    from umi_task5_primitives import freeze_task5_directions, rms64
    from umi_task6_primitives import ALPHAS, DIRECTION_IDS, build_generation_plan, build_run_status, evaluate_resources, stable_hash, STATE_IDS, SEEDS, parse_action


class PreflightError(ValueError):
    """Generation-free configuration or provenance mismatch."""


class ResourceStop(RuntimeError):
    """A resource gate stopped a run; completed samples remain valid."""


def _sha(value: Any) -> str:
    return stable_hash(value)


def _array_sha(value: Any) -> str:
    # Task 5 direction manifests use its established array hash format.
    return sha256_array(np.asarray(value))


def build_task6_hash_binding(runtime: Any, inputs: Any, config: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Derive the six canonical hashes from actual runtime/input identities."""
    actual = runtime.actual_identity() if hasattr(runtime, "actual_identity") else {"runtime": type(runtime).__name__}
    source_root = Path(__file__).resolve().parent
    sources = [source_root / name for name in ("umi_task6_runtime.py", "run_umi_task6_experiment.py",
        "umi_task6_primitives.py", "umi_fd_post_vae_bridge.py",
        "umi_precision_runtime.py", "umi_precision_storage.py", "umi_precision_official.py",
        "umi_task5_runtime.py", "umi_task5_primitives.py") if (source_root / name).is_file()]
    code_digest = hashlib.sha256()
    for source in sources:
        code_digest.update(source.name.encode("utf-8")); code_digest.update(b"\0"); code_digest.update(source.read_bytes())
    identity = inputs.identity() if hasattr(inputs, "identity") else {}
    return {"code": code_digest.hexdigest(), "model": _sha(actual), "config": _sha(config or {}),
            "direction": _sha(identity.get("directions", {})), "input": _sha(identity),
            "noise": _sha({"seed": identity.get("seed"), "strategy": "runtime_prepare_noise", "routes": ["prepare", "sampler", "scheduler"]})}


def task6_binding_config(inputs: Any) -> dict[str, Any]:
    return {"schema_version": "umi-task6-v1", "group": {"state": inputs.state, "seed": inputs.seed},
            "settings": {"num_steps": 30, "guidance": 1.0, "shift": 10.0, "autocast": False,
                         "tf32": False, "diffusion_cache": False, "batch_size": 1}}




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
    if expected_hashes is None or set(expected_hashes) != set(DIRECTION_IDS):
        raise ValueError("direction hash manifest must contain exactly the five frozen direction ids")
    for key, value in result.items():
        if _array_sha(value) != str(expected_hashes[key]):
            raise ValueError(f"direction hash mismatch: {key}")
    return result


def _derive_frozen_directions_unpinned(direction_bank: Any, mask: Any) -> dict[str, np.ndarray]:
    """Test-fixture-only derivation; operational callers must use a manifest."""
    if isinstance(direction_bank, Mapping):
        direction_bank = np.stack([direction_bank[key] for key in ("v0", "v1", "v2")])
    bank = np.asarray(direction_bank)
    frozen = freeze_task5_directions(bank, np.asarray(mask, dtype=bool))
    return {key: np.ascontiguousarray(frozen["directions"][key], dtype=np.float32) for key in DIRECTION_IDS}


class Task6Inputs:
    """Immutable state/action/prompt and runtime carrier geometry for one group."""
    def __init__(self, carrier: Any, condition_indexes: Any, packed_mask: Any, direction_bank: Any,
                 *, z_bar: Any | None = None, task4_reference: Mapping[str, Any] | None = None,
                 action: Any | None = None, prompt: str = "", state: str = "bridge_0", seed: int = 0,
                 direction_hashes: Mapping[str, str] | None = None):
        self.z0 = projection(carrier)
        if self.z0.shape != (1, 48, 5, 16, 16):
            raise ValueError("Task 6 carrier must have exact shape (1, 48, 5, 16, 16)")
        self.geometry = mask_geometry(condition_indexes, packed_mask, self.z0.shape)
        if self.geometry.condition_indexes != (0,) or self.geometry.predicted_indexes != (1, 2, 3, 4):
            raise ValueError("Task 6 geometry requires condition index [0] and predicted indexes [1, 2, 3, 4]")
        self.z_bar = projection(self.z0 if z_bar is None else z_bar)
        if self.z_bar.shape != self.z0.shape:
            raise ValueError("Task 6 z_bar must match carrier shape")
        if direction_hashes is None:
            raise ValueError("Task 6 requires the complete Task 5 direction hash manifest")
        self.directions = load_frozen_directions(direction_bank, self.geometry.mask, expected_hashes=direction_hashes)
        self.s_z = rms64(self.z_bar[self.geometry.mask])
        if not np.isfinite(self.s_z) or self.s_z == 0.0:
            raise ValueError("Task 6 z_bar masked RMS must be finite and nonzero")
        self.direction_ids = tuple(DIRECTION_IDS)
        self.direction_bank_hash = _array_sha(np.stack([self.directions[key] for key in ("v0", "v1", "v2")]))
        if action is None: raise ValueError("Task 6 action is required")
        self.action = parse_action(action)
        if not isinstance(prompt, str) or not prompt.strip(): raise ValueError("Task 6 prompt must be a nonempty string")
        self.prompt = prompt
        if state not in STATE_IDS or isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or int(seed) not in SEEDS:
            raise ValueError("Task 6 inputs require an approved state and seed 0 or 1")
        self.state = str(state); self.seed = int(seed)
        self.task4_reference = json.loads(canonical_json(dict(task4_reference or {})))
        for value in (self.z0, self.z_bar, *self.directions.values()):
            value.setflags(write=False)

    def for_spec(self, spec: Mapping[str, Any]) -> np.ndarray:
        raw_state = spec.get("state", self.state); raw_seed = spec.get("seed", self.seed)
        if not isinstance(raw_state, str) or isinstance(raw_seed, bool) or not isinstance(raw_seed, numbers.Integral):
            raise ValueError("sample spec state/seed types are invalid")
        if raw_state != self.state or int(raw_seed) != self.seed:
            raise ValueError("sample spec is not bound to this Task 6 group")
        if spec.get("kind") == "baseline":
            alpha = spec.get("alpha", 0.0); sign = spec.get("sign", 0)
            if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real) or isinstance(sign, bool) or not isinstance(sign, numbers.Integral) or float(alpha) != 0.0 or int(sign) != 0:
                raise ValueError("baseline must have zero alpha and sign")
            return self.z_bar.copy()
        if spec.get("kind") != "perturbation" or spec.get("direction_id") not in self.directions:
            raise ValueError("unknown frozen Task 6 direction")
        raw_alpha = spec.get("alpha"); raw_sign = spec.get("sign", 0)
        if isinstance(raw_alpha, bool) or not isinstance(raw_alpha, numbers.Real):
            raise ValueError("Task 6 alpha must be a numeric value from the fixed grid")
        if isinstance(raw_sign, bool) or not isinstance(raw_sign, numbers.Integral):
            raise ValueError("Task 6 sign must be integer -1 or +1")
        alpha = float(raw_alpha); sign = int(raw_sign)
        if alpha not in ALPHAS or sign not in (-1, 1):
            raise ValueError("Task 6 amplitude/sign is outside the fixed plan")
        result = construct_delta(self.z_bar, self.geometry.mask, self.directions[spec["direction_id"]],
                                 alpha=alpha, sign=sign).latent
        if not np.array_equal(result[~self.geometry.mask], self.z_bar[~self.geometry.mask]):
            raise ValueError("condition perturbation escaped runtime mask")
        return result

    def identity(self) -> dict[str, Any]:
        return {"z0": _sha(self.z0), "z_bar": _sha(self.z_bar), "geometry": self.geometry.metadata(),
                "directions": {key: _array_sha(value) for key, value in self.directions.items()},
                "direction_bank": self.direction_bank_hash, "action": None if self.action is None else _sha(self.action),
                "prompt": self.prompt, "state": self.state, "seed": self.seed, "s_z": self.s_z,
                "task4_reference": self.task4_reference}


class Task6RuntimeAdapter:
    """Narrow adapter allowing one resident validated Task 5 runtime to be reused."""
    def __init__(self, runtime: Any, inputs: Task6Inputs):
        self.runtime = runtime
        self.inputs = inputs
        # OfficialPrecisionRuntime validates identity against the inputs it
        # constructed during its generation-free prepare seam.  Reuse that
        # exact object (or a caller-supplied factory result) rather than
        # passing a Task6Inputs wrapper with a different schema.
        runtime_inputs = getattr(runtime, "inputs", None)
        if runtime_inputs is None:
            raise ValueError("wrapped runtime must expose inputs created with Task6 inputs_factory")
        if runtime_inputs is not None and hasattr(runtime_inputs, "identity") and hasattr(inputs, "identity"):
            if runtime_inputs.identity() != inputs.identity():
                raise ValueError("runtime inputs identity differs; construct OfficialPrecisionRuntime with Task6 inputs_factory")
        self.runtime_inputs = inputs

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
        if target is not self.inputs and target.identity() != self.inputs.identity():
            raise ValueError("Task 6 adapter inputs do not match bound group")
        request = dict(spec)
        # The validated Task 5 seam names its single FP32 execution path C;
        # Task 6 state/seed are carried alongside it, never used to branch the
        # denoiser implementation.
        request.setdefault("group", "C")
        request.setdefault("model_seed", request.get("seed", self.inputs.seed))
        if scope not in ("full", "module"):
            raise ValueError("official runtime supports only full or module scope")
        record = dict(self.runtime.execute(request, target, scope=scope))
        consumed = target.for_spec({**request, "state": target.state, "seed": target.seed})
        actual_delta = np.subtract(consumed, target.z_bar, dtype=np.float32)
        direction = np.zeros_like(target.z_bar, dtype=np.float32)
        expected_delta = np.zeros_like(target.z_bar, dtype=np.float32)
        if spec.get("kind") == "perturbation":
            direction = np.array(target.directions[spec["direction_id"]], dtype=np.float32, copy=True)
            expected_delta = np.multiply(np.float32(spec["sign"] * float(spec["alpha"]) * target.s_z), direction, dtype=np.float32)
            expected_delta[~target.geometry.mask] = 0.0
        fields = {"z_bar": np.array(target.z_bar, dtype=np.float32, copy=True), "mask": np.array(target.geometry.mask, dtype=bool, copy=True),
                  "direction": direction, "actual_delta_fp32": actual_delta, "target_delta_fp32": expected_delta,
                  "s_z": float(target.s_z), "spec": dict(spec), "group": {"state": target.state, "seed": target.seed}, "seed": target.seed, "model_seed": target.seed}
        for name, value in fields.items():
            if name in record:
                existing = record[name]
                if isinstance(value, np.ndarray):
                    if not np.array_equal(np.asarray(existing), value): raise ValueError(f"runtime returned mismatched Task 6 field: {name}")
                elif existing != value: raise ValueError(f"runtime returned mismatched Task 6 field: {name}")
            record[name] = value
        if "predicted_latent" not in record:
            if "output_full" not in record: raise ValueError("runtime record lacks predicted latent/output_full")
            record["predicted_latent"] = np.array(record["output_full"], dtype=np.float32, copy=True)
        else:
            record["predicted_latent"] = np.array(record["predicted_latent"], dtype=np.float32, copy=True)
        return record

    def cleanup(self) -> None:
        """Release per-call state while retaining the resident validated model."""
        method = getattr(self.runtime, "cleanup", None) or getattr(self.runtime, "reset_cache", None)
        if callable(method): method()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except (ImportError, AttributeError):
            pass
        cache = getattr(self.runtime, "request_cache", None)
        if cache:
            raise RuntimeError("runtime request cache remains after cleanup")


Task6Runtime = Task6RuntimeAdapter


def preflight_task6(config: Mapping[str, Any], *, strict: bool = False, runtime: Any | None = None,
                    inputs: Any | None = None) -> dict[str, Any]:
    """Validate generation-free evidence; no model/runtime calls are made."""
    if not isinstance(config, Mapping):
        raise PreflightError("preflight configuration must be a mapping")
    failures: list[str] = []
    if strict and (runtime is None or inputs is None): failures.append("actual_runtime_inputs")
    if runtime is not None and inputs is not None:
        observed_runtime = runtime.actual_identity() if hasattr(runtime, "actual_identity") else None
        observed_inputs = inputs.identity() if hasattr(inputs, "identity") else None
        expected_runtime = config.get("observed_runtime_identity", config.get("runtime_identity"))
        expected_inputs = config.get("observed_input_identity", config.get("input_identity"))
        if expected_runtime != observed_runtime: failures.append("runtime_identity")
        if expected_inputs != observed_inputs: failures.append("input_identity")
        if config.get("prompt") is not None and config.get("prompt") != getattr(inputs, "prompt", None): failures.append("prompt")
        if config.get("action_hash") is not None and config.get("action_hash") != observed_inputs.get("action"): failures.append("action_hash")
        if config.get("direction_hashes") is not None and config.get("direction_hashes") != observed_inputs.get("directions"): failures.append("direction_hashes")
        if config.get("carrier_hash") is not None and config.get("carrier_hash") != observed_inputs.get("z0"): failures.append("carrier_hash")
        if config.get("z_bar_hash") is not None and config.get("z_bar_hash") != observed_inputs.get("z_bar"): failures.append("z_bar_hash")
        if config.get("mask_hash") is not None and config.get("mask_hash") != inputs.geometry.metadata().get("mask_sha256"): failures.append("mask_hash")
        if strict and config.get("action_hash") != observed_inputs.get("action"): failures.append("action_hash")
        if strict and config.get("seed_config", {}).get("seed") != observed_inputs.get("seed"): failures.append("seed_config")
    required = ("environment", "provenance", "asset_hashes", "carrier_shape", "condition_indexes",
                "predicted_indexes", "mask_shape", "action_shape", "direction_hashes", "settings", "seed_config")
    if strict:
        failures.extend(key for key in required if key not in config)
        failures.extend(key for key in ("asset_paths",
                                         "observed_runtime_identity", "observed_input_identity") if key not in config)
        if "direction_bank_path" not in config: failures.append("direction_bank_path")
    shape = config.get("carrier_shape")
    if "carrier" in config:
        try:
            actual_shape = list(projection(config["carrier"]).shape)
            if shape is not None and actual_shape != list(shape): failures.append("carrier_shape")
            shape = actual_shape
        except Exception: failures.append("carrier")
    if shape is not None and list(shape) != [1, 48, 5, 16, 16]: failures.append("carrier_shape")
    if "condition_indexes" in config and list(config["condition_indexes"]) != [0]: failures.append("condition_indexes")
    if "predicted_indexes" in config and list(config["predicted_indexes"]) != [1, 2, 3, 4]: failures.append("predicted_indexes")
    if "mask_shape" in config and shape is not None and list(config["mask_shape"]) != list(shape): failures.append("mask_shape")
    if strict and "mask_hash" not in config: failures.append("mask_hash")
    if "action_shape" in config and list(config["action_shape"]) != [16, 10]: failures.append("action_shape")
    if "action" in config:
        try:
            action = parse_action(config["action"])
            if "action_hash" in config and _array_sha(action) != str(config["action_hash"]): failures.append("action_hash")
        except Exception: failures.append("action")
    if strict and not str(config.get("prompt", "")).strip(): failures.append("prompt")
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
    if strict and (not isinstance(assets, Mapping) or not assets): failures.append("asset_hashes")
    if isinstance(assets, Mapping):
        for key, value in assets.items():
            if not isinstance(value, str) or len(value) != 64: failures.append(f"asset_hashes.{key}")
    asset_paths = config.get("asset_paths", config.get("actual_asset_paths"))
    if strict and (not isinstance(asset_paths, Mapping) or not asset_paths): failures.append("asset_paths")
    if isinstance(asset_paths, Mapping):
        if not isinstance(assets, Mapping): failures.append("asset_hashes")
        for key, raw_path in asset_paths.items():
            path = Path(raw_path)
            if not path.is_file(): failures.append(f"asset_paths.{key}"); continue
            if assets.get(key) != _file_sha(path): failures.append(f"asset_hashes.{key}")
    for key in ("checkpoint_path", "vae_path"):
        raw_path = config.get(key)
        if raw_path is not None:
            path = Path(raw_path)
            if not path.is_file(): failures.append(key)
            elif isinstance(assets, Mapping) and assets.get(key.removesuffix("_path")) != _file_sha(path): failures.append(f"asset_hashes.{key.removesuffix('_path')}")
    if "direction_bank_path" in config:
        try:
            expected = config.get("direction_hashes")
            if expected is None: failures.append("direction_hashes")
            else:
                bank_path = Path(config["direction_bank_path"])
                if not bank_path.is_file(): raise ValueError("direction bank missing")
                file_hash = config.get("direction_bank_file_hash", config.get("direction_bank_hash"))
                if strict and (not isinstance(file_hash, str) or _file_sha(bank_path) != file_hash): raise ValueError("direction bank file hash mismatch")
                if "mask" not in config and inputs is not None: mask = inputs.geometry.mask
                else: mask = config["mask"]
                load_frozen_directions(bank_path, mask, expected_hashes=expected)
        except Exception: failures.append("direction_bank")
    provenance = config.get("provenance")
    if isinstance(provenance, Mapping) and provenance.get("source_commit") is not None:
        if str(provenance.get("source_commit")) != "2b17a2413bd86b2cf9b03823637108851e4ddf2d":
            failures.append("provenance.source_commit")
    for section in (config.get("environment"), provenance):
        if isinstance(section, Mapping):
            if section.get("source_commit") is not None and str(section["source_commit"]) != "2b17a2413bd86b2cf9b03823637108851e4ddf2d": failures.append("source_commit")
            if section.get("expected_framework_commit") is not None and section.get("framework_commit") != section.get("expected_framework_commit"): failures.append("framework_commit")
    if failures:
        raise PreflightError("Task 6 preflight failed: " + ", ".join(dict.fromkeys(failures)))
    return {"status": "PASS", "generation_started": False, "checked": sorted(set(config.keys()))}


def verify_reference_reuse(runtime: Any, inputs: Any, *, reference_artifacts: str | os.PathLike[str] | Mapping[str, Any] | None = None,
                           reference_manifest: str | os.PathLike[str] | Mapping[str, str] | None = None, execute: bool = True) -> dict[str, Any]:
    """Perform exactly four engineering-equivalence calls for reference reuse."""
    specs = [
        {"sample_id": "baseline_pre", "kind": "baseline", "alpha": 0.0, "sign": 0, "direction_id": None},
        {"sample_id": "baseline_post", "kind": "baseline", "alpha": 0.0, "sign": 0, "direction_id": None},
        {"sample_id": "v0_alpha_00_plus", "kind": "perturbation", "alpha": 0.001, "sign": 1, "direction_id": "v0"},
        {"sample_id": "v0_alpha_00_minus", "kind": "perturbation", "alpha": 0.001, "sign": -1, "direction_id": "v0"},
    ]
    if isinstance(reference_artifacts, Mapping) or not isinstance(reference_artifacts, (str, os.PathLike)):
        raise ValueError("reference artifacts must be one on-disk Task 5 sample root")
    reference_root = Path(reference_artifacts)
    if not reference_root.is_dir(): raise ValueError("Task 5 reference sample root is missing")
    if isinstance(reference_manifest, (str, os.PathLike)):
        manifest_path = Path(reference_manifest)
        if not manifest_path.is_file(): raise ValueError("Task 5 reference manifest is missing")
        entries = {}
        for line in manifest_path.read_text(encoding="ascii").splitlines():
            parts = line.split(None, 1)
            if len(parts) != 2 or len(parts[0]) != 64: raise ValueError("malformed Task 5 reference manifest")
            entries[parts[1]] = parts[0]
        reference_manifest = entries
    if not isinstance(reference_manifest, Mapping):
        raise ValueError("Task 5 reference manifest hashes are required")
    saved: dict[str, Any] = {}
    for spec in specs:
        path = reference_root / spec["sample_id"]
        if path.is_dir():
            record = json.loads((path / "sample.json").read_text(encoding="utf-8"))
            state = json.loads((path / "status.json").read_text(encoding="utf-8"))
            if state.get("status") != "success": raise ValueError(f"reference sample is not successful: {spec['sample_id']}")
            for filename, digest in state.get("artifact_sha256", {}).items():
                artifact = path / filename
                if not artifact.is_file() or _file_sha(artifact) != digest: raise ValueError(f"reference artifact hash mismatch: {spec['sample_id']}")
                rel = str(artifact.relative_to(reference_root)).replace("\\", "/")
                expected_digest = reference_manifest.get(rel, reference_manifest.get(str(artifact), reference_manifest.get(artifact.as_posix())))
                if expected_digest is None or expected_digest != digest: raise ValueError(f"reference manifest hash mismatch: {spec['sample_id']}")
            def decode(value):
                if isinstance(value, Mapping) and isinstance(value.get("artifact"), str):
                    return np.load(path / value["artifact"], allow_pickle=False)
                if isinstance(value, Mapping): return {key: decode(item) for key, item in value.items()}
                if isinstance(value, list): return [decode(item) for item in value]
                return value
            decoded = decode(record)
            decoded.update({p.stem: np.load(p, allow_pickle=False) for p in path.glob("*.npy")})
            saved[spec["sample_id"]] = decoded
        else:
            raise ValueError(f"reference sample directory is missing: {path}")
    records = []
    for spec in specs:
        if not execute: raise ValueError("reference reuse verification requires execution of all four calls")
        records.append(runtime.execute({**spec, "group": "C", "model_seed": spec.get("seed", 0)}, inputs, scope="full"))
    def bits(record):
        if isinstance(record, Mapping):
            required = {"common_input_fp32", "initial_state", "consumed_initial_state", "sampler_input_state", "output_full"}
            if not required.issubset(record):
                nested = [value for value in record.values() if isinstance(value, Mapping)]
                for candidate in nested:
                    if required.issubset(candidate): return bits(candidate)
            if not required.issubset(record): return ("MISSING_REQUIRED_RAW_FIELDS", tuple(sorted(required - set(record))))
            optional = ["decoded_final"] if "decoded_final" in record else []
            raw = tuple((key, np.asarray(record[key]).dtype.str, np.asarray(record[key]).shape, np.asarray(record[key]).tobytes()) for key in sorted(required | set(optional)))
            return raw
        return ("MISSING_REQUIRED_RAW_FIELDS",)
    evidence = {}
    for spec, record in zip(specs, records):
        current = bits(record); reference = bits(saved[spec["sample_id"]])
        evidence[spec["sample_id"]] = {"bitwise_equal": current == reference}
    reusable = all(row["bitwise_equal"] for row in evidence.values())
    return {"reusable": reusable, "calls": 4, "required_calls": [x["sample_id"] for x in specs], "evidence": evidence,
            "reason": None if reusable else "engineering-equivalence mismatch; all 32 calls must be rerun"}


def _status_hashes(runtime: Any, inputs: Task6Inputs, config: Mapping[str, Any]) -> dict[str, str]:
    return build_task6_hash_binding(runtime, inputs, config)


def _write_manifest(root: Path) -> None:
    """Write the root raw-evidence manifest without self-reference."""
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"MANIFEST.sha256", ".runner.lock"}:
            lines.append(f"{_file_sha(path)}  {path.relative_to(root).as_posix()}")
    _atomic_text(root / "MANIFEST.sha256", "\n".join(lines) + "\n")


def _validate_immutable_manifest(root: Path) -> None:
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file(): return
    mutable = {"run_status.json", "invocation_history.jsonl", "gpu_samples.csv", "ram_samples.csv", "disk_samples.csv", "sample_resource_snapshots.csv", "sample_resource_snapshots.jsonl"}
    for line in manifest.read_text(encoding="ascii").splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2: raise ValueError("malformed immutable manifest")
        digest, rel = parts
        rel = rel.strip()
        if Path(rel).name in mutable: continue
        path = root / rel
        if not path.is_file() or _file_sha(path) != digest: raise ValueError(f"immutable manifest mismatch: {rel}")


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _run_task6_group(runtime: Any, inputs: Task6Inputs, run_dir: str | Path, *, resume: bool = False,
                    authorization: Mapping[str, Any], monitor: Any) -> dict[str, Any]:
    """Execute one exact 32-call group serially with immutable samples."""
    root = Path(run_dir); root.mkdir(parents=True, exist_ok=True)
    group = (inputs.state, inputs.seed)
    plan = build_generation_plan(*group)
    if not isinstance(authorization, Mapping) or authorization.get("status") != "AWAITING_RESOURCE_REVIEW" or authorization.get("smoke_decision_accepted") is not True:
        raise ValueError("generation requires an accepted resource-smoke authorization")
    config_binding = task6_binding_config(inputs)
    expected_binding = build_task6_hash_binding(runtime, inputs, config_binding)
    if dict(authorization.get("hashes", {})) != expected_binding:
        raise ValueError("accepted resource-smoke hash binding differs from current runtime/input/config")
    if monitor is None: raise ValueError("generation requires an integrated resource monitor")
    config = {**config_binding,
              "plan": plan, "inputs": inputs.identity(), "provenance": getattr(runtime, "provenance", {}),
              "actual_runtime": runtime.actual_identity() if hasattr(runtime, "actual_identity") else {}}
    identity = _sha(config)
    plan_path = root / "task6_plan.json"
    if plan_path.exists():
        if not resume: raise FileExistsError("Task 6 run exists; explicit resume is required")
        if json.loads(plan_path.read_text(encoding="utf-8")) != json.loads(canonical_json(config)): raise ValueError("strict Task 6 plan identity mismatch")
    else:
        _atomic_text(plan_path, canonical_json(config) + "\n")
    history = root / "invocation_history.jsonl"
    samples = PrecisionSampleStore(root / "samples")
    completed: list[str] = []; failed: list[str] = []; skipped: list[str] = []
    status_path = root / "run_status.json"
    hashes = _status_hashes(runtime, inputs, config_binding)
    monitor_stopped = False
    failure_evidence: list[dict[str, Any]] = []
    def write_status(status, reason_code=None, reason=None):
        nonlocal monitor_stopped
        if monitor is not None and not monitor_stopped and hasattr(monitor, "stop"):
            try: monitor.stop()
            except Exception as error:
                monitor_stopped = True
                failure_evidence.append({"kind": "monitor_stop", "type": type(error).__name__, "message": str(error)})
                if not (status == "RESOURCE_STOP" and reason_code == "CUDA_OOM"):
                    status, reason_code, reason = "RESOURCE_STOP", "MONITOR_FAILURE", str(error)
            else: monitor_stopped = True
        last_resources = {}
        if monitor is not None:
            last_resources = getattr(monitor, "last_resources", {}) or {}
        smoke_run_id = None
        if status_path.is_file():
            try: smoke_run_id = json.loads(status_path.read_text(encoding="utf-8")).get("smoke_run_id")
            except (OSError, ValueError, json.JSONDecodeError): smoke_run_id = None
        if smoke_run_id is None:
            acceptance_path = root / "smoke_acceptance.json"
            if acceptance_path.is_file():
                try: smoke_run_id = json.loads(acceptance_path.read_text(encoding="utf-8")).get("smoke_run_id")
                except (OSError, ValueError, json.JSONDecodeError): smoke_run_id = None
        payload = build_run_status(status, reason_code=reason_code, reason=reason, completed=completed,
                                   failed=failed, skipped=skipped, hashes=hashes,
                                   group={"state": inputs.state, "seed": inputs.seed}, successful_samples=len(completed),
                                   last_resource_snapshots=last_resources, smoke_run_id=smoke_run_id,
                                   secondary_errors=failure_evidence)
        _atomic_text(status_path, canonical_json(payload) + "\n"); return payload
    with ProcessLock(root / ".runner.lock"):
        if resume and (root / "task6_plan.json").is_file() and not (root / "MANIFEST.sha256").is_file():
            raise ValueError("Task 6 resume requires the immutable root MANIFEST.sha256")
        _validate_immutable_manifest(root)
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
        if monitor is not None and hasattr(monitor, "start"):
            try:
                monitor.start()
            except Exception as error:
                payload = write_status("RESOURCE_STOP", "MONITOR_FAILURE", str(error)); _write_manifest(root); return payload
        for ordinal, spec in enumerate(plan):
            # Give incomplete prior attempts deterministic, reviewable names.
            sample_path = root / "samples" / spec["sample_id"]
            try:
                if resume and sample_path.is_dir() and (sample_path / "status.json").is_file():
                    existing_state = json.loads((sample_path / "status.json").read_text(encoding="utf-8"))
                    if existing_state.get("status") != "success":
                        attempt_no = 1
                        while (sample_path.parent / f"{sample_path.name}.attempt.{attempt_no}").exists(): attempt_no += 1
                        sample_path.rename(sample_path.parent / f"{sample_path.name}.attempt.{attempt_no}")
            except Exception as error:
                payload = write_status("FAILED", type(error).__name__, str(error)); _write_manifest(root); return payload
            try:
                disposition = samples.prepare(spec["sample_id"], resume=resume, required_files=("sample.json",))
            except Exception as error:
                failed.append(spec["sample_id"])
                payload = write_status("FAILED", type(error).__name__, str(error)); _write_manifest(root); return payload
            if disposition == "skip":
                skipped.append(spec["sample_id"]); completed.append(spec["sample_id"]); continue
            if monitor is not None and getattr(monitor, "failure", None) is not None:
                payload = write_status("RESOURCE_STOP", "MONITOR_FAILURE", "resource monitor failed"); _write_manifest(root); return payload
            if monitor is not None and hasattr(monitor, "check"):
                try:
                    decision = monitor.check(phase="pilot", starting_new_sample=True,
                                             remaining_samples=32 - len(completed), run_dir=root)
                except Exception as error:
                    payload = write_status("RESOURCE_STOP", "MONITOR_FAILURE", str(error)); _write_manifest(root); return payload
                if decision.get("status") == "HARD_STOP":
                    payload = write_status("RESOURCE_STOP", decision.get("reason_code"), decision.get("reason")); _write_manifest(root); return payload
            if monitor is not None and hasattr(monitor, "capture_sample"):
                try: capture_decision = monitor.capture_sample(spec["sample_id"], "pre_sample", 32 - len(completed), root)
                except Exception as error:
                    payload = write_status("RESOURCE_STOP", "MONITOR_FAILURE", str(error)); _write_manifest(root); return payload
                if capture_decision.get("decision_status") == "HARD_STOP":
                    payload = write_status("RESOURCE_STOP", capture_decision.get("reason_code"), "resource gate stopped before sample"); _write_manifest(root); return payload
            try:
                with history.open("a", encoding="utf-8") as stream:
                    stream.write(canonical_json({"sample_id": spec["sample_id"], "identity": identity, "started_at_unix": time.time()}) + "\n")
            except Exception as error:
                payload = write_status("FAILED", type(error).__name__, str(error)); _write_manifest(root); return payload
            cleanup_attempted = False; cleanup_failure = None
            try:
                record = dict(runtime.execute(spec, inputs, scope="full"))
                if "output_full" not in record: raise ValueError("runtime did not return output_full")
                record.update({"status": "success", "identity": identity, "spec": spec})
                arrays = {key + ".npy": value for key, value in record.items() if isinstance(value, np.ndarray)}
                samples.write_success(spec["sample_id"], {key: value for key, value in record.items() if not isinstance(value, np.ndarray)}, artifacts=arrays)
                cleanup = getattr(runtime, "cleanup", None) or getattr(runtime, "reset_cache", None)
                if callable(cleanup):
                    cleanup_attempted = True
                    try: cleanup()
                    except Exception as error:
                        cleanup_failure = error; raise
                completed.append(spec["sample_id"])
                if monitor is not None and hasattr(monitor, "capture_sample"):
                    try: capture_decision = monitor.capture_sample(spec["sample_id"], "post_cleanup", 32 - len(completed), root)
                    except Exception as error:
                        payload = write_status("RESOURCE_STOP", "MONITOR_FAILURE", str(error)); _write_manifest(root); return payload
                    if capture_decision.get("decision_status") == "HARD_STOP":
                        payload = write_status("RESOURCE_STOP", capture_decision.get("reason_code"), "resource cleanup growth exceeded gate"); _write_manifest(root); return payload
                if monitor is not None and hasattr(monitor, "check"):
                    try:
                        decision = monitor.check(phase="pilot", starting_new_sample=False,
                                                 remaining_samples=32 - len(completed), run_dir=root)
                    except Exception as error:
                        payload = write_status("RESOURCE_STOP", "MONITOR_FAILURE", str(error)); _write_manifest(root); return payload
                    if decision.get("status") == "HARD_STOP":
                        payload = write_status("RESOURCE_STOP", decision.get("reason_code"), decision.get("reason")); _write_manifest(root); return payload
            except KeyboardInterrupt as primary_error:
                secondary_errors = []
                if not cleanup_attempted:
                    cleanup = getattr(runtime, "cleanup", None) or getattr(runtime, "reset_cache", None)
                    if callable(cleanup):
                        cleanup_attempted = True
                        try: cleanup()
                        except Exception as cleanup_error: secondary_errors.append({"type": type(cleanup_error).__name__, "message": str(cleanup_error), "kind": "cleanup"})
                    if cleanup_attempted and monitor is not None and hasattr(monitor, "capture_sample"):
                        try: monitor.capture_sample(spec["sample_id"], "post_cleanup", 32 - len(completed), root)
                        except Exception as capture_error: secondary_errors.append({"type": type(capture_error).__name__, "message": str(capture_error), "kind": "monitor"})
                failed.append(spec["sample_id"])
                failure_evidence.extend({"sample_id": spec["sample_id"], **item} for item in secondary_errors)
                payload = write_status("INTERRUPTED", "SIGNAL_INTERRUPTED", "interrupt received"); _write_manifest(root); return payload
            except Exception as primary_error:
                secondary_errors = []
                if not cleanup_attempted:
                    cleanup = getattr(runtime, "cleanup", None) or getattr(runtime, "reset_cache", None)
                    if callable(cleanup):
                        cleanup_attempted = True
                        try: cleanup()
                        except Exception as cleanup_error: secondary_errors.append({"type": type(cleanup_error).__name__, "message": str(cleanup_error), "kind": "cleanup"})
                    if cleanup_attempted and monitor is not None and hasattr(monitor, "capture_sample"):
                        try: monitor.capture_sample(spec["sample_id"], "post_cleanup", 32 - len(completed), root)
                        except Exception as capture_error: secondary_errors.append({"type": type(capture_error).__name__, "message": str(capture_error), "kind": "monitor"})
                failed.append(spec["sample_id"])
                failure_evidence.append({"sample_id": spec["sample_id"], "primary": {"type": type(primary_error).__name__, "message": str(primary_error)}, "secondary": secondary_errors})
                try:
                    samples.write_failure(spec["sample_id"], {"status": "fail", "identity": identity,
                        "spec": spec, "error": {"type": type(primary_error).__name__, "message": str(primary_error), "secondary": secondary_errors}})
                except Exception:
                    # The terminal status remains authoritative even if a
                    # failure record cannot be published after a hard stop.
                    pass
                message = str(primary_error); lower = (type(primary_error).__name__ + " " + message).lower()
                oom = "outofmemory" in lower or "out of memory" in lower or "cuda oom" in lower
                monitor_secondary = any(item["kind"] == "monitor" for item in secondary_errors)
                cleanup_secondary = any(item["kind"] == "cleanup" for item in secondary_errors)
                cleanup_primary = cleanup_failure is not None
                terminal_status = "RESOURCE_STOP" if oom or cleanup_primary or cleanup_secondary or monitor_secondary else "FAILED"
                reason_code = "CUDA_OOM" if oom else ("MONITOR_FAILURE" if monitor_secondary else ("RESOURCE_CLEANUP_FAILURE" if cleanup_primary or cleanup_secondary else type(primary_error).__name__))
                if secondary_errors: message += "; secondary=" + json.dumps(secondary_errors, sort_keys=True)
                payload = write_status(terminal_status, reason_code, message); _write_manifest(root); return payload
        if len(completed) != 32:
            payload = write_status("FAILED", "INCOMPLETE_GROUP", "group did not produce exactly 32 successes"); _write_manifest(root); return payload
        payload = write_status("AWAITING_REVIEW", "PILOT_COMPLETE", "32 generation successes; analysis pending")
        payload["analysis_pending"] = True
        _atomic_text(status_path, canonical_json(payload) + "\n")
        _write_manifest(root)
        return payload


validate_task6_preflight = preflight_task6
load_task5_direction_bank = load_frozen_directions
verify_umi_reference_reuse = verify_reference_reuse

__all__ = ["PreflightError", "ResourceStop", "Task6Inputs", "Task6Runtime", "Task6RuntimeAdapter", "build_task6_hash_binding", "task6_binding_config",
           "load_frozen_directions", "load_task5_direction_bank", "preflight_task6",
           "validate_task6_preflight", "verify_reference_reuse", "verify_umi_reference_reuse"]
