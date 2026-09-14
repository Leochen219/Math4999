"""Fail-closed runner for Task 5's fixed 32-call directional experiment.

This module deliberately reuses the Task 4 observation seam.  It changes only
which already-encoded condition token is supplied for each request; all model,
sampler, cache, dtype, noise, and output-capture checks remain in the official
runtime and :func:`umi_precision_runtime.validate_capture`.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from .umi_fd_post_vae_bridge import construct_delta, sha256_array
    from .umi_precision_primitives import canonical_json, mask_geometry
    from .umi_precision_runtime import EvidenceError, TensorEvidence, projection, validate_capture
    from .umi_precision_storage import PrecisionSampleStore as SampleStore, ProcessLock
    from .umi_fd_post_vae_scan import sha256_file
    from .umi_task5_primitives import ALPHAS, build_task5_call_plan, freeze_task5_directions, rms64
except ImportError:
    from umi_fd_post_vae_bridge import construct_delta, sha256_array
    from umi_precision_primitives import canonical_json, mask_geometry
    from umi_precision_runtime import EvidenceError, TensorEvidence, projection, validate_capture
    from umi_precision_storage import PrecisionSampleStore as SampleStore, ProcessLock
    from umi_fd_post_vae_scan import sha256_file
    from umi_task5_primitives import ALPHAS, build_task5_call_plan, freeze_task5_directions, rms64


class NativeFixture:
    """Tiny TensorEvidence-compatible tensor used only by CPU unit tests."""
    def __init__(self, value: Any, dtype: str):
        self.value = np.asarray(value, dtype=np.float32).copy()
        self.dtype = dtype

    def detach(self):
        return self

    def float(self):
        return NativeFixture(self.value, "float32")

    def cpu(self):
        return self

    def numpy(self):
        return self.value.copy()


class Task5Inputs:
    """Frozen C-group condition geometry, directions, and Task 5 plan inputs."""
    def __init__(self, carrier: Any, condition_indexes: Any, packed_mask: Any, direction_bank: Any, *,
                 z_bar: Any, task4_reference: Mapping[str, Any]):
        self.z0 = projection(carrier)
        self.geometry = mask_geometry(condition_indexes, packed_mask, self.z0.shape)
        self.z_bar = projection(z_bar)
        if self.z_bar.shape != self.z0.shape:
            raise ValueError("Task 4 C baseline condition does not match current carrier shape")
        if not np.all(np.isfinite(self.z_bar)):
            raise ValueError("Task 4 C baseline condition is non-finite")
        frozen = freeze_task5_directions(direction_bank, self.geometry.mask)
        self.directions = {key: np.ascontiguousarray(value, dtype=np.float32) for key, value in frozen["directions"].items()}
        self.direction_ids = tuple(frozen["ids"])
        self.c01, self.c12 = float(frozen["c01"]), float(frozen["c12"])
        self.gram = np.asarray(frozen["gram"], dtype=np.float64)
        self.masked_direction_rms = dict(frozen["masked_rms"])
        self.s_z = rms64(self.z_bar[self.geometry.mask])
        if self.s_z == 0.0:
            raise ValueError("Task 4 C baseline has zero masked RMS")
        self.task4_reference = json.loads(canonical_json(dict(task4_reference)))
        self.direction_bank_hash = sha256_array(np.asarray(direction_bank, dtype=np.float32))
        self.zero_direction = np.zeros_like(self.z_bar, dtype=np.float32)
        for value in (self.z0, self.z_bar, self.geometry.mask, self.gram, self.zero_direction, *self.directions.values()):
            value.setflags(write=False)

    def for_spec(self, spec: Mapping[str, Any]) -> np.ndarray:
        if str(spec.get("group")) != "C":
            raise ValueError("Task 5 may only run the verified FP32 C path")
        kind = str(spec.get("kind"))
        if kind == "baseline":
            if float(spec.get("alpha", 0.0)) != 0.0 or int(spec.get("sign", 0)) != 0:
                raise ValueError("baseline spec must have zero amplitude and sign")
            return self.z_bar.copy()
        if kind != "perturbation":
            raise ValueError("unknown Task 5 call kind")
        direction_id = str(spec.get("direction_id"))
        if direction_id not in self.directions:
            raise ValueError("unknown frozen Task 5 direction")
        alpha = float(spec.get("alpha"))
        if alpha not in ALPHAS:
            raise ValueError("Task 5 amplitude is outside the fixed three-point grid")
        value = construct_delta(self.z_bar, self.geometry.mask, self.directions[direction_id],
                                alpha=alpha, sign=int(spec.get("sign"))).latent
        if not np.array_equal(value[~self.geometry.mask], self.z_bar[~self.geometry.mask]):
            raise EvidenceError("Task 5 perturbation escaped the condition mask")
        return value

    def direction_for_spec(self, spec: Mapping[str, Any]) -> np.ndarray:
        if str(spec.get("kind")) == "baseline":
            return self.zero_direction
        return self.directions[str(spec["direction_id"])]

    def identity(self) -> dict[str, Any]:
        return {"z0": sha256_array(self.z0), "z_bar": sha256_array(self.z_bar),
                "geometry": self.geometry.metadata(), "direction_bank": self.direction_bank_hash,
                "directions": {key: sha256_array(value) for key, value in self.directions.items()},
                "c01": self.c01, "c12": self.c12, "gram": self.gram.tolist(), "s_z": self.s_z,
                "task4_reference": self.task4_reference}

    def task5_plan(self) -> dict[str, Any]:
        return {"formal_call_count": 32, "alphas": list(ALPHAS), "direction_ids": list(self.direction_ids),
                "direction_hashes": {key: sha256_array(value) for key, value in self.directions.items()},
                "masked_direction_rms": self.masked_direction_rms, "gram_v0_v1_v2": self.gram.tolist(),
                "combination_coefficients": {"c01": self.c01, "c12": self.c12}, "s_z": self.s_z,
                "task4_reference": self.task4_reference,
                "criteria": {"direction_input_cosine_min": 0.99, "opposite_input_cosine_max": -0.99,
                             "slope_range": [0.8, 1.2], "r2_min": 0.98, "secant_cosine_min": 0.95,
                             "secant_relative_change_max": 0.25, "response_noise_multiple": 10.0,
                             "additivity_relative_error_max": 0.10, "prediction_relative_error_max": 0.10}}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _source_hashes(runtime: Any) -> dict[str, str]:
    paths = {Path(__file__).resolve()}
    for function in (Task5Inputs, freeze_task5_directions, construct_delta, validate_capture, type(runtime)):
        source = inspect.getsourcefile(function)
        if source:
            paths.add(Path(source).resolve())
    return {str(path): sha256_file(path) for path in sorted(paths)}


def _strict_success(path: Path, identity: str) -> None:
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "success":
        return
    hashes = status.get("artifact_sha256", {})
    if "sample.json" not in hashes or any(not (path / name).is_file() or sha256_file(path / name) != digest
                                            for name, digest in hashes.items()):
        raise ValueError("successful Task 5 sample integrity mismatch")
    if json.loads((path / "sample.json").read_text(encoding="utf-8")).get("identity") != identity:
        raise ValueError("Task 5 resume identity mismatch")


def run_task5_experiment(runtime: Any, inputs: Task5Inputs, run_dir: str | Path, *, resume: bool = False,
                         cpu_calibration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run exactly the pre-frozen 32 full C-path calls, serially and durably."""
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    with ProcessLock(root / ".runner.lock"):
        plan = []
        for entry in build_task5_call_plan():
            item = dict(entry)
            item["group"] = "C"
            plan.append(item)
        config = {"schema_version": "umi-task5-directional-linearity-v1", "runtime": runtime.provenance,
                  "actual_runtime": runtime.actual_identity(), "inputs": inputs.identity(), "plan": plan,
                  "plan_detail": inputs.task5_plan(), "source_sha256": _source_hashes(runtime),
                  "scope": "full", "diffusion_cache": "off", "model_seed": 0,
                  "cpu_linear_formula_calibration": dict(cpu_calibration or {})}
        identity = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
        plan_path = root / "task5_plan.json"
        if plan_path.exists():
            if not resume:
                raise FileExistsError("Task 5 directory exists; explicit --resume is required")
            if json.loads(plan_path.read_text(encoding="utf-8")) != json.loads(canonical_json(config)):
                raise ValueError("strict Task 5 plan/config/source identity mismatch")
        elif any(path.name != ".runner.lock" for path in root.iterdir()):
            raise ValueError("Task 5 run directory is nonempty without its frozen plan")
        else:
            _atomic_json(plan_path, config)

        for path in (root / "samples").glob("*") if (root / "samples").exists() else ():
            if path.is_dir() and ".attempt." not in path.name and (path / "status.json").is_file():
                _strict_success(path, identity)
        status_path = root / "status.json"
        previous = json.loads(status_path.read_text(encoding="utf-8")) if resume and status_path.exists() else {}
        if previous.get("status") == "complete":
            if int(previous.get("formal_successful", 0)) != 32:
                raise ValueError("completed Task 5 status does not contain 32 formal samples")
            return previous
        summary: dict[str, Any] = {"status": "running", "identity": identity, "scope": "full",
                                   "formal_successful": 0, "formal_attempts": 0, "failures": 0,
                                   "retries": 0, "diagnostic_attempts": 0, "full_sampling_claim": False,
                                   "image_claim": False, "started_at_unix": time.time()}
        paired_noise: str | None = None
        schedule: list[float] | None = None
        store = SampleStore(root / "samples")
        for spec in plan:
            disposition = store.prepare(spec["sample_id"], resume=resume, required_files=("sample.json", "output_full.npy"))
            if disposition == "skip":
                record = store.load_record(spec["sample_id"])
                paired_noise = validate_capture(record, spec, inputs, "full", paired_noise=paired_noise)
                current_schedule = [float(row["timestep"]) for row in record["steps"]]
                if schedule is not None and schedule != current_schedule:
                    raise EvidenceError("Task 5 resumed sample has a different sampler schedule")
                schedule = current_schedule
                summary["formal_successful"] += 1
                continue
            summary["formal_attempts"] += 1
            _atomic_json(status_path, summary)
            started = time.perf_counter()
            record: dict[str, Any] = {}
            try:
                record = dict(runtime.execute(spec, inputs, scope="full"))
                paired_noise = validate_capture(record, spec, inputs, "full", paired_noise=paired_noise)
                current_schedule = [float(row["timestep"]) for row in record["steps"]]
                if schedule is not None and schedule != current_schedule:
                    raise EvidenceError("Task 5 sampler schedule changed across calls")
                schedule = current_schedule
                if record.get("decode_id") != runtime.provenance.get("decode"):
                    raise EvidenceError("Task 5 decoder implementation changed during generation")
                record["decoder_input_full_latent"] = projection(record["output_full"])
                record.update({"status": "success", "identity": identity, "spec": spec,
                               "elapsed_seconds": time.perf_counter() - started})
                arrays = {key + ".npy": value for key, value in record.items() if isinstance(value, np.ndarray)}
                store.write_success(spec["sample_id"], {key: value for key, value in record.items()
                                                         if not isinstance(value, np.ndarray)}, artifacts=arrays)
                summary["formal_successful"] += 1
            except Exception as error:
                summary["failures"] += 1
                failure = {"status": "fail", "identity": identity, "spec": spec, "scope": "full",
                           "elapsed_seconds": time.perf_counter() - started,
                           "error": {"type": type(error).__name__, "message": str(error)},
                           "traceback": traceback.format_exc(), "partial_capture": getattr(error, "capture", record)}
                store.write_failure(spec["sample_id"], failure)
                summary.update({"status": "blocked", "blocked_sample": spec["sample_id"], "completed_at_unix": time.time()})
                _atomic_json(status_path, summary)
                return summary
            _atomic_json(status_path, summary)
        if summary["formal_successful"] != 32:
            raise AssertionError("Task 5 runner cannot complete with a partial formal call set")
        summary.update({"status": "complete", "full_sampling_claim": True, "image_claim": True,
                        "paired_initial_noise_hash": paired_noise, "sampler_timestep_schedule": schedule,
                        "completed_at_unix": time.time()})
        _atomic_json(status_path, summary)
        return summary


__all__ = ["NativeFixture", "Task5Inputs", "run_task5_experiment"]
