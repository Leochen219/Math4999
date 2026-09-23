"""Pure-CPU geometry and spectrum calculations for Task 9.

Columns of each response matrix correspond to independently probed input
directions at one fixed scene and diffusion seed. Different base points are
never pooled into a purported single Jacobian.
"""
from __future__ import annotations

import math
import hashlib
import csv
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


def generate_direction_bank(mask, *, seed: int, train_count: int = 32,
                            holdout_count: int = 8) -> np.ndarray:
    active = np.asarray(mask, dtype=bool)
    if not active.size:
        raise ValueError("mask must be nonempty")
    count = int(train_count) + int(holdout_count)
    coordinates = int(np.count_nonzero(active))
    if train_count < 1 or holdout_count < 1 or coordinates < count:
        raise ValueError("active mask must have at least one coordinate per direction")
    samples = np.random.default_rng(seed).standard_normal((coordinates, count))
    basis, _ = np.linalg.qr(samples, mode="reduced")
    result = np.zeros((count,) + active.shape, dtype=np.float32)
    result[:, active] = (basis.T * math.sqrt(coordinates)).astype(np.float32)
    return result


def build_call_plan(*, train_count: int = 32, holdout_count: int = 8,
                    alpha: float = 0.003, calibration_alpha: float = 0.0015) -> list[dict]:
    if min(train_count, holdout_count) < 1:
        raise ValueError("invalid direction counts")
    if not (alpha > 0 and np.isclose(calibration_alpha, alpha / 2, rtol=0, atol=1e-15)):
        raise ValueError("calibration alpha must be exactly half the primary alpha")
    plan = [{"sample_id": "baseline_pre", "kind": "baseline", "direction_id": None,
             "alpha": 0.0, "sign": 0}]
    families = (("train", train_count, alpha), ("holdout", holdout_count, alpha))
    for kind, count, step in families:
        for index in range(count):
            direction_id = f"{kind}_{index:02d}"
            for sign, label in ((1, "plus"), (-1, "minus")):
                plan.append({"sample_id": f"{direction_id}_{label}", "kind": kind,
                             "direction_id": direction_id, "alpha": float(step), "sign": sign})
    for family, count, _ in families:
        for index in range(count):
            direction_id = f"{family}_{index:02d}"
            for sign, label in ((1, "plus"), (-1, "minus")):
                plan.append({"sample_id": f"calibration_{direction_id}_{label}", "kind": "calibration",
                             "direction_id": direction_id, "alpha": float(calibration_alpha), "sign": sign})
    plan.append({"sample_id": "baseline_post", "kind": "baseline", "direction_id": None,
                 "alpha": 0.0, "sign": 0})
    return plan


def central_response(plus, minus, step: float) -> np.ndarray:
    positive = np.asarray(plus, dtype=np.float32)
    negative = np.asarray(minus, dtype=np.float32)
    if positive.shape != negative.shape:
        raise ValueError("positive and negative response shape mismatch")
    if not positive.size or not np.all(np.isfinite(positive)) or not np.all(np.isfinite(negative)):
        raise ValueError("responses must be finite and nonempty")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    return (positive.astype(np.float64) - negative.astype(np.float64)) / (2 * step)


def spectrum_metrics(train_columns, holdout_columns, *, singular_resolution: float = 0.0) -> dict:
    train = np.asarray(train_columns, dtype=np.float64)
    holdout = np.asarray(holdout_columns, dtype=np.float64)
    if train.ndim != 2 or holdout.ndim != 2 or train.shape[0] != holdout.shape[0] or train.shape[1] < 1:
        raise ValueError("train and holdout response matrices must share output rows")
    if not np.all(np.isfinite(train)) or not np.all(np.isfinite(holdout)):
        raise ValueError("response matrices must be finite")
    if not np.isfinite(singular_resolution) or singular_resolution < 0:
        raise ValueError("singular resolution must be finite and nonnegative")
    output_basis, singular, _ = np.linalg.svd(train, full_matrices=False)
    raw_values = singular * singular
    tolerance = max(singular_resolution,
                    np.finfo(np.float64).eps * max(train.shape) * (singular[0] if singular.size else 0.0))
    rank = int(np.count_nonzero(singular > tolerance))
    values = raw_values.copy()
    values[rank:] = 0.0
    unresolved_fraction = (float(raw_values[rank:].sum() / raw_values.sum())
                           if raw_values.sum() > 0 else None)
    if rank == 0 or values.sum() == 0.0:
        return {"status": "NO_NONZERO_RESPONSE" if not np.any(singular) else "BELOW_NUMERICAL_RESOLUTION",
                "observed_rank": 0, "singular_resolution": float(tolerance),
                "unresolved_energy_fraction": unresolved_fraction,
                "singular_values": singular.tolist(), "cumulative_energy": [],
                "effective_rank": None, "k90": None, "k95": None, "k99": None,
                "heldout_relative_residual_by_k": None}
    probabilities = values / values.sum()
    cumulative = np.cumsum(probabilities)
    nonzero = probabilities[probabilities > 0]
    effective = float(np.exp(-np.sum(nonzero * np.log(nonzero))))
    heldout_norms = np.linalg.norm(holdout, axis=0)
    projections = np.zeros_like(holdout)
    residuals = []
    for index in range(rank):
        column = output_basis[:, index]
        projections += np.outer(column, column @ holdout)
        errors = np.linalg.norm(holdout - projections, axis=0)
        residuals.append([None if norm == 0 else float(error / norm)
                          for error, norm in zip(errors, heldout_norms)])
    def threshold_k(target):
        return int(np.searchsorted(cumulative, target) + 1)
    return {"status": "OK", "observed_rank": rank,
            "singular_resolution": float(tolerance),
            "unresolved_energy_fraction": unresolved_fraction,
            "singular_values": singular.tolist(), "cumulative_energy": cumulative.tolist(),
            "effective_rank": effective, "k90": threshold_k(0.90),
            "k95": threshold_k(0.95), "k99": threshold_k(0.99),
            "heldout_relative_residual_by_k": residuals}


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _verify_success(destination: Path, identity_hash: str) -> str:
    status_path = destination / "status.json"
    if not status_path.is_file():
        raise ValueError(f"sample lacks status: {destination.name}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "success" or status.get("identity_hash") != identity_hash:
        raise ValueError(f"sample identity or success status mismatch: {destination.name}")
    required = {"condition.npy", "predicted_latent.npy", "decoded_final_rgb.npy",
                "next_condition.npy", "record.json"}
    hashes = status.get("artifact_sha256", {})
    if set(hashes) != required or any(not (destination / name).is_file() or
                                      _file_hash(destination / name) != digest
                                      for name, digest in hashes.items()):
        raise ValueError(f"sample artifact integrity mismatch: {destination.name}")
    record = json.loads((destination / "record.json").read_text(encoding="utf-8"))
    return str(record["prediction_noise_hash"])


def _run_samples_impl(root: str | Path, base_condition: Any, mask: Any,
                directions: Mapping[str, Any], plan: list[Mapping[str, Any]], *,
                seed: int, identity: Mapping[str, Any],
                step: Callable[[np.ndarray, int], Mapping[str, Any]],
                resume: bool = False, resource_check: Callable[[str], Any] | None = None,
                start_free_bytes: int = 0, reserve_bytes: int = 0,
                forecast_factor: float = 1.3) -> dict[str, Any]:
    """Run one stratum serially, saving only quantitative float tensors.

    The model callback is responsible for the already-verified G/D/E precision
    path. This function checks the exact consumed condition and paired noise,
    then publishes each compact sample by a no-overwrite directory rename.
    """
    destination = Path(root)
    base = np.asarray(base_condition, dtype=np.float32)
    active = np.asarray(mask, dtype=bool)
    if base.shape != active.shape or not base.size or not active.any() or not np.all(np.isfinite(base)):
        raise ValueError("invalid finite base condition/mask geometry")
    if seed not in (0, 1) or isinstance(seed, bool):
        raise ValueError("seed must be 0 or 1")
    frozen: dict[str, np.ndarray] = {}
    for name, value in directions.items():
        item = np.asarray(value, dtype=np.float32)
        if item.shape != base.shape or not np.all(np.isfinite(item)) or np.any(item[~active] != 0):
            raise ValueError(f"invalid masked direction: {name}")
        norm = float(np.sqrt(np.mean(item[active].astype(np.float64) ** 2)))
        if not np.isclose(norm, 1.0, rtol=0, atol=2e-6):
            raise ValueError(f"direction is not unit masked RMS: {name}")
        frozen[str(name)] = item.copy()
    specs = [dict(spec) for spec in plan]
    names = [str(spec.get("sample_id", "")) for spec in specs]
    if not specs or len(names) != len(set(names)) or any(re.fullmatch(r"[A-Za-z0-9_]+", name) is None for name in names):
        raise ValueError("plan requires unique safe sample ids")
    for spec in specs:
        if spec.get("kind") == "baseline":
            if spec.get("alpha") != 0.0 or spec.get("sign") != 0:
                raise ValueError("baseline must have zero alpha/sign")
        elif spec.get("direction_id") not in frozen or spec.get("sign") not in (-1, 1) or float(spec.get("alpha", 0)) <= 0:
            raise ValueError("perturbation spec is not bound to a direction")
    if start_free_bytes < 0 or reserve_bytes < 0 or forecast_factor < 1:
        raise ValueError("invalid disk safety budget")
    if shutil.disk_usage(destination.parent).free < start_free_bytes:
        raise RuntimeError("disk startup free-space gate failed")
    configuration = {"schema": "umi-task9-spectrum-v1", "identity": dict(identity),
                     "base_hash": _array_hash(base), "mask_hash": _array_hash(active),
                     "direction_hashes": {name: _array_hash(value) for name, value in sorted(frozen.items())},
                     "plan": specs, "seed": seed,
                     "runner_source_sha256": _file_hash(Path(__file__))}
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)
    identity_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise FileExistsError("nonempty Task 9 directory requires explicit resume")
    destination.mkdir(parents=True, exist_ok=True)
    config_path = destination / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != configuration:
            raise ValueError("run identity mismatch on resume")
    elif any(destination.iterdir()):
        raise ValueError("nonempty run lacks immutable config identity")
    else:
        _write_json_atomic(config_path, configuration)
        np.save(destination / "base_condition.npy", base, allow_pickle=False)
        np.save(destination / "mask.npy", active, allow_pickle=False)
        for name, value in frozen.items():
            # The direction id is already checked through the bound plan.
            if re.fullmatch(r"[A-Za-z0-9_]+", name) is None:
                raise ValueError("unsafe direction id")
            np.save(destination / f"direction_{name}.npy", value, allow_pickle=False)
    samples_root = destination / "samples"
    samples_root.mkdir(exist_ok=True)
    scale = float(np.sqrt(np.mean(base[active].astype(np.float64) ** 2)))
    if scale == 0.0:
        raise ValueError("base masked RMS is zero")
    expected_noise = None
    largest_sample_bytes = 0
    completed = 0
    for index, spec in enumerate(specs):
        name = spec["sample_id"]
        sample = samples_root / name
        if sample.exists():
            if not resume:
                raise FileExistsError(sample)
            noise = _verify_success(sample, identity_hash)
            if expected_noise is not None and noise != expected_noise:
                raise ValueError("resumed samples have nonpaired prediction noise")
            expected_noise = noise
            largest_sample_bytes = max(largest_sample_bytes, sum(path.stat().st_size for path in sample.iterdir() if path.is_file()))
            completed += 1
            continue
        if resource_check is not None:
            resource_check("before")
        free = shutil.disk_usage(destination).free
        remaining = len(specs) - completed
        forecast = int(largest_sample_bytes * remaining * forecast_factor) + reserve_bytes
        if free < max(reserve_bytes, forecast):
            raise RuntimeError("disk forecast/reserve gate stopped new sample")
        if spec["kind"] == "baseline":
            condition = base.copy()
        else:
            delta = np.float32(float(spec["alpha"]) * scale * int(spec["sign"])) * frozen[spec["direction_id"]]
            condition = (base + delta).astype(np.float32)
            if not np.array_equal(condition[~active], base[~active]):
                raise ValueError("perturbation changed mask exterior")
        started = time.perf_counter()
        record = dict(step(condition.copy(), seed))
        if resource_check is not None:
            resource_check("after")
        consumed = np.asarray(record["actual_consumed_condition"], dtype=np.float32)
        if consumed.shape != condition.shape or not np.array_equal(consumed, condition):
            raise ValueError("model did not consume the requested FP32 condition")
        noise = str(record["prediction_noise_hash"])
        if not noise or (expected_noise is not None and noise != expected_noise):
            raise ValueError("prediction noise differs across paired calls")
        arrays = {"condition.npy": condition,
                  "predicted_latent.npy": np.asarray(record["predicted_latent"], dtype=np.float32),
                  "decoded_final_rgb.npy": np.asarray(record["decoded_last_rgb"], dtype=np.float32),
                  "next_condition.npy": np.asarray(record["next_condition_fp32"], dtype=np.float32)}
        if arrays["next_condition.npy"].shape != base.shape or arrays["decoded_final_rgb.npy"].ndim != 3 or arrays["decoded_final_rgb.npy"].shape[0] != 3:
            raise ValueError("output shape is not validated feedback/RGB geometry")
        if any(not value.size or not np.all(np.isfinite(value)) for value in arrays.values()):
            raise ValueError("nonfinite or empty quantitative tensor")
        step_evidence = record.get("step_evidence", {})
        if not isinstance(step_evidence, Mapping):
            raise ValueError("step evidence must be a JSON mapping")
        encoded_evidence = json.dumps(step_evidence, sort_keys=True, allow_nan=False)
        if len(encoded_evidence.encode("utf-8")) > 1_000_000:
            raise ValueError("step evidence exceeds the compact per-sample budget")
        stage = Path(tempfile.mkdtemp(prefix=f".{name}.stage.", dir=samples_root))
        try:
            for filename, value in arrays.items():
                np.save(stage / filename, value, allow_pickle=False)
            metadata = {"sample_id": name, "spec": spec, "seed": seed, "identity_hash": identity_hash,
                        "prediction_noise_hash": noise, "elapsed_seconds": time.perf_counter() - started,
                        "actual_input_hash": _array_hash(condition),
                        "step_evidence": step_evidence,
                        "predicted_latent_hash": _array_hash(arrays["predicted_latent.npy"]),
                        "decoded_final_rgb_hash": _array_hash(arrays["decoded_final_rgb.npy"]),
                        "next_condition_hash": _array_hash(arrays["next_condition.npy"])}
            _write_json_atomic(stage / "record.json", metadata)
            hashes = {filename: _file_hash(stage / filename) for filename in (*arrays, "record.json")}
            _write_json_atomic(stage / "status.json", {"status": "success", "identity_hash": identity_hash,
                                                        "artifact_sha256": hashes})
            if sample.exists():
                raise FileExistsError(sample)
            stage.rename(sample)
        except BaseException:
            attempt = samples_root / f"{name}.attempt.1"
            number = 1
            while attempt.exists():
                number += 1
                attempt = samples_root / f"{name}.attempt.{number}"
            stage.rename(attempt)
            raise
        expected_noise = noise
        largest_sample_bytes = max(largest_sample_bytes, sum(path.stat().st_size for path in sample.iterdir() if path.is_file()))
        completed += 1
        _write_json_atomic(destination / "run_status.json", {"status": "RUNNING", "completed": completed,
                                                           "total": len(specs), "last_sample": name,
                                                           "paired_noise_hash": expected_noise,
                                                           "largest_sample_bytes": largest_sample_bytes})
    result = {"status": "COMPLETE", "completed": completed, "total": len(specs),
              "paired_noise_hash": expected_noise, "largest_sample_bytes": largest_sample_bytes,
              "identity_hash": identity_hash}
    _write_json_atomic(destination / "run_status.json", result)
    return result


def run_samples(root: str | Path, base_condition: Any, mask: Any,
                directions: Mapping[str, Any], plan: list[Mapping[str, Any]], *,
                seed: int, identity: Mapping[str, Any],
                step: Callable[[np.ndarray, int], Mapping[str, Any]],
                resume: bool = False, resource_check: Callable[[str], Any] | None = None,
                start_free_bytes: int = 0, reserve_bytes: int = 0,
                forecast_factor: float = 1.3) -> dict[str, Any]:
    destination = Path(root)
    try:
        return _run_samples_impl(destination, base_condition, mask, directions, plan,
                                 seed=seed, identity=identity, step=step, resume=resume,
                                 resource_check=resource_check,
                                 start_free_bytes=start_free_bytes,
                                 reserve_bytes=reserve_bytes,
                                 forecast_factor=forecast_factor)
    except BaseException as error:
        config_path = destination / "config.json"
        status_path = destination / "run_status.json"
        prior = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        if (config_path.is_file() and prior.get("status") != "COMPLETE"
                and not isinstance(error, FileExistsError)
                and "identity mismatch" not in str(error)):
            samples_root = destination / "samples"
            completed = sorted(path.name for path in samples_root.iterdir()
                               if path.is_dir() and (path / "status.json").is_file()
                               and json.loads((path / "status.json").read_text(encoding="utf-8")).get("status") == "success") if samples_root.is_dir() else []
            _write_json_atomic(status_path, {"status": "BLOCKED", "completed": len(completed),
                                                   "completed_sample_ids": completed,
                                                   "total": len(plan), "error_type": type(error).__name__,
                                                   "reason": str(error)})
        raise


def _rms(value: Any) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array, dtype=np.float64)))


def _cosine(left: Any, right: Any) -> float | None:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denominator == 0.0 else float(np.dot(a, b) / denominator)


def analyze_stratum(root: str | Path) -> dict[str, Any]:
    """Independently recompute response spectra from immutable float samples."""
    run_root = Path(root)
    configuration = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    status = json.loads((run_root / "run_status.json").read_text(encoding="utf-8"))
    plan = configuration["plan"]
    if status.get("status") != "COMPLETE" or status.get("completed") != len(plan):
        raise ValueError("scan is not complete")
    identity_hash = status["identity_hash"]
    canonical_config = json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(canonical_config.encode("utf-8")).hexdigest() != identity_hash:
        raise ValueError("run identity hash does not match immutable configuration")
    noise = None
    for spec in plan:
        value = _verify_success(run_root / "samples" / spec["sample_id"], identity_hash)
        if noise is not None and value != noise:
            raise ValueError("saved samples do not share paired prediction noise")
        noise = value
    mask = np.load(run_root / "mask.npy", allow_pickle=False).astype(bool)
    base = np.load(run_root / "base_condition.npy", allow_pickle=False)
    if _array_hash(base) != configuration["base_hash"] or _array_hash(mask) != configuration["mask_hash"]:
        raise ValueError("saved base condition or mask differs from immutable configuration")
    directions = {name: np.load(run_root / f"direction_{name}.npy", allow_pickle=False)
                  for name in configuration["direction_hashes"]}
    for name, value in directions.items():
        if _array_hash(value) != configuration["direction_hashes"][name]:
            raise ValueError(f"saved direction differs from immutable configuration: {name}")
    if mask.shape != base.shape or not mask.any() or not np.all(np.isfinite(base)):
        raise ValueError("saved base/mask geometry is invalid")
    scale = float(np.sqrt(np.mean(base[mask].astype(np.float64) ** 2)))
    for spec in plan:
        sample_id = spec["sample_id"]
        sample_root = run_root / "samples" / sample_id
        saved_record = json.loads((sample_root / "record.json").read_text(encoding="utf-8"))
        if saved_record.get("sample_id") != sample_id or saved_record.get("spec") != spec:
            raise ValueError(f"sample record/plan mismatch: {sample_id}")
        expected = base if spec["kind"] == "baseline" else (
            base + np.float32(float(spec["alpha"]) * scale * int(spec["sign"]))
            * directions[spec["direction_id"]]
        ).astype(np.float32)
        saved_input = np.load(sample_root / "condition.npy", allow_pickle=False)
        if not np.array_equal(saved_input, expected):
            raise ValueError(f"sample input differs from immutable perturbation plan: {sample_id}")
    spaces = {"predicted_latent": "predicted_latent.npy", "feedback_condition": "next_condition.npy",
              "float_rgb_final": "decoded_final_rgb.npy"}
    primary = {spec["direction_id"]: float(spec["alpha"]) for spec in plan
               if spec["kind"] in {"train", "holdout"}}
    half = {spec["direction_id"]: float(spec["alpha"]) for spec in plan
            if spec["kind"] == "calibration"}
    if set(primary) != set(half) or any(not np.isclose(half[key], primary[key] / 2, rtol=0, atol=1e-15)
                                      for key in primary):
        raise ValueError("every independent direction needs an exact half-step pair")
    train_ids = sorted(name for name in primary if name.startswith("train_"))
    holdout_ids = sorted(name for name in primary if name.startswith("holdout_"))
    if not train_ids or not holdout_ids or len(train_ids) + len(holdout_ids) != len(primary):
        raise ValueError("independent train and holdout direction sets are required")

    def tensor(sample_id: str, filename: str) -> np.ndarray:
        return np.load(run_root / "samples" / sample_id / filename, allow_pickle=False)

    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for space, filename in spaces.items():
        before = tensor("baseline_pre", filename)
        after = tensor("baseline_post", filename)
        if before.shape != after.shape:
            raise ValueError("baseline output shape drift")
        repeat_floor = _rms(after.astype(np.float64) - before.astype(np.float64))
        precision_floor = float(8 * np.finfo(np.float32).eps * max(1.0, _rms(before), _rms(after)))
        floor = max(repeat_floor, precision_floor)
        derivatives: dict[str, np.ndarray] = {}
        primary_steps: list[float] = []
        passing = 0
        for direction_id in train_ids + holdout_ids:
            direction = directions[direction_id]
            estimates: dict[str, np.ndarray] = {}
            input_cosines: list[float | None] = []
            response_multiples: list[float | None] = []
            for label, prefix in (("primary", direction_id), ("half", f"calibration_{direction_id}")):
                plus_id, minus_id = f"{prefix}_plus", f"{prefix}_minus"
                x_plus = tensor(plus_id, "condition.npy").astype(np.float64)
                x_minus = tensor(minus_id, "condition.npy").astype(np.float64)
                if x_plus.shape != direction.shape or x_minus.shape != direction.shape:
                    raise ValueError("input geometry drift")
                pair = x_plus - x_minus
                if np.any(pair[~mask] != 0):
                    raise ValueError("input perturbation escaped condition mask")
                realized_step = float(np.mean(pair[mask] * direction[mask].astype(np.float64)) / 2)
                if not np.isfinite(realized_step) or realized_step <= 0:
                    raise ValueError("actual paired input step vanished")
                if label == "primary":
                    primary_steps.append(realized_step)
                input_cosines.append(_cosine(pair[mask], direction[mask]))
                y_plus = tensor(plus_id, filename)
                y_minus = tensor(minus_id, filename)
                if y_plus.shape != before.shape or y_minus.shape != before.shape:
                    raise ValueError("output shape drift")
                estimates[label] = central_response(y_plus, y_minus, realized_step)
                amplitude = _rms(y_plus.astype(np.float64) - y_minus.astype(np.float64)) / 2
                response_multiples.append(None if floor == 0 else amplitude / floor)
            derivative = estimates["primary"]
            smaller = estimates["half"]
            cosine = _cosine(derivative, smaller)
            denominator = max(_rms(derivative), _rms(smaller))
            change = None if denominator == 0 else _rms(derivative.astype(np.float64) - smaller.astype(np.float64)) / denominator
            distinguishable = all(multiple is not None and multiple > 10
                                  for multiple in response_multiples)
            passed = (all(value is not None and value >= 0.99 for value in input_cosines)
                      and distinguishable and cosine is not None and cosine >= 0.95
                      and change is not None and change <= 0.25)
            passing += int(passed)
            derivatives[direction_id] = derivative.reshape(-1)
            rows.append({"space": space, "direction_id": direction_id,
                         "primary_alpha": primary[direction_id], "half_alpha": half[direction_id],
                         "primary_input_cosine": input_cosines[0], "half_input_cosine": input_cosines[1],
                         "derivative_cosine": cosine, "derivative_relative_rms_change": change,
                         "primary_response_floor_multiple": response_multiples[0],
                         "half_response_floor_multiple": response_multiples[1], "pass": passed})
        training = np.column_stack([derivatives[name] for name in train_ids])
        heldout = np.column_stack([derivatives[name] for name in holdout_ids])
        singular_resolution = (math.sqrt(training.shape[0] * training.shape[1])
                               * floor / min(primary_steps))
        spectrum = spectrum_metrics(training, heldout, singular_resolution=singular_resolution)
        results[space] = {**spectrum, "n_train": len(train_ids), "n_holdout": len(holdout_ids),
                          "baseline_noise_rms": repeat_floor, "fp32_precision_floor_rms": precision_floor,
                          "effective_resolution_floor_rms": floor,
                          "half_step_pass_count": passing,
                          "half_step_total": len(primary),
                          "interpretation": "LOCAL_JACOBIAN_CANDIDATE" if passing == len(primary)
                          else "FINITE_AMPLITUDE_RESPONSE_ONLY"}
    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    with (analysis_dir / "half_step_consistency.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"status": "COMPLETE", "scene_identity": configuration["identity"],
               "seed": configuration["seed"], "paired_noise_hash": noise,
               "direction_count": len(primary), "spaces": results,
               "effective_rank_definition": "exp(-sum(p_i*log(p_i))), p_i=sigma_i^2/sum_j sigma_j^2",
               "scope": "one fixed scene/action/seed; sampled input subspace only"}
    _write_json_atomic(analysis_dir / "scan_summary.json", summary)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    for space, result in results.items():
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        singular = np.asarray(result["singular_values"], dtype=np.float64)
        indexes = np.arange(1, len(singular) + 1)
        if singular.size and singular[0] > 0:
            axes[0].semilogy(indexes, singular / singular[0], marker="o", markersize=3)
        axes[0].set(xlabel="Index", ylabel="Singular value / largest", title="Response spectrum")
        cumulative = result["cumulative_energy"]
        if cumulative:
            axes[1].plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", markersize=3)
        for target in (0.90, 0.95, 0.99):
            axes[1].axhline(target, linestyle="--", linewidth=0.7, color="gray")
        axes[1].set(xlabel="Truncation rank k", ylabel="Cumulative response energy", ylim=(0, 1.02))
        residuals = result["heldout_relative_residual_by_k"]
        if residuals:
            valid = [[value for value in row if value is not None] for row in residuals]
            means = [float(np.mean(row)) if row else float("nan") for row in valid]
            maxima = [float(np.max(row)) if row else float("nan") for row in valid]
            axes[2].plot(np.arange(1, len(means) + 1), means, label="mean")
            axes[2].plot(np.arange(1, len(maxima) + 1), maxima, label="max")
            axes[2].legend()
        axes[2].set(xlabel="Truncation rank k", ylabel="Held-out projection residual", title="Unseen directions")
        fig.suptitle(f"{space} | {result['interpretation']}")
        fig.tight_layout()
        fig.savefig(analysis_dir / f"{space}_spectrum.png", dpi=150)
        fig.savefig(analysis_dir / f"{space}_spectrum.svg")
        plt.close(fig)
    with (analysis_dir / "singular_values.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("space", "index", "singular_value", "cumulative_energy"))
        for space, result in results.items():
            for index, singular in enumerate(result["singular_values"]):
                cumulative = result["cumulative_energy"]
                writer.writerow((space, index + 1, singular,
                                 cumulative[index] if index < len(cumulative) else ""))
    report = ["# UMI Task 9: local response spectrum", "",
              f"Scene identity: `{json.dumps(configuration['identity'], sort_keys=True)}`. Seed: {configuration['seed']}.",
              f"Independent directions: {len(train_ids)} fit, {len(holdout_ids)} held out. Matched noise hash: `{noise}`.", "",
              "| Space | Half-step passes | Interpretation | k90 | k95 | k99 | Effective rank | Unresolved energy fraction | Max held-out residual at k95 |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for space, result in results.items():
        k95 = result["k95"]
        residual = result["heldout_relative_residual_by_k"]
        valid = None if k95 is None or residual is None else [value for value in residual[k95 - 1] if value is not None]
        worst = "N/A" if not valid else f"{max(valid):.6g}"
        rank = result["effective_rank"]
        unresolved = result["unresolved_energy_fraction"]
        report.append(f"| {space} | {result['half_step_pass_count']}/{result['half_step_total']} | "
                      f"{result['interpretation']} | {result['k90']} | {k95} | {result['k99']} | "
                      f"{'N/A' if rank is None else f'{rank:.6g}'} | "
                      f"{'N/A' if unresolved is None else f'{unresolved:.6g}'} | {worst} |")
    report += ["", "Cumulative energy uses squared singular values. Effective rank is the entropy rank "
               "exp(-sum p_i log p_i), p_i = sigma_i^2 / sum sigma_j^2).",
               "The heuristic response distinguishability gate uses the larger of measured baseline repeat RMS and "
               "8 times FP32 machine epsilon times max(1, baseline RMS). Singular values below a "
               "conservative output-resolution/actual-step threshold are excluded from rank and energy claims; "
               "raw singular values remain available in CSV. This FP32-scale estimate is not a rigorous "
               "error bound for the entire inference pipeline.",
               "Held-out residual is ||r - U_k U_k^T r|| / ||r|| on directions excluded from SVD; "
               "zero-norm responses are N/A.",
               "A passing half-step gate is empirical evidence for a local linear approximation in these sampled directions, "
               "not a proof that a Jacobian exists or that the full operator is low-rank.",
               "The predicted latent, FP32 re-encoded feedback condition, and RGB spaces are separate measurements. "
               "Do not pool matrices from different scenes or seeds into one Jacobian.", ""]
    (analysis_dir / "experiment_report.md").write_text("\n".join(report), encoding="utf-8", newline="\n")
    manifest = run_root / "MANIFEST.sha256"
    lines = [f"{_file_hash(path)}  {path.relative_to(run_root).as_posix()}" for path in sorted(run_root.rglob("*"))
             if path.is_file() and path != manifest]
    manifest.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    return summary
