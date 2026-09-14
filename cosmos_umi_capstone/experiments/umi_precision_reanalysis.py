"""Immutable, CPU-only reanalysis of the old UMI post-VAE run.

The reanalysis consumes only detached JSON/NumPy evidence from an existing
run.  It validates the original manifest before reading quantitative data,
never writes to the source run, and publishes a new revision with an atomic
no-overwrite directory move.  Images and videos are intentionally outside
the metric path.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import numpy as np

try:  # package import
    from .umi_precision_primitives import (
        decompose_response_vectors,
        build_precision_contract,
        even_symmetry_residual,
        fixed_noise_hash,
        metric_with_reason,
        quantize_bf16_fp32,
        slice_predicted_output,
    )
except ImportError:  # direct import from experiments/
    from umi_precision_primitives import (
        decompose_response_vectors,
        build_precision_contract,
        even_symmetry_residual,
        fixed_noise_hash,
        metric_with_reason,
        quantize_bf16_fp32,
        slice_predicted_output,
    )


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_EXCLUSIONS = {"MANIFEST.sha256", "MANIFEST.analysis.sha256", "review_bundle.zip"}
_TASK4_RAW_ROOT_EXCLUSIONS = {"MANIFEST.sha256", ".runner.lock"}
_MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".mov", ".avi", ".mkv"}
_SAMPLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _is_root_generated_path(relative: str) -> bool:
    """Only the exact root generated paths are excluded from raw inventory."""

    return relative in _MANIFEST_EXCLUSIONS


def is_task4_raw_manifest_excluded(relative: str) -> bool:
    """Shared exact relative-path predicate for Task4 raw-run manifests."""

    pure = PurePosixPath(relative)
    return relative in _TASK4_RAW_ROOT_EXCLUSIONS or bool(pure.parts) and pure.parts[0] == "precision_analysis"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(relative: str) -> str:
    pure = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
        or relative.endswith(" ")
    ):
        raise ValueError(f"unsafe manifest path: {relative!r}")
    return relative


def _validate_sample_id(value: Any, directory_name: str, *, require_match: bool = True) -> str:
    if not isinstance(value, str) or not value or not _SAMPLE_ID_RE.fullmatch(value):
        raise ValueError(f"unsafe sample_id: {value!r}")
    if require_match and value != directory_name:
        raise ValueError(f"sample_id must equal sample directory name: {value!r} != {directory_name!r}")
    return value


def _record_input_hash(root: Path, path: Path, read_hashes: dict[str, str] | None) -> None:
    if read_hashes is None:
        return
    read_hashes[path.resolve().relative_to(root.resolve()).as_posix()] = sha256_file(path)


def _manifest_entries(
    root: Path,
    manifest_name: str = "MANIFEST.sha256",
    *,
    allow_known_generated_entries: bool = False,
) -> dict[str, str]:
    path = root / manifest_name
    if not path.is_file():
        raise ValueError(f"{manifest_name} is missing")
    entries: dict[str, str] = {}
    seen_paths: set[str] = set()
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"manifest cannot be read: {error}") from error
    if not lines:
        raise ValueError(f"{manifest_name} is empty")
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError(f"malformed manifest line: {line!r}")
        digest, relative = match.groups()
        relative = _safe_relative(relative)
        if relative in seen_paths:
            raise ValueError(f"duplicate manifest entry: {relative}")
        seen_paths.add(relative)
        if allow_known_generated_entries and _is_root_generated_path(relative):
            file_path = root / Path(relative)
            if not file_path.is_file() or sha256_file(file_path) != digest:
                raise ValueError(f"manifest mismatch: {relative}")
            # Older presentation publications accidentally included their
            # analysis manifest and bundle in the root manifest.  They are
            # verified above but remain outside the quantitative input
            # inventory, so immutable reanalysis can consume the raw run
            # without rewriting or trusting those generated files.
            continue
        entries[relative] = digest
        file_path = root / Path(relative)
        if not file_path.is_file() or sha256_file(file_path) != digest:
            raise ValueError(f"manifest mismatch: {relative}")
    actual = {
        relative
        for file_path in root.rglob("*")
        if file_path.is_file()
        for relative in [file_path.relative_to(root).as_posix()]
        if not _is_root_generated_path(relative)
    }
    if set(entries) != actual:
        missing = sorted(actual - set(entries))
        extra = sorted(set(entries) - actual)
        raise ValueError(f"manifest inventory mismatch; missing={missing}, extra={extra}")
    return entries


def validate_original_manifest(root: str | Path, *, allow_known_generated_entries: bool = False) -> dict[str, Any]:
    """Validate and return the immutable original-manifest evidence."""

    path = Path(root).resolve()
    entries = _manifest_entries(path, allow_known_generated_entries=allow_known_generated_entries)
    return {
        "path": str(path / "MANIFEST.sha256"),
        "sha256": sha256_file(path / "MANIFEST.sha256"),
        "entry_count": len(entries),
        "entries": entries,
    }


def write_sha256_manifest(root: str | Path) -> Path:
    """Write a strict manifest for a fixture or a newly staged revision."""

    directory = Path(root).resolve()
    files = sorted(
        (
            path.relative_to(directory).as_posix(),
            path,
        )
        for path in directory.rglob("*")
        if path.is_file() and not _is_root_generated_path(path.relative_to(directory).as_posix())
    )
    if not files:
        raise ValueError("cannot write a manifest for an empty directory")
    lines = [f"{sha256_file(path)}  {relative}" for relative, path in files]
    manifest = directory / "MANIFEST.sha256"
    temp = manifest.with_name(f".{manifest.name}.tmp-{os.getpid()}")
    temp.write_text("\n".join(lines) + "\n", encoding="ascii")
    os.replace(temp, manifest)
    return manifest


def _json_load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _safe_array(value: Any, *, name: str, dtype: Any | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} is empty or non-finite")
    return np.ascontiguousarray(array)


def _load_root_arrays(root: Path, *, read_hashes: dict[str, str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = ("z0.npy", "mask.npy", "direction_bank.npy")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError("required root arrays are missing: " + ", ".join(missing))
    try:
        z0 = _safe_array(np.load(root / "z0.npy", allow_pickle=False), name="z0", dtype=np.float32)
        _record_input_hash(root, root / "z0.npy", read_hashes)
        mask = np.asarray(np.load(root / "mask.npy", allow_pickle=False), dtype=bool)
        _record_input_hash(root, root / "mask.npy", read_hashes)
        directions = _safe_array(np.load(root / "direction_bank.npy", allow_pickle=False), name="direction bank", dtype=np.float32)
        _record_input_hash(root, root / "direction_bank.npy", read_hashes)
    except (OSError, ValueError) as error:
        raise ValueError(f"root quantitative arrays cannot be read: {error}") from error
    if mask.shape != z0.shape:
        raise ValueError("z0 and mask shapes differ")
    if directions.ndim != z0.ndim + 1 or directions.shape[1:] != z0.shape:
        raise ValueError("direction bank shape does not match z0")
    if not np.any(mask) or not np.any(~mask):
        raise ValueError("mask must contain both conditioned and predicted values")
    return z0, mask, directions


def _load_samples(root: Path, *, read_hashes: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    samples_root = root / "samples"
    if not samples_root.is_dir():
        raise ValueError("samples directory is missing")
    records: dict[str, dict[str, Any]] = {}
    for directory in sorted(item for item in samples_root.iterdir() if item.is_dir() and ".attempt" not in item.name):
        sample_json = directory / "sample.json"
        status_json = directory / "status.json"
        if not sample_json.is_file() or not status_json.is_file():
            raise ValueError(f"sample metadata is incomplete: {directory.name}")
        record = _json_load(sample_json)
        _record_input_hash(root, sample_json, read_hashes)
        status = _json_load(status_json)
        _record_input_hash(root, status_json, read_hashes)
        if status.get("status") not in {"success", "COMPLETE"}:
            raise ValueError(f"sample is not successful: {directory.name}")
        sample_id = record.get("sample_id", directory.name)
        if not isinstance(sample_id, str):
            raise ValueError(f"unsafe sample_id: {sample_id!r}")
        if sample_id in records:
            raise ValueError(f"duplicate sample_id: {sample_id!r}")
        record["sample_id"] = _validate_sample_id(sample_id, directory.name, require_match=False)
        record["_sample_directory_name"] = directory.name
        record["_sample_dir"] = str(directory)
        for path in sorted(directory.glob("*.npy")):
            try:
                record[path.stem] = np.load(path, allow_pickle=False)
                _record_input_hash(root, path, read_hashes)
            except (OSError, ValueError) as error:
                raise ValueError(f"sample tensor cannot be read: {path}: {error}") from error
        records[record["sample_id"]] = record
    if not records:
        raise ValueError("samples directory is empty")
    for record in records.values():
        _validate_sample_id(record["sample_id"], record["_sample_directory_name"])
    return records


def _spec(record: Mapping[str, Any]) -> tuple[int | None, float, int]:
    nested = record.get("spec") if isinstance(record.get("spec"), Mapping) else record
    try:
        direction = nested.get("direction_index")
        direction_value = None if direction is None else int(direction)
        alpha = float(nested.get("alpha", 0.0))
        sign = int(nested.get("sign", 0))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid sample perturbation spec: {record.get('sample_id')}") from error
    if not np.isfinite(alpha) or alpha < 0.0 or sign not in {-1, 0, 1}:
        raise ValueError(f"invalid sample perturbation values: {record.get('sample_id')}")
    return direction_value, alpha, sign


def _rms(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("RMS is undefined")
    return float(np.sqrt(np.mean(array * array, dtype=np.float64)))


def _cosine(left: Any, right: Any) -> float | None:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size == 0 or a.size != b.size or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return None
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denominator == 0.0 else float(np.dot(a, b) / denominator)


def _target_delta(record: Mapping[str, Any], z0: np.ndarray, mask: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    direction_index, alpha, sign = _spec(record)
    if direction_index is None:
        direction_index = 0
    if direction_index < 0 or direction_index >= directions.shape[0]:
        raise ValueError(f"direction index is outside direction bank: {record.get('sample_id')}")
    direction = directions[direction_index]
    expected = np.zeros_like(z0, dtype=np.float32)
    if sign:
        s_z = _rms(z0[mask])
        expected[mask] = np.float32(sign * alpha * s_z) * direction[mask]
    stored = record.get("target_delta_fp32")
    if stored is not None:
        observed = np.asarray(stored, dtype=np.float32)
        if observed.shape != expected.shape or observed.tobytes(order="C") != expected.tobytes(order="C"):
            raise ValueError(f"stored target delta does not match recomputed target: {record.get('sample_id')}")
    return expected, {
        "direction_index": direction_index,
        "alpha": alpha,
        "sign": sign,
        "target_rms_fp32": _rms(expected[mask]),
        "target_delta_sha256": _array_hash(expected),
    }


def _array_hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _delta_evidence(
    record: Mapping[str, Any], carrier_before: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate stored realized FP32/BF16 deltas against both carrier states."""

    after_value = record.get("realized_carrier_fp32")
    if after_value is None:
        after_value = record.get("carrier_after_fp32")
    stored_fp32 = record.get("actual_delta_fp32")
    stored_bf16 = record.get("actual_delta_bf16")
    sample_id = record.get("sample_id")
    if after_value is None:
        raise ValueError(f"realized carrier is required: {sample_id}")
    carrier_after = _safe_array(after_value, name=f"realized carrier {sample_id}", dtype=np.float32)
    if carrier_after.shape != carrier_before.shape:
        raise ValueError(f"realized carrier shape mismatch: {sample_id}")
    realized = (carrier_after - carrier_before).astype(np.float32, copy=False)
    if stored_fp32 is not None:
        observed_fp32 = np.asarray(stored_fp32, dtype=np.float32)
        if observed_fp32.shape != realized.shape or observed_fp32.tobytes(order="C") != realized.tobytes(order="C"):
            raise ValueError(f"stored actual FP32 delta does not match carrier_after-carrier_before: {sample_id}")
    effective = (quantize_bf16_fp32(carrier_after) - quantize_bf16_fp32(carrier_before)).astype(np.float32)
    if stored_bf16 is not None:
        observed_bf16 = np.asarray(stored_bf16, dtype=np.float32)
        if observed_bf16.shape != effective.shape or observed_bf16.tobytes(order="C") != effective.tobytes(order="C"):
            raise ValueError(f"stored BF16 effective delta does not match realized carriers: {sample_id}")
    return realized, effective


def _slice_output(record: Mapping[str, Any], mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    full = record.get("final_latent_full")
    if full is None:
        raise ValueError(f"final_latent_full is required: {record.get('sample_id')}")
    # ``latent_slicing`` is the persisted key written by the runner.  Do not
    # accept reanalysis-only aliases: doing so could silently turn a guessed
    # layout into evidence for an old run.
    observed_metadata = record.get("latent_slicing")
    if not isinstance(observed_metadata, Mapping):
        raise ValueError(f"latent_slicing metadata is required: {record.get('sample_id')}")
    output, metadata = slice_predicted_output(full, mask)
    required_metadata = ("axis", "source_shape", "selected_shape", "condition_indexes", "predicted_indexes")
    observed_keys = set(observed_metadata)
    required_keys = set(required_metadata)
    if observed_keys != required_keys:
        missing = sorted(required_keys - observed_keys)
        extra = sorted(observed_keys - required_keys)
        raise ValueError(
            f"latent_slicing schema does not match runner metadata "
            f"(missing={missing}, extra={extra}): {record.get('sample_id')}"
        )
    for key in required_metadata:
        observed_value = observed_metadata[key]
        if key == "axis":
            if isinstance(observed_value, bool) or not isinstance(observed_value, int):
                raise ValueError(f"latent_slicing axis must be an integer: {record.get('sample_id')}")
        else:
            if not isinstance(observed_value, list):
                raise ValueError(f"latent_slicing {key} must be a list: {record.get('sample_id')}")
        if observed_value != metadata[key]:
            raise ValueError(f"latent_slicing does not match raw final latent: {record.get('sample_id')}")
    predicted = record.get("predicted_latent")
    if predicted is not None:
        observed = np.asarray(predicted, dtype=np.float32)
        if observed.shape != output.shape or observed.tobytes(order="C") != output.astype(np.float32).tobytes(order="C"):
            raise ValueError(f"stored predicted output does not match raw output slicing: {record.get('sample_id')}")
    metadata = dict(metadata)
    metadata["source"] = "final_latent_full"
    return output.astype(np.float64, copy=False), metadata


def _response_metric_row(record: Mapping[str, Any], z0: np.ndarray, mask: np.ndarray, directions: np.ndarray) -> dict[str, Any]:
    carrier_value = record.get("carrier_fp32")
    if carrier_value is None:
        carrier_value = record.get("carrier_before_fp32")
    if carrier_value is None:
        raise ValueError(f"carrier_before is required: {record.get('sample_id')}")
    carrier = np.asarray(carrier_value, dtype=np.float32)
    if carrier.shape != z0.shape or carrier.tobytes(order="C") != z0.tobytes(order="C"):
        raise ValueError(f"sample carrier does not equal root z0: {record.get('sample_id')}")
    target, spec = _target_delta(record, z0, mask, directions)
    realized, effective = _delta_evidence(record, carrier, target)
    direction = directions[spec["direction_index"]]
    target_values = target[mask].astype(np.float64, copy=False)
    realized_values = realized[mask].astype(np.float64, copy=False)
    effective_values = effective[mask].astype(np.float64, copy=False)
    requested_rms = _rms(target_values)
    realized_rms = _rms(realized_values)
    effective_rms = _rms(effective_values)
    requested_nonzero_ratio = float(np.count_nonzero(target_values) / target_values.size)
    realized_nonzero_ratio = float(np.count_nonzero(realized_values) / realized_values.size)
    effective_nonzero_ratio = float(np.count_nonzero(effective_values) / effective_values.size)
    requested_direction_cosine = _cosine(target_values, direction[mask])
    realized_direction_cosine = _cosine(realized_values, direction[mask])
    effective_direction_cosine = _cosine(effective_values, direction[mask])
    return {
        "sample_id": str(record["sample_id"]),
        **spec,
        "requested_input_rms_fp32": requested_rms,
        "realized_input_rms_fp32": realized_rms,
        "actual_input_rms": realized_rms,
        "requested_input_rms_reason": "requested input is exactly zero" if requested_rms == 0.0 else None,
        "realized_input_rms_reason": "realized FP32 input is exactly zero" if realized_rms == 0.0 else None,
        "effective_input_rms_bf16": effective_rms,
        "bf16_input_rms": effective_rms,
        "effective_input_rms_reason": "effective BF16 input is exactly zero" if effective_rms == 0.0 else None,
        "requested_input_nonzero_ratio": requested_nonzero_ratio,
        "realized_input_nonzero_ratio": realized_nonzero_ratio,
        "effective_input_nonzero_ratio": effective_nonzero_ratio,
        "nonzero_ratio": realized_nonzero_ratio,
        "bf16_nonzero_ratio": effective_nonzero_ratio,
        "requested_input_direction_cosine": requested_direction_cosine,
        "realized_input_direction_cosine": realized_direction_cosine,
        "effective_input_direction_cosine": effective_direction_cosine,
        "direction_cosine": realized_direction_cosine,
        "bf16_direction_cosine": effective_direction_cosine,
        "realized_input_direction_cosine_reason": "realized input or direction norm is zero" if realized_direction_cosine is None else None,
        "effective_input_direction_cosine_reason": "effective input or direction norm is zero" if effective_direction_cosine is None else None,
        "requested_input_direction_cosine_reason": "requested input or direction norm is zero" if requested_direction_cosine is None else None,
        "network_visible_effective_delta_sha256": _array_hash(effective),
        "target_delta": target,
        "realized_delta": realized,
        "effective_delta": effective,
    }


def _baseline_record(records: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    for name in ("scan_pre", "baseline", "scan_post", "A", "C"):
        if name in records:
            return records[name]
    return next(iter(records.values()))


def _mapped_noise_mask(record: Mapping[str, Any], state: Any, carrier_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    initial_mask_value = record.get("initial_condition_mask")
    sample_id = record.get("sample_id")
    if initial_mask_value is None:
        raise ValueError(f"initial condition mask is required for sampler state: {sample_id}")
    state_array = _safe_array(state, name=f"initial sampler state {sample_id}", dtype=np.float32)
    initial_mask = np.asarray(initial_mask_value, dtype=bool)
    if state_array.size != carrier_mask.size or initial_mask.size != carrier_mask.size:
        raise ValueError(f"sampler state/carrier geometry mismatch: {sample_id}")
    if not np.array_equal(initial_mask.reshape(-1), carrier_mask.reshape(-1)):
        raise ValueError(f"initial condition mask does not match carrier geometry: {sample_id}")
    try:
        mapped = np.ascontiguousarray(initial_mask.reshape(state_array.shape), dtype=bool)
    except ValueError as error:
        raise ValueError(f"sampler state layout cannot map carrier geometry: {sample_id}") from error
    return state_array, mapped


def _paired_noise(records: Mapping[str, Mapping[str, Any]], mask: np.ndarray, baseline_name: str) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    for sample_id, record in records.items():
        state = record.get("initial_state")
        if state is None:
            raise ValueError(f"initial sampler state is required: {sample_id}")
        state_array, mapped_mask = _mapped_noise_mask(record, state, mask)
        try:
            hashes[sample_id] = fixed_noise_hash(state_array, mapped_mask)
        except (TypeError, ValueError) as error:
            raise ValueError(f"sampler noise evidence is invalid: {sample_id}: {error}") from error
    reference = hashes.get(baseline_name)
    if reference is None:
        raise ValueError(f"baseline sampler noise evidence is missing: {baseline_name}")
    mismatches = sorted(sample_id for sample_id, value in hashes.items() if value != reference)
    return {
        "reference_sample": baseline_name,
        "hashes": hashes,
        "mismatches": mismatches,
        "passed": bool(hashes) and not mismatches,
        "reasons": {},
    }


def _network_condition_observation(record: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize the genuine run09 network-condition capture, if present.

    The old runner stores the BF16 condition tensor as an FP32-readable NumPy
    artifact and records its native dtype separately.  This is evidence about
    the network-condition capture only; it is not evidence that A/B/C shared a
    common interface tensor.
    """

    value = record.get("network_condition_bf16")
    dtype = record.get("network_condition_dtype")
    if dtype is None:
        dtype = record.get("network_dtype") or record.get("native_dtype")
    if dtype is None and isinstance(record.get("condition_capture"), Mapping):
        dtype = record["condition_capture"].get("network_condition_dtype")
    expected_hash = record.get("expected_network_condition_hash")
    step_hashes = record.get("condition_step_hashes")
    if step_hashes is None:
        step_hashes = record.get("denoise_condition_hashes")
    observation: dict[str, Any] = {
        "observed": False,
        "native_dtype": None if dtype is None else str(dtype),
        "stored_array_dtype": None,
        "shape": None,
        "sha256": None,
        "expected_hash": expected_hash if isinstance(expected_hash, str) else None,
        "condition_step_hash_count": len(step_hashes) if isinstance(step_hashes, (list, tuple)) else 0,
        "reason": None,
    }
    if value is None:
        observation["reason"] = "network_condition_bf16 artifact is missing"
        return observation
    if dtype is None:
        observation["reason"] = "network_condition_bf16 native dtype observation is missing"
        return observation
    try:
        array = _safe_array(value, name=f"network condition {record.get('sample_id')}", dtype=np.float32)
    except (TypeError, ValueError) as error:
        observation["reason"] = f"network_condition_bf16 artifact is invalid: {error}"
        return observation
    observation.update({
        "observed": True,
        "stored_array_dtype": str(np.asarray(value).dtype),
        "shape": list(array.shape),
        "sha256": _array_hash(array),
    })
    return observation


def _precision_contract_from_records(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    network_observations = {
        name: _network_condition_observation(record) for name, record in records.items()
    }
    unavailable_reason = (
        "run09 lacks actual A/B/C common interface_fp32 observations; "
        "full cross-group identity validation is reserved for Task 3 artifacts"
    )
    required = ("A", "B", "C")
    if not all(name in records for name in required):
        contract = build_precision_contract()
        contract["observation_reason"] = unavailable_reason
        contract["network_condition_observations"] = network_observations
        contract["cross_group_identity_observed"] = False
        contract["cross_group_identity_reason"] = unavailable_reason
        contract["cross_group_identity"] = {
            "observed": False,
            "value": None,
            "reason": unavailable_reason,
        }
        return contract
    interface: dict[str, Any] = {}
    dtypes: dict[str, str] = {}
    for name in required:
        record = records[name]
        value = record.get("interface_fp32")
        dtype = record.get("network_dtype") or record.get("native_dtype") or record.get("network_condition_dtype")
        if dtype is None and isinstance(record.get("condition_capture"), Mapping):
            dtype = record["condition_capture"].get("network_condition_dtype")
        if value is None or dtype is None:
            contract = build_precision_contract()
            contract["observation_reason"] = unavailable_reason
            contract["network_condition_observations"] = network_observations
            contract["cross_group_identity_observed"] = False
            contract["cross_group_identity_reason"] = unavailable_reason
            contract["cross_group_identity"] = {
                "observed": False,
                "value": None,
                "reason": unavailable_reason,
            }
            return contract
        interface[name] = value
        dtypes[name] = str(dtype)
    contract = build_precision_contract(interface_tensors=interface, network_dtypes=dtypes)
    contract["network_condition_observations"] = network_observations
    contract["cross_group_identity_observed"] = True
    contract["cross_group_identity_reason"] = None
    contract["cross_group_identity"] = {
        "observed": True,
        "value": contract["checks"],
        "reason": None,
    }
    return contract


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_metrics_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    keys: list[str] = []
    for row in values:
        for key, value in row.items():
            if isinstance(value, (np.ndarray, list, tuple, dict)):
                continue
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys or ["status"])
        writer.writeheader()
        for row in values:
            writer.writerow({key: "" if row.get(key) is None else _json_safe(row.get(key)) for key in (keys or ["status"])})


def _write_npy(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _stage_output_path(stage: Path, *parts: str) -> Path:
    """Resolve a generated path and prove it remains beneath the stage root."""

    if not parts or any(not isinstance(part, str) or not part or part in {".", ".."} for part in parts):
        raise ValueError(f"unsafe staged output path: {parts!r}")
    candidate = stage.joinpath(*parts).resolve()
    stage_root = stage.resolve()
    try:
        candidate.relative_to(stage_root)
    except ValueError as error:
        raise ValueError(f"staged output escaped stage directory: {candidate}") from error
    return candidate


def _atomic_rename_noreplace(stage: Path, output: Path) -> None:
    """Atomically move a staged directory without replacing ``output``.

    ``os.rename`` replaces an empty destination directory on POSIX.  The
    platform primitives below provide the no-replace guarantee needed for a
    revision publication race: Linux ``renameat2(RENAME_NOREPLACE)`` and
    Windows ``MoveFileW`` (whose default semantics reject an existing target).
    """

    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file = kernel32.MoveFileW
        move_file.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        move_file.restype = ctypes.c_bool
        if move_file(str(stage), str(output)):
            return
        error = ctypes.get_last_error()
        if error in (80, 183):  # ERROR_FILE_EXISTS / ERROR_ALREADY_EXISTS
            raise FileExistsError(error, os.strerror(error), str(output))
        raise OSError(error, os.strerror(error), str(output))

    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOSYS, "renameat2 is unavailable; no no-replace publish primitive")
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        # AT_FDCWD = -100, RENAME_NOREPLACE = 1.
        result = renameat2(-100, os.fsencode(stage), -100, os.fsencode(output), 1)
        if result == 0:
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), str(output))
        raise OSError(error, os.strerror(error), str(output))

    raise OSError(errno.ENOTSUP, "no atomic no-replace directory publish primitive for this platform")


def _publish_revision(stage: Path, output: Path) -> None:
    write_sha256_manifest(stage)
    if output.exists() or os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing revision directory: {output}")
    try:
        _atomic_rename_noreplace(stage, output)
    except FileExistsError as error:
        raise FileExistsError(f"output appeared during publish: {output}") from error


def reanalyze_old_run(source_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Recompute precision evidence from an immutable old run.

    The source must contain a strict ``MANIFEST.sha256`` and the raw arrays
    used by the old runner.  The new revision is staged beside the destination
    and published with one atomic no-overwrite directory move; an existing
    destination is never overwritten.
    """

    source = Path(source_dir).resolve()
    if not source.is_dir():
        raise ValueError(f"source run is not a directory: {source}")
    if output_dir is None:
        output = source.parent / f"{source.name}_precision_reanalysis"
    else:
        output = Path(output_dir).resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("source and output directories must not be equal or nested")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing revision directory: {output}")
    manifest_before = validate_original_manifest(source, allow_known_generated_entries=True)
    read_hashes: dict[str, str] = {}
    z0, mask, directions = _load_root_arrays(source, read_hashes=read_hashes)
    records = _load_samples(source, read_hashes=read_hashes)
    baseline = _baseline_record(records)
    baseline_row = _response_metric_row(baseline, z0, mask, directions)
    baseline_output, baseline_slice = _slice_output(baseline, mask)
    rows: list[dict[str, Any]] = []
    for record in records.values():
        row = _response_metric_row(record, z0, mask, directions)
        output_array, slicing = _slice_output(record, mask)
        response = output_array.astype(np.float64, copy=False) - baseline_output.astype(np.float64, copy=False)
        output_rms = _rms(response)
        row.update({
            "output_rms": output_rms,
            "output_rms_reason": "response is exactly zero" if output_rms == 0.0 else None,
            "output_response": response,
            "predicted_output": output_array,
            "output_slicing": slicing,
        })
        rows.append(row)
    by_key = {(row["direction_index"], row["alpha"], row["sign"]): row for row in rows if row["sign"] in {-1, 1}}
    adjacent: list[dict[str, Any]] = []
    for direction in range(directions.shape[0]):
        for sign in (-1, 1):
            ordered = sorted((row for row in rows if row["direction_index"] == direction and row["sign"] == sign), key=lambda row: row["alpha"])
            for left, right in zip(ordered, ordered[1:]):
                input_metric = metric_with_reason("realized FP32 input adjacent change", left["realized_delta"][mask], right["realized_delta"][mask])
                output_metric = metric_with_reason("output adjacent change", left["output_response"], right["output_response"])
                adjacent.append({
                    "direction_index": direction,
                    "sign": sign,
                    "alpha_left": left["alpha"],
                    "alpha_right": right["alpha"],
                    "input_direction_cosine": _cosine(left["realized_delta"][mask], right["realized_delta"][mask]),
                    "input_relative_change": input_metric["value"],
                    "input_relative_change_reason": input_metric["reason"],
                    "output_relative_change": output_metric["value"],
                    "output_relative_change_reason": output_metric["reason"],
                })
    direction_adjacency: list[dict[str, Any]] = []
    alpha_sign_groups = sorted({(row["alpha"], row["sign"]) for row in rows if row["sign"] in {-1, 1}})
    for alpha, sign in alpha_sign_groups:
        ordered = [by_key.get((direction, alpha, sign)) for direction in range(directions.shape[0])]
        for left_direction, (left, right) in enumerate(zip(ordered, ordered[1:])):
            if left is None or right is None:
                direction_adjacency.append({
                    "alpha": alpha,
                    "sign": sign,
                    "direction_left": left_direction,
                    "direction_right": left_direction + 1,
                    "input_direction_cosine": None,
                    "input_direction_cosine_reason": "one adjacent direction sample is missing",
                    "output_direction_cosine": None,
                    "output_direction_cosine_reason": "one adjacent direction sample is missing",
                })
                continue
            input_cosine = _cosine(left["realized_delta"][mask], right["realized_delta"][mask])
            output_cosine = _cosine(left["output_response"], right["output_response"])
            direction_adjacency.append({
                "alpha": alpha,
                "sign": sign,
                "direction_left": left_direction,
                "direction_right": left_direction + 1,
                "input_direction_cosine": input_cosine,
                "input_direction_cosine_reason": "one input norm is zero" if input_cosine is None else None,
                "output_direction_cosine": output_cosine,
                "output_direction_cosine_reason": "one response norm is zero" if output_cosine is None else None,
            })
    sign_reversal: list[dict[str, Any]] = []
    for (direction, alpha, sign), plus in sorted(by_key.items()):
        if sign != 1 or (direction, alpha, -1) not in by_key:
            continue
        minus = by_key[(direction, alpha, -1)]
        input_sum = plus["realized_delta"][mask].astype(np.float64) + minus["realized_delta"][mask].astype(np.float64)
        response_plus = plus["predicted_output"]
        response_minus = minus["predicted_output"]
        y0 = baseline_output
        input_cosine = _cosine(plus["realized_delta"][mask], -minus["realized_delta"][mask])
        output_cosine = _cosine(response_plus - y0, -(response_minus - y0))
        sign_reversal.append({
            "direction_index": direction,
            "alpha": alpha,
            "input_sign_reversal_cosine": input_cosine,
            "input_sign_reversal_cosine_reason": "effective input norm is zero" if input_cosine is None else None,
            "input_sign_reversal_residual_rms": _rms(input_sum),
            "output_even_symmetry_residual": even_symmetry_residual(response_plus, response_minus, y0),
            "output_sign_reversal_cosine": output_cosine,
            "output_sign_reversal_cosine_reason": "response norm is zero" if output_cosine is None else None,
        })
    noise = _paired_noise(records, mask, str(baseline["sample_id"]))
    precision_contract = _precision_contract_from_records(records)
    vector_identity: dict[str, Any] | None = None
    vector_identity_reason: str | None = precision_contract.get("cross_group_identity_reason")
    if (
        all(name in records for name in ("A", "B", "C"))
        and precision_contract.get("cross_group_identity_observed") is True
    ):
        outputs = {name: _slice_output(records[name], mask)[0] for name in ("A", "B", "C")}
        vector_identity = decompose_response_vectors(outputs["A"], outputs["B"], outputs["C"])
        vector_identity = {key: value for key, value in vector_identity.items() if key not in {"identity_error", "reconstructed", "r_A_minus_r_C", "r_B_minus_r_C", "r_A_minus_r_B"}}
        vector_identity_reason = None
    public_rows: list[dict[str, Any]] = []
    for row in rows:
        public_rows.append({
            key: value
            for key, value in row.items()
            if key not in {"target_delta", "realized_delta", "effective_delta", "output_response", "predicted_output"}
        })
    metrics = {
        "status": "COMPLETE",
        "source_run": str(source),
        "original_manifest_sha256": manifest_before["sha256"],
        "original_manifest_entry_count": manifest_before["entry_count"],
        "baseline_sample": str(baseline["sample_id"]),
        "target_delta_recomputed": True,
        "network_visible_effective_delta": "BF16(realized_carrier)-BF16(carrier_before), represented as FP32",
        "source_input_hashes": dict(sorted(read_hashes.items())),
        "precision_contract": precision_contract,
        "rows": public_rows,
        "adjacent_direction_consistency": adjacent,
        "direction_adjacency_consistency": direction_adjacency,
        "sign_reversal_consistency": sign_reversal,
        "paired_sampler_noise": noise,
        "predicted_output_slicing": {"baseline": baseline_slice, "samples": {row["sample_id"]: row["output_slicing"] for row in rows}},
        "vector_decomposition": vector_identity,
        "vector_decomposition_reason": vector_identity_reason,
        "media_metric_policy": "MP4/PNG are not read or used for metrics",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=str(output.parent)))
    try:
        _write_json(stage / "metrics.json", metrics)
        _write_metrics_csv(stage / "input_output_metrics.csv", public_rows)
        for row in rows:
            sample_label = _validate_sample_id(str(row["sample_id"]), str(row["sample_id"]))
            _write_npy(_stage_output_path(stage, "analysis_tensors", "target_delta", f"{sample_label}.npy"), row["target_delta"])
            _write_npy(_stage_output_path(stage, "analysis_tensors", "realized_delta_fp32", f"{sample_label}.npy"), row["realized_delta"])
            _write_npy(_stage_output_path(stage, "analysis_tensors", "effective_delta_bf16", f"{sample_label}.npy"), row["effective_delta"])
            _write_npy(_stage_output_path(stage, "analysis_tensors", "predicted_output", f"{sample_label}.npy"), row["predicted_output"])
        _write_json(stage / "original_manifest_snapshot.json", manifest_before)
        # Persist only scalar, source-binding metadata.  Raw old tensors stay
        # at the source path and are never copied or modified.
        _write_json(stage / "reanalysis_provenance.json", {
            "type": "umi_precision_reanalysis",
            "source_run": str(source),
            "source_manifest_sha256": manifest_before["sha256"],
            "source_manifest_entry_count": manifest_before["entry_count"],
            "source_input_hashes": dict(sorted(read_hashes.items())),
            "source_read_only": True,
            "metric_differences": {
                "relative_derivative_change_denominator": "left",
                "even_symmetry_residual": "RMS(Y_plus + Y_minus - 2Y0), unnormalised",
            },
        })
        # Re-read the complete source manifest immediately before publication.
        # Hashing MANIFEST.sha256 alone would miss a mutated input file.
        manifest_after = validate_original_manifest(source, allow_known_generated_entries=True)
        if manifest_after["entries"] != manifest_before["entries"] or manifest_after["sha256"] != manifest_before["sha256"]:
            raise ValueError("source manifest changed during reanalysis")
        _publish_revision(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "status": "COMPLETE",
        "output_dir": str(output),
        "source_manifest_sha256": manifest_before["sha256"],
        "sample_count": len(records),
    }


reanalyze_run = reanalyze_old_run
reanalyse_old_run = reanalyze_old_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Immutable UMI old-run precision reanalysis")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = reanalyze_old_run(args.source_dir, args.output_dir)
    print(json.dumps(_json_safe(result), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "is_task4_raw_manifest_excluded",
    "reanalyse_old_run",
    "reanalyze_old_run",
    "reanalyze_run",
    "sha256_file",
    "validate_original_manifest",
    "write_sha256_manifest",
]
