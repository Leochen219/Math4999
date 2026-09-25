"""Offline true-error spectrum analysis of immutable Task 10 float outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from umi_task10_real_scenes import LOCKED_RECORDS, SEED_PAIRS, build_call_plan
from umi_task11_error_analysis import error_decomposition, error_spectrum, prediction_error, rgb_endpoint_metrics, leave_one_episode_out


TASK10_METRIC_TOLERANCE = {"rtol": 1e-10, "atol": 1e-12}
TASK10_REPRODUCED_METRICS = ("g0_latent_rms", "tf_rmse", "ar_rmse", "tf2_latent_rms", "ar2_latent_rms")
_TASK10_METRIC_CSV_REQUIRED_FIELDS = ("record_index", "seed_pair", *TASK10_REPRODUCED_METRICS)
TASK11_ANALYSIS_CONFIG = {"requested_loeo_ranks": [1, 2, 4, 8, 10],
                          "error_and_reduction_dtype": "float64",
                          "task10_metric_reproduction_tolerance": TASK10_METRIC_TOLERANCE,
                          "prediction_error_convention": "prediction_minus_aligned_truth"}
_TASK8_ARRAY_FIELDS = ("condition_input_fp32", "action", "output_full", "decoded_rgb_full",
                       "decoded_raw", "generated_rgb", "decoded_last_rgb", "encoded_condition",
                       "condition_steps_fp32")
_ERROR_MODES = {"E1": ("G0", "gt_condition_x16", 16),
                "ETF2": ("TF2", "gt_condition_x32", 32),
                "EAR2": ("AR2", "gt_condition_x32", 32)}


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


def _task8_descriptor_hash(value: Any) -> str:
    """Match task8_live._sha, used by Task 8's JSON-safe array descriptors."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii")); digest.update(b"\0")
    digest.update(repr(tuple(int(dim) for dim in array.shape)).encode("ascii")); digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def verify_ar_feedback_condition(g0_encoded_condition: Any, ar_condition_input: Any) -> None:
    """Require AR2 to consume the exact FP32 encoded endpoint produced by G0."""
    g0 = np.asarray(g0_encoded_condition)
    ar_input = np.asarray(ar_condition_input)
    if (g0.dtype != np.float32 or ar_input.dtype != np.float32 or g0.shape != ar_input.shape
            or not np.array_equal(np.ascontiguousarray(g0).view(np.uint8),
                                  np.ascontiguousarray(ar_input).view(np.uint8))):
        raise ValueError("Task 10 AR2 condition input differs from G0 encoded condition")


def verify_masked_condition_consumption(condition_input: Any, condition_steps: Any,
                                        generation_mask: Any, truth_full_mask: Any) -> None:
    """Verify every consumed condition-carrier slot, not generated-region state."""
    condition = np.asarray(condition_input)
    steps = np.asarray(condition_steps)
    mask = np.asarray(generation_mask)
    expected_mask = np.asarray(truth_full_mask)
    if (condition.dtype != np.float32 or steps.dtype != np.float32
            or mask.dtype != np.bool_ or expected_mask.dtype != np.bool_
            or not np.array_equal(mask, expected_mask)):
        raise ValueError("Task 10 generation mask differs from the saved full condition mask")
    if (steps.ndim != condition.ndim + 1 or steps.shape[0] != 30
            or mask.shape != steps.shape[1:] or condition.size == 0
            or not np.isfinite(condition).all() or not np.isfinite(steps).all()):
        raise ValueError("Task 10 condition carrier has invalid dtype/shape/values")
    try:
        expanded_condition = np.broadcast_to(condition, steps.shape)
        expanded_mask = np.broadcast_to(mask, steps.shape[1:])
    except ValueError as error:
        raise ValueError("Task 10 condition input cannot broadcast to the actual carrier") from error
    if (not np.any(expanded_mask)
            or not np.array_equal(steps[:, expanded_mask], expanded_condition[:, expanded_mask])):
        raise ValueError("Task 10 consumed condition positions differ from FP32 condition input")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence is not an object: {path}")
    return value


def validate_source_identity(identity: Mapping[str, Any], status: Mapping[str, Any]) -> None:
    """Require exactly the immutable Task 10 six-record/two-schedule matrix."""
    expected_records = list(LOCKED_RECORDS)
    expected_pairs = [list(pair) for pair in SEED_PAIRS]
    if (identity.get("schema") != "umi-task10-v1"
            or identity.get("record_indices") != expected_records
            or identity.get("seed_pairs") != expected_pairs):
        raise ValueError("Task 10 source identity differs from the locked records or seed schedules")
    if (status.get("status") != "GENERATION_COMPLETE" or status.get("completed_samples") != 48
            or status.get("record_indices") != expected_records or status.get("seed_pairs") != expected_pairs):
        raise ValueError("Task 10 generation is not complete for the exact 48-call matrix")


def verify_task10_manifest(run_root: str | Path) -> dict[str, Any]:
    """Verify every file bound by Task 10's manifest and reject unlisted source files."""
    root = Path(run_root).resolve()
    manifest_path = root / "MANIFEST.sha256"
    if not manifest_path.is_file():
        raise ValueError("Task 10 run lacks MANIFEST.sha256")
    entries: dict[str, str] = {}
    for number, line in enumerate(manifest_path.read_text(encoding="ascii").splitlines(), 1):
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise ValueError(f"invalid Task 10 manifest row {number}") from error
        path = Path(relative)
        candidate = (root / path).resolve()
        if (len(digest) != 64 or path.is_absolute() or ".." in path.parts
                or not candidate.is_relative_to(root) or relative in entries):
            raise ValueError(f"unsafe or duplicate Task 10 manifest path on row {number}")
        if not candidate.is_file() or sha256_file(candidate) != digest.lower():
            raise ValueError(f"Task 10 manifest hash mismatch: {relative}")
        entries[relative.replace("\\", "/")] = digest.lower()
    auxiliary_paths = {"analysis/review_bundle.zip", "analysis/review_bundle.sha256"}
    actual_files = {path.relative_to(root).as_posix(): path for path in root.rglob("*")
                    if path.is_file() and path != manifest_path}
    auxiliary_present = (auxiliary_paths & set(actual_files)) - set(entries)
    sidecar = actual_files.get("analysis/review_bundle.sha256")
    sidecar_digest = None
    if sidecar is not None:
        lines = sidecar.read_text(encoding="ascii").splitlines()
        if (len(lines) != 1 or len(lines[0].split("  ", 1)) != 2
                or lines[0].split("  ", 1)[1] != "review_bundle.zip"
                or len(lines[0].split("  ", 1)[0]) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in lines[0].split("  ", 1)[0])):
            raise ValueError("Task 10 review-bundle SHA sidecar is malformed")
        sidecar_digest = lines[0].split("  ", 1)[0].lower()
        bundle = actual_files.get("analysis/review_bundle.zip")
        if bundle is not None and sha256_file(bundle) != sidecar_digest:
            raise ValueError("Task 10 review-bundle SHA sidecar does not match its archive")
    if ("analysis/review_bundle.zip" in auxiliary_present
            and (sidecar is None or sidecar_digest is None)):
        raise ValueError("unmanifested Task 10 review-bundle archive lacks a valid checksum sidecar")
    actual = set(actual_files)
    actual -= auxiliary_present
    if actual != set(entries):
        missing, unlisted = sorted(set(entries) - actual), sorted(actual - set(entries))
        raise ValueError(f"Task 10 manifest inventory mismatch (missing={missing[:3]}, unlisted={unlisted[:3]})")
    auxiliary_evidence = []
    if "analysis/review_bundle.zip" in auxiliary_present:
        auxiliary_evidence.append({"path": "analysis/review_bundle.zip",
                                   "sha256": sha256_file(actual_files["analysis/review_bundle.zip"]),
                                   "source_manifest_bound": False,
                                   "validation_scope": "sidecar-matched review-package auxiliary; not source evidence"})
    if sidecar is not None and "analysis/review_bundle.sha256" in auxiliary_present:
        auxiliary_evidence.append({"path": "analysis/review_bundle.sha256", "sha256": sha256_file(sidecar),
                                   "declared_archive_sha256": sidecar_digest,
                                   "archive_present": "analysis/review_bundle.zip" in actual_files,
                                   "sidecar_matches_archive": (None if "analysis/review_bundle.zip" not in actual_files
                                                                else sha256_file(actual_files["analysis/review_bundle.zip"])
                                                                == sidecar_digest),
                                   "source_manifest_bound": False,
                                   "validation_scope": "sidecar syntax/archive checksum only; not source evidence"})
    return {"file_count": len(entries), "manifest_sha256": sha256_file(manifest_path),
            "files": entries, "excluded_auxiliary_files": sorted(auxiliary_present),
            "excluded_auxiliary_evidence": auxiliary_evidence}


def _walk_array_descriptors(value: Any):
    if isinstance(value, Mapping):
        if "artifact" in value:
            yield value
            return
        for child in value.values():
            yield from _walk_array_descriptors(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_array_descriptors(child)


def validate_task8_sample(sample_root: str | Path) -> dict[str, Any]:
    """Verify all Task 8 sample artifact hashes without loading tensor payloads."""
    root = Path(sample_root).resolve()
    status_path, record_path = root / "status.json", root / "record.json"
    status = _load_json(status_path)
    if status.get("status") != "success":
        raise ValueError(f"Task 10 sample is not a successful immutable sample: {root}")
    hashes = status.get("artifact_sha256")
    if not isinstance(hashes, Mapping) or "record.json" not in hashes:
        raise ValueError(f"Task 10 sample hash manifest is missing: {root}")
    for name, expected in hashes.items():
        if (not isinstance(name, str) or Path(name).name != name or not isinstance(expected, str)
                or len(expected) != 64):
            raise ValueError(f"unsafe Task 10 sample artifact manifest: {root}")
        artifact = root / name
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError(f"Task 10 sample artifact hash mismatch: {artifact}")
    record = _load_json(record_path)
    for descriptor in _walk_array_descriptors(record):
        name = descriptor.get("artifact")
        if (not isinstance(name, str) or Path(name).name != name or name not in hashes
                or not isinstance(descriptor.get("dtype"), str)
                or not isinstance(descriptor.get("shape"), list)):
            raise ValueError(f"invalid array descriptor in Task 10 sample: {root}")
    return record


def load_record_array(sample_root: str | Path, record: Mapping[str, Any], key: str) -> np.ndarray:
    """Open one verified .npy tensor by memory map; caller closes it after use."""
    descriptor = record.get(key)
    return load_array_descriptor(sample_root, descriptor, key)


def load_array_descriptor(sample_root: str | Path, descriptor: Any, label: str) -> np.ndarray:
    """Open one nested or top-level Task 8 array descriptor by memory map."""
    if not isinstance(descriptor, Mapping) or "artifact" not in descriptor:
        raise ValueError(f"Task 10 sample lacks array field {label}")
    path = Path(sample_root) / str(descriptor["artifact"])
    if not path.is_file():
        raise ValueError(f"Task 10 sample array is missing: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if str(array.dtype) != descriptor.get("dtype") or list(array.shape) != descriptor.get("shape"):
        array._mmap.close()
        raise ValueError(f"Task 10 sample array descriptor mismatch: {path}")
    return array


def validate_metric_reproduction(source_row: Mapping[str, Any], reproduced: Mapping[str, Any], *,
                                 rtol: float = 1e-10, atol: float = 1e-12) -> dict[str, Any]:
    """Compare the float64 reproduction to Task 10's published stratum metrics."""
    metrics = {}
    for key in TASK10_REPRODUCED_METRICS:
        try:
            source_value, actual_value = float(source_row[key]), float(reproduced[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Task 10 metric reproduction lacks finite numeric field {key}") from error
        if not np.isfinite(source_value) or not np.isfinite(actual_value):
            raise ValueError(f"Task 10 metric reproduction has nonfinite field {key}")
        difference = abs(source_value - actual_value)
        allowed = atol + rtol * abs(source_value)
        metrics[key] = {"task10": source_value, "reproduced": actual_value,
                        "absolute_difference": difference, "allowed_difference": allowed,
                        "within_tolerance": difference <= allowed}
        if difference > allowed:
            raise ValueError(f"Task 10 metric reproduction failed for {key}: {difference} > {allowed}")
    return {"status": "WITHIN_DECLARED_TOLERANCE", "tolerance": {"rtol": float(rtol), "atol": float(atol)},
            "metrics": metrics}


def _read_npz_array(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise ValueError(f"preflight NPZ lacks {key}: {path}")
        return np.array(archive[key], copy=True)


def _read_npz_first_axis_slice(path: Path, key: str, index: int) -> np.ndarray:
    """Read one NPY row from a compressed NPZ member without retaining all frames."""
    try:
        with zipfile.ZipFile(path, "r") as archive:
            member_name = key + ".npy"
            if member_name not in archive.namelist():
                raise ValueError(f"preflight NPZ lacks {key}: {path}")
            with archive.open(member_name, "r") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"unsupported NPY header version {version} for {key}")
                if not shape or dtype.hasobject or fortran_order:
                    raise ValueError(f"preflight array {key} is not a row-major numeric array")
                if index < 0 or index >= shape[0]:
                    raise ValueError(f"frame index {index} is outside {key} with shape {shape}")
                row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
                bytes_to_skip = index * row_bytes
                while bytes_to_skip:
                    skipped = stream.read(min(bytes_to_skip, 1024 * 1024))
                    if not skipped:
                        raise ValueError(f"truncated preflight array {key}")
                    bytes_to_skip -= len(skipped)
                payload = stream.read(row_bytes)
                if len(payload) != row_bytes:
                    raise ValueError(f"truncated preflight frame {key}[{index}]")
                return np.frombuffer(payload, dtype=dtype).reshape(shape[1:]).copy()
    except (OSError, zipfile.BadZipFile, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(("preflight", "frame index", "truncated")):
            raise
        raise ValueError(f"cannot read preflight NPZ array {key} from {path}") from error


def _read_csv_rows(path: Path) -> dict[tuple[int, tuple[int, int]], dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Task 10 source metric table is missing: {path}")
    rows = {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing_fields = sorted(set(_TASK10_METRIC_CSV_REQUIRED_FIELDS) - set(reader.fieldnames or ()))
        if missing_fields:
            raise ValueError(f"Task 10 stratum metrics lack required fields: {missing_fields}")
        for row in reader:
            try:
                key = int(row["record_index"]), tuple(json.loads(row["seed_pair"]))
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid Task 10 stratum metrics row in {path}") from error
            if key in rows:
                raise ValueError(f"duplicate Task 10 metric row: {key}")
            if len(key[1]) != 2 or key[1] not in SEED_PAIRS or key[0] not in LOCKED_RECORDS:
                raise ValueError(f"unlocked Task 10 metric row: {key}")
            for metric in TASK10_REPRODUCED_METRICS:
                try:
                    value = float(row[metric])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Task 10 stratum metric is not numeric: {metric}/{key}") from error
                if not np.isfinite(value):
                    raise ValueError(f"Task 10 stratum metric is nonfinite: {metric}/{key}")
            rows[key] = row
    if set(rows) != {(record, pair) for record in LOCKED_RECORDS for pair in SEED_PAIRS}:
        raise ValueError("Task 10 stratum metrics do not contain the exact 12 episode/schedule rows")
    return rows


def _close(array: Any) -> None:
    closer = getattr(array, "_mmap", None)
    if closer is not None:
        closer.close()


def _expected_noise_binding(pair: tuple[int, int]) -> str:
    payload = {"policy": "prediction-region-seeded",
               "seeds": [pair[0], pair[0], pair[1], pair[1]], "pair": "TF2-AR2"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _preflight_paths(root: Path, record_index: int) -> tuple[Path, Path]:
    folder = root / "preflight" / f"record_{record_index:02d}"
    return folder / "task8_preflight.json", folder / "inputs.npz"


def _verify_episode_receipts(root: Path, record_index: int, identity: Mapping[str, Any]) -> dict[str, Any]:
    report_path, input_path = _preflight_paths(root, record_index)
    report = _load_json(report_path)
    selected = report.get("selected") or {}
    input_metadata = selected.get("inputs_npz") or {}
    if (report.get("status") != "PREFLIGHT_PASSED_GENERATION_NOT_PERFORMED"
            or report.get("record_count") != 52 or report.get("manifest_sha256_match") is not True
            or selected.get("record_index") != record_index
            or input_metadata.get("rgb_float32_shape") != [33, 3, 256, 256]
            or input_metadata.get("actions_normalized_shape") != [2, 16, 10]
            or not input_path.is_file() or sha256_file(input_path) != input_metadata.get("sha256")):
        raise ValueError(f"Task 10 preflight identity/hash/interface failed for record {record_index}")
    camera = report.get("camera") or {}
    if (camera.get("feature_key") != "steps/observation/image_0"
            or camera.get("all_window_frames_rgb_256x256") is not True
            or camera.get("all_window_frames_nonconstant") is not True
            or int(camera.get("unique_encoded_frames", 0)) <= 1):
        raise ValueError(f"Task 10 preflight camera evidence failed for record {record_index}")
    if not str(selected.get("language", "")).strip():
        raise ValueError(f"Task 10 preflight language is empty for record {record_index}")
    temporal = report.get("temporal_alignment") or {}
    if any(temporal.get(key) is not True for key in
           ("sequence_index_order_proven", "frequency_continuity_proven_from_official_source",
            "flags_prove_first_last_boundaries")):
        raise ValueError(f"Task 10 preflight temporal evidence failed for record {record_index}")

    condition_path = root / "records" / f"record_{record_index:02d}" / "ground_truth_conditions.npz"
    receipt_path = condition_path.with_name("ground_truth_conditions.sha256.json")
    evidence_path = condition_path.with_name("ground_truth_condition_evidence.json")
    receipt = _load_json(receipt_path)
    if (not condition_path.is_file() or not evidence_path.is_file()
            or sha256_file(condition_path) != receipt.get("npz_sha256")
            or sha256_file(evidence_path) != receipt.get("evidence_sha256")
            or receipt.get("record_index") != record_index
            or receipt.get("input_sha256") != sha256_file(input_path)
            or receipt.get("vae_sha256") != identity.get("vae_sha256")):
        raise ValueError(f"Task 10 ground-truth receipt/hash mismatch for record {record_index}")
    evidence = _load_json(evidence_path)
    if (evidence.get("source") != "fresh_temporary_fp32_encoder"
            or evidence.get("setup_bf16_not_reused") is not True
            or not isinstance(evidence.get("conditions"), Mapping)):
        raise ValueError(f"Task 10 ground-truth encoding evidence failed for record {record_index}")
    geometry = _read_truth_geometry(condition_path)
    return {"report": report, "report_path": report_path, "input_path": input_path,
            "input_sha256": sha256_file(input_path), "condition_path": condition_path,
            "receipt": receipt, "evidence": evidence, **geometry,
            "episode_id": selected.get("episode_id"), "file_path": selected.get("file_path"),
            "language": selected.get("language")}


def _validate_array_contracts(record: Mapping[str, Any], call: str, sample_root: Path,
                              geometry: Mapping[str, Any]) -> None:
    condition_shape = tuple(geometry["condition_shape"])
    for name in _TASK8_ARRAY_FIELDS:
        if not isinstance(record.get(name), Mapping) or "artifact" not in record[name]:
            raise ValueError(f"Task 10 {call} lacks required quantitative array {name}")
    for name in ("condition_input_fp32", "encoded_condition"):
        array = load_record_array(sample_root, record, name)
        try:
            if array.dtype != np.float32 or tuple(array.shape) != condition_shape or not np.isfinite(array).all():
                raise ValueError(f"Task 10 {call} {name} has invalid dtype/shape/values")
        finally:
            _close(array)
    output = load_record_array(sample_root, record, "output_full")
    try:
        if (output.dtype != np.float32 or tuple(output.shape) != tuple(geometry["full_condition_mask"].shape)
                or not np.isfinite(output).all()):
            raise ValueError(f"Task 10 {call} output_full has invalid dtype/shape/values")
    finally:
        _close(output)
    condition = load_record_array(sample_root, record, "condition_input_fp32")
    steps = load_record_array(sample_root, record, "condition_steps_fp32")
    generation = record.get("generation") or {}
    generation_mask = load_array_descriptor(sample_root, generation.get("mask"), "generation.mask")
    try:
        verify_masked_condition_consumption(condition, steps, generation_mask,
                                            geometry["full_condition_mask"])
    finally:
        _close(condition); _close(steps); _close(generation_mask)
    for name in ("generated_rgb", "decoded_rgb_full", "decoded_raw", "decoded_last_rgb"):
        array = load_record_array(sample_root, record, name)
        try:
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"Task 10 {call} {name} must be finite float32")
            if name == "generated_rgb" and (array.ndim != 4 or array.shape[:2] != (3, 16)):
                raise ValueError(f"Task 10 {call} generated_rgb must be [3,16,H,W]")
            if name == "decoded_rgb_full" and (array.ndim != 4 or array.shape[:2] != (3, 17)):
                raise ValueError(f"Task 10 {call} decoded_rgb_full must be [3,17,H,W]")
            if name == "decoded_raw" and (array.ndim != 5 or array.shape[:3] != (1, 3, 17)):
                raise ValueError(f"Task 10 {call} decoded_raw must be [1,3,17,H,W]")
            if name == "decoded_last_rgb" and (array.ndim != 3 or array.shape[0] != 3):
                raise ValueError(f"Task 10 {call} decoded_last_rgb must be CHW")
            if name in {"generated_rgb", "decoded_rgb_full", "decoded_last_rgb"}:
                if float(array.min()) < 0.0 or float(array.max()) > 1.0:
                    raise ValueError(f"Task 10 {call} {name} must be decoded RGB in [0,1]")
        finally:
            _close(array)
    raw = load_record_array(sample_root, record, "decoded_raw")
    decoded_full = load_record_array(sample_root, record, "decoded_rgb_full")
    try:
        if raw.shape[2:] != decoded_full.shape[1:]:
            raise ValueError(f"Task 10 {call} decoder raw/normalized geometry differs")
        normalized = np.clip((raw.astype(np.float64) + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)[0]
        if not np.array_equal(normalized, decoded_full):
            raise ValueError(f"Task 10 {call} saved RGB does not match frozen decoder normalization")
    finally:
        _close(raw); _close(decoded_full)
    generated = load_record_array(sample_root, record, "generated_rgb")
    decoded_full = load_record_array(sample_root, record, "decoded_rgb_full")
    decoded_last = load_record_array(sample_root, record, "decoded_last_rgb")
    try:
        if (not np.array_equal(decoded_full[:, 1:], generated)
                or not np.array_equal(generated[:, -1], decoded_last)):
            raise ValueError(f"Task 10 {call} decoded-last/full-chunk RGB aliases disagree")
    finally:
        _close(generated); _close(decoded_full); _close(decoded_last)
    action = load_record_array(sample_root, record, "action")
    try:
        if action.dtype != np.float32 or action.shape != (16, 10) or not np.isfinite(action).all():
            raise ValueError(f"Task 10 {call} action must be finite float32 [16,10]")
        expected_action_hash = _array_hash(action)
        if record.get("action_hash") != expected_action_hash:
            raise ValueError(f"Task 10 {call} requested action hash mismatch")
    finally:
        _close(action)
    provenance = record.get("provenance") or {}
    action_evidence = provenance.get("action_evidence") or {}
    effective_descriptor = action_evidence.get("effective_action")
    if not isinstance(effective_descriptor, Mapping):
        raise ValueError(f"Task 10 {call} lacks effective-action descriptor")
    requested = load_record_array(sample_root, record, "action")
    try:
        effective_shape = effective_descriptor.get("shape")
        if (effective_descriptor.get("dtype") != "float32" or not isinstance(effective_shape, list)
                or len(effective_shape) != 2 or effective_shape[0] != 16
                or effective_shape[1] < requested.shape[1]):
            raise ValueError(f"Task 10 {call} effective-action descriptor has invalid shape/dtype")
        # Task 8 stores only a JSON-safe hash descriptor for this transformed
        # value, not an artifact. Rebuild the frozen zero-padding route from the
        # requested action and verify it under both historical hash schemes.
        effective = np.zeros(tuple(effective_shape), dtype=np.float32)
        effective[:, :requested.shape[1]] = requested
        if (_task8_descriptor_hash(effective) != effective_descriptor.get("sha256")
                or action_evidence.get("action_hash") != expected_action_hash
                or action_evidence.get("effective_action_hash") != _array_hash(effective)):
            raise ValueError(f"Task 10 {call} action routing hash evidence differs")
    finally:
        _close(requested)
    precision = record.get("precision") or {}
    if any(precision.get(key) != "float32" for key in ("G", "D", "E")):
        raise ValueError(f"Task 10 {call} lacks actual FP32 G/D/E evidence")
    generation = record.get("generation") or {}
    noise = generation.get("noise_evidence") or {}
    seed = noise.get("seed")
    if (not isinstance(seed, int) or noise.get("prepare_seed") != seed
            or generation.get("sampler_generator_seeds") != [seed] * 30):
        raise ValueError(f"Task 10 {call} actual seed/sampler evidence differs")
    consumption = record.get("action_consumption") or {}
    packed_hashes = record.get("packed_action_token_hashes", ())
    expected_token_hash = action_evidence.get("effective_action_hash")
    if (consumption.get("steps") != 30 or consumption.get("all_steps_match") is not True
            or consumption.get("expected_token_hash") != expected_token_hash
            or len(consumption.get("consumed_token_hashes", ())) != 30
            or len(packed_hashes) != 30
            or any(item != expected_token_hash for item in consumption.get("consumed_token_hashes", ()))
            or any(item != expected_token_hash for item in packed_hashes)):
        raise ValueError(f"Task 10 {call} actual action-token consumption evidence failed")


def _validate_formal_source(root: Path, identity: Mapping[str, Any], episode_info: Mapping[int, Mapping[str, Any]]) -> dict[tuple[int, tuple[int, int], str], tuple[Path, dict[str, Any]]]:
    """Validate all 12 groups and 48 hash-published samples before analysis."""
    mask_hashes = {episode_info[index]["condition_mask_sha256"] for index in LOCKED_RECORDS}
    mask_shapes = {episode_info[index]["condition_shape"] for index in LOCKED_RECORDS}
    full_mask_hashes = {episode_info[index]["full_condition_mask_sha256"] for index in LOCKED_RECORDS}
    index_bindings = {(tuple(episode_info[index]["condition_indexes"]), episode_info[index]["temporal_axis"])
                      for index in LOCKED_RECORDS}
    if (len(mask_hashes) != 1 or len(mask_shapes) != 1 or len(full_mask_hashes) != 1
            or len(index_bindings) != 1
            or any(not np.array_equal(episode_info[index]["full_condition_mask"],
                                      episode_info[LOCKED_RECORDS[0]]["full_condition_mask"])
                   for index in LOCKED_RECORDS[1:])):
        raise ValueError("Task 10 condition masks/index bindings differ across episodes")
    sample_index: dict[tuple[int, tuple[int, int], str], tuple[Path, dict[str, Any]]] = {}
    for record_index in LOCKED_RECORDS:
        info = episode_info[record_index]
        actions = _read_npz_array(info["input_path"], "actions_normalized")
        if actions.dtype != np.float32 or actions.shape != (2, 16, 10) or not np.isfinite(actions).all():
            raise ValueError(f"Task 10 normalized actions are invalid for record {record_index}")
        for pair in SEED_PAIRS:
            formal = root / "records" / f"record_{record_index:02d}" / f"seeds_{pair[0]}_{pair[1]}" / "formal"
            formal_status = _load_json(formal / "run_status.json")
            if (formal_status.get("status") != "COMPLETE" or formal_status.get("formal_calls") != 4
                    or formal_status.get("plan") != [dict(row) for row in build_call_plan(*pair)]):
                raise ValueError(f"Task 10 formal group is incomplete or has the wrong plan: {record_index}/{pair}")
            binding = formal_status.get("binding") or {}
            if (binding.get("model") != identity.get("checkpoint_sha256")
                    or binding.get("vae") != identity.get("vae_sha256")
                    or binding.get("data") != info["input_sha256"]
                    or binding.get("actions") != _array_hash(actions)
                    or binding.get("noise") != _expected_noise_binding(pair)):
                raise ValueError(f"Task 10 formal group binding differs: {record_index}/{pair}")
            group: dict[str, tuple[Path, dict[str, Any]]] = {}
            for call, expected_seed, action_index in (("G0", pair[0], 0), ("G0_repeat", pair[0], 0),
                                                       ("TF2", pair[1], 1), ("AR2", pair[1], 1)):
                sample_root = formal / "samples" / call
                record = validate_task8_sample(sample_root)
                if (record.get("call") != call or record.get("sample_id") != call
                        or record.get("status") != "success"):
                    raise ValueError(f"Task 10 sample identity differs: {record_index}/{pair}/{call}")
                _validate_array_contracts(record, call, sample_root, info)
                expected_condition_source = {"G0": "real_x0", "G0_repeat": "real_x0",
                                             "TF2": "real_x16", "AR2": "g0_float_last_fp32"}[call]
                expected_action_source = "a0" if action_index == 0 else "a1"
                provenance = record.get("provenance") or {}
                if (provenance.get("condition_source") != expected_condition_source
                        or provenance.get("action_source") != expected_action_source):
                    raise ValueError(f"Task 10 {call} route provenance differs: {record_index}/{pair}")
                generation = record.get("generation") or {}
                noise = generation.get("noise_evidence") or {}
                if (noise.get("seed") != expected_seed or noise.get("prepare_seed") != expected_seed
                        or generation.get("sampler_generator_seeds") != [expected_seed] * 30):
                    raise ValueError(f"Task 10 sample identity/seed differs: {record_index}/{pair}/{call}")
                action = load_record_array(sample_root, record, "action")
                try:
                    if not np.array_equal(action, actions[action_index]):
                        raise ValueError(f"Task 10 {call} action differs from preflight for {record_index}/{pair}")
                finally:
                    _close(action)
                group[call] = (sample_root, record)
                sample_index[(record_index, pair, call)] = (sample_root, record)
            _validate_sample_pair(group, record_index, pair)
        del actions
    return sample_index


def _read_truth_geometry(condition_path: Path) -> dict[str, Any]:
    with np.load(condition_path, allow_pickle=False) as archive:
        if not {"gt_condition_x16", "gt_condition_x32", "condition_mask", "full_condition_mask",
                "condition_indexes", "temporal_axis"} <= set(archive.files):
            raise ValueError(f"Task 10 truth NPZ is missing required arrays: {condition_path}")
        gt16 = archive["gt_condition_x16"]
        gt32 = archive["gt_condition_x32"]
        mask = archive["condition_mask"]
        full_mask = archive["full_condition_mask"]
        indexes = np.asarray(archive["condition_indexes"])
        temporal_axis = int(np.asarray(archive["temporal_axis"]).item())
        if (gt16.dtype != np.float32 or gt32.dtype != np.float32 or mask.dtype != np.bool_
                or gt16.shape != gt32.shape or gt16.shape != mask.shape or not np.any(mask)
                or not np.isfinite(gt16).all() or not np.isfinite(gt32).all()
                or full_mask.dtype != np.bool_ or indexes.dtype.kind not in "iu"
                or indexes.ndim != 1 or indexes.size == 0 or temporal_axis < 0
                or temporal_axis >= full_mask.ndim):
            raise ValueError(f"Task 10 encoded truth/mask contract failed: {condition_path}")
        if (np.any(indexes < 0) or np.any(indexes >= full_mask.shape[temporal_axis])
                or tuple(np.take(full_mask, indexes.astype(np.int64), axis=temporal_axis).shape) != tuple(mask.shape)
                or not np.array_equal(np.take(full_mask, indexes.astype(np.int64), axis=temporal_axis), mask)):
            raise ValueError(f"Task 10 condition indexes do not select the saved condition mask: {condition_path}")
        return {"condition_shape": tuple(gt16.shape), "condition_mask_sha256": _array_hash(mask),
                "full_condition_mask": np.array(full_mask, copy=True),
                "full_condition_mask_sha256": _array_hash(full_mask),
                "condition_indexes": indexes.astype(np.int64).tolist(), "temporal_axis": temporal_axis}


def _same_artifact_array(left_root: Path, left: Mapping[str, Any], right_root: Path,
                         right: Mapping[str, Any], key: str) -> bool:
    a = load_record_array(left_root, left, key)
    b = load_record_array(right_root, right, key)
    try:
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    finally:
        _close(a); _close(b)


def _validate_sample_pair(records: Mapping[str, Mapping[str, Any]], record_index: int,
                          pair: tuple[int, int]) -> None:
    first_root, first = records["G0"]
    repeat_root, repeat = records["G0_repeat"]
    tf_root, tf = records["TF2"]
    ar_root, ar = records["AR2"]
    # Call ids and artifact hashes were already checked; bytewise array equality
    # here is the exact-repeat scientific gate, not a count-based completion test.
    for key in ("condition_input_fp32", "action", "output_full", "decoded_rgb_full",
                "generated_rgb", "decoded_last_rgb", "encoded_condition", "condition_steps_fp32"):
        if not _same_artifact_array(first_root, first, repeat_root, repeat, key):
            raise ValueError(f"Task 10 G0 exact repeat differs in {key}: {record_index}/{pair}")
    if first.get("prediction_noise_hash") != repeat.get("prediction_noise_hash"):
        raise ValueError(f"Task 10 G0 exact repeat prediction noise differs: {record_index}/{pair}")
    for key in ("action",):
        if not _same_artifact_array(tf_root, tf, ar_root, ar, key):
            raise ValueError(f"Task 10 TF2/AR2 differs in {key}: {record_index}/{pair}")
    if tf.get("prediction_noise_hash") != ar.get("prediction_noise_hash"):
        raise ValueError(f"Task 10 TF2/AR2 prediction noise differs: {record_index}/{pair}")
    g0_encoded = load_record_array(first_root, first, "encoded_condition")
    ar_condition = load_record_array(ar_root, ar, "condition_input_fp32")
    try:
        verify_ar_feedback_condition(g0_encoded, ar_condition)
    finally:
        _close(g0_encoded); _close(ar_condition)


def validate_task10_run(source_run: str | Path) -> dict[str, Any]:
    """Validate the locked Task 10 run and every formal sample without analysis writes."""
    root = Path(source_run).resolve()
    identity = _load_json(root / "run_identity.json")
    status = _load_json(root / "run_status.json")
    validate_source_identity(identity, status)
    manifest = verify_task10_manifest(root)
    metric_rows = _read_csv_rows(root / "analysis" / "stratum_metrics.csv")
    episode_info = {index: _verify_episode_receipts(root, index, identity) for index in LOCKED_RECORDS}
    sample_index = _validate_formal_source(root, identity, episode_info)
    return {"source_root": root, "identity": identity, "status": status,
            "manifest": manifest, "metric_rows": metric_rows,
            "episode_info": episode_info, "sample_index": sample_index}


def _atomic_json(path: Path, value: Any) -> None:
    def clean(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): clean(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not math.isfinite(item):
            return "inf" if item > 0 else ("-inf" if item < 0 else "N/A")
        return item

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean(value), sort_keys=True, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    names = list(fieldnames) if fieldnames is not None else list(dict.fromkeys(
        key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("N/A" if value is None else
                                   "inf" if isinstance(value, (float, np.floating)) and np.isposinf(value) else
                                   "-inf" if isinstance(value, (float, np.floating)) and np.isneginf(value) else value)
                             for key, value in row.items()})


def _mean_square_rms(error: np.ndarray) -> float:
    values = np.asarray(error, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values, dtype=np.float64)))


def analyze_task10_stratum(validated: Mapping[str, Any], record_index: int,
                           pair: tuple[int, int], column_index: int = 0) -> dict[str, Any]:
    """Compute one immutable episode/seed stratum entirely in memory for auditing."""
    info = validated["episode_info"][record_index]
    with np.load(info["condition_path"], allow_pickle=False) as archive:
        gt16 = np.array(archive["gt_condition_x16"], copy=True)
        gt32 = np.array(archive["gt_condition_x32"], copy=True)
        condition_mask = np.array(archive["condition_mask"], copy=True)
    truth_rgb = {16: _read_npz_first_axis_slice(info["input_path"], "rgb_float32", 16),
                 32: _read_npz_first_axis_slice(info["input_path"], "rgb_float32", 32)}
    pair_vectors: dict[tuple[str, str], np.ndarray] = {}
    pair_predictions: dict[tuple[str, str], np.ndarray] = {}
    column_rows: list[dict[str, Any]] = []
    for mode, (call, truth_key, truth_frame) in _ERROR_MODES.items():
        sample_root, record = validated["sample_index"][(record_index, pair, call)]
        encoded = load_record_array(sample_root, record, "encoded_condition")
        generated = load_record_array(sample_root, record, "generated_rgb")
        try:
            latent_prediction = np.asarray(encoded).copy()
            latent_truth = gt16 if truth_key == "gt_condition_x16" else gt32
            latent_error = prediction_error(latent_prediction, latent_truth, mask=condition_mask)
            rgb_prediction = np.asarray(generated[:, -1]).copy()
            rgb_error = prediction_error(rgb_prediction, truth_rgb[truth_frame])
            latent_predicted_values = latent_prediction[condition_mask].astype(np.float64)
            latent_truth_values = latent_truth[condition_mask].astype(np.float64)
            latent_cosine_denom = float(np.linalg.norm(latent_predicted_values) * np.linalg.norm(latent_truth_values))
            latent_cosine = (None if latent_cosine_denom == 0 else
                             float(np.dot(latent_predicted_values, latent_truth_values) / latent_cosine_denom))
            rgb_metrics = rgb_endpoint_metrics(rgb_prediction, truth_rgb[truth_frame])
            pair_vectors[(mode, "latent")] = latent_error
            pair_vectors[(mode, "rgb")] = rgb_error
            pair_predictions[(mode, "latent")] = latent_prediction[condition_mask].copy()
            pair_predictions[(mode, "rgb")] = rgb_prediction.copy()
            column_rows.extend((
                {"column_index": column_index, "record_index": record_index,
                 "seed_pair": json.dumps(list(pair)), "mode": mode, "space": "latent",
                 "feature_count": int(latent_error.size), "error_rms": _mean_square_rms(latent_error),
                 "error_mae": float(np.mean(np.abs(latent_error), dtype=np.float64)),
                 "column_norm": float(np.linalg.norm(latent_error)), "cosine_to_truth": latent_cosine},
                {"column_index": column_index, "record_index": record_index,
                 "seed_pair": json.dumps(list(pair)), "mode": mode, "space": "rgb",
                 "feature_count": int(rgb_error.size), "error_rms": rgb_metrics["rmse"],
                 "error_mae": rgb_metrics["mae"], "column_norm": float(np.linalg.norm(rgb_error)),
                 "psnr_db": rgb_metrics["psnr_db"], "cosine_to_truth": None},
            ))
        finally:
            _close(encoded); _close(generated)
    reproduced = {"g0_latent_rms": _mean_square_rms(pair_vectors[("E1", "latent")]),
                  "tf_rmse": _mean_square_rms(pair_vectors[("ETF2", "rgb")]),
                  "ar_rmse": _mean_square_rms(pair_vectors[("EAR2", "rgb")]),
                  "tf2_latent_rms": _mean_square_rms(pair_vectors[("ETF2", "latent")]),
                  "ar2_latent_rms": _mean_square_rms(pair_vectors[("EAR2", "latent")])}
    source_row = validated["metric_rows"][(record_index, pair)]
    reproduction = validate_metric_reproduction(source_row, reproduced)
    decomposition_rows = []
    for space in ("latent", "rgb"):
        baseline = pair_vectors[("ETF2", space)]
        feedback = prediction_error(pair_predictions[("EAR2", space)], pair_predictions[("ETF2", space)])
        decomposition = error_decomposition(baseline, feedback,
                                            combined_error=pair_vectors[("EAR2", space)])
        decomposition_rows.append({"record_index": record_index, "seed_pair": list(pair),
                                   "space": space, **decomposition})
    return {"pair_vectors": pair_vectors, "pair_predictions": pair_predictions,
            "column_rows": column_rows,
            "reproduction_row": {"record_index": record_index, "seed_pair": list(pair), **reproduction},
            "decomposition_rows": decomposition_rows,
            "reproduced_metrics": reproduced}


def _analysis_implementation_identity() -> dict[str, Any]:
    here = Path(__file__).resolve().parent
    return {"configuration": TASK11_ANALYSIS_CONFIG,
            "analyzer_sha256": sha256_file(Path(__file__)),
            "math_module_sha256": sha256_file(here / "umi_task11_error_analysis.py"),
            "task10_contract_sha256": sha256_file(here / "umi_task10_real_scenes.py")}


def _summarize_column_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    metric_names = ("error_rms", "error_mae", "cosine_to_truth")
    for mode in ("E1", "ETF2", "EAR2"):
        for space in ("latent", "rgb"):
            matching = [row for row in rows if row["mode"] == mode and row["space"] == space]
            for record_index in LOCKED_RECORDS:
                selected = [row for row in matching if row["record_index"] == record_index]
                if len(selected) != 2:
                    raise ValueError(f"Task 11 needs two seed columns per episode: {mode}/{space}/{record_index}")
                summary = {"mode": mode, "space": space, "record_index": record_index,
                           "seed_pair_count": 2}
                for metric in metric_names:
                    values = [float(row[metric]) for row in selected if row.get(metric) is not None]
                    summary["mean_" + metric] = None if not values else float(np.mean(values, dtype=np.float64))
                    summary[metric + "_available_seed_pairs"] = len(values)
                result.append(summary)
    return result


def _descriptive_episode_summary(episode_rows: Sequence[Mapping[str, Any]], group_keys: Sequence[str],
                                 metric_names: Sequence[str]) -> list[dict[str, Any]]:
    result = []
    key_tuples = sorted({tuple(row[key] for key in group_keys) for row in episode_rows}, key=str)
    for key_tuple in key_tuples:
        matching = [row for row in episode_rows if tuple(row[key] for key in group_keys) == key_tuple]
        by_episode = []
        for record_index in LOCKED_RECORDS:
            seed_rows = [row for row in matching if row["record_index"] == record_index]
            if not seed_rows:
                continue
            episode = {"record_index": record_index, "seed_pair_count": len(seed_rows)}
            for metric in metric_names:
                values = [float(row[metric]) for row in seed_rows if row.get(metric) is not None]
                episode["mean_" + metric] = None if not values else float(np.mean(values, dtype=np.float64))
            by_episode.append(episode)
        summary = {key: value for key, value in zip(group_keys, key_tuple)}
        summary["episode_means"] = by_episode
        summary["episode_count"] = len(by_episode)
        for metric in metric_names:
            values = [row["mean_" + metric] for row in by_episode if row["mean_" + metric] is not None]
            summary["six_episode_mean_" + metric] = None if not values else float(np.mean(values, dtype=np.float64))
            summary["six_episode_min_" + metric] = None if not values else float(np.min(values))
            summary["six_episode_max_" + metric] = None if not values else float(np.max(values))
            summary[metric + "_nonzero_episode_count"] = int(sum(value != 0.0 for value in values))
            summary[metric + "_zero_episode_count"] = int(sum(value == 0.0 for value in values))
        result.append(summary)
    return result


def analyze_task10_run(source_run: str | Path, output_run: str | Path, *, resume: bool = False,
                       supersedes_failed_attempt: str | Path | None = None) -> dict[str, Any]:
    """Validate and compute Task 11 true-error spectra from immutable Task 10 outputs."""
    validated = validate_task10_run(source_run)
    root = validated["source_root"]
    source_identity = {"schema": "umi-task11-offline-v1", "source_run": str(root),
                       "source_manifest_sha256": validated["manifest"]["manifest_sha256"],
                       "source_identity_sha256": sha256_file(root / "run_identity.json"),
                       "source_status_sha256": sha256_file(root / "run_status.json"),
                       "analysis_implementation": _analysis_implementation_identity(),
                       "record_indices": list(LOCKED_RECORDS),
                       "seed_pairs": [list(pair) for pair in SEED_PAIRS]}
    if supersedes_failed_attempt is not None:
        superseded_root = Path(supersedes_failed_attempt).resolve()
        superseded_status = _load_json(superseded_root / "run_status.json")
        if (superseded_status.get("status") != "FAILED"
                or superseded_status.get("source_manifest_sha256") != source_identity["source_manifest_sha256"]):
            raise ValueError("superseded offline attempt is not a failed run over this identical Task 10 source")
        source_identity["supersedes_failed_attempt"] = {
            "run_directory": str(superseded_root), "status": "FAILED",
            "attempt": superseded_status.get("attempt"),
            "source_manifest_sha256": superseded_status.get("source_manifest_sha256")}
    destination = Path(output_run).resolve()
    if destination.exists():
        if not resume or not (destination / "run_identity.json").is_file():
            raise ValueError("Task 11 output exists; pass --resume with the identical source or choose a new run-dir")
        if _load_json(destination / "run_identity.json") != source_identity:
            raise ValueError("Task 11 resume source identity differs from the existing run")
        current = _load_json(destination / "run_status.json") if (destination / "run_status.json").is_file() else {}
        if current.get("status") == "COMPLETE":
            raise ValueError("Task 11 completed outputs are immutable; choose a new run-dir")
        attempts = [int(path.name.split("_")[-1]) for path in destination.glob("attempt_*")
                    if path.is_dir() and path.name.split("_")[-1].isdigit()]
        attempt_number = max(attempts, default=0) + 1
    else:
        destination.mkdir(parents=True)
        _atomic_json(destination / "run_identity.json", source_identity)
        attempt_number = 1
    attempt_dir = destination / f"attempt_{attempt_number:02d}"
    attempt_dir.mkdir()
    status_path = destination / "run_status.json"
    _atomic_json(status_path, {"schema": "umi-task11-offline-status-v1", "status": "RUNNING",
                               "attempt": attempt_number, "completed_columns": 0,
                               "source_manifest_sha256": source_identity["source_manifest_sha256"]})

    matrices: dict[tuple[str, str], np.memmap] = {}
    paths: dict[tuple[str, str], Path] = {}
    arrays: dict[str, np.ndarray] = {}
    arrays_closed = False
    column_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    reproduction_rows: list[dict[str, Any]] = []
    try:
        first_geometry = validated["episode_info"][LOCKED_RECORDS[0]]
        with np.load(first_geometry["condition_path"], allow_pickle=False) as archive:
            latent_features = int(np.count_nonzero(archive["condition_mask"]))
        rgb_shape = tuple(validated["episode_info"][LOCKED_RECORDS[0]]["report"]["selected"]["inputs_npz"]["rgb_float32_shape"][1:])
        rgb_features = int(np.prod(rgb_shape, dtype=np.int64))
        for mode in _ERROR_MODES:
            for space, feature_count in (("latent", latent_features), ("rgb", rgb_features)):
                matrix_path = attempt_dir / f"errors_{mode}_{space}.npy"
                arrays[(mode, space)] = np.lib.format.open_memmap(
                    matrix_path, mode="w+", dtype=np.float64, shape=(feature_count, 12))
                matrices[(mode, space)] = arrays[(mode, space)]
                paths[(mode, space)] = matrix_path

        column_index = 0
        for record_index in LOCKED_RECORDS:
            for pair in SEED_PAIRS:
                stratum = analyze_task10_stratum(validated, record_index, pair, column_index)
                for mode in _ERROR_MODES:
                    for space in ("latent", "rgb"):
                        matrices[(mode, space)][:, column_index] = stratum["pair_vectors"][(mode, space)]
                column_rows.extend(stratum["column_rows"])
                reproduction_rows.append(stratum["reproduction_row"])
                decomposition_rows.extend(stratum["decomposition_rows"])
                column_index += 1
                _atomic_json(status_path, {"schema": "umi-task11-offline-status-v1", "status": "RUNNING",
                                           "attempt": attempt_number, "completed_columns": column_index,
                                           "source_manifest_sha256": source_identity["source_manifest_sha256"]})
                del stratum

        for array in arrays.values():
            array.flush()
            array._mmap.close()
        arrays_closed = True
        episode_metric_rows = _summarize_column_metrics(column_rows)
        ensemble_summary: dict[str, Any] = {}
        for mode in _ERROR_MODES:
            ensemble_summary[mode] = {}
            for space in ("latent", "rgb"):
                matrix = np.load(paths[(mode, space)], mmap_mode="r", allow_pickle=False)
                try:
                    spectrum = error_spectrum(matrix)
                    loeo = leave_one_episode_out(matrix, [index for index in LOCKED_RECORDS for _ in SEED_PAIRS])
                finally:
                    _close(matrix)
                ensemble_summary[mode][space] = {
                    "matrix_file": paths[(mode, space)].name,
                    "matrix_sha256": sha256_file(paths[(mode, space)]),
                    "spectrum": spectrum, "leave_one_episode_out": loeo}
        decomposition_summary = _descriptive_episode_summary(
            decomposition_rows, ("space",),
            ("baseline_mse", "feedback_change_mse", "cross_term_2dot_over_n", "combined_error_mse",
             "squared_error_difference", "identity_residual", "cosine"))
        metrics_summary = _descriptive_episode_summary(
            column_rows, ("mode", "space"), ("error_rms", "error_mae", "cosine_to_truth"))
        _write_csv(attempt_dir / "error_columns.csv", column_rows)
        _write_csv(attempt_dir / "episode_column_means.csv", episode_metric_rows)
        _write_csv(attempt_dir / "error_decomposition.csv", decomposition_rows)
        report = {"schema": "umi-task11-error-analysis-v1", "status": "COMPLETE",
                  "source": {"run_directory": str(root), "manifest_sha256": validated["manifest"]["manifest_sha256"],
                             "manifest_file_count": validated["manifest"]["file_count"],
                             "excluded_auxiliary_evidence": validated["manifest"]["excluded_auxiliary_evidence"],
                             "records": list(LOCKED_RECORDS), "seed_pairs": [list(pair) for pair in SEED_PAIRS],
                             "immutable_source_validated": True},
                  "analysis_implementation": source_identity["analysis_implementation"],
                  "supersedes_failed_attempt": source_identity.get("supersedes_failed_attempt"),
                  "definitions": {"error": "float64 prediction minus aligned truth",
                                  "latent_space": "condition_mask-selected FP32 VAE condition latent",
                                  "rgb_space": "saved final decoded RGB frame; E1 against x16, ETF2/EAR2 against x32",
                                  "svd": "uncentered thin SVD of full-amplitude 12-column error matrix",
                                  "loeo": "six episode-held-out projections; both seed columns held out; training mean only in centered sensitivity",
                                  "decomposition": "b=TF2-truth; p=AR2-TF2; observed AR2-vs-TF2 squared-error difference checked against p MSE + 2dot cross",
                                  "loeo_caveat": "projection residual describes optimal representation of known held-out error, not a trained predictor"},
                  "task10_metric_reproduction": reproduction_rows,
                  "ensembles": ensemble_summary,
                  "column_metrics": column_rows,
                  "episode_mean_metrics": metrics_summary,
                  "error_decomposition": {"stratum_rows": decomposition_rows,
                                          "six_episode_summary": decomposition_summary},
                  "zero_handling": {"ensemble_zero_column_count": {
                      f"{mode}/{space}": len(ensemble_summary[mode][space]["spectrum"]["zero_column_indices"])
                      for mode in _ERROR_MODES for space in ("latent", "rgb")},
                      "zero_baseline_or_feedback_geometry_n_a_count": sum(
                          row["cosine"] is None for row in decomposition_rows)}}
        _atomic_json(attempt_dir / "analysis.json", report)
        summary_lines = ["# Task 11 offline true-error analysis", "",
                         f"Source: `{root}` (Task 10 manifest {validated['manifest']['manifest_sha256']}; "
                         f"{validated['manifest']['file_count']} files verified).", "",
                         "The source contains six locked episodes and two seed schedules per episode. "
                         "All metrics use true saved outputs; no model was run.", "",
                         "| Mode | Space | k95 | Effective rank | LOEO folds |", "|---|---|---:|---:|---:|"]
        for mode in _ERROR_MODES:
            for space in ("latent", "rgb"):
                item = ensemble_summary[mode][space]
                spectrum = item["spectrum"]
                summary_lines.append(f"| {mode} | {space} | {spectrum['k95'] if spectrum['k95'] is not None else 'N/A'} | "
                                     f"{spectrum['effective_rank'] if spectrum['effective_rank'] is not None else 'N/A'} | "
                                     f"{item['leave_one_episode_out']['fold_count']} |")
        summary_lines.extend(("", "LOEO projection residuals are descriptive geometry for known held-out errors; "
                              "they are not trained predictions. Per-episode means are reported before the six-episode "
                              "descriptive summary. Zero matrices, zero columns, and zero-vector cosines are marked N/A.", ""))
        (attempt_dir / "review_report.md").write_text("\n".join(summary_lines), encoding="utf-8")
        manifest_files = sorted(path for path in attempt_dir.iterdir() if path.is_file())
        manifest_rows = [f"{sha256_file(path)}  {path.name}" for path in manifest_files]
        (attempt_dir / "MANIFEST.sha256").write_text("\n".join(manifest_rows) + "\n", encoding="ascii")
        _atomic_json(status_path, {"schema": "umi-task11-offline-status-v1", "status": "COMPLETE",
                                   "attempt": attempt_number, "completed_columns": 12,
                                   "artifact_count": len(manifest_files),
                                   "manifest_sha256": sha256_file(attempt_dir / "MANIFEST.sha256"),
                                   "source_manifest_sha256": source_identity["source_manifest_sha256"]})
        return {"output_run": destination, "attempt_dir": attempt_dir,
                "status": "COMPLETE", "manifest_sha256": sha256_file(attempt_dir / "MANIFEST.sha256")}
    except Exception as error:
        if not arrays_closed:
            for array in arrays.values():
                try:
                    array.flush()
                    array._mmap.close()
                except Exception:
                    pass
        _atomic_json(status_path, {"schema": "umi-task11-offline-status-v1", "status": "FAILED",
                                   "attempt": attempt_number, "completed_columns": column_index if 'column_index' in locals() else 0,
                                   "error": f"{type(error).__name__}: {error}",
                                   "source_manifest_sha256": source_identity["source_manifest_sha256"]})
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Task 10 provenance and compute Task 11 true-error spectra.")
    parser.add_argument("--source-run", required=True, help="immutable Task 10 run directory")
    parser.add_argument("--run-dir", help="Task 11 output directory (required unless --validate-only)")
    parser.add_argument("--resume", action="store_true", help="start a new immutable attempt after exact identity validation")
    parser.add_argument("--supersedes-failed-attempt", help="explicit failed Task 11 run directory superseded by this new run")
    parser.add_argument("--validate-only", action="store_true", help="verify source provenance without analysis writes")
    args = parser.parse_args(argv)
    if args.validate_only:
        result = validate_task10_run(args.source_run)
        print(json.dumps({"status": "VALIDATED", "source_run": str(result["source_root"]),
                          "manifest_sha256": result["manifest"]["manifest_sha256"],
                          "manifest_file_count": result["manifest"]["file_count"],
                          "formal_sample_count": len(result["sample_index"]),
                          "episode_count": len(result["episode_info"])}, sort_keys=True))
        return 0
    if not args.run_dir:
        parser.error("--run-dir is required unless --validate-only is set")
    result = analyze_task10_run(args.source_run, args.run_dir, resume=args.resume,
                                supersedes_failed_attempt=args.supersedes_failed_attempt)
    print(json.dumps({"status": result["status"], "output_run": str(result["output_run"]),
                      "attempt_dir": str(result["attempt_dir"]),
                      "manifest_sha256": result["manifest_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
