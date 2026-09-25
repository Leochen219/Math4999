"""Pure NumPy Task 8 ground-truth error analysis.

No model, CUDA, framework, or remote data is imported here.  The analyzer
consumes the immutable float tensors emitted by the runner and reports
explicit ``None`` cosine values for zero norms and ``inf`` PSNR for exact
zero error.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class AnalysisError(ValueError):
    """Saved evidence cannot support a trustworthy metric."""


FORMAL_SAMPLE_IDS = (
    "G0_real_x0_seed0",
    "G0_repeat_real_x0_seed0",
    "TF2_real_x16_seed1",
    "AR2_g0_float_last_fp32_seed1",
)


def _f32(value: Any, *, name: str) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise AnalysisError(f"{name} must be finite and non-empty")
    return result


def _canonical_frames(value: Any, *, name: str, require_rgb: bool = False) -> np.ndarray:
    array = _f32(value, name=name)
    if array.ndim != 4:
        raise AnalysisError(f"{name} must have four dimensions")
    # Internal evidence is [T,C,H,W]; accept the framework's [C,T,H,W]
    # decoder layout and normalize it once for offline calculations.
    if array.shape[1] in (1, 3):
        result = np.ascontiguousarray(array)
    elif array.shape[0] in (1, 3):
        result = np.ascontiguousarray(np.moveaxis(array, 0, 1))
    elif array.shape[-1] in (1, 3):
        result = np.ascontiguousarray(np.moveaxis(array, -1, 1))
    else:
        raise AnalysisError(f"{name} has no recognizable channel axis: {array.shape}")
    if require_rgb and result.shape[1] != 3:
        raise AnalysisError(f"{name} must be RGB with 3 channels, got {result.shape}")
    if float(result.min()) < 0.0 or float(result.max()) > 1.0:
        raise AnalysisError(f"{name} must be in [0, 1], got range [{result.min()}, {result.max()}]")
    return result


def cosine_or_none(left: Any, right: Any, mask: Any | None = None) -> float | None:
    lhs = _f32(left, name="left"); rhs = _f32(right, name="right")
    if lhs.shape != rhs.shape:
        raise AnalysisError("cosine operands have different shapes")
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != lhs.shape:
            raise AnalysisError("cosine mask shape differs from operands")
        lhs, rhs = lhs[selected], rhs[selected]
    else:
        lhs, rhs = lhs.reshape(-1), rhs.reshape(-1)
    left_norm = float(np.sqrt(np.sum(lhs.astype(np.float64) ** 2)))
    right_norm = float(np.sqrt(np.sum(rhs.astype(np.float64) ** 2)))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return float(np.sum(lhs.astype(np.float64) * rhs.astype(np.float64)) / (left_norm * right_norm))


def psnr(predicted: Any, truth: Any, *, data_range: float = 1.0) -> float:
    estimate = _f32(predicted, name="predicted"); target = _f32(truth, name="truth")
    if estimate.shape != target.shape:
        raise AnalysisError("PSNR operands have different shapes")
    mse = float(np.mean((estimate.astype(np.float64) - target.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((float(data_range) ** 2) / mse))


def analyze_rgb_metrics(predicted: Any, truth: Any) -> dict[str, float]:
    estimate = _canonical_frames(predicted, name="predicted RGB", require_rgb=True)
    target = _canonical_frames(truth, name="truth RGB", require_rgb=True)
    if estimate.shape != target.shape:
        raise AnalysisError(f"RGB shapes differ: {estimate.shape} vs {target.shape}")
    diff = estimate.astype(np.float64) - target.astype(np.float64)
    return {"rmse": float(np.sqrt(np.mean(diff * diff))), "mae": float(np.mean(np.abs(diff))),
            "psnr": psnr(estimate, target), "frames": int(estimate.shape[0])}


def _json_psnr(value: float) -> float | str:
    return "inf" if np.isinf(value) else float(value)


def psnr_delta_higher_is_better(ar_psnr: float, tf_psnr: float) -> float | str:
    """Return AR2 minus TF2 PSNR, preserving the higher-is-better direction."""
    ar_inf = bool(np.isposinf(ar_psnr))
    tf_inf = bool(np.isposinf(tf_psnr))
    if ar_inf and tf_inf:
        return "N/A"
    if ar_inf:
        return "inf"
    if tf_inf:
        return "-inf"
    return float(ar_psnr - tf_psnr)


def _json_metric_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    if isinstance(result.get("psnr"), (float, int)):
        result["psnr"] = _json_psnr(float(result["psnr"]))
    return result


def condition_latent_metrics(actual: Any, ground_truth: Any, mask: Any | None = None) -> dict[str, float | None]:
    lhs = _f32(actual, name="actual condition latent"); rhs = _f32(ground_truth, name="ground-truth condition latent")
    if lhs.shape != rhs.shape:
        raise AnalysisError("condition latent shapes differ")
    delta = lhs.astype(np.float64) - rhs.astype(np.float64)
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != lhs.shape:
            raise AnalysisError("condition mask shape differs from latent")
        delta = delta[selected]
    return {"rms": float(np.sqrt(np.mean(delta * delta))),
            "cosine": cosine_or_none(lhs, rhs, mask)}


def _find(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _action_identity(record: Mapping[str, Any]) -> Any:
    values = _find(record, "packed_action_token_hashes")
    if values is not None:
        return tuple(values) if isinstance(values, (list, tuple)) else values
    consumption = record.get("action_consumption")
    if isinstance(consumption, Mapping):
        hashes = consumption.get("consumed_token_hashes")
        if isinstance(hashes, (list, tuple)):
            return tuple(hashes)
        return consumption.get("expected_token_hash")
    return None


def _prediction_frames(record: Mapping[str, Any], *, key: str) -> np.ndarray:
    generated = _find(record, "generated_rgb", "decoded_generated_rgb", "decoded_frames")
    if generated is None:
        raise AnalysisError(f"{key} lacks generated RGB float tensor")
    frames = _canonical_frames(generated, name=f"{key} RGB", require_rgb=True)
    # A 17-frame decode includes the conditioning frame at index 0. It is
    # never a prediction row and must be excluded from all 16-row metrics.
    if frames.shape[0] == 17:
        frames = frames[1:]
    if frames.shape[0] != 16:
        raise AnalysisError(f"{key} must contain 16 generated frames (or full 17-frame decode)")
    return frames


def analyze_task8_outputs(records: Mapping[str, Mapping[str, Any]], *, truth_rgb: Any,
                          gt_condition_x16: Any | None = None,
                          gt_condition_x32: Any | None = None,
                          condition_mask: Any | None = None) -> dict[str, Any]:
    """Compute E1/ETF2/EAR2 and condition-chain metrics from saved tensors."""
    required = FORMAL_SAMPLE_IDS
    missing = [key for key in required if key not in records]
    if missing:
        raise AnalysisError(f"formal records missing: {missing}")
    truth = _canonical_frames(truth_rgb, name="ground-truth RGB", require_rgb=True)
    if truth.shape[0] != 33:
        raise AnalysisError("Task 8 requires all 33 ground-truth RGB observations")
    for key in required:
        record = records[key]
        if record.get("status") not in (None, "success"):
            raise AnalysisError(f"{key} is not a successful formal sample")
        if "output_full" not in record:
            raise AnalysisError(f"{key} lacks full generated latent evidence")
        _prediction_frames(record, key=key)
        if not any(name in record for name in ("prediction_noise_hash", "noise_hash", "initial_noise_hash")):
            raise AnalysisError(f"{key} lacks prediction-noise evidence")
        if "encoded_condition" not in record:
            raise AnalysisError(f"{key} lacks condition evidence")
        if not isinstance(record.get("precision"), Mapping):
            raise AnalysisError(f"{key} lacks G/D/E precision evidence")
        consumption = record.get("action_consumption")
        if not ((isinstance(consumption, Mapping) and consumption.get("all_steps_match") is True and int(consumption.get("steps", 0)) == 30)
                or (isinstance(record.get("packed_action_token_hashes"), (list, tuple)) and len(record["packed_action_token_hashes"]) == 30)):
            raise AnalysisError(f"{key} lacks all-step action-token evidence")
    tf_noise = _find(records["TF2_real_x16_seed1"], "prediction_noise_hash", "noise_hash", "initial_noise_hash")
    ar_noise = _find(records["AR2_g0_float_last_fp32_seed1"], "prediction_noise_hash", "noise_hash", "initial_noise_hash")
    if tf_noise != ar_noise:
        raise AnalysisError("TF2 and AR2 do not share the approved seed-1 prediction noise")
    tf_action = _action_identity(records["TF2_real_x16_seed1"])
    ar_action = _action_identity(records["AR2_g0_float_last_fp32_seed1"])
    if tf_action is None or ar_action is None or tf_action != ar_action:
        raise AnalysisError("TF2 and AR2 packed.action.tokens evidence differs")
    metrics: dict[str, Any] = {}
    mapping = {"E1": ("G0_real_x0_seed0", 16), "E_TF2": ("TF2_real_x16_seed1", 32), "E_AR2": ("AR2_g0_float_last_fp32_seed1", 32)}
    metrics["target_truth_indices"] = {label: truth_index for label, (_, truth_index) in mapping.items()}
    per_frame_truth = {"G0": truth[1:17], "TF2": truth[17:33], "AR2": truth[17:33]}
    per_frame: dict[str, Any] = {}
    per_frame_rows: list[dict[str, Any]] = []
    for label, (key, truth_index) in mapping.items():
        generated_frames = _prediction_frames(records[key], key=key)
        last = generated_frames[-1:]
        metrics[label] = _json_metric_row(analyze_rgb_metrics(last, truth[truth_index:truth_index + 1]))
        call_name = label.removeprefix("E_").replace("E1", "G0")
        per_frame[call_name] = _json_metric_row(analyze_rgb_metrics(generated_frames, per_frame_truth[call_name]))
        for frame_index, (predicted_frame, truth_frame) in enumerate(zip(generated_frames, per_frame_truth[call_name], strict=True)):
            row = _json_metric_row(analyze_rgb_metrics(predicted_frame[None], truth_frame[None]))
            per_frame_rows.append({"call": call_name, "frame_index": frame_index,
                                   "truth_index": (frame_index + 1 if call_name == "G0" else frame_index + 17), **row})
    metrics["per_frame"] = per_frame
    tf_psnr = metrics["E_TF2"]["psnr"]; ar_psnr = metrics["E_AR2"]["psnr"]
    psnr_delta = psnr_delta_higher_is_better(
        float("inf") if ar_psnr == "inf" else float(ar_psnr),
        float("inf") if tf_psnr == "inf" else float(tf_psnr),
    )
    tf_frames = _prediction_frames(records["TF2_real_x16_seed1"], key="TF2_real_x16_seed1")
    ar_frames = _prediction_frames(records["AR2_g0_float_last_fp32_seed1"], key="AR2_g0_float_last_fp32_seed1")
    common_truth = truth[17:33]
    # The preregistered E_TF2/E_AR2 comparison is the final generated frame
    # against x32. Keep that endpoint delta as the primary result; the
    # all-16-frame aggregate is useful but answers a different question.
    primary_rmse_delta = float(metrics["E_AR2"]["rmse"]) - float(metrics["E_TF2"]["rmse"])
    primary_mae_delta = float(metrics["E_AR2"]["mae"]) - float(metrics["E_TF2"]["mae"])
    tf_error = tf_frames.astype(np.float64) - common_truth.astype(np.float64)
    ar_error = ar_frames.astype(np.float64) - common_truth.astype(np.float64)
    prediction_delta = ar_frames.astype(np.float64) - tf_frames.astype(np.float64)
    tf_full_metrics = analyze_rgb_metrics(tf_frames, common_truth)
    ar_full_metrics = analyze_rgb_metrics(ar_frames, common_truth)
    tf_error_norm = float(np.sqrt(np.mean(tf_error * tf_error)))
    ar_error_norm = float(np.sqrt(np.mean(ar_error * ar_error)))
    full_sequence_rmse_delta = ar_error_norm - tf_error_norm
    full_sequence_mae_delta = float(ar_full_metrics["mae"] - tf_full_metrics["mae"])
    full_sequence_psnr_delta = psnr_delta_higher_is_better(
        float(ar_full_metrics["psnr"]), float(tf_full_metrics["psnr"]))
    prediction_delta_norm = float(np.sqrt(np.mean(prediction_delta * prediction_delta)))
    squared_error_delta = float(np.mean(ar_error * ar_error) - np.mean(tf_error * tf_error))
    squared_error_rhs = float(np.mean(prediction_delta * prediction_delta) + 2.0 * np.mean(tf_error * prediction_delta))
    metrics["delta_feedback_error"] = {
        "primary_scope": "last_frame",
        "rmse": primary_rmse_delta,
        "mae": primary_mae_delta,
        "psnr_delta_higher_is_better": psnr_delta,
        "full_16_frame_aggregate": {
            "scope": "all_16_generated_frames",
            "rmse_delta": full_sequence_rmse_delta,
            "mae_delta": full_sequence_mae_delta,
            "psnr_delta_higher_is_better": full_sequence_psnr_delta,
            "prediction_delta_rmse": prediction_delta_norm,
            "reverse_triangle_bound_rmse": prediction_delta_norm,
            "reverse_triangle_residual": abs(full_sequence_rmse_delta) - prediction_delta_norm,
            "reverse_triangle_sanity": bool(abs(full_sequence_rmse_delta) <= prediction_delta_norm + 1e-12),
            "squared_error_delta": squared_error_delta,
            "squared_error_decomposition_rhs": squared_error_rhs,
            "squared_error_decomposition_residual": squared_error_delta - squared_error_rhs,
            "squared_error_decomposition_sanity": bool(abs(squared_error_delta - squared_error_rhs) <= 1e-12),
        },
    }
    def exact_field(name: str) -> bool:
        left, right = records[required[0]].get(name), records[required[1]].get(name)
        if left is None or right is None:
            return False
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return np.array_equal(np.asarray(left), np.asarray(right))
        return left == right
    # The repeatability gate is deliberately independent of the three metric
    # calls: both the latent and decoded products, the actual condition input,
    # the consumed action, and the prediction noise must all be present and
    # byte-identical.  A pair of absent optional aliases must never pass.
    repeat_fields = ("output_full", "generated_rgb", "encoded_condition")
    def exact_alias(*names: str) -> bool:
        left = _find(records[required[0]], *names)
        right = _find(records[required[1]], *names)
        if left is None or right is None:
            return False
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return np.array_equal(np.asarray(left), np.asarray(right))
        return left == right
    repeat_noise_a = _find(records[required[0]], "prediction_noise_hash", "noise_hash", "initial_noise_hash")
    repeat_noise_b = _find(records[required[1]], "prediction_noise_hash", "noise_hash", "initial_noise_hash")
    repeat_action_a = _find(records[required[0]], "action_consumption", "packed_action_token_hashes")
    repeat_action_b = _find(records[required[1]], "action_consumption", "packed_action_token_hashes")
    repeat_exact = (all(exact_field(name) for name in repeat_fields)
                    and exact_alias("generated_rgb", "decoded_generated_rgb", "decoded_frames")
                    and repeat_noise_a == repeat_noise_b and repeat_action_a == repeat_action_b
                    and records[required[0]].get("action_hash") is not None
                    and records[required[0]].get("action_hash") == records[required[1]].get("action_hash")
                    and exact_alias("condition_input", "condition_input_fp32", "condition_actual", "condition_fp32"))
    metrics["g0_repeat_exact"] = bool(repeat_exact)
    if gt_condition_x16 is None or gt_condition_x32 is None:
        raise AnalysisError("both GT16 and GT32 encoded condition tensors are required")
    conditions: dict[str, Any] = {}
    for label, key, gt in (("G0_x16", "G0_real_x0_seed0", gt_condition_x16),
                           ("TF2_x32", "TF2_real_x16_seed1", gt_condition_x32),
                           ("AR2_x32", "AR2_g0_float_last_fp32_seed1", gt_condition_x32)):
        actual = records[key].get("encoded_condition")
        if actual is None or gt is None:
            raise AnalysisError(f"{label} requires FP32 encoded output condition and encoded ground truth")
        values = condition_latent_metrics(actual, gt, condition_mask)
        conditions[label] = {"rms": values["rms"], "cosine": values["cosine"] if values["cosine"] is not None else "N/A"}
    return {"schema_version": "umi-task8-analysis-v1", "metrics": metrics, "per_frame_rows": per_frame_rows, "conditions": conditions,
            "information_availability": "future observed poses are included in conditioning; actions are not control-only",
            "status": "PASS" if repeat_exact else "FAIL_REPEATABILITY"}


def analyze_task8_run(run_dir: str | Path, *, truth_rgb: Any, output_dir: str | Path | None = None,
                      gt_condition_x16: Any | None = None, gt_condition_x32: Any | None = None,
                      condition_mask: Any | None = None) -> dict[str, Any]:
    root = Path(run_dir).resolve(); samples = root / "samples"
    if not samples.is_dir():
        raise AnalysisError(f"Task 8 samples directory is missing: {samples}")
    entries = {path.name for path in samples.iterdir()}
    unexpected = sorted(entries.difference(FORMAL_SAMPLE_IDS))
    if unexpected:
        raise AnalysisError(f"samples contains non-formal or failed-attempt entries: {unexpected}")
    records: dict[str, Mapping[str, Any]] = {}
    for sample_id in FORMAL_SAMPLE_IDS:
        path = samples / sample_id
        if not path.is_dir():
            raise AnalysisError(f"formal sample directory is missing: {path}")
        records[sample_id] = _load_verified_record(path)
    result = analyze_task8_outputs(records, truth_rgb=truth_rgb, gt_condition_x16=gt_condition_x16,
                                   gt_condition_x32=gt_condition_x32, condition_mask=condition_mask)
    destination = Path(output_dir).resolve() if output_dir is not None else root / "task8_analysis"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "analysis_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with (destination / "per_frame_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        rows = result.get("per_frame_rows", [])
        fields = ["call", "frame_index", "truth_index", "rmse", "mae", "psnr", "frames"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return result


def _decode(value: Any, root: Path) -> Any:
    if isinstance(value, Mapping) and "artifact" in value:
        return _load_numpy_array(root / str(value["artifact"]))
    if isinstance(value, Mapping): return {k: _decode(v, root) for k, v in value.items()}
    if isinstance(value, list): return [_decode(v, root) for v in value]
    return value


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_verified_record(sample_path: Path) -> Mapping[str, Any]:
    status_path = sample_path / "status.json"
    record_path = sample_path / "record.json"
    if not status_path.is_file() or not record_path.is_file():
        raise AnalysisError(f"{sample_path} lacks status.json or record.json")
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"invalid sample status: {status_path}") from error
    if not isinstance(status, Mapping) or status.get("status") != "success":
        raise AnalysisError(f"formal sample is not successful: {sample_path.name}")
    hashes = status.get("artifact_sha256")
    if not isinstance(hashes, Mapping) or "record.json" not in hashes:
        raise AnalysisError(f"{sample_path} lacks artifact_sha256 evidence")
    for name, expected in hashes.items():
        relative = Path(str(name))
        if relative.is_absolute() or relative.name != str(name):
            raise AnalysisError(f"unsafe artifact path in {status_path}: {name!r}")
        artifact = sample_path / relative
        if not artifact.is_file() or _file_hash(artifact) != str(expected):
            raise AnalysisError(f"sample artifact hash mismatch: {artifact}")
    try:
        decoded = _decode(json.loads(record_path.read_text(encoding="utf-8")), sample_path)
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"invalid sample record: {record_path}") from error
    if not isinstance(decoded, Mapping):
        raise AnalysisError(f"sample record is not an object: {record_path}")
    return decoded


def _load_numpy_array(path: str | Path) -> np.ndarray:
    """Load one .npy or one-entry .npz and close an archive immediately."""
    loaded = np.load(Path(path), allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        return np.ascontiguousarray(loaded)
    try:
        names = list(loaded.files)
        if len(names) != 1:
            raise AnalysisError(f"{path} must contain exactly one array, found {names}")
        return np.ascontiguousarray(loaded[names[0]])
    finally:
        loaded.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True); parser.add_argument("--truth-rgb", required=True)
    parser.add_argument("--gt-condition-x16", required=True)
    parser.add_argument("--gt-condition-x32", required=True)
    parser.add_argument("--condition-mask")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    result = analyze_task8_run(
        args.run_dir,
        truth_rgb=_load_numpy_array(args.truth_rgb),
        gt_condition_x16=_load_numpy_array(args.gt_condition_x16),
        gt_condition_x32=_load_numpy_array(args.gt_condition_x32),
        condition_mask=_load_numpy_array(args.condition_mask) if args.condition_mask else None,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False)); return 0 if result["status"] == "PASS" else 2


__all__ = ["AnalysisError", "analyze_rgb_metrics", "analyze_task8_outputs", "analyze_task8_run",
           "condition_latent_metrics", "cosine_or_none", "psnr", "psnr_delta_higher_is_better"]


if __name__ == "__main__":
    raise SystemExit(main())
