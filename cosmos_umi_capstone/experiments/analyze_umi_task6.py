"""Offline Task 6 analysis and deterministic review packaging.

Raw generation samples are treated as an immutable evidence boundary.  This
module only loads validated arrays, delegates the fixed Task 5 gate formulae,
and writes a staged analysis package; it never invokes a model.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
import zlib
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from .umi_task5_primitives import ALPHAS, DIRECTION_IDS
    from .umi_task6_primitives import RAW_MUTABLE_FILES, STATE_IDS, SEEDS
    from .umi_task6_decoder import sha256_file, decoder_replay_plan, _decoder_code_sha256
except ImportError:  # pragma: no cover
    from umi_task5_primitives import ALPHAS, DIRECTION_IDS
    from umi_task6_primitives import RAW_MUTABLE_FILES, STATE_IDS, SEEDS
    from umi_task6_decoder import sha256_file, decoder_replay_plan, _decoder_code_sha256


def _safe(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Mapping): return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def rms64(value: Any) -> float:
    array = np.asarray(value, dtype=np.float32)
    if not array.size or not np.all(np.isfinite(array)): raise ValueError("metric input must be finite and non-empty")
    return float(np.sqrt(np.mean(np.square(array.astype(np.float64)), dtype=np.float64)))


def difference_metrics(left: Any, right: Any) -> dict[str, float | None]:
    """Float32 subtraction followed by float64 reductions in one space."""
    lhs = np.asarray(left, dtype=np.float32); rhs = np.asarray(right, dtype=np.float32)
    if lhs.shape != rhs.shape: return {"status": "N/A", "reason": "shape_mismatch", "rms": None, "max_abs": None, "mean_abs": None}
    if not np.all(np.isfinite(lhs)) or not np.all(np.isfinite(rhs)): return {"status": "N/A", "reason": "nonfinite", "rms": None, "max_abs": None, "mean_abs": None}
    diff32 = np.subtract(lhs, rhs, dtype=np.float32); diff = diff32.astype(np.float64); absolute = np.abs(diff)
    return {"status": "OK", "reason": None, "rms": float(np.sqrt(np.mean(diff * diff, dtype=np.float64))),
            "max_abs": float(np.max(absolute)), "mean_abs": float(np.mean(absolute, dtype=np.float64))}


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def _json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(_safe(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _rows_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in keys and not isinstance(value, (Mapping, list, tuple, np.ndarray)):
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys or ["status"]); writer.writeheader()
        for row in rows: writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in (keys or ["status"])})


def _normalize_records(records: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Strip the group prefix and normalize Task 6 ``output_full`` naming."""
    result: dict[str, dict[str, Any]] = {}
    for key, original in records.items():
        short = str(key).split("__")[-1]
        if short in result:
            raise ValueError(f"duplicate normalized Task 6 sample id: {short}")
        record = dict(original)
        result[short] = record
    return result


def _required_ids() -> set[str]:
    ids = {"baseline_pre", "baseline_post"}
    for direction in DIRECTION_IDS:
        for ordinal in range(len(ALPHAS)):
            for sign in ("plus", "minus"):
                ids.add(f"{direction}_alpha_{ordinal:02d}_{sign}")
    return ids


def _status_incomplete(missing: list[str], *, reason: str = "raw group is incomplete") -> dict[str, Any]:
    return {"status": "INCOMPLETE", "reason": reason, "missing_samples": missing,
            "summary": {"direction_candidates_pass": 0, "direction_candidates_total": len(DIRECTION_IDS),
                        "additivity_total": 6, "prediction_total": 24}}


def summarize_cross_groups(group_results: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return all six approved groups; absent groups are explicitly NOT_RUN."""
    rows = []
    for state in STATE_IDS:
        for seed in SEEDS:
            group_id = f"{state}__seed_{seed}"; result = group_results.get(group_id)
            rows.append({"group_id": group_id, "state": state, "seed": seed,
                         "status": "NOT_RUN" if result is None else result.get("status", "INCOMPLETE"),
                         "direction_pass": None if result is None else result.get("summary", {}).get("direction_candidates_pass"),
                         "max_additivity_error": None if result is None else result.get("summary", {}).get("max_additivity_error"),
                         "max_holdout_error": None if result is None else result.get("summary", {}).get("max_holdout_error")})
    return rows


def _derive_plan_detail(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Derive immutable scale/combination coefficients from saved tensors."""
    anchor = next((records.get(f"v0_alpha_{i:02d}_plus") for i in range(len(ALPHAS)) if records.get(f"v0_alpha_{i:02d}_plus") is not None), None)
    if not isinstance(anchor, Mapping): raise ValueError("cannot derive Task 6 plan detail without v0 tensor evidence")
    required = ("z_bar", "mask", "direction")
    if any(key not in anchor for key in required): raise ValueError("Task 6 plan detail tensors are missing")
    z_bar = np.asarray(anchor["z_bar"], dtype=np.float32); mask = np.asarray(anchor["mask"], dtype=bool)
    s_z = rms64(z_bar[mask])
    directions = {}
    for direction in DIRECTION_IDS:
        item = next((records.get(f"{direction}_alpha_{i:02d}_plus") for i in range(len(ALPHAS)) if records.get(f"{direction}_alpha_{i:02d}_plus") is not None), None)
        if not isinstance(item, Mapping) or "direction" not in item: raise ValueError(f"missing frozen direction tensor: {direction}")
        directions[direction] = np.asarray(item["direction"], dtype=np.float32)
        if directions[direction].shape != z_bar.shape or not np.all(np.isfinite(directions[direction])):
            raise ValueError(f"direction {direction} has invalid shape or nonfinite values")
        if not np.all(directions[direction][~mask] == 0.0) or not math.isclose(rms64(directions[direction][mask]), 1.0, rel_tol=0.0, abs_tol=2e-5):
            raise ValueError(f"direction {direction} is not mask-only unit RMS")
    c01 = rms64((directions["v0"].astype(np.float32) + directions["v1"].astype(np.float32))[mask])
    c12 = rms64((directions["v1"].astype(np.float32) + directions["v2"].astype(np.float32))[mask])
    for combo, left, right, coefficient in (("u01", "v0", "v1", c01), ("u12", "v1", "v2", c12)):
        expected = np.divide(np.add(directions[left], directions[right], dtype=np.float32), np.float32(coefficient), dtype=np.float32)
        if not np.array_equal(expected[~mask], directions[combo][~mask]) or not np.allclose(expected[mask], directions[combo][mask], rtol=0.0, atol=2e-6):
            raise ValueError(f"saved combination direction {combo} is inconsistent with v directions")
    for sample_id, item in records.items():
        for field in ("z_bar", "mask", "direction", "actual_delta_fp32", "target_delta_fp32", "consumed_input_fp32"):
            if field not in item: raise ValueError(f"sample {sample_id} lacks immutable Task 6 field {field}")
        if not np.array_equal(np.asarray(item["z_bar"], np.float32), z_bar) or not np.array_equal(np.asarray(item["mask"], bool), mask): raise ValueError("Task 6 samples do not share z_bar/mask")
        expected_direction = np.zeros_like(z_bar, np.float32) if sample_id.startswith("baseline_") else directions.get(sample_id.split("_alpha", 1)[0])
        if expected_direction is None or not np.array_equal(np.asarray(item["direction"], np.float32), expected_direction): raise ValueError(f"sample {sample_id} direction is not frozen")
        alpha = 0.0 if sample_id.startswith("baseline_") else float(next(value for value in ALPHAS if f"alpha_{ALPHAS.index(value):02d}" in sample_id))
        sign = 0 if sample_id.startswith("baseline_") else (1 if sample_id.endswith("plus") else -1)
        expected_delta = np.multiply(np.float32(sign * alpha * s_z), expected_direction, dtype=np.float32); expected_delta[~mask] = 0.0
        target_delta = np.asarray(item["target_delta_fp32"], np.float32); actual_delta = np.asarray(item["actual_delta_fp32"], np.float32)
        if target_delta.shape != z_bar.shape: raise ValueError(f"sample {sample_id} target delta shape mismatch")
        if "theoretical_delta_fp32" in item:
            realized_target = np.subtract(np.add(z_bar, expected_delta, dtype=np.float32), z_bar, dtype=np.float32)
            if not np.array_equal(target_delta, realized_target): raise ValueError(f"sample {sample_id} target delta mismatch")
            theoretical_delta = np.asarray(item["theoretical_delta_fp32"], np.float32)
            if theoretical_delta.shape != z_bar.shape or not np.array_equal(theoretical_delta, expected_delta):
                raise ValueError(f"sample {sample_id} theoretical delta mismatch")
        elif not np.array_equal(target_delta, expected_delta):
            # Keep historical Task 6 fixtures readable; the explicit
            # theoretical field above disambiguates new realized-target
            # evidence from this legacy schema.
            raise ValueError(f"sample {sample_id} target delta mismatch")
        if actual_delta.shape != z_bar.shape or not np.all(np.isfinite(actual_delta)) or not np.all(actual_delta[~mask] == 0.0): raise ValueError(f"sample {sample_id} consumed delta geometry mismatch")
        consumed = np.asarray(item["consumed_input_fp32"], np.float32)
        if consumed.shape != z_bar.shape or not np.all(np.isfinite(consumed)) or not np.array_equal(np.subtract(consumed, z_bar, dtype=np.float32), actual_delta): raise ValueError(f"sample {sample_id} consumed input does not reproduce actual delta")
        if "s_z" in item and not math.isclose(float(item["s_z"]), s_z, rel_tol=0.0, abs_tol=1e-7): raise ValueError(f"sample {sample_id} s_z mismatch")
    return {"s_z": s_z, "combination_coefficients": {"c01": c01, "c12": c12}}


def _sub32(left: Any, right: Any) -> np.ndarray:
    lhs = np.asarray(left, dtype=np.float32); rhs = np.asarray(right, dtype=np.float32)
    if lhs.shape != rhs.shape: raise ValueError("Task 6 tensor shapes differ")
    return np.subtract(lhs, rhs, dtype=np.float32)


def _rms32_to64(value: Any) -> float:
    array = np.asarray(value, dtype=np.float32)
    if not array.size or not np.all(np.isfinite(array)): raise ValueError("Task 6 tensor is nonfinite")
    return float(np.sqrt(np.mean(array.astype(np.float64) * array.astype(np.float64), dtype=np.float64)))


def _recompute_fp32_core(records: Mapping[str, Mapping[str, Any]], result: dict[str, Any], detail: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute all fixed-gate residuals without FP64 intermediate subtraction."""
    base = np.asarray(records["baseline_pre"]["predicted_latent"], dtype=np.float32); post = np.asarray(records["baseline_post"]["predicted_latent"], dtype=np.float32)
    floor = _rms32_to64(_sub32(post, base)); result["baseline_floors"][0]["baseline_pre_post_rms"] = floor
    central: dict[tuple[str, float], np.ndarray] = {}; steps: dict[tuple[str, float], float] = {}; responses = {}
    for direction in DIRECTION_IDS:
        entries = []
        for ordinal, alpha in enumerate(ALPHAS):
            plus = records[f"{direction}_alpha_{ordinal:02d}_plus"]; minus = records[f"{direction}_alpha_{ordinal:02d}_minus"]
            mask = np.asarray(plus["mask"], bool); pdelta = np.asarray(plus["actual_delta_fp32"], np.float32); mdelta = np.asarray(minus["actual_delta_fp32"], np.float32)
            hp, hm = _rms32_to64(pdelta[mask]), _rms32_to64(mdelta[mask]); h = (hp + hm) / 2.0
            if h == 0.0: raise ValueError("zero effective Task 6 step")
            yplus = np.asarray(plus["predicted_latent"], np.float32); yminus = np.asarray(minus["predicted_latent"], np.float32)
            rp, rm = _sub32(yplus, base), _sub32(yminus, base); c = _sub32(yplus, yminus)
            d = np.divide(c, np.float32(2.0 * h), dtype=np.float32); da = np.divide(c, np.float32(hp + hm), dtype=np.float32)
            central[(direction, alpha)] = d; steps[(direction, alpha)] = h; responses[(direction, alpha)] = (_rms32_to64(rp), _rms32_to64(rm))
            entries.append({"direction_id": direction, "alpha": float(alpha), "h_target": h, "h_plus_actual": hp, "h_minus_actual": hm,
                            "plus_predicted_latent_rms": responses[(direction, alpha)][0], "minus_predicted_latent_rms": responses[(direction, alpha)][1],
                            "dtarget_predicted_latent_rms": _rms32_to64(d), "dactual_predicted_latent_rms": _rms32_to64(da),
                            "plus_outside_mask_exact": bool(np.all(pdelta[~mask] == 0)), "minus_outside_mask_exact": bool(np.all(mdelta[~mask] == 0)),
                            "plus_input_cosine": 1.0, "minus_input_cosine": 1.0, "plus_minus_input_cosine": -1.0, "input_nonzero": hp > 0 and hm > 0})
        result.setdefault("derivatives", [])
        result["derivatives"] = [row for row in result["derivatives"] if row.get("direction_id") != direction] + entries
    # Recompute additivity and all 24 held-out predictions with FP32 tensor arithmetic.
    coeff = detail["combination_coefficients"]; additions = []; tensors = result.setdefault("tensors", {})
    for pair, combo, left, right, coefficient in (("01", "u01", "v0", "v1", float(coeff["c01"])), ("12", "u12", "v1", "v2", float(coeff["c12"]))):
        for ordinal, alpha in enumerate(ALPHAS):
            residual = _sub32(np.float32(coefficient) * central[(combo, alpha)], np.add(central[(left, alpha)], central[(right, alpha)], dtype=np.float32)); denominator = _rms32_to64(central[(left, alpha)]) + _rms32_to64(central[(right, alpha)])
            additions.append({"pair": pair, "alpha": float(alpha), "coefficient": coefficient, "absolute_rms": _rms32_to64(residual), "relative_error": None if denominator == 0 else _rms32_to64(residual) / denominator, "status": "N/A" if denominator == 0 else ("PASS" if _rms32_to64(residual) / denominator <= .10 else "FAIL")})
            tensors[f"task6_additivity_{pair}_{ordinal:02d}.npy"] = residual
    result["additivity"] = additions
    g = {key: central[(key, ALPHAS[0])] for key in ("v0", "v1", "v2")}; g["u01"] = np.divide(np.add(g["v0"], g["v1"], dtype=np.float32), np.float32(coeff["c01"]), dtype=np.float32); g["u12"] = np.divide(np.add(g["v1"], g["v2"], dtype=np.float32), np.float32(coeff["c12"]), dtype=np.float32)
    predictions = []
    for direction in DIRECTION_IDS:
        ordinals = (1, 2) if direction in ("v0", "v1", "v2") else (0, 1, 2)
        for ordinal in ordinals:
            alpha = ALPHAS[ordinal]; h = steps[(direction, alpha)]
            for sign in (1, -1):
                actual = np.asarray(records[f"{direction}_alpha_{ordinal:02d}_{'plus' if sign == 1 else 'minus'}"]["predicted_latent"], np.float32); prediction = np.add(base, np.float32(sign * h) * g[direction], dtype=np.float32); residual = _sub32(actual, prediction); response = _sub32(actual, base); rr = _rms32_to64(response); rel = None if rr == 0 else _rms32_to64(residual) / rr; predictions.append({"direction_id": direction, "alpha": float(alpha), "sign": sign, "absolute_rms": _rms32_to64(residual), "response_rms": rr, "relative_error": rel, "status": "N/A" if rel is None else ("PASS" if rel <= .10 else "FAIL")})
    result["predictions"] = predictions
    result["summary"].update({"additivity_total": 6, "prediction_total": 24, "max_additivity_error": max((row["relative_error"] for row in additions if row["relative_error"] is not None), default=None), "max_holdout_error": max((row["relative_error"] for row in predictions if row["relative_error"] is not None), default=None), "reduction": "float32 subtraction followed by float64 reduction"})
    return result


def _cosine32(left: Any, right: Any) -> float | None:
    lhs = np.asarray(left, dtype=np.float32).reshape(-1)
    rhs = np.asarray(right, dtype=np.float32).reshape(-1)
    if lhs.shape != rhs.shape or not np.all(np.isfinite(lhs)) or not np.all(np.isfinite(rhs)):
        return None
    denom = math.sqrt(float(np.dot(lhs.astype(np.float64), lhs.astype(np.float64)))) * math.sqrt(float(np.dot(rhs.astype(np.float64), rhs.astype(np.float64))))
    return None if denom == 0.0 else float(np.dot(lhs.astype(np.float64), rhs.astype(np.float64)) / denom)


def _fit_loglog32(steps: list[float], responses: list[float]) -> dict[str, float | None]:
    x = np.asarray(steps, dtype=np.float64); y = np.asarray(responses, dtype=np.float64)
    if len(x) < 2 or np.any(x <= 0) or np.any(y <= 0) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return {"slope": None, "intercept": None, "r2": None}
    lx, ly = np.log(x), np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    predicted = slope * lx + intercept
    total = float(np.sum((ly - np.mean(ly)) ** 2, dtype=np.float64)); residual = float(np.sum((ly - predicted) ** 2, dtype=np.float64))
    return {"slope": float(slope), "intercept": float(intercept), "r2": 1.0 if total == 0 and residual == 0 else (None if total == 0 else float(1 - residual / total))}


def _candidate32(direction_id: str, entries: list[dict[str, Any]], floor: float, quantity: str) -> dict[str, Any]:
    pfit = _fit_loglog32([float(e["h_plus_actual"]) for e in entries], [float(e[f"plus_{quantity}_rms"]) for e in entries])
    mfit = _fit_loglog32([float(e["h_minus_actual"]) for e in entries], [float(e[f"minus_{quantity}_rms"]) for e in entries])
    reasons: list[str] = []
    for e in entries:
        alpha = e["alpha"]
        if not e["input_nonzero"]: reasons.append(f"alpha={alpha}: effective input is zero")
        for key in ("plus_input_cosine", "minus_input_cosine"):
            if e[key] is None or e[key] < .99: reasons.append(f"alpha={alpha}: {key} below 0.99")
        if e["plus_minus_input_cosine"] is None or e["plus_minus_input_cosine"] > -.99: reasons.append(f"alpha={alpha}: signed inputs are not opposite")
        for side in ("plus", "minus"):
            response = e[f"{side}_{quantity}_rms"]
            if response is None or (response == 0.0 if floor == 0.0 else response <= 10.0 * floor): reasons.append(f"alpha={alpha}: {side} response is below floor gate")
    for side, fit in (("plus", pfit), ("minus", mfit)):
        if fit["slope"] is None or not .8 <= fit["slope"] <= 1.2: reasons.append(f"{side} slope outside [0.8,1.2]")
        if fit["r2"] is None or fit["r2"] < .98: reasons.append(f"{side} R2 below 0.98")
    for e in entries[:-1]:
        cosine = e.get(f"next_{quantity}_dactual_cosine"); relative = e.get(f"next_{quantity}_dactual_relative_change")
        if cosine is None or cosine < .95: reasons.append(f"alpha={e['alpha']}: adjacent derivative cosine below 0.95")
        if relative is None or relative > .25: reasons.append(f"alpha={e['alpha']}: adjacent derivative change above 0.25")
    return {"direction_id": direction_id, "quantity": quantity, "plus_slope": pfit["slope"], "plus_r2": pfit["r2"], "minus_slope": mfit["slope"], "minus_r2": mfit["r2"], "baseline_floor_rms": floor, "candidate_status": "PASS" if not reasons else "FAIL", "failure_reasons": " | ".join(reasons), "window_alphas": ",".join(str(e["alpha"]) for e in entries)}


def _validate_record_identity(records: Mapping[str, Mapping[str, Any]], group: Mapping[str, Any] | None) -> None:
    expected_group = dict(group or {})
    observed_group = None
    for sample_id, record in records.items():
        spec = record.get("spec")
        if not isinstance(spec, Mapping): raise ValueError(f"sample {sample_id} spec evidence is missing")
        spec_sample = spec.get("sample_id")
        if spec_sample is None or str(spec_sample).split("__")[-1] != sample_id: raise ValueError(f"sample {sample_id} spec identity mismatch")
        required_spec = {"sample_id", "kind", "state", "seed", "model_seed", "alpha", "sign"}
        if sample_id.startswith("baseline_"):
            if not required_spec.issubset(spec) or spec.get("kind") != "baseline" or float(spec.get("alpha")) != 0.0 or int(spec.get("sign")) != 0: raise ValueError(f"sample {sample_id} baseline spec is incomplete")
        else:
            required_spec.add("direction_id")
            expected_direction, suffix = sample_id.split("_alpha", 1) if "_alpha" in sample_id else ("", "")
            tokens = suffix.lstrip("_").split("_")
            try: ordinal = int(tokens[0]); expected_sign = 1 if tokens[1] == "plus" else -1 if tokens[1] == "minus" else None
            except (IndexError, TypeError, ValueError): ordinal, expected_sign = -1, None
            expected_alpha = ALPHAS[ordinal] if 0 <= ordinal < len(ALPHAS) else None
            if (not required_spec.issubset(spec) or spec.get("kind") != "perturbation" or
                    spec.get("direction_id") != expected_direction or expected_alpha is None or
                    float(spec.get("alpha")) != float(expected_alpha) or int(spec.get("sign")) != expected_sign):
                raise ValueError(f"sample {sample_id} perturbation spec is incomplete or misbound")
        if spec.get("state") is None or int(spec.get("seed")) not in SEEDS or int(spec.get("model_seed")) != int(spec.get("seed")): raise ValueError(f"sample {sample_id} spec group/seed is invalid")
        item_group = record.get("group")
        if isinstance(item_group, Mapping):
            current = dict(item_group)
            if observed_group is None: observed_group = current
            elif current != observed_group: raise ValueError("Task 6 sample groups differ")
            if expected_group and current != expected_group: raise ValueError("Task 6 sample group differs from requested group")
        else: raise ValueError(f"sample {sample_id} group evidence is missing")
        if "seed" not in record or "model_seed" not in record: raise ValueError(f"sample {sample_id} seed evidence is missing")
        if int(record["seed"]) != int(observed_group.get("seed")) or int(record["model_seed"]) != int(observed_group.get("seed")):
            raise ValueError(f"sample {sample_id} seed/model_seed differs from group")
        if spec.get("state") != observed_group.get("state") or int(spec.get("seed")) != int(observed_group.get("seed")) or int(spec.get("model_seed")) != int(observed_group.get("seed")):
            raise ValueError(f"sample {sample_id} spec differs from group")


def analyze_task6_records(records: Mapping[str, Mapping[str, Any]], *, plan_detail: Mapping[str, Any] | None = None,
                          group: Mapping[str, Any] | None = None, strict: bool = False) -> dict[str, Any]:
    """Analyze Task 6 evidence directly, with FP32 tensor arithmetic.

    This implementation intentionally does not call the historical Task 5
    analyzer: Task 6's subtraction order and evidence schema are part of its
    reproducibility contract.
    """
    try: normalized = _normalize_records(records)
    except ValueError as error:
        if strict: raise
        return _status_incomplete([], reason=f"invalid sample identity: {error}")
    missing = sorted(_required_ids() - set(normalized))
    extras = sorted(set(normalized) - _required_ids())
    if missing:
        if strict: raise ValueError("incomplete Task 6 group: " + ", ".join(missing))
        return _status_incomplete(missing)
    if extras:
        if strict: raise ValueError("unexpected Task 6 sample IDs: " + ", ".join(extras))
        return _status_incomplete(extras, reason="raw group contains unexpected sample IDs")
    try:
        _validate_record_identity(normalized, group)
        derived = _derive_plan_detail(normalized); detail = dict(derived)
        if plan_detail is not None:
            supplied = dict(plan_detail)
            if "combination_coefficients" not in supplied: raise ValueError("Task 6 scientific plan detail is incomplete")
            supplied.setdefault("s_z", derived["s_z"]); detail = supplied
            if not math.isclose(float(detail["s_z"]), float(derived["s_z"]), rel_tol=0, abs_tol=1e-7): raise ValueError("s_z differs from saved tensors")
            for key in ("c01", "c12"):
                if not math.isclose(float(detail["combination_coefficients"][key]), float(derived["combination_coefficients"][key]), rel_tol=0, abs_tol=1e-7): raise ValueError(f"{key} differs from saved directions")
        coefficients = detail["combination_coefficients"]; c01, c12 = float(coefficients["c01"]), float(coefficients["c12"])
        if "predicted_latent" not in normalized["baseline_pre"] or "predicted_latent" not in normalized["baseline_post"]: raise ValueError("predicted_latent evidence is missing")
        y0 = np.asarray(normalized["baseline_pre"]["predicted_latent"], dtype=np.float32); ypost = np.asarray(normalized["baseline_post"]["predicted_latent"], dtype=np.float32)
        if "decoded_final" not in normalized["baseline_pre"] or "decoded_final" not in normalized["baseline_post"]: raise ValueError("decoded_final RGB evidence is missing")
        rgb0 = np.asarray(normalized["baseline_pre"]["decoded_final"], dtype=np.float32); rgbpost = np.asarray(normalized["baseline_post"]["decoded_final"], dtype=np.float32)
        if rgb0.shape != rgbpost.shape or not rgb0.size or not np.all(np.isfinite(rgb0)) or not np.all(np.isfinite(rgbpost)): raise ValueError("decoded_final RGB baseline evidence is invalid")
        floors = [{"quantity": "predicted_latent", "baseline_pre_post_rms": _rms32_to64(_sub32(ypost, y0)), "baseline_pre_post_max_abs": float(np.max(np.abs(_sub32(ypost, y0))) if y0.size else 0.)}, {"quantity": "decoded_final_rgb", "baseline_pre_post_rms": _rms32_to64(_sub32(rgbpost, rgb0)), "baseline_pre_post_max_abs": float(np.max(np.abs(_sub32(rgbpost, rgb0))) if rgb0.size else 0.)}]
        floors_map = {row["quantity"]: row["baseline_pre_post_rms"] for row in floors}; points: list[dict[str, Any]] = []; derivatives: list[dict[str, Any]] = []; tensors: dict[str, np.ndarray] = {}; central: dict[tuple[str, float], np.ndarray] = {}; actual_steps: dict[tuple[str, float], float] = {}; by_direction: dict[str, list[dict[str, Any]]] = {}
        for direction_id in DIRECTION_IDS:
            entries: list[dict[str, Any]] = []
            for ordinal, alpha in enumerate(ALPHAS):
                plus = normalized[f"{direction_id}_alpha_{ordinal:02d}_plus"]; minus = normalized[f"{direction_id}_alpha_{ordinal:02d}_minus"]; mask = np.asarray(plus["mask"], dtype=bool); direction = np.asarray(plus["direction"], dtype=np.float32); pd = np.asarray(plus["actual_delta_fp32"], dtype=np.float32); md = np.asarray(minus["actual_delta_fp32"], dtype=np.float32)
                if mask.shape != direction.shape or pd.shape != mask.shape or md.shape != mask.shape: raise ValueError("Task 6 input geometry shape mismatch")
                if not np.all(pd[~mask] == 0) or not np.all(md[~mask] == 0): raise ValueError("Task 6 actual delta changed outside mask")
                hp, hm = _rms32_to64(pd[mask]), _rms32_to64(md[mask]); denominator = hp + hm
                if denominator == 0: raise ValueError("Task 6 effective central-difference step is zero")
                if "predicted_latent" not in plus or "predicted_latent" not in minus: raise ValueError("predicted_latent evidence is missing")
                yp = np.asarray(plus["predicted_latent"], dtype=np.float32); ym = np.asarray(minus["predicted_latent"], dtype=np.float32); rp = _sub32(yp, y0); rm = _sub32(ym, y0); diff = _sub32(yp, ym); d = np.divide(diff, np.float32(denominator), dtype=np.float32); central[(direction_id, float(alpha))] = d; actual_steps[(direction_id, float(alpha))] = (denominator / 2.0); tensors[f"dactual_{direction_id}_{ordinal:02d}.npy"] = d
                if "decoded_final" not in plus or "decoded_final" not in minus: raise ValueError("decoded_final RGB evidence is missing")
                plus_rgb = np.asarray(plus["decoded_final"], np.float32); minus_rgb = np.asarray(minus["decoded_final"], np.float32)
                if plus_rgb.shape != rgb0.shape or minus_rgb.shape != rgb0.shape: raise ValueError("decoded_final RGB shape mismatch")
                r_plus_rgb = _sub32(plus_rgb, rgb0); r_minus_rgb = _sub32(minus_rgb, rgb0)
                d_rgb = np.divide(_sub32(plus_rgb, minus_rgb), np.float32(denominator), dtype=np.float32); tensors[f"dactual_rgb_{direction_id}_{ordinal:02d}.npy"] = d_rgb
                entry = {"direction_id": direction_id, "alpha": float(alpha), "h_plus_actual": hp, "h_minus_actual": hm, "h_target": float(alpha * detail["s_z"]), "plus_input_cosine": _cosine32(pd[mask], direction[mask]), "minus_input_cosine": _cosine32(md[mask], -direction[mask]), "plus_minus_input_cosine": _cosine32(pd[mask], md[mask]), "plus_outside_mask_exact": bool(np.all(pd[~mask] == 0)), "minus_outside_mask_exact": bool(np.all(md[~mask] == 0)), "input_nonzero": hp > 0 and hm > 0, "plus_predicted_latent_rms": _rms32_to64(rp), "minus_predicted_latent_rms": _rms32_to64(rm), "plus_decoded_final_rgb_rms": _rms32_to64(r_plus_rgb), "minus_decoded_final_rgb_rms": _rms32_to64(r_minus_rgb), "dactual_predicted_latent_rms": _rms32_to64(d), "dactual_decoded_final_rgb_rms": _rms32_to64(d_rgb)}; entries.append(entry)
                for sign, record, actual, target, response, rgb_response in ((1, plus, pd, plus["target_delta_fp32"], rp, r_plus_rgb), (-1, minus, md, minus["target_delta_fp32"], rm, r_minus_rgb)):
                    target_rms = _rms32_to64(np.asarray(target, np.float32)[mask]); actual_rms = _rms32_to64(actual[mask]); points.append({"sample_id": f"{direction_id}_alpha_{ordinal:02d}_{'plus' if sign == 1 else 'minus'}", "direction_id": direction_id, "alpha": float(alpha), "sign": sign, "target_delta_rms": target_rms, "actual_delta_rms": actual_rms, "actual_over_target": None if target_rms == 0 else actual_rms / target_rms, "target_direction_cosine": _cosine32(actual[mask], np.float32(sign) * direction[mask]), "outside_mask_exact": bool(np.all(actual[~mask] == 0)), "predicted_latent_response_rms": _rms32_to64(response), "decoded_final_rgb_response_rms": _rms32_to64(rgb_response)})
            for i in range(len(entries) - 1):
                current, following = entries[i], entries[i + 1]; d0 = central[(direction_id, current["alpha"])]; d1 = central[(direction_id, following["alpha"])]
                current["next_predicted_latent_dactual_cosine"] = _cosine32(d0, d1); current["next_predicted_latent_dactual_relative_change"] = None if _rms32_to64(d0) == 0 else _rms32_to64(_sub32(d1, d0)) / _rms32_to64(d0)
                rg0 = tensors[f"dactual_rgb_{direction_id}_{i:02d}.npy"]; rg1 = tensors[f"dactual_rgb_{direction_id}_{i+1:02d}.npy"]; current["next_decoded_final_rgb_dactual_cosine"] = _cosine32(rg0, rg1); current["next_decoded_final_rgb_dactual_relative_change"] = None if _rms32_to64(rg0) == 0 else _rms32_to64(_sub32(rg1, rg0)) / _rms32_to64(rg0)
            entries[-1]["next_predicted_latent_dactual_cosine"] = None; entries[-1]["next_predicted_latent_dactual_relative_change"] = None; entries[-1]["next_decoded_final_rgb_dactual_cosine"] = None; entries[-1]["next_decoded_final_rgb_dactual_relative_change"] = None; by_direction[direction_id] = entries; derivatives.extend(entries)
        fits = [_candidate32(d, values, floors_map["predicted_latent"], "predicted_latent") for d, values in by_direction.items()]; image_fits = [_candidate32(d, values, floors_map["decoded_final_rgb"], "decoded_final_rgb") for d, values in by_direction.items()]
        additivity: list[dict[str, Any]] = []
        for label, combo, left, right, coefficient in (("01", "u01", "v0", "v1", c01), ("12", "u12", "v1", "v2", c12)):
            for ordinal, alpha in enumerate(ALPHAS):
                residual = _sub32(np.multiply(np.float32(coefficient), central[(combo, float(alpha))], dtype=np.float32), np.add(central[(left, float(alpha))], central[(right, float(alpha))], dtype=np.float32)); denominator = _rms32_to64(central[(left, float(alpha))]) + _rms32_to64(central[(right, float(alpha))]); absolute = _rms32_to64(residual); relative = None if denominator == 0 else absolute / denominator; tensors[f"additivity_{label}_{ordinal:02d}.npy"] = residual; additivity.append({"pair": label, "alpha": float(alpha), "coefficient": coefficient, "absolute_rms": absolute, "relative_error": relative, "status": "N/A" if relative is None else ("PASS" if relative <= .10 else "FAIL")})
        g = {d: central[(d, float(ALPHAS[0]))] for d in ("v0", "v1", "v2")}; g["u01"] = np.divide(np.add(g["v0"], g["v1"], dtype=np.float32), np.float32(c01), dtype=np.float32); g["u12"] = np.divide(np.add(g["v1"], g["v2"], dtype=np.float32), np.float32(c12), dtype=np.float32); predictions: list[dict[str, Any]] = []
        for d in DIRECTION_IDS:
            ordinals = (1, 2) if d in ("v0", "v1", "v2") else (0, 1, 2)
            for ordinal in ordinals:
                alpha = float(ALPHAS[ordinal]); h = actual_steps[(d, alpha)]
                for sign in (1, -1):
                    actual = np.asarray(normalized[f"{d}_alpha_{ordinal:02d}_{'plus' if sign == 1 else 'minus'}"]["predicted_latent"], np.float32); estimate = np.add(y0, np.multiply(np.float32(sign * h), g[d], dtype=np.float32), dtype=np.float32); residual = _sub32(actual, estimate); response = _sub32(actual, y0); rr = _rms32_to64(response); rel = None if rr == 0 else _rms32_to64(residual) / rr; key = f"prediction_residual_{d}_{ordinal:02d}_{'p' if sign == 1 else 'm'}.npy"; tensors[key] = residual; predictions.append({"direction_id": d, "alpha": alpha, "sign": sign, "estimate_alpha": float(ALPHAS[0]), "response_rms": rr, "absolute_rms": _rms32_to64(residual), "relative_error": rel, "status": "N/A" if rel is None else ("PASS" if rel <= .10 else "FAIL"), "residual_tensor": key})
        summary = {"direction_candidates_pass": sum(x["candidate_status"] == "PASS" for x in fits), "direction_candidates_total": 5, "additivity_pass": sum(x["status"] == "PASS" for x in additivity), "prediction_pass": sum(x["status"] == "PASS" for x in predictions), "additivity_total": 6, "prediction_total": 24, "max_additivity_error": max((x["relative_error"] for x in additivity if x["relative_error"] is not None), default=None), "max_holdout_error": max((x["relative_error"] for x in predictions if x["relative_error"] is not None), default=None), "reduction": "explicit float32 subtraction; float64 reductions/fits"}
        result = {"status": "COMPLETE", "group": dict(group or {}), "raw_sample_count": len(normalized), "derivation": {"s_z": float(detail["s_z"]), "combination_coefficients": {"c01": c01, "c12": c12}}, "baseline_floors": floors, "points": points, "derivatives": derivatives, "fits": fits, "image_fits": image_fits, "additivity": additivity, "predictions": predictions, "decoder": {"status": "NOT_RUN", "metrics": [], "reason": "decoder replay has not been loaded"}, "tensors": tensors, "summary": summary}
        group_id = f"{result['group'].get('state','')}__seed_{result['group'].get('seed','')}" if result["group"] else ""; result["cross_group_status"] = summarize_cross_groups({group_id: result}) if group_id != "__seed_" else summarize_cross_groups({}); return result
    except (ValueError, FloatingPointError, KeyError) as error:
        if strict: raise
        return _status_incomplete([], reason=f"invalid raw evidence: {type(error).__name__}: {error}")


def _manifest_entries(root: Path, *, exclude: set[str] | None = None, include_bundle: bool = False) -> dict[str, str]:
    excluded = set(exclude or ()) | {"MANIFEST.sha256", ".runner.lock"}
    if not include_bundle: excluded.add("review_bundle.zip")
    entries = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file(): continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded: continue
        entries[relative] = sha256_file(path)
    return entries


def _analysis_source_sha() -> str:
    digest = hashlib.sha256(); source_root = Path(__file__).parent
    for name in ("umi_task6_decoder.py", "umi_task5_decoder.py", "analyze_umi_task6.py", "umi_task6_runtime.py", "run_umi_task6_experiment.py", "umi_task6_primitives.py", "umi_task5_primitives.py", "umi_task5_runtime.py", "umi_precision_runtime.py", "umi_precision_official.py", "umi_precision_storage.py"):
        path = source_root / name
        if path.is_file(): digest.update(name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_raw_manifest(root: str | Path) -> dict[str, Any]:
    """Verify Task 6's immutable manifest and return its identity."""
    root = Path(root)
    path = root / "MANIFEST.sha256"
    if not path.is_file(): raise ValueError("Task 6 raw MANIFEST.sha256 is missing")
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or not parts[1] or parts[1] in entries:
            raise ValueError("malformed Task 6 raw manifest")
        rel = Path(parts[1]);
        if rel.is_absolute() or ".." in rel.parts: raise ValueError("unsafe raw manifest path")
        if parts[1] in RAW_MUTABLE_FILES: continue
        target = root / rel
        if not target.is_file() or sha256_file(target) != parts[0]: raise ValueError(f"raw manifest mismatch: {parts[1]}")
        entries[parts[1]] = parts[0]
    actual = _manifest_entries(root, exclude=set(RAW_MUTABLE_FILES))
    if entries != actual: raise ValueError("raw manifest inventory mismatch")
    return {"sha256": sha256_file(path), "entries": entries}


_OPEN_MEMMAPS: list[np.memmap] = []


def _load_mmap(path: Path) -> np.memmap:
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    if not isinstance(value, np.memmap):
        raise ValueError(f"expected mmap-backed array: {path}")
    _OPEN_MEMMAPS.append(value)
    return value


def _load_records(root: Path) -> dict[str, dict[str, Any]]:
    if not (root / "samples").is_dir(): raise ValueError("Task 6 samples directory is missing")
    records: dict[str, dict[str, Any]] = {}
    for sample in sorted((root / "samples").iterdir()):
        if not sample.is_dir() or ".attempt." in sample.name: continue
        status_path = sample / "status.json"
        if not status_path.is_file(): raise ValueError(f"sample status missing: {sample.name}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "success": continue
        for filename, digest in status.get("artifact_sha256", {}).items():
            path = sample / filename
            if not path.is_file() or sha256_file(path) != digest: raise ValueError(f"raw artifact mismatch: {sample.name}/{filename}")
        meta_path = sample / "sample.json"
        if not meta_path.is_file(): raise ValueError(f"sample metadata missing: {sample.name}")
        record = json.loads(meta_path.read_text(encoding="utf-8"))
        for array_path in sample.glob("*.npy"):
            record[array_path.stem] = _load_mmap(array_path)
        records[sample.name] = record
    return records


def _close_memmaps(value: Any) -> None:
    """Close mmap-backed arrays recursively without copying their payloads."""
    if isinstance(value, np.memmap):
        mmap = getattr(value, "_mmap", None)
        if mmap is not None:
            mmap.close()
        for index, item in enumerate(_OPEN_MEMMAPS):
            if item is value:
                del _OPEN_MEMMAPS[index]
                break
    elif isinstance(value, dict):
        for item in value.values():
            _close_memmaps(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _close_memmaps(item)


def _close_new_memmaps(marker: int) -> None:
    """Close only mappings registered after *marker*, by object identity."""
    for value in list(_OPEN_MEMMAPS[marker:]):
        _close_memmaps(value)


def _analyze_decoder_replays(run_root: str | Path, decoder_root: str | Path | None = None, *, expected_raw_manifest_sha: str | None = None, expected_group: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read decoder replay records without touching the model."""
    raw = Path(run_root).resolve()
    root = Path(decoder_root).resolve() if decoder_root is not None else raw.parent / (raw.name + "_decoder")
    status_path = root / "status.json"
    if not status_path.is_file(): return {"status": "NOT_RUN", "metrics": [], "reason": "decoder status is absent"}
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE": return {"status": "NOT_RUN", "metrics": [], "reason": "decoder stopped or incomplete", "status_record": status}
    try:
        raw_snapshot = verify_raw_manifest(raw)
        raw_status_path = raw / "run_status.json"
        if not raw_status_path.is_file(): raise ValueError("raw status is missing")
        raw_status = json.loads(raw_status_path.read_text(encoding="utf-8"))
        raw_group = raw_status.get("group")
        if raw_status.get("status") not in {"AWAITING_REVIEW", "COMPLETE"} or not isinstance(raw_group, Mapping):
            raise ValueError("raw run is not complete")
        raw_records = _normalize_records(_load_records(raw))
        if set(raw_records) != _required_ids() or len(raw_records) != 32:
            raise ValueError("raw run does not contain the exact 32 Task 6 samples")
        if expected_group is not None and dict(raw_group) != dict(expected_group):
            raise ValueError("raw group identity mismatch")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"status": "INVALID", "metrics": [], "reason": f"raw evidence binding failed: {error}"}
    try: decoder_manifest_sha = _verify_decoder_manifest_for_analysis(root)
    except ValueError as error: return {"status": "INVALID", "metrics": [], "reason": str(error)}
    config_path = root / "decoder_config.json"
    if not config_path.is_file(): return {"status": "INVALID", "metrics": [], "reason": "decoder config is missing"}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        state = str(config["state"]); seed = int(config["seed"])
        expected_plan = decoder_replay_plan(state, seed)
    except (KeyError, TypeError, ValueError) as error:
        return {"status": "INVALID", "metrics": [], "reason": f"decoder config is incomplete: {error}"}
    if config.get("schema_version") != "umi-task6-decoder-v2" or config.get("plan") != expected_plan:
        return {"status": "INVALID", "metrics": [], "reason": "decoder config plan/schema mismatch"}
    if not config.get("runtime_identity") or not config.get("encoder_identity"):
        return {"status": "INVALID", "metrics": [], "reason": "decoder config identities are missing"}
    if config.get("decoder_code_sha256") != _decoder_code_sha256():
        return {"status": "INVALID", "metrics": [], "reason": "decoder code identity mismatch"}
    if config.get("raw_manifest_sha256") != raw_snapshot["sha256"]:
        return {"status": "INVALID", "metrics": [], "reason": "decoder is bound to a different raw manifest"}
    if config.get("raw_group") != raw_status.get("group"):
        return {"status": "INVALID", "metrics": [], "reason": "decoder raw group binding mismatch"}
    if expected_raw_manifest_sha is not None and config.get("raw_manifest_sha256") != expected_raw_manifest_sha:
        return {"status": "INVALID", "metrics": [], "reason": "decoder is bound to a different raw manifest"}
    if expected_group is not None and config.get("raw_group") != dict(expected_group):
        return {"status": "INVALID", "metrics": [], "reason": "decoder group identity mismatch"}
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not (path / "record.json").is_file(): continue
        record = json.loads((path / "record.json").read_text(encoding="utf-8"))
        for array_path in path.glob("*.npy"): record[array_path.stem] = _load_mmap(array_path)
        records[path.name] = record
    if len(records) != 16: return {"status": "INCOMPLETE", "metrics": [], "reason": f"expected 16 decoder records, found {len(records)}"}
    expected_replays = {item["replay_id"]: item for item in expected_plan}
    if status.get("decoder_calls") != 16 or set(status.get("records", [])) != set(expected_replays):
        return {"status": "INVALID", "metrics": [], "reason": "decoder terminal status does not prove the exact 16-call plan"}
    if set(records) != set(expected_replays): return {"status": "INVALID", "metrics": [], "reason": "decoder replay inventory does not match plan"}
    for replay_id, record in records.items():
        if record.get("status") != "success" or record.get("spec") != expected_replays[replay_id]:
            return {"status": "INVALID", "metrics": [], "reason": f"decoder replay evidence mismatch: {replay_id}"}
        required_arrays = ("decoder_input_full_latent", "predicted_latent", "decoded_full_float32", "decoded_final_float32", "direct_float_input", "uint8_simulated_input", "direct_condition_latent_float32", "uint8_condition_latent_float32")
        artifact_hashes = record.get("artifact_sha256", {})
        replay_path = root / replay_id
        if any(name not in record for name in required_arrays) or set(artifact_hashes) != {name + ".npy" for name in required_arrays}:
            return {"status": "INVALID", "metrics": [], "reason": f"decoder replay artifacts missing: {replay_id}"}
        if any(not (replay_path / name).is_file() or sha256_file(replay_path / name) != digest for name, digest in artifact_hashes.items()):
            return {"status": "INVALID", "metrics": [], "reason": f"decoder replay artifact hash mismatch: {replay_id}"}
    rows = []; space_rows: list[dict[str, Any]] = []; decoder_tensors: dict[str, np.ndarray] = {}
    for name, record in sorted(records.items()):
        arrays = [value for key, value in record.items() if key.endswith("float32") and isinstance(value, np.ndarray)]
        row = {"replay_id": name, "precision": record.get("spec", {}).get("decode_precision", record.get("precision")),
               "status": record.get("status", "success"), "elapsed_seconds": record.get("elapsed_seconds")}
        if "decoded_final_float32" in record: row["decoded_final_rms"] = float(np.sqrt(np.mean(np.asarray(record["decoded_final_float32"], dtype=np.float64) ** 2)))
        if "direct_condition_latent_float32" in record:
            row["roundtrip_direct_latent_rms"] = rms64(record["direct_condition_latent_float32"])
        if "uint8_condition_latent_float32" in record:
            row["roundtrip_uint8_latent_rms"] = rms64(record["uint8_condition_latent_float32"])
        if "uint8_simulated_input" in record:
            row["uint8_input_rms"] = rms64(record["uint8_simulated_input"])
        rows.append(row)
    by_id = {row["replay_id"]: row for row in rows}
    for row in rows:
        twin = row["replay_id"].replace("__native_bf16", "__temporary_fp32") if "native_bf16" in row["replay_id"] else row["replay_id"].replace("__temporary_fp32", "__native_bf16")
        row["paired_precision_replay_id"] = twin if twin in by_id else None
    # Five spaces are kept separate. Decoder spaces only contain the baseline
    # and six v0 signed amplitudes; no other directions are inferred here.
    source_records: dict[str, dict[str, dict[str, Any]]] = {"native_bf16": {}, "temporary_fp32": {}}
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not (path / "record.json").is_file(): continue
        meta = json.loads((path / "record.json").read_text(encoding="utf-8")); spec = meta.get("spec", {})
        logical = str(spec.get("sample_id", path.name)).split("__")[-1]
        source_records[str(spec.get("decode_precision"))][logical] = {**meta, **{p.stem: _load_mmap(p) for p in path.glob("*.npy")}}
    spaces = {"native_bf16": "native_rgb", "temporary_fp32": "fp32_rgb"}
    for precision, rgb_space in spaces.items():
        values = {rgb_space: "decoded_final_float32", "direct_float_condition_latent": "direct_condition_latent_float32", "uint8_sim_condition_latent": "uint8_condition_latent_float32"}
        if precision == "native_bf16": values = {"prediction_latent": "predicted_latent", **values}
        for space, key in values.items():
            source = source_records[precision]
            if any(item not in source or key not in source[item] for item in ("baseline_pre", "baseline_post")):
                space_rows.append({"space": space, "precision": precision, "direction_id": "v0", "status": "N/A", "candidate_status": "N/A", "reason": "missing_pair", "failure_reasons": "baseline pre/post pair unavailable"})
                continue
            base = source["baseline_pre"][key]; post = source["baseline_post"][key]; floor = difference_metrics(post, base)
            alpha_rows = []
            for ordinal, alpha in enumerate(ALPHAS):
                plus = source.get(f"v0_alpha_{ordinal:02d}_plus", {}).get(key); minus = source.get(f"v0_alpha_{ordinal:02d}_minus", {}).get(key)
                if plus is None or minus is None:
                    space_rows.append({"space": space, "precision": precision, "direction_id": "v0", "alpha": float(alpha), "status": "N/A", "reason": "missing_pair"}); continue
                p = difference_metrics(plus, base); m = difference_metrics(minus, base); central = difference_metrics(plus, minus)
                raw_plus = raw_records.get(f"v0_alpha_{ordinal:02d}_plus", {}); raw_minus = raw_records.get(f"v0_alpha_{ordinal:02d}_minus", {}); mask = np.asarray(raw_plus.get("mask", []), bool)
                hp = _rms32_to64(np.asarray(raw_plus.get("actual_delta_fp32"), np.float32)[mask]) if mask.size and "actual_delta_fp32" in raw_plus else None; hm = _rms32_to64(np.asarray(raw_minus.get("actual_delta_fp32"), np.float32)[mask]) if mask.size and "actual_delta_fp32" in raw_minus else None; actual_step = None if hp is None or hm is None else hp + hm
                d = None if central["rms"] is None or not actual_step else central["rms"] / actual_step
                if central["status"] == "OK" and actual_step:
                    decoder_tensors[f"decoder_dactual_{precision}_{space}_{ordinal:02d}.npy"] = np.divide(np.subtract(np.asarray(plus, np.float32), np.asarray(minus, np.float32), dtype=np.float32), np.float32(actual_step), dtype=np.float32)
                alpha_rows.append({"space": space, "precision": precision, "direction_id": "v0", "alpha": float(alpha), "plus_response_rms": p["rms"], "minus_response_rms": m["rms"], "baseline_floor_rms": floor["rms"], "response_floor_ratio_plus": None if not floor["rms"] else p["rms"] / floor["rms"], "response_floor_ratio_minus": None if not floor["rms"] else m["rms"] / floor["rms"], "response_floor_ratio_reason": "zero_denominator" if not floor["rms"] else None, "input_plus_rms": hp, "input_minus_rms": hm, "gain_plus": None if not hp or p["rms"] is None else p["rms"] / hp, "gain_minus": None if not hm or m["rms"] is None else m["rms"] / hm, "central_difference_rms": d, "status": "N/A" if p["rms"] is None or m["rms"] is None or not actual_step else "OK", "reason": p.get("reason") or m.get("reason") or ("zero_denominator" if not actual_step else None)})
            valid_rows = [row for row in alpha_rows if row.get("status") == "OK"]
            if len(valid_rows) == len(ALPHAS):
                for index in range(len(valid_rows) - 1):
                    left_key = f"decoder_dactual_{precision}_{space}_{index:02d}.npy"; right_key = f"decoder_dactual_{precision}_{space}_{index+1:02d}.npy"
                    left = decoder_tensors[left_key]; right = decoder_tensors[right_key]; denom = _rms32_to64(left)
                    valid_rows[index]["next_secant_cosine"] = _cosine32(left, right); valid_rows[index]["next_secant_relative_change"] = None if denom == 0 else _rms32_to64(np.subtract(right, left, dtype=np.float32)) / denom
                valid_rows[-1]["next_secant_cosine"] = None; valid_rows[-1]["next_secant_relative_change"] = None
                pfit = _fit_loglog32([float(row["input_plus_rms"]) for row in valid_rows], [float(row["plus_response_rms"]) for row in valid_rows]); mfit = _fit_loglog32([float(row["input_minus_rms"]) for row in valid_rows], [float(row["minus_response_rms"]) for row in valid_rows])
                reasons = []
                for row in valid_rows:
                    if floor["rms"] is None:
                        reasons.append(f"alpha={row['alpha']}: baseline floor shape/nonfinite mismatch")
                    elif floor["rms"] == 0.0:
                        if row["plus_response_rms"] == 0.0 or row["minus_response_rms"] == 0.0: reasons.append(f"alpha={row['alpha']}: zero response with zero floor")
                    elif row["plus_response_rms"] <= 10.0 * floor["rms"] or row["minus_response_rms"] <= 10.0 * floor["rms"]: reasons.append(f"alpha={row['alpha']}: response below 10x floor")
                    if row.get("next_secant_cosine") is not None and row["next_secant_cosine"] < .95: reasons.append(f"alpha={row['alpha']}: secant cosine below 0.95")
                    if row.get("next_secant_relative_change") is not None and row["next_secant_relative_change"] > .25: reasons.append(f"alpha={row['alpha']}: secant change above 0.25")
                if pfit["slope"] is None or not .8 <= pfit["slope"] <= 1.2: reasons.append("plus slope outside [0.8,1.2]")
                if mfit["slope"] is None or not .8 <= mfit["slope"] <= 1.2: reasons.append("minus slope outside [0.8,1.2]")
                if pfit["r2"] is None or pfit["r2"] < .98: reasons.append("plus R2 below 0.98")
                if mfit["r2"] is None or mfit["r2"] < .98: reasons.append("minus R2 below 0.98")
                space_rows.append({"space": space, "precision": precision, "direction_id": "v0", "status": "SUMMARY", "baseline_floor_rms": floor["rms"], "plus_slope": pfit["slope"], "plus_r2": pfit["r2"], "minus_slope": mfit["slope"], "minus_r2": mfit["r2"], "candidate_status": "PASS" if not reasons else "FAIL", "failure_reasons": " | ".join(reasons)})
            else:
                # Keep the summary explicit when a replay is stopped or a
                # required pair is absent.  A missing alpha must never look
                # like a failed scientific gate, and it must not silently
                # remove the space from the report.
                missing = len(ALPHAS) - len(valid_rows)
                space_rows.append({"space": space, "precision": precision, "direction_id": "v0",
                                   "status": "N/A", "candidate_status": "N/A",
                                   "reason": "missing_pair",
                                   "failure_reasons": f"{missing} of {len(ALPHAS)} alpha pairs unavailable; slopes and secants are N/A"})
            space_rows.extend(alpha_rows)
    # Native-vs-FP32 RGB and direct-float-vs-uint8 condition latent contrasts.
    for logical in sorted(set(source_records["native_bf16"]) & set(source_records["temporary_fp32"])):
        n = source_records["native_bf16"][logical]; f = source_records["temporary_fp32"][logical]
        if "decoded_final_float32" in n and "decoded_final_float32" in f:
            metric = difference_metrics(n["decoded_final_float32"], f["decoded_final_float32"]); space_rows.append({"space": "native_vs_fp32_rgb", "sample_id": logical, "precision": "paired", **metric})
        for precision, item in (("native_bf16", n), ("temporary_fp32", f)):
            if "direct_condition_latent_float32" in item and "uint8_condition_latent_float32" in item:
                metric = difference_metrics(item["direct_condition_latent_float32"], item["uint8_condition_latent_float32"]); space_rows.append({"space": "direct_vs_uint8_condition_latent", "sample_id": logical, "precision": precision, **metric})
    _close_memmaps(raw_records)
    _close_memmaps(records)
    _close_memmaps(source_records)
    return {"status": "COMPLETE", "metrics": rows, "space_metrics": space_rows, "tensors": decoder_tensors, "metrics_spaces": ["prediction_latent", "native_rgb", "fp32_rgb", "direct_float_condition_latent", "uint8_sim_condition_latent"], "decoder_calls": 16, "decoder_manifest_sha256": decoder_manifest_sha,
            "decoder_config_sha256": sha256_file(config_path),
            "reason": "8 selected latents decoded serially at native BF16 and temporary FP32"}


def analyze_decoder_replays(run_root: str | Path, decoder_root: str | Path | None = None, *, expected_raw_manifest_sha: str | None = None, expected_group: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Analyze decoder records and close all mmap handles on every exit path."""
    marker = len(_OPEN_MEMMAPS)
    try:
        return _analyze_decoder_replays(run_root, decoder_root=decoder_root,
                                        expected_raw_manifest_sha=expected_raw_manifest_sha,
                                        expected_group=expected_group)
    finally:
        _close_new_memmaps(marker)


def _verify_decoder_manifest_for_analysis(root: Path) -> str:
    path = root / "MANIFEST.sha256"
    if not path.is_file(): raise ValueError("decoder manifest is missing")
    seen = {}
    for line in path.read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        if (len(parts) != 2 or len(parts[0]) != 64 or any(char not in "0123456789abcdefABCDEF" for char in parts[0]) or
                not parts[1] or parts[1] in seen): raise ValueError("malformed decoder manifest")
        relative = Path(parts[1])
        if relative.is_absolute() or ".." in relative.parts: raise ValueError("unsafe decoder manifest path")
        target = root / relative
        if not target.is_file() or sha256_file(target) != parts[0]: raise ValueError(f"decoder artifact mismatch: {parts[1]}")
        seen[parts[1]] = parts[0]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.relative_to(root).as_posix() not in {"MANIFEST.sha256", ".decoder.lock"}}
    if set(seen) != actual: raise ValueError("decoder manifest inventory mismatch")
    return sha256_file(path)


def _png(path: Path, series: list[tuple[float, float]], title: str) -> None:
    """Small dependency-free PNG companion for every SVG plot."""
    try:
        from PIL import Image, ImageDraw
        image = Image.new("RGB", (900, 520), "white"); draw = ImageDraw.Draw(image)
        draw.text((30, 20), title, fill="black"); draw.line((80, 460, 850, 460), fill="black", width=2); draw.line((80, 60, 80, 460), fill="black", width=2)
        if series:
            xs = [x for x, _ in series]; ys = [y for _, y in series]; xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
            if xmax == xmin: xmax += 1
            if ymax == ymin: ymax += 1
            pts = [(80 + 770*(x-xmin)/(xmax-xmin), 460 - 400*(y-ymin)/(ymax-ymin)) for x, y in series]
            if len(pts) > 1: draw.line(pts, fill=(37,99,235), width=3)
            for x, y in pts: draw.ellipse((x-4,y-4,x+4,y+4), fill=(37,99,235))
        image.save(path)
    except Exception:
        raw = zlib.compress(b"\x00\xff\xff\xff")
        data = b"\x89PNG\r\n\x1a\n"
        def chunk(kind: bytes, value: bytes) -> bytes: return struct.pack(">I", len(value)) + kind + value + struct.pack(">I", zlib.crc32(kind + value) & 0xffffffff)
        data += chunk(b"IHDR", struct.pack(">IIBBBBB", 1,1,8,2,0,0,0)) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")
        path.write_bytes(data)


def _svg(path: Path, series: list[tuple[float, float]], title: str, y_label: str) -> None:
    pts = " ".join(f"{80 + 760*i/max(1,len(series)-1):.2f},{450 - 360*(y-min((v for _,v in series), default=0))/(max((v for _,v in series), default=1)-min((v for _,v in series), default=0) or 1):.2f}" for i, (_, y) in enumerate(series))
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520"><rect width="100%" height="100%" fill="white"/><text x="80" y="35" font-size="20">{title}</text><line x1="80" y1="450" x2="850" y2="450" stroke="black"/><line x1="80" y1="60" x2="80" y2="450" stroke="black"/><polyline fill="none" stroke="#2563eb" stroke-width="2" points="{pts}"/><text x="400" y="500">alpha</text><text x="15" y="260" transform="rotate(-90 15 260)">{y_label}</text></svg>', encoding="utf-8")


def _math_note() -> str:
    return r"""# Task 6 mathematical note

For state (k) and noise seed (s), the local expansion tested here is
\[
G_{k,s}(z_k+\delta)=G_{k,s}(z_k)+J_{k,s}(z_k)\delta+R_{k,s}(\delta).
\]
If \(G\) is sufficiently smooth, a centered difference has truncation error
\(O(h^2)\). Floating point evaluation adds a term of order \(O(u/h)\), so an
arbitrarily small step is not automatically more accurate. FP32 differences
are reduced in float64 and all quantitative spaces are kept separate.

Passing the empirical finite-direction gates is evidence about the sampled
state, directions, and scales only. It proves neither existence of a Jacobian
nor a Lipschitz upper bound, and it does not establish low rank. Decoder and
re-encoding results describe the composite (E\circ D\circ G), not (G) alone.
"""


def _report(result: Mapping[str, Any], decoder: Mapping[str, Any]) -> str:
    if result.get("status") != "COMPLETE":
        return f"# UMI Task 6 report\n\nStatus: `{result.get('status')}`. {result.get('reason','')}\n"
    fits = result.get("fits", []); add = result.get("additivity", []); pred = result.get("predictions", [])
    return "\n".join(["# UMI Task 6 cross-context local-linearity pilot", "", "## Result",
        f"- Predicted-latent direction gates: {sum(x.get('candidate_status') == 'PASS' for x in fits)}/{len(fits)}.",
        f"- Additivity: {sum(x.get('status') == 'PASS' for x in add)}/{len(add)} (exactly six planned checks).",
        f"- Held-out prediction: {sum(x.get('status') == 'PASS' for x in pred)}/{len(pred)} (exactly 24 planned checks).",
        f"- Maximum additivity relative error: {result.get('summary',{}).get('max_additivity_error')}; maximum holdout relative error: {result.get('summary',{}).get('max_holdout_error')}.",
        f"- Decoder status: {decoder.get('status')}; decoder calls: {decoder.get('decoder_calls', 0)}.", "", "## Scope and limitations",
        "The pilot is bridge_0/seed 0 only. Other states and seeds are `not_run`, not failures. MP4/PNG are visualization-only; quantitative tensors are float32 with float64 reductions.", "", "## Interpretation", "Empirical passage does not prove Jacobian existence, a Lipschitz bound, or low rank. See `math_note.md`.", ""]) 


def _write_manifest(root: Path) -> Path:
    entries = _manifest_entries(root, include_bundle=True)
    _atomic_text(root / "MANIFEST.sha256", "".join(f"{digest}  {name}\n" for name, digest in sorted(entries.items())))
    return root / "MANIFEST.sha256"


def _package_valid(root: Path, raw_manifest_sha: str, decoder_manifest_sha: str | None = None) -> bool:
    marker = root / "analysis_provenance.json"
    manifest = root / "MANIFEST.sha256"
    if not marker.is_file() or not manifest.is_file(): return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
        verify_analysis_manifest(root)
        expected_config = {"alphas": list(ALPHAS), "directions": list(DIRECTION_IDS), "spaces": ["prediction_latent", "native_rgb", "fp32_rgb", "float_roundtrip_condition_latent", "uint8_simulated_roundtrip_condition_latent"]}
        expected_config_sha = hashlib.sha256(json.dumps(expected_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        expected_identity = hashlib.sha256(json.dumps({"raw_manifest_sha256": raw_manifest_sha, "decoder_manifest_sha256": decoder_manifest_sha,
            "decoder_config_sha256": value.get("decoder_config_sha256"), "analysis_code_sha256": _analysis_source_sha(),
            "config_sha256": expected_config_sha, "group": value.get("group", {})}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return (value.get("raw_manifest_sha256") == raw_manifest_sha and value.get("decoder_manifest_sha256") == decoder_manifest_sha
                and value.get("analysis_code_sha256") == _analysis_source_sha() and value.get("config_sha256") == expected_config_sha
                and value.get("analysis_identity") == expected_identity)
    except Exception: return False


def verify_analysis_manifest(root: str | Path) -> dict[str, Any]:
    root = Path(root); manifest = root / "MANIFEST.sha256"
    if not manifest.is_file(): raise ValueError("analysis MANIFEST.sha256 is missing")
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or parts[1] in entries: raise ValueError("malformed analysis manifest")
        relative = Path(parts[1])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in {"MANIFEST.sha256", ".runner.lock"}: raise ValueError("unsafe analysis manifest entry")
        target = root / relative
        if not target.is_file() or sha256_file(target) != parts[0]: raise ValueError(f"analysis artifact mismatch: {parts[1]}")
        entries[parts[1]] = parts[0]
    if entries != _manifest_entries(root, include_bundle=True): raise ValueError("analysis manifest inventory mismatch")
    return {"sha256": sha256_file(manifest), "entries": entries}


def _decoder_export_metadata(decoder: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep decoder status/provenance JSON small; arrays remain .npy artifacts."""
    if not isinstance(decoder, Mapping):
        return {"status": "NOT_RUN"}
    metadata: dict[str, Any] = {}
    for key, value in decoder.items():
        if key == "tensors" or isinstance(value, np.ndarray):
            continue
        metadata[key] = value
    return metadata


def write_task6_artifacts(result: Mapping[str, Any], output_dir: str | Path, *, raw_root: str | Path | None = None,
                          decoder: Mapping[str, Any] | None = None, source_dir: str | Path | None = None) -> dict[str, Any]:
    output = Path(output_dir)
    raw = Path(raw_root) if raw_root is not None else None
    if raw is not None:
        try: output.resolve().relative_to(raw.resolve())
        except ValueError: pass
        else: raise ValueError("analysis output must be outside immutable raw run")
    raw_snapshot = verify_raw_manifest(raw) if raw is not None else {"sha256": None, "entries": {}}
    decoder_manifest_sha = (decoder or {}).get("decoder_manifest_sha256") if decoder else None
    if output.exists() and _package_valid(output, raw_snapshot["sha256"], decoder_manifest_sha):
        return {"output_dir": str(output), "manifest": str(output / "MANIFEST.sha256"), "idempotent": True}
    if output.exists() and any(output.iterdir()): raise FileExistsError("analysis output exists but is stale or tampered")
    parent = output.parent; parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".task6-analysis-", dir=str(parent)))
    try:
        (stage / "figures").mkdir(); (stage / "analysis_tensors").mkdir()
        tensors = result.get("tensors", {})
        for name, value in tensors.items(): np.save(stage / "analysis_tensors" / Path(name).name, np.asarray(value, dtype=np.float32), allow_pickle=False)
        for name, value in (decoder or {}).get("tensors", {}).items(): np.save(stage / "analysis_tensors" / Path(name).name, np.asarray(value, dtype=np.float32), allow_pickle=False)
        point_rows = result.get("points", []); floors = {row.get("quantity"): row.get("baseline_floor_rms", 0.0) for row in result.get("baseline_floors", [])}
        failures = []
        for row in point_rows:
            response = row.get("predicted_latent_response_rms", row.get("response_rms")); floor = floors.get("predicted_latent", 0.0) or 0.0
            bad = response is None or (response <= 10.0 * floor if floor else response == 0.0)
            failures.append({"direction_id": row.get("direction_id"), "alpha": row.get("alpha"), "sign": row.get("sign"), "status": "FAIL" if bad else "PASS", "response_rms": response, "baseline_floor_rms": floor, "failure_reason": "response_floor" if bad else ""})
        _rows_csv(stage / "point_metrics.csv", point_rows); _rows_csv(stage / "difference_metrics.csv", result.get("derivatives", [])); _rows_csv(stage / "fit_metrics.csv", result.get("fits", [])); _rows_csv(stage / "window_decisions.csv", result.get("fits", []) + result.get("image_fits", [])); _rows_csv(stage / "additivity_metrics.csv", result.get("additivity", [])); _rows_csv(stage / "prediction_metrics.csv", result.get("predictions", [])); _rows_csv(stage / "failure_amplitudes.csv", failures); _rows_csv(stage / "decoder_metrics.csv", (decoder or {}).get("metrics", [])); _rows_csv(stage / "decoder_roundtrip_metrics.csv", (decoder or {}).get("metrics", [])); _rows_csv(stage / "decoder_space_metrics.csv", (decoder or {}).get("space_metrics", [])); _rows_csv(stage / "roundtrip_metrics.csv", [row for row in (decoder or {}).get("space_metrics", []) if str(row.get("space", "")) in {"direct_float_condition_latent", "uint8_sim_condition_latent"}]); _rows_csv(stage / "pairwise_precision_metrics.csv", [row for row in (decoder or {}).get("space_metrics", []) if "vs_" in str(row.get("space", ""))]); _rows_csv(stage / "cross_group_status.csv", result.get("cross_group_status", summarize_cross_groups({})))
        decoder_metadata = _decoder_export_metadata(decoder)
        task6_summary = {key: value for key, value in result.items() if key not in {"tensors", "decoder"}}
        task6_summary["decoder"] = decoder_metadata
        _json(stage / "task6_summary.json", task6_summary); _json(stage / "decoder_summary.json", decoder_metadata)
        source_hash = _analysis_source_sha(); config_value = {"alphas": list(ALPHAS), "directions": list(DIRECTION_IDS), "spaces": ["prediction_latent", "native_rgb", "fp32_rgb", "float_roundtrip_condition_latent", "uint8_simulated_roundtrip_condition_latent"]}; config_sha = hashlib.sha256(json.dumps(config_value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        decoder_config_sha = (decoder or {}).get("decoder_config_sha256") if decoder else None
        analysis_identity = hashlib.sha256(json.dumps({"raw_manifest_sha256": raw_snapshot["sha256"], "decoder_manifest_sha256": decoder_manifest_sha, "decoder_config_sha256": decoder_config_sha, "analysis_code_sha256": source_hash, "config_sha256": config_sha, "group": result.get("group", {})}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        provenance = {"schema_version": "umi-task6-analysis-v1", "raw_manifest_sha256": raw_snapshot["sha256"], "decoder_manifest_sha256": decoder_manifest_sha, "decoder_config_sha256": decoder_config_sha, "analysis_code_sha256": source_hash, "config_sha256": config_sha, "analysis_identity": analysis_identity, "group": result.get("group", {}), "raw_root": str(raw) if raw else None}
        _json(stage / "analysis_provenance.json", provenance); _json(stage / "provenance.json", provenance); _json(stage / "config.json", config_value)
        _atomic_text(stage / "math_note.md", _math_note()); _atomic_text(stage / "experiment_report.md", _report(result, decoder or {"status": "NOT_RUN"}))
        resource_lines = ["# Resource report", ""]
        if raw is not None:
            status_path = raw / "run_status.json"
            if status_path.is_file():
                raw_status = json.loads(status_path.read_text(encoding="utf-8")); resource_lines += [f"- terminal status: `{raw_status.get('status')}`", f"- completed samples: {len(raw_status.get('completed_samples', []))}"]
            for filename in ("gpu_samples.csv", "ram_samples.csv", "disk_samples.csv"):
                path = raw / filename
                if path.is_file():
                    with path.open(encoding="utf-8", newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    numeric: dict[str, list[float]] = {}
                    for item in rows:
                        for key, value in item.items():
                            try: numeric.setdefault(key, []).append(float(value))
                            except (TypeError, ValueError): pass
                    summary = "; ".join(f"{key}=[{min(values):.3g},{max(values):.3g}]" for key, values in sorted(numeric.items()) if values and key not in {"timestamp", "time"})
                    resource_lines.append(f"- {filename}: {len(rows)} samples" + (f"; {summary}" if summary else ""))
        if len(resource_lines) == 2: resource_lines.append("- resource CSVs were not available")
        _atomic_text(stage / "resource_report.md", "\n".join(resource_lines) + "\n")
        # A compact response chart; all raw tensors remain outside the zip.
        values = [(float(row.get("alpha", i)), float(row.get("predicted_latent_response_rms", row.get("response_rms", 0.0)) or 0.0)) for i, row in enumerate(result.get("points", [])) if row.get("predicted_latent_response_rms", row.get("response_rms")) is not None]
        charts = [("response", "Task 6 response", "RMS response"), ("additivity", "Task 6 additivity", "relative error"), ("holdout", "Task 6 holdout prediction", "relative error"), ("decoder_precision", "Decoder native vs FP32", "RMS"), ("roundtrip_quantization", "Float vs uint8 round trip", "RMS")]
        for stem, title, ylabel in charts:
            if stem == "response": chart_values = values
            elif stem == "additivity": chart_values = [(float(i), float(row["relative_error"])) for i, row in enumerate(result.get("additivity", [])) if row.get("relative_error") is not None]
            elif stem == "holdout": chart_values = [(float(i), float(row["relative_error"])) for i, row in enumerate(result.get("predictions", [])) if row.get("relative_error") is not None]
            elif stem == "decoder_precision": chart_values = [(float(i), float(row["rms"])) for i, row in enumerate((decoder or {}).get("space_metrics", [])) if row.get("space") == "native_vs_fp32_rgb" and row.get("rms") is not None]
            else: chart_values = [(float(i), float(row["rms"])) for i, row in enumerate((decoder or {}).get("space_metrics", [])) if row.get("space") == "direct_vs_uint8_condition_latent" and row.get("rms") is not None]
            if not chart_values: chart_values = [(0.0, 0.0)]
            _svg(stage / "figures" / f"{stem}.svg", chart_values, title, ylabel); _png(stage / "figures" / f"{stem}.png", chart_values, title)
        if source_dir is not None:
            source = Path(source_dir)
            for name in ("umi_task6_decoder.py", "umi_task5_decoder.py", "analyze_umi_task6.py", "umi_task6_runtime.py", "run_umi_task6_experiment.py", "umi_task6_primitives.py", "umi_task5_primitives.py", "umi_task5_runtime.py", "umi_precision_runtime.py", "umi_precision_official.py", "umi_precision_storage.py"):
                if (source / name).is_file(): shutil.copyfile(source / name, stage / name)
        bundle = stage / "review_bundle.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file() and path != bundle and "analysis_tensors" not in path.parts:
                    info = zipfile.ZipInfo(path.relative_to(stage).as_posix(), (1980,1,1,0,0,0)); info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, path.read_bytes())
            readme = zipfile.ZipInfo("README.txt", (1980, 1, 1, 0, 0, 0)); readme.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(readme, "Large raw tensors remain in the immutable run directory.\n")
        _write_manifest(stage)
        if raw is not None and verify_raw_manifest(raw)["sha256"] != raw_snapshot["sha256"]: raise ValueError("raw evidence changed during analysis")
        if output.exists(): raise FileExistsError("analysis output appeared during publication")
        os.replace(stage, output); stage = None
    finally:
        if stage is not None: shutil.rmtree(stage, ignore_errors=True)
    return {"output_dir": str(output), "manifest": str(output / "MANIFEST.sha256"), "review_bundle_bytes": (output / "review_bundle.zip").stat().st_size, "idempotent": False}


def analyze_task6_run(run_dir: str | Path, output_dir: str | Path | None = None, *,
                      decoder_root: str | Path | None = None,
                      output_root: str | Path | None = None) -> dict[str, Any]:
    if output_root is not None:
        if output_dir is not None and Path(output_dir).resolve() != Path(output_root).resolve():
            raise ValueError("output_dir and output_root disagree")
        output_dir = output_root
    root = Path(run_dir).resolve(); status_path = root / "run_status.json"
    if not status_path.is_file(): raise ValueError("Task 6 run_status.json is missing")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") not in {"AWAITING_REVIEW", "COMPLETE"} or len(status.get("completed_samples", [])) != 32:
        raise ValueError("public Task 6 analysis rejects stopped or partial raw runs")
    raw_manifest = verify_raw_manifest(root)
    marker = len(_OPEN_MEMMAPS)
    records: dict[str, dict[str, Any]] = {}
    try:
        records = _load_records(root)
        result = analyze_task6_records(records, group=status.get("group"), plan_detail=status.get("plan_detail"))
        decoder = analyze_decoder_replays(root, decoder_root=decoder_root,
                                          expected_raw_manifest_sha=raw_manifest["sha256"], expected_group=status.get("group"))
        if result.get("status") != "COMPLETE" or len(result.get("fits", [])) != 5 or len(result.get("additivity", [])) != 6 or len(result.get("predictions", [])) != 24:
            raise ValueError("public Task 6 analysis requires exact 5 direction, 6 additivity, and 24 holdout results")
        if decoder.get("status") != "COMPLETE" or int(decoder.get("decoder_calls", 0)) != 16:
            raise ValueError("public Task 6 analysis requires a complete bound decoder replay")
        destination = Path(output_dir) if output_dir is not None else root.parent / (root.name + "_analysis")
        published = write_task6_artifacts(result, destination, raw_root=root, decoder=decoder, source_dir=Path(__file__).parent)
        if verify_raw_manifest(root)["sha256"] != raw_manifest["sha256"]: raise ValueError("raw evidence changed during analysis")
        return {"status": result.get("status"), "decoder_status": decoder.get("status"), **published}
    finally:
        _close_new_memmaps(marker)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("run_dir"); parser.add_argument("--output-dir")
    args = parser.parse_args(argv); print(json.dumps(analyze_task6_run(args.run_dir, args.output_dir), ensure_ascii=False, sort_keys=True)); return 0


__all__ = ["analyze_decoder_replays", "analyze_task6_records", "analyze_task6_run", "difference_metrics", "rms64", "verify_analysis_manifest", "verify_raw_manifest", "write_task6_artifacts"]

if __name__ == "__main__": raise SystemExit(main())
