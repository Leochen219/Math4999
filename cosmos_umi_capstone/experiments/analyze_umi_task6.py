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
    from .analyze_umi_task5 import analyze_task5_records
    from .umi_task5_primitives import ALPHAS, DIRECTION_IDS
    from .umi_task6_primitives import STATE_IDS, SEEDS
    from .umi_task6_decoder import sha256_file
except ImportError:  # pragma: no cover
    from analyze_umi_task5 import analyze_task5_records
    from umi_task5_primitives import ALPHAS, DIRECTION_IDS
    from umi_task6_primitives import STATE_IDS, SEEDS
    from umi_task6_decoder import sha256_file


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
        record = dict(original)
        if "predicted_latent" not in record and "output_full" in record:
            record["predicted_latent"] = record["output_full"]
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
    c01 = rms64((directions["v0"].astype(np.float32) + directions["v1"].astype(np.float32))[mask])
    c12 = rms64((directions["v1"].astype(np.float32) + directions["v2"].astype(np.float32))[mask])
    for combo, left, right, coefficient in (("u01", "v0", "v1", c01), ("u12", "v1", "v2", c12)):
        expected = np.divide(np.add(directions[left], directions[right], dtype=np.float32), np.float32(coefficient), dtype=np.float32)
        if not np.array_equal(expected[~mask], directions[combo][~mask]) or not np.allclose(expected[mask], directions[combo][mask], rtol=0.0, atol=2e-6):
            raise ValueError(f"saved combination direction {combo} is inconsistent with v directions")
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


def analyze_task6_records(records: Mapping[str, Mapping[str, Any]], *, plan_detail: Mapping[str, Any] | None = None,
                          group: Mapping[str, Any] | None = None, strict: bool = False) -> dict[str, Any]:
    """Recompute the unchanged Task 5 metrics from one Task 6 group.

    Missing, stopped, or nonfinite evidence is a truthful incomplete result by
    default.  ``strict=True`` is useful for callers that want an exception.
    """
    normalized = _normalize_records(records)
    missing = sorted(_required_ids() - set(normalized))
    if missing:
        if strict: raise ValueError("incomplete Task 6 group: " + ", ".join(missing))
        return _status_incomplete(missing)
    derived = _derive_plan_detail(normalized)
    detail = dict(plan_detail or {})
    if not detail: detail = derived
    else:
        if "combination_coefficients" not in detail: raise ValueError("Task 6 scientific plan detail is incomplete")
        detail.setdefault("s_z", derived["s_z"])
        if not math.isclose(float(detail["s_z"]), float(derived["s_z"]), rel_tol=0.0, abs_tol=1e-7): raise ValueError("s_z differs from saved tensors")
        for key in ("c01", "c12"):
            if not math.isclose(float(detail["combination_coefficients"][key]), float(derived["combination_coefficients"][key]), rel_tol=0.0, abs_tol=1e-7): raise ValueError(f"{key} differs from saved directions")
    try:
        result = analyze_task5_records(normalized, plan_detail=detail)
    except (ValueError, FloatingPointError) as error:
        if strict: raise
        return _status_incomplete([], reason=f"invalid raw evidence: {type(error).__name__}: {error}")
    result = dict(result)
    result.update({"status": "COMPLETE", "group": dict(group or {}), "raw_sample_count": len(normalized)})
    result["summary"] = {**dict(result.get("summary", {})), "additivity_total": 6, "prediction_total": 24,
                          "max_additivity_error": max((row.get("relative_error") or 0.0 for row in result.get("additivity", [])), default=None),
                          "max_holdout_error": max((row.get("relative_error") or 0.0 for row in result.get("predictions", [])), default=None)}
    group_id = f"{result.get('group', {}).get('state', '')}__seed_{result.get('group', {}).get('seed', '')}" if result.get("group") else ""
    result["cross_group_status"] = summarize_cross_groups({group_id: result}) if group_id != "__seed_" else summarize_cross_groups({})
    return _recompute_fp32_core(normalized, result, detail)


def _manifest_entries(root: Path, *, exclude: set[str] | None = None, include_bundle: bool = False) -> dict[str, str]:
    excluded = set(exclude or ()) | {"MANIFEST.sha256"}
    if not include_bundle: excluded.add("review_bundle.zip")
    return {path.relative_to(root).as_posix(): sha256_file(path)
            for path in sorted(root.rglob("*")) if path.is_file() and path.name not in excluded}


def _analysis_source_sha() -> str:
    digest = hashlib.sha256(); source_root = Path(__file__).parent
    for name in ("umi_task6_decoder.py", "analyze_umi_task6.py", "umi_task6_runtime.py", "run_umi_task6_experiment.py", "umi_task6_primitives.py", "umi_task5_primitives.py", "umi_task5_runtime.py", "umi_precision_runtime.py", "umi_precision_official.py", "umi_precision_storage.py"):
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
        target = root / rel
        if not target.is_file() or sha256_file(target) != parts[0]: raise ValueError(f"raw manifest mismatch: {parts[1]}")
        entries[parts[1]] = parts[0]
    actual = _manifest_entries(root)
    if entries != actual: raise ValueError("raw manifest inventory mismatch")
    return {"sha256": sha256_file(path), "entries": entries}


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
            record[array_path.stem] = np.load(array_path, allow_pickle=False)
        records[sample.name] = record
    return records


def analyze_decoder_replays(run_root: str | Path, decoder_root: str | Path | None = None, *, expected_raw_manifest_sha: str | None = None, expected_group: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read decoder replay records without touching the model."""
    raw = Path(run_root).resolve()
    root = Path(decoder_root).resolve() if decoder_root is not None else raw.parent / (raw.name + "_decoder")
    status_path = root / "status.json"
    if not status_path.is_file(): return {"status": "NOT_RUN", "metrics": [], "reason": "decoder status is absent"}
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE": return {"status": "NOT_RUN", "metrics": [], "reason": "decoder stopped or incomplete", "status_record": status}
    try: decoder_manifest_sha = _verify_decoder_manifest_for_analysis(root)
    except ValueError as error: return {"status": "INVALID", "metrics": [], "reason": str(error)}
    config_path = root / "decoder_config.json"
    if not config_path.is_file(): return {"status": "INVALID", "metrics": [], "reason": "decoder config is missing"}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if expected_raw_manifest_sha is not None and config.get("raw_manifest_sha256") != expected_raw_manifest_sha:
        return {"status": "INVALID", "metrics": [], "reason": "decoder is bound to a different raw manifest"}
    if expected_group is not None and config.get("raw_group") != dict(expected_group):
        return {"status": "INVALID", "metrics": [], "reason": "decoder group identity mismatch"}
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not (path / "record.json").is_file(): continue
        record = json.loads((path / "record.json").read_text(encoding="utf-8"))
        for array_path in path.glob("*.npy"): record[array_path.stem] = np.load(array_path, allow_pickle=False)
        records[path.name] = record
    if len(records) != 16: return {"status": "INCOMPLETE", "metrics": [], "reason": f"expected 16 decoder records, found {len(records)}"}
    rows = []; space_rows: list[dict[str, Any]] = []
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
        source_records[str(spec.get("decode_precision"))][logical] = {**meta, **{p.stem: np.load(p, allow_pickle=False) for p in path.glob("*.npy")}}
    spaces = {"native_bf16": "native_rgb", "temporary_fp32": "fp32_rgb"}
    for precision, rgb_space in spaces.items():
        values = {rgb_space: "decoded_final_float32", "direct_float_condition_latent": "direct_condition_latent_float32", "uint8_sim_condition_latent": "uint8_condition_latent_float32"}
        for space, key in values.items():
            source = source_records[precision]
            if any(item not in source or key not in source[item] for item in ("baseline_pre", "baseline_post")): continue
            base = source["baseline_pre"][key]; post = source["baseline_post"][key]; floor = difference_metrics(post, base)
            alpha_rows = []
            for ordinal, alpha in enumerate(ALPHAS):
                plus = source.get(f"v0_alpha_{ordinal:02d}_plus", {}).get(key); minus = source.get(f"v0_alpha_{ordinal:02d}_minus", {}).get(key)
                if plus is None or minus is None: continue
                p = difference_metrics(plus, base); m = difference_metrics(minus, base); central = difference_metrics(plus, minus)
                central_step = float(2.0 * alpha); d = None if central["rms"] is None else central["rms"] / central_step
                alpha_rows.append({"space": space, "precision": precision, "direction_id": "v0", "alpha": float(alpha), "plus_response_rms": p["rms"], "minus_response_rms": m["rms"], "baseline_floor_rms": floor["rms"], "central_difference_rms": d, "status": "N/A" if p["rms"] is None or m["rms"] is None else "OK"})
            space_rows.extend(alpha_rows)
    # Native-vs-FP32 RGB and direct-float-vs-uint8 condition latent contrasts.
    for logical in sorted(set(source_records["native_bf16"]) & set(source_records["temporary_fp32"])):
        n = source_records["native_bf16"][logical]; f = source_records["temporary_fp32"][logical]
        if "decoded_final_float32" in n and "decoded_final_float32" in f:
            metric = difference_metrics(n["decoded_final_float32"], f["decoded_final_float32"]); space_rows.append({"space": "native_vs_fp32_rgb", "sample_id": logical, **metric})
        if "direct_condition_latent_float32" in n and "uint8_condition_latent_float32" in n:
            metric = difference_metrics(n["direct_condition_latent_float32"], n["uint8_condition_latent_float32"]); space_rows.append({"space": "direct_vs_uint8_condition_latent", "sample_id": logical, **metric})
    return {"status": "COMPLETE", "metrics": rows, "space_metrics": space_rows, "decoder_calls": 16, "decoder_manifest_sha256": decoder_manifest_sha,
            "reason": "8 selected latents decoded serially at native BF16 and temporary FP32"}


def _verify_decoder_manifest_for_analysis(root: Path) -> str:
    path = root / "MANIFEST.sha256"
    if not path.is_file(): raise ValueError("decoder manifest is missing")
    seen = {}
    for line in path.read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or parts[1] in seen: raise ValueError("malformed decoder manifest")
        target = root / parts[1]
        if not target.is_file() or sha256_file(target) != parts[0]: raise ValueError(f"decoder artifact mismatch: {parts[1]}")
        seen[parts[1]] = parts[0]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name not in {"MANIFEST.sha256", ".decoder.lock"}}
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
[
G_{k,s}(z_k+delta)=G_{k,s}(z_k)+J_{k,s}(z_k)delta+R_{k,s}(delta).
]
If (G) is sufficiently smooth, a centered difference has truncation error
(O(h^2)). Floating point evaluation adds a term of order (O(u/h)), so an
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
        return (value.get("raw_manifest_sha256") == raw_manifest_sha and value.get("decoder_manifest_sha256") == decoder_manifest_sha
                and value.get("analysis_code_sha256") == _analysis_source_sha() and value.get("config_sha256") == expected_config_sha)
    except Exception: return False


def verify_analysis_manifest(root: str | Path) -> dict[str, Any]:
    root = Path(root); manifest = root / "MANIFEST.sha256"
    if not manifest.is_file(): raise ValueError("analysis MANIFEST.sha256 is missing")
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or parts[1] in entries: raise ValueError("malformed analysis manifest")
        relative = Path(parts[1])
        if relative.is_absolute() or ".." in relative.parts or relative.name == "MANIFEST.sha256": raise ValueError("unsafe analysis manifest entry")
        target = root / relative
        if not target.is_file() or sha256_file(target) != parts[0]: raise ValueError(f"analysis artifact mismatch: {parts[1]}")
        entries[parts[1]] = parts[0]
    if entries != _manifest_entries(root, include_bundle=True): raise ValueError("analysis manifest inventory mismatch")
    return {"sha256": sha256_file(manifest), "entries": entries}


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
        point_rows = result.get("points", []); floors = {row.get("quantity"): row.get("baseline_floor_rms", 0.0) for row in result.get("baseline_floors", [])}
        failures = []
        for row in point_rows:
            response = row.get("predicted_latent_response_rms", row.get("response_rms")); floor = floors.get("predicted_latent", 0.0) or 0.0
            bad = response is None or (response <= 10.0 * floor if floor else response == 0.0)
            failures.append({"direction_id": row.get("direction_id"), "alpha": row.get("alpha"), "sign": row.get("sign"), "status": "FAIL" if bad else "PASS", "response_rms": response, "baseline_floor_rms": floor, "failure_reason": "response_floor" if bad else ""})
        _rows_csv(stage / "point_metrics.csv", point_rows); _rows_csv(stage / "difference_metrics.csv", result.get("derivatives", [])); _rows_csv(stage / "fit_metrics.csv", result.get("fits", [])); _rows_csv(stage / "window_decisions.csv", result.get("fits", []) + result.get("image_fits", [])); _rows_csv(stage / "additivity_metrics.csv", result.get("additivity", [])); _rows_csv(stage / "prediction_metrics.csv", result.get("predictions", [])); _rows_csv(stage / "failure_amplitudes.csv", failures); _rows_csv(stage / "decoder_metrics.csv", (decoder or {}).get("metrics", [])); _rows_csv(stage / "decoder_roundtrip_metrics.csv", (decoder or {}).get("metrics", [])); _rows_csv(stage / "decoder_space_metrics.csv", (decoder or {}).get("space_metrics", [])); _rows_csv(stage / "roundtrip_metrics.csv", [row for row in (decoder or {}).get("space_metrics", []) if "roundtrip" in str(row.get("space", ""))]); _rows_csv(stage / "pairwise_precision_metrics.csv", [row for row in (decoder or {}).get("space_metrics", []) if "vs_" in str(row.get("space", ""))]); _rows_csv(stage / "cross_group_status.csv", result.get("cross_group_status", summarize_cross_groups({})))
        _json(stage / "task6_summary.json", {key: value for key, value in result.items() if key != "tensors"}); _json(stage / "decoder_summary.json", decoder or {"status": "NOT_RUN"})
        source_hash = _analysis_source_sha(); config_value = {"alphas": list(ALPHAS), "directions": list(DIRECTION_IDS), "spaces": ["prediction_latent", "native_rgb", "fp32_rgb", "float_roundtrip_condition_latent", "uint8_simulated_roundtrip_condition_latent"]}; config_sha = hashlib.sha256(json.dumps(config_value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        provenance = {"schema_version": "umi-task6-analysis-v1", "raw_manifest_sha256": raw_snapshot["sha256"], "decoder_manifest_sha256": decoder_manifest_sha, "analysis_code_sha256": source_hash, "config_sha256": config_sha, "raw_root": str(raw) if raw else None}
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
            chart_values = values if stem == "response" else [(float(i), float(row.get("relative_error", row.get("rms", 0.0)) or 0.0)) for i, row in enumerate((result.get("additivity", []) if stem == "additivity" else result.get("predictions", []) if stem == "holdout" else (decoder or {}).get("space_metrics", [])))][:64]
            _svg(stage / "figures" / f"{stem}.svg", chart_values, title, ylabel); _png(stage / "figures" / f"{stem}.png", chart_values, title)
        if source_dir is not None:
            source = Path(source_dir)
            for name in ("umi_task6_decoder.py", "analyze_umi_task6.py", "umi_task6_runtime.py", "run_umi_task6_experiment.py", "umi_task6_primitives.py", "umi_task5_primitives.py", "umi_task5_runtime.py", "umi_precision_runtime.py", "umi_precision_official.py", "umi_precision_storage.py"):
                if (source / name).is_file(): shutil.copyfile(source / name, stage / name)
        bundle = stage / "review_bundle.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file() and path != bundle and "analysis_tensors" not in path.parts:
                    info = zipfile.ZipInfo(path.relative_to(stage).as_posix(), (1980,1,1,0,0,0)); info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, path.read_bytes())
            archive.writestr("README.txt", "Large raw tensors remain in the immutable run directory.\n")
        _write_manifest(stage)
        if raw is not None and verify_raw_manifest(raw)["sha256"] != raw_snapshot["sha256"]: raise ValueError("raw evidence changed during analysis")
        if output.exists(): raise FileExistsError("analysis output appeared during publication")
        os.replace(stage, output); stage = None
    finally:
        if stage is not None: shutil.rmtree(stage, ignore_errors=True)
    return {"output_dir": str(output), "manifest": str(output / "MANIFEST.sha256"), "review_bundle_bytes": (output / "review_bundle.zip").stat().st_size, "idempotent": False}


def analyze_task6_run(run_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(run_dir).resolve(); status_path = root / "run_status.json"
    if not status_path.is_file(): raise ValueError("Task 6 run_status.json is missing")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") not in {"AWAITING_REVIEW", "COMPLETE"} or len(status.get("completed_samples", [])) != 32:
        raise ValueError("public Task 6 analysis rejects stopped or partial raw runs")
    raw_manifest = verify_raw_manifest(root)
    records = _load_records(root)
    result = analyze_task6_records(records, group=status.get("group"), plan_detail=status.get("plan_detail"))
    decoder = analyze_decoder_replays(root, expected_raw_manifest_sha=raw_manifest["sha256"], expected_group=status.get("group"))
    if result.get("status") != "COMPLETE" or len(result.get("fits", [])) != 5 or len(result.get("additivity", [])) != 6 or len(result.get("predictions", [])) != 24:
        raise ValueError("public Task 6 analysis requires exact 5 direction, 6 additivity, and 24 holdout results")
    if decoder.get("status") != "COMPLETE" or int(decoder.get("decoder_calls", 0)) != 16:
        raise ValueError("public Task 6 analysis requires a complete bound decoder replay")
    destination = Path(output_dir) if output_dir is not None else root.parent / (root.name + "_analysis")
    published = write_task6_artifacts(result, destination, raw_root=root, decoder=decoder, source_dir=Path(__file__).parent)
    if verify_raw_manifest(root)["sha256"] != raw_manifest["sha256"]: raise ValueError("raw evidence changed during analysis")
    return {"status": result.get("status"), "decoder_status": decoder.get("status"), **published}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("run_dir"); parser.add_argument("--output-dir")
    args = parser.parse_args(argv); print(json.dumps(analyze_task6_run(args.run_dir, args.output_dir), ensure_ascii=False, sort_keys=True)); return 0


__all__ = ["analyze_decoder_replays", "analyze_task6_records", "analyze_task6_run", "difference_metrics", "rms64", "verify_analysis_manifest", "verify_raw_manifest", "write_task6_artifacts"]

if __name__ == "__main__": raise SystemExit(main())
