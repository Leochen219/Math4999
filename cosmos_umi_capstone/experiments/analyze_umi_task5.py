"""Offline FP32/FP64 analysis and review artifacts for the bounded Task 5 run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

try:
    from .umi_fd_post_vae_scan import sha256_file
    from .umi_task5_primitives import ALPHAS, DIRECTION_IDS, cosine64, pair_additivity_metrics, rms64
except ImportError:
    from umi_fd_post_vae_scan import sha256_file
    from umi_task5_primitives import ALPHAS, DIRECTION_IDS, cosine64, pair_additivity_metrics, rms64


def _array(value: Any, name: str) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    if not result.size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite and nonempty")
    return result


def _diff(left: Any, right: Any) -> np.ndarray:
    return (np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)).astype(np.float32)


def _fit_loglog(steps: list[float], responses: list[float]) -> dict[str, float | None]:
    x = np.asarray(steps, dtype=np.float64)
    y = np.asarray(responses, dtype=np.float64)
    if len(x) < 2 or np.any(x <= 0.0) or np.any(y <= 0.0) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return {"slope": None, "intercept": None, "r2": None}
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    predicted = slope * np.log(x) + intercept
    total = float(np.sum((np.log(y) - np.mean(np.log(y))) ** 2))
    residual = float(np.sum((np.log(y) - predicted) ** 2))
    return {"slope": float(slope), "intercept": float(intercept), "r2": 1.0 if total == 0.0 and residual == 0.0 else (None if total == 0 else float(1.0 - residual / total))}


def _pass_threshold(value: float | None, *, low: float | None = None, high: float | None = None) -> bool:
    if value is None or not math.isfinite(value):
        return False
    return (low is None or value >= low) and (high is None or value <= high)


def _sample_id(direction_id: str, ordinal: int, sign: int) -> str:
    return f"{direction_id}_alpha_{ordinal:02d}_{'plus' if sign == 1 else 'minus'}"


def _output(record: Mapping[str, Any], quantity: str) -> np.ndarray:
    if quantity not in record:
        raise ValueError(f"sample lacks required {quantity}")
    return _array(record[quantity], quantity)


def _rows_to_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key.startswith("_") or isinstance(value, (dict, list, tuple, np.ndarray)):
                continue
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys or ["status"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in (keys or ["status"])})


def _safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _candidate_for_direction(direction_id: str, entries: list[dict[str, Any]], *, floor: float, quantity: str) -> dict[str, Any]:
    plus_steps = [entry["h_plus_actual"] for entry in entries]
    minus_steps = [entry["h_minus_actual"] for entry in entries]
    plus_response = [entry[f"plus_{quantity}_rms"] for entry in entries]
    minus_response = [entry[f"minus_{quantity}_rms"] for entry in entries]
    plus_fit = _fit_loglog(plus_steps, plus_response)
    minus_fit = _fit_loglog(minus_steps, minus_response)
    reasons: list[str] = []
    for entry in entries:
        if not entry["input_nonzero"]:
            reasons.append(f"alpha={entry['alpha']}: effective input is zero")
        if not _pass_threshold(entry["plus_input_cosine"], low=0.99):
            reasons.append(f"alpha={entry['alpha']}: plus input cosine below 0.99")
        if not _pass_threshold(entry["minus_input_cosine"], low=0.99):
            reasons.append(f"alpha={entry['alpha']}: minus input cosine below 0.99")
        opposite = entry["plus_minus_input_cosine"]
        if opposite is None or opposite > -0.99:
            reasons.append(f"alpha={entry['alpha']}: plus/minus input is not opposite (cosine > -0.99)")
        for side in ("plus", "minus"):
            response = entry[f"{side}_{quantity}_rms"]
            if floor == 0.0:
                if response == 0.0:
                    reasons.append(f"alpha={entry['alpha']}: {side} response is zero while baseline floor is zero")
            elif response <= 10.0 * floor:
                reasons.append(f"alpha={entry['alpha']}: {side} response is not above 10x baseline floor")
    for side, fit in (("plus", plus_fit), ("minus", minus_fit)):
        if not _pass_threshold(fit["slope"], low=0.8, high=1.2):
            reasons.append(f"{side} log-log slope outside [0.8,1.2]")
        if not _pass_threshold(fit["r2"], low=0.98):
            reasons.append(f"{side} log-log R2 below 0.98")
    for entry in entries[:-1]:
        if not _pass_threshold(entry[f"next_{quantity}_dactual_cosine"], low=0.95):
            reasons.append(f"alpha={entry['alpha']}: paired secant cosine below 0.95")
        relative = entry[f"next_{quantity}_dactual_relative_change"]
        if relative is None or relative > 0.25:
            reasons.append(f"alpha={entry['alpha']}: paired secant relative change above 0.25")
    return {"direction_id": direction_id, "quantity": quantity, "plus_slope": plus_fit["slope"], "plus_r2": plus_fit["r2"],
            "minus_slope": minus_fit["slope"], "minus_r2": minus_fit["r2"], "baseline_floor_rms": floor,
            "candidate_status": "PASS" if not reasons else "FAIL", "failure_reasons": " | ".join(reasons) if reasons else "",
            "window_alphas": ",".join(str(entry["alpha"]) for entry in entries)}


def analyze_task5_records(records: Mapping[str, Mapping[str, Any]], *, plan_detail: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every Task 5 metric from immutable saved tensors.

    All tensor subtraction starts from float32 data and is reduced in float64;
    the returned residual tensors retain float32 storage for review/reuse.
    """
    required = {"baseline_pre", "baseline_post"}
    for direction_id in DIRECTION_IDS:
        for ordinal in range(len(ALPHAS)):
            required.update({_sample_id(direction_id, ordinal, 1), _sample_id(direction_id, ordinal, -1)})
    missing = sorted(required.difference(records))
    if missing:
        raise ValueError(f"Task 5 record set is incomplete: {missing}")
    coefficients = plan_detail.get("combination_coefficients", {})
    c01, c12 = float(coefficients["c01"]), float(coefficients["c12"])
    y0 = _output(records["baseline_pre"], "predicted_latent")
    ypost = _output(records["baseline_post"], "predicted_latent")
    rgb0 = _output(records["baseline_pre"], "decoded_final")
    rgbpost = _output(records["baseline_post"], "decoded_final")
    floors = {"quantity": "predicted_latent", "baseline_pre_post_rms": rms64(_diff(ypost, y0)),
              "baseline_pre_post_max_abs": float(np.max(np.abs(_diff(ypost, y0)), initial=0.0))}, {
              "quantity": "decoded_final_rgb", "baseline_pre_post_rms": rms64(_diff(rgbpost, rgb0)),
              "baseline_pre_post_max_abs": float(np.max(np.abs(_diff(rgbpost, rgb0)), initial=0.0))}
    floor_map = {row["quantity"]: row["baseline_pre_post_rms"] for row in floors}
    point_rows: list[dict[str, Any]] = []
    derivative_rows: list[dict[str, Any]] = []
    tensors: dict[str, np.ndarray] = {}
    entries_by_direction: dict[str, list[dict[str, Any]]] = {}
    dtarget: dict[tuple[str, float], np.ndarray] = {}
    dactual: dict[tuple[str, float], np.ndarray] = {}
    for direction_id in DIRECTION_IDS:
        entries: list[dict[str, Any]] = []
        for ordinal, alpha in enumerate(ALPHAS):
            plus = records[_sample_id(direction_id, ordinal, 1)]
            minus = records[_sample_id(direction_id, ordinal, -1)]
            plus_delta = _array(plus["actual_delta_fp32"], "plus actual delta")
            minus_delta = _array(minus["actual_delta_fp32"], "minus actual delta")
            target_plus = _array(plus["target_delta_fp32"], "plus target delta")
            target_minus = _array(minus["target_delta_fp32"], "minus target delta")
            mask = np.asarray(plus["mask"], dtype=bool)
            direction = _array(plus["direction"], "frozen direction")
            if not (mask.shape == direction.shape == plus_delta.shape == minus_delta.shape):
                raise ValueError("Task 5 input geometry shape mismatch")
            if np.any(plus_delta[~mask]) or np.any(minus_delta[~mask]):
                raise ValueError("Task 5 actual condition delta changed outside mask")
            if not np.array_equal(mask, np.asarray(minus["mask"], dtype=bool)):
                raise ValueError("Task 5 plus/minus masks differ")
            hp, hm = rms64(plus_delta[mask]), rms64(minus_delta[mask])
            ht_plus, ht_minus = rms64(target_plus[mask]), rms64(target_minus[mask])
            h_target = (ht_plus + ht_minus) / 2.0
            if h_target == 0.0:
                raise ValueError("perturbation target step is zero")
            yplus, yminus = _output(plus, "predicted_latent"), _output(minus, "predicted_latent")
            rplus, rminus = _output(plus, "decoded_final"), _output(minus, "decoded_final")
            central = _diff(yplus, yminus)
            d_t = (central.astype(np.float64) / (2.0 * h_target)).astype(np.float32)
            d_a = None if hp + hm == 0.0 else (central.astype(np.float64) / (hp + hm)).astype(np.float32)
            if d_a is None:
                raise ValueError("Task 5 effective central-difference step is zero")
            dtarget[(direction_id, alpha)] = d_t
            dactual[(direction_id, alpha)] = d_a
            tensors[f"dtarget_{direction_id}_{ordinal:02d}.npy"] = d_t
            tensors[f"dactual_{direction_id}_{ordinal:02d}.npy"] = d_a
            entry = {"direction_id": direction_id, "alpha": float(alpha), "h_target": h_target,
                     "h_plus_actual": hp, "h_minus_actual": hm, "target_plus_rms": ht_plus, "target_minus_rms": ht_minus,
                     "plus_input_cosine": cosine64(plus_delta[mask], direction[mask]),
                     "minus_input_cosine": cosine64(minus_delta[mask], -direction[mask]),
                     "plus_minus_input_cosine": cosine64(plus_delta[mask], minus_delta[mask]),
                     "plus_outside_mask_exact": bool(np.all(plus_delta[~mask] == 0.0)),
                     "minus_outside_mask_exact": bool(np.all(minus_delta[~mask] == 0.0)), "input_nonzero": hp > 0.0 and hm > 0.0,
                     "plus_predicted_latent_rms": rms64(_diff(yplus, y0)), "minus_predicted_latent_rms": rms64(_diff(yminus, y0)),
                     "plus_decoded_final_rgb_rms": rms64(_diff(rplus, rgb0)), "minus_decoded_final_rgb_rms": rms64(_diff(rminus, rgb0)),
                     "even_predicted_latent_rms": rms64(_diff(_diff(yplus, y0), _diff(y0, yminus))),
                     "even_decoded_final_rgb_rms": rms64(_diff(_diff(rplus, rgb0), _diff(rgb0, rminus))),
                     "dtarget_predicted_latent_rms": rms64(d_t), "dactual_predicted_latent_rms": rms64(d_a)}
            entries.append(entry)
            for sign, record, actual, target in ((1, plus, plus_delta, target_plus), (-1, minus, minus_delta, target_minus)):
                point_rows.append({"sample_id": _sample_id(direction_id, ordinal, sign), "direction_id": direction_id, "alpha": float(alpha),
                                   "sign": sign, "target_delta_rms": rms64(target[mask]), "actual_delta_rms": rms64(actual[mask]),
                                   "actual_over_target": None if rms64(target[mask]) == 0 else rms64(actual[mask]) / rms64(target[mask]),
                                   "target_direction_cosine": cosine64(actual[mask], sign * direction[mask]),
                                   "outside_mask_exact": bool(np.all(actual[~mask] == 0)),
                                   "predicted_latent_response_rms": rms64(_diff(_output(record, "predicted_latent"), y0)),
                                   "decoded_final_rgb_response_rms": rms64(_diff(_output(record, "decoded_final"), rgb0))})
        for index in range(len(entries) - 1):
            next_entry = entries[index + 1]
            current, following = entries[index], next_entry
            current_d, following_d = dactual[(direction_id, current["alpha"])], dactual[(direction_id, following["alpha"])]
            current["next_predicted_latent_dactual_cosine"] = cosine64(current_d, following_d)
            denominator = rms64(current_d)
            current["next_predicted_latent_dactual_relative_change"] = None if denominator == 0.0 else rms64(_diff(following_d, current_d)) / denominator
            # RGB secants use the same actual denominator but retain their own output tensor.
            rplus, rminus = _output(records[_sample_id(direction_id, index, 1)], "decoded_final"), _output(records[_sample_id(direction_id, index, -1)], "decoded_final")
            nplus, nminus = _output(records[_sample_id(direction_id, index + 1, 1)], "decoded_final"), _output(records[_sample_id(direction_id, index + 1, -1)], "decoded_final")
            rgb_d = (_diff(rplus, rminus).astype(np.float64) / (current["h_plus_actual"] + current["h_minus_actual"])).astype(np.float32)
            rgb_next = (_diff(nplus, nminus).astype(np.float64) / (following["h_plus_actual"] + following["h_minus_actual"])).astype(np.float32)
            current["next_decoded_final_rgb_dactual_cosine"] = cosine64(rgb_d, rgb_next)
            rgb_denominator = rms64(rgb_d)
            current["next_decoded_final_rgb_dactual_relative_change"] = None if rgb_denominator == 0.0 else rms64(_diff(rgb_next, rgb_d)) / rgb_denominator
        for final in entries[-1:]:
            for quantity in ("predicted_latent", "decoded_final_rgb"):
                final[f"next_{quantity}_dactual_cosine"] = None
                final[f"next_{quantity}_dactual_relative_change"] = None
        entries_by_direction[direction_id] = entries
        derivative_rows.extend(entries)
    fits = [_candidate_for_direction(direction_id, values, floor=floor_map["predicted_latent"], quantity="predicted_latent")
            for direction_id, values in entries_by_direction.items()]
    image_fits = [_candidate_for_direction(direction_id, values, floor=floor_map["decoded_final_rgb"], quantity="decoded_final_rgb")
                  for direction_id, values in entries_by_direction.items()]

    additivity: list[dict[str, Any]] = []
    for label, combo, left, right, coefficient in (("01", "u01", "v0", "v1", c01), ("12", "u12", "v1", "v2", c12)):
        for ordinal, alpha in enumerate(ALPHAS):
            metric = pair_additivity_metrics(dtarget[(combo, alpha)], dtarget[(left, alpha)], dtarget[(right, alpha)], coefficient)
            tensors[f"additivity_{label}_{ordinal:02d}.npy"] = metric.pop("residual")
            plus_combo = _array(records[_sample_id(combo, ordinal, 1)]["actual_delta_fp32"], "combo actual delta")
            plus_left = _array(records[_sample_id(left, ordinal, 1)]["actual_delta_fp32"], "left actual delta")
            plus_right = _array(records[_sample_id(right, ordinal, 1)]["actual_delta_fp32"], "right actual delta")
            mask = np.asarray(records[_sample_id(combo, ordinal, 1)]["mask"], dtype=bool)
            input_residual = coefficient * plus_combo - plus_left - plus_right
            input_denominator = rms64(plus_left[mask]) + rms64(plus_right[mask])
            row = {"pair": label, "alpha": float(alpha), "coefficient": coefficient,
                   "input_combo_residual_rms": rms64(input_residual[mask]),
                   "input_combo_relative_error": None if input_denominator == 0 else rms64(input_residual[mask]) / input_denominator,
                   "status": "N/A" if metric["relative_error"] is None else ("PASS" if metric["relative_error"] <= .10 else "FAIL"), **metric}
            additivity.append(row)
            tensors[f"input_combo_residual_{label}_{ordinal:02d}.npy"] = input_residual.astype(np.float32)

    predictions: list[dict[str, Any]] = []
    g = {direction_id: dtarget[(direction_id, ALPHAS[0])] for direction_id in ("v0", "v1", "v2")}
    g.update({"u01": ((g["v0"].astype(np.float64) + g["v1"].astype(np.float64)) / c01).astype(np.float32),
              "u12": ((g["v1"].astype(np.float64) + g["v2"].astype(np.float64)) / c12).astype(np.float32)})
    for direction_id in DIRECTION_IDS:
        ordinals = (1, 2) if direction_id in ("v0", "v1", "v2") else (0, 1, 2)
        for ordinal in ordinals:
            alpha = ALPHAS[ordinal]
            h = entries_by_direction[direction_id][ordinal]["h_target"]
            for sign in (1, -1):
                actual = _output(records[_sample_id(direction_id, ordinal, sign)], "predicted_latent")
                prediction = (y0.astype(np.float64) + sign * h * g[direction_id].astype(np.float64)).astype(np.float32)
                residual = _diff(actual, prediction)
                response = _diff(actual, y0)
                response_rms = rms64(response)
                relative = None if response_rms == 0.0 else rms64(residual) / response_rms
                key = f"prediction_residual_{direction_id}_{ordinal:02d}_{'p' if sign == 1 else 'm'}.npy"
                tensors[key] = residual
                predictions.append({"direction_id": direction_id, "alpha": float(alpha), "sign": sign,
                                    "estimate_alpha": ALPHAS[0], "response_rms": response_rms, "absolute_rms": rms64(residual),
                                    "relative_error": relative, "status": "N/A" if relative is None else ("PASS" if relative <= .10 else "FAIL"),
                                    "residual_tensor": key})
    return {"status": "COMPLETE", "baseline_floors": list(floors), "points": point_rows, "derivatives": derivative_rows,
            "fits": fits, "image_fits": image_fits, "additivity": additivity, "predictions": predictions,
            "decoder": {"status": "NOT_RUN", "metrics": [], "reason": "decoder replay has not been loaded"}, "tensors": tensors, "summary": {"direction_candidates_pass": sum(row["candidate_status"] == "PASS" for row in fits),
                                                 "direction_candidates_total": len(fits),
                                                 "additivity_pass": sum(row["status"] == "PASS" for row in additivity if row["status"] != "N/A"),
                                                 "prediction_pass": sum(row["status"] == "PASS" for row in predictions if row["status"] != "N/A")}}


def _simple_svg(title: str, series: Mapping[str, list[tuple[float, float | None]]], y_label: str, *, threshold: float | None = None) -> str:
    width, height, left, top, right, bottom = 920, 540, 90, 65, 30, 90
    points = [(float(x), float(y)) for values in series.values() for x, y in values if y is not None and math.isfinite(float(y))]
    xs, ys = [item[0] for item in points] or [0., 1.], [item[1] for item in points] or [0., 1.]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    if xmin == xmax: xmax = xmin + 1
    if ymin == ymax: ymax = ymin + 1
    def px(x): return left + (width-left-right)*(x-xmin)/(xmax-xmin)
    def py(y): return top + (height-top-bottom)*(1-(y-ymin)/(ymax-ymin))
    colors = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c"]
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>', f'<text x="{left}" y="32" font-size="20" font-family="Arial">{title}</text>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
            f'<text x="{width/2}" y="{height-28}" text-anchor="middle" font-family="Arial">alpha</text>',
            f'<text x="22" y="{height/2}" transform="rotate(-90 22 {height/2})" text-anchor="middle" font-family="Arial">{y_label}</text>']
    if threshold is not None and ymin <= threshold <= ymax:
        body.append(f'<line x1="{left}" y1="{py(threshold):.2f}" x2="{width-right}" y2="{py(threshold):.2f}" stroke="#777" stroke-dasharray="7,4"/><text x="{width-right}" y="{py(threshold)-5:.2f}" text-anchor="end" font-size="12">threshold {threshold:.2g}</text>')
    for index, (label, values) in enumerate(series.items()):
        color = colors[index % len(colors)]
        clean = [(x, y) for x, y in values if y is not None and math.isfinite(float(y))]
        if clean:
            body.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>' % (color, " ".join(f"{px(x):.2f},{py(float(y)):.2f}" for x, y in clean)))
            body.append(f'<text x="{left+index*150}" y="{height-55}" fill="{color}" font-family="Arial" font-size="13">{label}</text>')
    body.append("</svg>")
    return "".join(body)


def _render_png_chart(path: Path, title: str, series: Mapping[str, list[tuple[float, float | None]]], y_label: str,
                      *, threshold: float | None = None) -> None:
    """Render the same numeric series as the SVG; PNG is not a placeholder."""
    from PIL import Image, ImageDraw
    width, height, left, top, right, bottom = 920, 540, 90, 65, 30, 90
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    points = [(float(x), float(y)) for values in series.values() for x, y in values if y is not None and math.isfinite(float(y))]
    xs, ys = [item[0] for item in points] or [0., 1.], [item[1] for item in points] or [0., 1.]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    if xmin == xmax: xmax = xmin + 1
    if ymin == ymax: ymax = ymin + 1
    def px(x): return left + (width-left-right)*(x-xmin)/(xmax-xmin)
    def py(y): return top + (height-top-bottom)*(1-(y-ymin)/(ymax-ymin))
    draw.text((left, 30), title, fill="black")
    draw.line((left, height-bottom, width-right, height-bottom), fill="black", width=2)
    draw.line((left, top, left, height-bottom), fill="black", width=2)
    draw.text((width//2 - 20, height-35), "alpha", fill="black")
    draw.text((8, height//2), y_label, fill="black")
    for tick in np.linspace(ymin, ymax, 5):
        y = py(float(tick)); draw.line((left-4, y, left, y), fill="black")
        draw.text((4, y-6), f"{tick:.2g}", fill="black")
    if threshold is not None and ymin <= threshold <= ymax:
        y = py(threshold); draw.line((left, y, width-right, y), fill="#777777", width=1)
        draw.text((width-right-115, y-14), f"threshold {threshold:.2g}", fill="#555555")
    colors = [(37,99,235), (220,38,38), (5,150,105), (147,51,234), (234,88,12)]
    for index, (label, values) in enumerate(series.items()):
        clean = [(x, float(y)) for x, y in values if y is not None and math.isfinite(float(y))]
        color = colors[index % len(colors)]
        if len(clean) >= 2: draw.line([(px(x), py(y)) for x, y in clean], fill=color, width=3)
        for x, y in clean: draw.ellipse((px(x)-3, py(y)-3, px(x)+3, py(y)+3), fill=color)
        draw.text((left + index*150, height-60), label, fill=color)
    image.save(path)


def _write_manifest(root: Path, name: str = "MANIFEST.sha256") -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != name:
            entries.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / name).write_text("\n".join(entries) + ("\n" if entries else ""), encoding="ascii")


def _report(result: Mapping[str, Any], *, task4_preflight: Mapping[str, Any] | None) -> str:
    fits = result["fits"]
    add = result["additivity"]
    prediction = result["predictions"]
    lines = ["# UMI Task 5：方向一致性、可加性与留出预测", "",
             "## 结果摘要", "",
             f"- 预测 latent 局部候选：{sum(x['candidate_status']=='PASS' for x in fits)}/{len(fits)} 个方向通过固定三点门。",
             f"- 可加性：{sum(x['status']=='PASS' for x in add if x['status']!='N/A')}/{len(add)} 个有效方向-幅度对满足 E_add ≤ 0.10。",
             f"- 留出预测：{sum(x['status']=='PASS' for x in prediction if x['status']!='N/A')}/{len(prediction)} 个有效留出点满足 E_pred ≤ 0.10。",
             "- 所有主量均为采样结束、decode 前的预测 latent；MP4/PNG 未参与任何定量归约。", "",
             "## 判读边界", "", "本轮只测一个基准点、一个动作 chunk、固定噪声和五个预先冻结方向。组合方向位于三个原方向张成的子空间，不能据此作 Jacobian 低秩或跨场景泛化结论。"]
    decoder = result.get("decoder", {})
    lines += ["", "## Decoder 对照", "", f"- decoder 状态：{decoder.get('status', 'N/A')}。{decoder.get('reason', '')}"]
    if task4_preflight:
        lines += ["", "## Task 4 核验摘要", "", f"- {task4_preflight.get('summary', '见 task4_preflight.json')}" ]
    return "\n".join(lines) + "\n"


def _math_note() -> str:
    return r"""# Task 5 数学说明

在相关局部区域内，若生成映射 \(G\) 二阶可微且二阶导数有界，则对单位 mask-RMS 方向 \(v\)：

\[G(\\bar z+h v)=G(\\bar z)+hJv+O(h^2).\]

若同一个线性映射 \(J\) 适用于所测方向，且 \(u_{01}=(v_0+v_1)/c_{01}\)，则
\[J u_{01}=(Jv_0+Jv_1)/c_{01}.\]
这给出以 \(c_{01}d(u_{01})-d(v_0)-d(v_1)\) 测量的可加性残差，以及只用小幅度原方向导数预测更大幅度与组合方向的理由。它们比单方向 RMS 斜率更强，因为同时检验输出的方向与幅度关系。

中心差分具有 \(O(h^2)\) 导数截断误差需要更强的三阶光滑性及正负有效步长对称假设；本实验同时保存 `d_target` 和按照实际消费步长得到的 `d_actual`，不把步长修正视为方向失真已经消失。浮点舍入、BF16/FP32 compute、采样器与 decode 都会带来额外有限精度项，FP32 不是数学真值。

所测是有限方向与有限尺度的局部一致性，不是完整 Jacobian，也不支持 SVD 或低秩结论。单 chunk latent 结果亦不能替代含 decode—选帧—再编码的长期闭环稳定性分析。
"""


def write_task5_artifacts(result: Mapping[str, Any], output_dir: str | Path, *, raw_root: str | Path | None = None,
                          task4_preflight: Mapping[str, Any] | None = None, source_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError("Task 5 analysis destination already exists")
    root.mkdir(parents=True)
    (root / "analysis_tensors").mkdir()
    (root / "figures").mkdir()
    for name, value in result["tensors"].items():
        np.save(root / "analysis_tensors" / name, value, allow_pickle=False)
    _rows_to_csv(root / "point_metrics.csv", result["points"])
    _rows_to_csv(root / "difference_metrics.csv", result["derivatives"])
    _rows_to_csv(root / "fit_metrics.csv", result["fits"])
    _rows_to_csv(root / "image_fit_metrics.csv", result["image_fits"])
    _rows_to_csv(root / "window_decisions.csv", result["fits"] + result["image_fits"])
    _rows_to_csv(root / "additivity_metrics.csv", result["additivity"])
    _rows_to_csv(root / "prediction_metrics.csv", result["predictions"])
    _rows_to_csv(root / "baseline_floors.csv", result["baseline_floors"])
    _rows_to_csv(root / "decoder_metrics.csv", list(result.get("decoder", {}).get("metrics", [])))
    (root / "task5_summary.json").write_text(json.dumps(_safe({key: value for key, value in result.items() if key != "tensors"}), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (root / "math_note.md").write_text(_math_note(), encoding="utf-8")
    (root / "experiment_report.md").write_text(_report(result, task4_preflight=task4_preflight), encoding="utf-8")
    (root / "implementation_notes.md").write_text("Task 5 reuses Task 4's official C-path observation seam; analysis subtracts float32 tensors and reduces in float64.  See task5_plan.json and raw samples for exact per-call evidence.\n", encoding="utf-8")
    if task4_preflight is not None:
        (root / "task4_preflight.json").write_text(json.dumps(_safe(task4_preflight), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    response = {direction: [(row["alpha"], (row["plus_predicted_latent_rms"] + row["minus_predicted_latent_rms"]) / 2.0)
                            for row in result["derivatives"] if row["direction_id"] == direction] for direction in DIRECTION_IDS}
    add = {f"E_add{row['pair']}": [(item["alpha"], item["relative_error"]) for item in result["additivity"] if item["pair"] == row["pair"]]
           for row in result["additivity"] if row["pair"] in ("01", "12")}
    pred = {"E_pred": [(row["alpha"], row["relative_error"]) for row in result["predictions"]]}
    charts = [("direction_response", "Five frozen directions: output response", response, "RMS predicted-latent response", None),
              ("additivity", "Combination additivity", add, "E_add", .10),
              ("heldout_prediction", "Held-out amplitude/direction prediction", pred, "E_pred", .10)]
    decoder_rows = list(result.get("decoder", {}).get("metrics", []))
    if decoder_rows:
        decoder_series = {path: [(row["alpha"], row["response_rms"]) for row in decoder_rows if row.get("precision_path") == path]
                          for path in ("native", "fp32")}
        charts.append(("decoder_response", "Decoder replay response (v0)", decoder_series, "RMS RGB response", None))
    for stem, title, series, ylabel, threshold in charts:
        (root / "figures" / f"{stem}.svg").write_text(_simple_svg(title, series, ylabel, threshold=threshold), encoding="utf-8")
        _render_png_chart(root / "figures" / f"{stem}.png", title, series, ylabel, threshold=threshold)
    # The compact review archive intentionally leaves complete tensor sets on the server.
    bundle = root / "review_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path != bundle and "analysis_tensors" not in path.parts:
                archive.write(path, path.relative_to(root).as_posix())
        if raw_root is not None:
            raw = Path(raw_root)
            selected = ["baseline_pre", "baseline_post", "v0_alpha_00_plus", "v0_alpha_02_minus", "v1_alpha_00_plus", "v1_alpha_02_minus", "v2_alpha_00_plus", "v2_alpha_02_minus", "u01_alpha_00_plus", "u01_alpha_02_minus", "u12_alpha_00_plus", "u12_alpha_02_minus"]
            for sample_id in selected:
                sample = raw / "samples" / sample_id
                for filename in ("output_full.npy", "predicted_latent.npy", "actual_delta_fp32.npy", "sample.json"):
                    path = sample / filename
                    if path.is_file():
                        archive.write(path, f"representative_samples/{sample_id}/{filename}")
        if source_dir is not None:
            for name in ("run_umi_task5_experiment.py", "umi_task5_runtime.py", "umi_task5_primitives.py", "analyze_umi_task5.py", "umi_precision_official.py", "umi_precision_runtime.py"):
                path = Path(source_dir) / name
                if path.is_file():
                    archive.write(path, f"source/{name}")
        archive.writestr("README.txt", "The full raw tensor set remains in the run directory; this bundle contains representative review tensors only.\n")
    _write_manifest(root)
    return {"output_dir": str(root), "review_bundle_bytes": bundle.stat().st_size, "manifest": str(root / "MANIFEST.sha256")}


def _load_records(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for sample in sorted((root / "samples").iterdir()):
        if not sample.is_dir() or ".attempt." in sample.name:
            continue
        status = json.loads((sample / "status.json").read_text(encoding="utf-8"))
        if status.get("status") != "success":
            raise ValueError(f"non-success Task 5 sample: {sample.name}")
        for filename, digest in status.get("artifact_sha256", {}).items():
            if sha256_file(sample / filename) != digest:
                raise ValueError(f"Task 5 raw hash mismatch: {sample.name}/{filename}")
        record = json.loads((sample / "sample.json").read_text(encoding="utf-8"))
        for array_path in sample.glob("*.npy"):
            record[array_path.stem] = np.load(array_path, allow_pickle=False)
        records[sample.name] = record
    return records


def _load_sample_tree(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for sample in sorted(root.iterdir()):
        if not sample.is_dir() or ".attempt." in sample.name:
            continue
        status = json.loads((sample / "status.json").read_text(encoding="utf-8"))
        if status.get("status") != "success":
            raise ValueError(f"non-success decoder sample: {sample}")
        for filename, digest in status.get("artifact_sha256", {}).items():
            if sha256_file(sample / filename) != digest:
                raise ValueError(f"decoder sample hash mismatch: {sample.name}/{filename}")
        record = json.loads((sample / "sample.json").read_text(encoding="utf-8"))
        for array_path in sample.glob("*.npy"):
            record[array_path.stem] = np.load(array_path, allow_pickle=False)
        records[sample.name] = record
    return records


def analyze_decoder_replays(run_root: Path) -> dict[str, Any]:
    decoder_root = run_root / "decoder"
    if not (decoder_root / "status.json").is_file():
        return {"status": "NOT_RUN", "metrics": [], "reason": "decoder/status.json is absent"}
    status = json.loads((decoder_root / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        return {"status": "BLOCKED", "metrics": [], "reason": json.dumps(status, ensure_ascii=False)}
    paths = {precision: _load_sample_tree(decoder_root / precision) for precision in ("native", "fp32")}
    metrics: list[dict[str, Any]] = []
    for precision, rows in paths.items():
        baseline = _output(rows["baseline_pre"], "decoded_final")
        post = _output(rows["baseline_post"], "decoded_final")
        floor = rms64(_diff(post, baseline))
        responses: list[float] = []
        for ordinal, alpha in enumerate(ALPHAS):
            plus, minus = rows[_sample_id("v0", ordinal, 1)], rows[_sample_id("v0", ordinal, -1)]
            plus_out, minus_out = _output(plus, "decoded_final"), _output(minus, "decoded_final")
            response = (rms64(_diff(plus_out, baseline)) + rms64(_diff(minus_out, baseline))) / 2.0
            responses.append(response)
            counterpart = paths["fp32" if precision == "native" else "native"][_sample_id("v0", ordinal, 1)]
            metrics.append({"precision_path": precision, "direction_id": "v0", "alpha": float(alpha), "baseline_floor_rms": floor,
                            "plus_response_rms": rms64(_diff(plus_out, baseline)), "minus_response_rms": rms64(_diff(minus_out, baseline)),
                            "response_rms": response, "even_residual_rms": rms64(_diff(_diff(plus_out, baseline), _diff(baseline, minus_out))),
                            "native_vs_fp32_plus_rms": rms64(_diff(plus_out, _output(counterpart, "decoded_final"))),
                            "native_original_task5_rgb_exact": plus.get("original_task5_rgb_exact") if precision == "native" else None,
                            "decoder_consumed_scaled_dtype": plus.get("decoder_consumed_scaled_dtype"),
                            "decoder_weight_dtype": plus.get("decoder_weight_dtype")})
        fit = _fit_loglog(list(ALPHAS), responses)
        for row in metrics:
            if row["precision_path"] == precision:
                row.update({"response_slope": fit["slope"], "response_r2": fit["r2"]})
    return {"status": "COMPLETE", "metrics": metrics, "reason": "8 native plus 8 FP32 real decoder replays"}


def analyze_task5_run(run_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete" or int(status.get("formal_successful", 0)) != 32:
        raise ValueError("Task 5 raw run is not a complete 32-call formal experiment")
    plan = json.loads((root / "task5_plan.json").read_text(encoding="utf-8"))
    records = _load_records(root)
    result = analyze_task5_records(records, plan_detail=plan["plan_detail"])
    result["decoder"] = analyze_decoder_replays(root)
    preflight_path = root / "task4_preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.is_file() else None
    destination = Path(output_dir) if output_dir else root / "task5_analysis"
    published = write_task5_artifacts(result, destination, raw_root=root, task4_preflight=preflight, source_dir=Path(__file__).parent)
    return {**result["summary"], **published, "status": "COMPLETE"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    print(json.dumps(analyze_task5_run(args.run_dir, args.output_dir), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
