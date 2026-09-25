"""Preregistered CPU contracts for six real Bridge trajectories.

The live orchestrator imports these functions; this module itself never imports
Torch or the Cosmos framework.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


LOCKED_RECORDS = (1, 2, 3, 6, 7, 14)
SEED_PAIRS = ((0, 1), (2, 3))
PREFLIGHT_RANKS = (0, 1, 2, 3, 4, 7)


def locked_preflight_ranks() -> tuple[int, ...]:
    """Ranks of the six preselected records among eligible episodes."""
    return PREFLIGHT_RANKS


def forecast_disk_after_calls(free_gib: float, measured_sample_bytes: int, remaining: int) -> float:
    if not np.isfinite(free_gib) or measured_sample_bytes <= 0 or remaining < 0:
        raise ValueError("invalid disk forecast input")
    forecast = float(free_gib) - 1.3 * measured_sample_bytes * remaining / 2**30
    if forecast < 5.0:
        raise ValueError("measured sample forecast breaches the 5 GiB reserve")
    return forecast


def select_locked_records(inventory: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    by_index = {int(row["record_index"]): row for row in inventory}
    selected = []
    for index in LOCKED_RECORDS:
        row = by_index.get(index)
        if row is None:
            raise ValueError(f"locked record {index} is missing")
        if (int(row.get("frames", 0)) < 33 or not row.get("has_language")
                or not row.get("primary_jpeg_nonempty") or not row.get("sequence_flags_aligned")):
            raise ValueError(f"locked record {index} fails two-chunk eligibility")
        selected.append(row)
    return tuple(selected)


def build_call_plan(first_seed: int, second_seed: int) -> tuple[dict[str, Any], ...]:
    if first_seed < 0 or second_seed < 0 or first_seed == second_seed:
        raise ValueError("seeds must be distinct nonnegative integers")
    return (
        {"sample_id": "G0", "call": "G0", "condition_source": "real_x0", "action_source": "a0", "chunk_index": 0, "seed": first_seed},
        {"sample_id": "G0_repeat", "call": "G0_repeat", "condition_source": "real_x0", "action_source": "a0", "chunk_index": 0, "seed": first_seed},
        {"sample_id": "TF2", "call": "TF2", "condition_source": "real_x16", "action_source": "a1", "chunk_index": 1, "seed": second_seed},
        {"sample_id": "AR2", "call": "AR2", "condition_source": "g0_float_last_fp32", "action_source": "a1", "chunk_index": 1, "seed": second_seed},
    )


def _same_array(left: Any, right: Any) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


def verify_call_records(records: Mapping[str, Mapping[str, Any]]) -> None:
    """Hard engineering gate over the values actually saved after inference."""
    if set(records) != {"G0", "G0_repeat", "TF2", "AR2"}:
        raise ValueError("exactly four call records are required")
    first, repeat = records["G0"], records["G0_repeat"]
    for key in ("condition_input_fp32", "action", "output_full", "decoded_rgb_full", "encoded_condition"):
        if not _same_array(first[key], repeat[key]):
            raise ValueError(f"G0 repeat differs in {key}")
    if first["prediction_noise_hash"] != repeat["prediction_noise_hash"]:
        raise ValueError("G0 repeat noise differs")
    tf, ar = records["TF2"], records["AR2"]
    if not _same_array(tf["action"], ar["action"]):
        raise ValueError("TF2/AR2 action differs")
    if tf["prediction_noise_hash"] != ar["prediction_noise_hash"]:
        raise ValueError("TF2/AR2 noise differs")
    if not _same_array(first["encoded_condition"], ar["condition_input_fp32"]):
        raise ValueError("AR2 feedback condition differs from G0 FP32 encode")


def _rgb_last(value: Any) -> np.ndarray:
    frames = np.asarray(value)
    if frames.dtype != np.float32 or frames.ndim != 4 or frames.shape[0] != 3 or frames.shape[1] != 16:
        raise ValueError("predicted RGB must be float32 [3,16,H,W]")
    if not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
        raise ValueError("predicted RGB must be finite in [0,1]")
    return frames[:, -1]


def compare_endpoint_rgb(tf_frames: Any, ar_frames: Any, truth_x32: Any) -> dict[str, float]:
    """Compute the preregistered endpoint, never whole-chunk, RGB metric."""
    tf, ar, truth = _rgb_last(tf_frames), _rgb_last(ar_frames), np.asarray(truth_x32)
    if truth.dtype != np.float32 or truth.shape != tf.shape or not np.isfinite(truth).all():
        raise ValueError("truth x32 must match float32 endpoint RGB")
    if truth.min() < 0 or truth.max() > 1:
        raise ValueError("truth RGB must be in [0,1]")
    tf_diff = tf.astype(np.float64) - truth.astype(np.float64)
    ar_diff = ar.astype(np.float64) - truth.astype(np.float64)
    tf_rmse = float(np.sqrt(np.mean(tf_diff * tf_diff)))
    ar_rmse = float(np.sqrt(np.mean(ar_diff * ar_diff)))
    return {"tf_rmse": tf_rmse, "ar_rmse": ar_rmse,
            "delta_feedback_rmse": ar_rmse - tf_rmse,
            "tf_mae": float(np.mean(np.abs(tf_diff))), "ar_mae": float(np.mean(np.abs(ar_diff))),
            "tf_psnr": float("inf") if tf_rmse == 0 else float(-20 * np.log10(tf_rmse)),
            "ar_psnr": float("inf") if ar_rmse == 0 else float(-20 * np.log10(ar_rmse))}


def aggregate_episodes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        index = int(row["record_index"])
        pair = tuple(row["seed_pair"])
        if index not in LOCKED_RECORDS or pair not in SEED_PAIRS:
            raise ValueError("unregistered episode or seed pair")
        groups[index].append(row)
    if set(groups) != set(LOCKED_RECORDS) or any(len(rows) != 2 or
            {tuple(row["seed_pair"]) for row in rows} != set(SEED_PAIRS) for rows in groups.values()):
        raise ValueError("all six episodes require both independent seed pairs")
    episodes = []
    for index in LOCKED_RECORDS:
        values = [float(row["delta_feedback_rmse"]) for row in groups[index]]
        if not all(np.isfinite(value) for value in values):
            raise ValueError("feedback deltas must be finite")
        episodes.append({"record_index": index, "mean_delta_feedback_rmse": float(np.mean(values)),
                         "seed_pair_deltas": values})
    episode_values = np.array([row["mean_delta_feedback_rmse"] for row in episodes], dtype=np.float64)
    return {"episode_count": 6, "seed_pair_count": 12, "episodes": episodes,
            "positive_episode_count": int(np.count_nonzero(episode_values > 0)),
            "negative_episode_count": int(np.count_nonzero(episode_values < 0)),
            "zero_episode_count": int(np.count_nonzero(episode_values == 0)),
            "median_episode_delta_feedback_rmse": float(np.median(episode_values)),
            "min_episode_delta_feedback_rmse": float(episode_values.min()),
            "max_episode_delta_feedback_rmse": float(episode_values.max())}


def _latent_metrics(predicted: Any, truth: Any, mask: Any) -> tuple[float, float | None]:
    estimate, target = np.asarray(predicted), np.asarray(truth)
    selected = np.asarray(mask, dtype=bool)
    if (estimate.dtype != np.float32 or target.dtype != np.float32 or estimate.shape != target.shape
            or selected.shape != estimate.shape or not np.any(selected)):
        raise ValueError("condition latent/mask interface differs")
    x, y = estimate[selected].astype(np.float64), target[selected].astype(np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("condition latent must be finite")
    rms = float(np.sqrt(np.mean((x - y) ** 2)))
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    cosine = None if denominator == 0 else float(np.dot(x, y) / denominator)
    return rms, cosine


def analyze_stratum_records(records: Mapping[str, Mapping[str, Any]], truth_rgb: Any,
                            gt16: Any, gt32: Any, condition_mask: Any, *,
                            record_index: int, seed_pair: tuple[int, int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return one paired endpoint row and aligned second-chunk frame rows."""
    verify_call_records(records)
    truth = np.asarray(truth_rgb)
    if truth.dtype != np.float32 or truth.shape[0:2] != (33, 3) or not np.isfinite(truth).all():
        raise ValueError("truth RGB must be float32 [33,3,H,W]")
    g0, tf, ar = (np.asarray(records[call]["generated_rgb"]) for call in ("G0", "TF2", "AR2"))
    if any(value.shape != (3, 16, *truth.shape[-2:]) or value.dtype != np.float32 for value in (g0, tf, ar)):
        raise ValueError("all generated chunks must be float32 [3,16,H,W]")
    endpoint = compare_endpoint_rgb(tf, ar, truth[32])
    residual = tf[:, -1].astype(np.float64) - truth[32].astype(np.float64)
    feedback_change = ar[:, -1].astype(np.float64) - tf[:, -1].astype(np.float64)
    squared_error_rhs = float(2 * np.mean(residual * feedback_change)
                              + np.mean(feedback_change * feedback_change))
    squared_error_lhs = endpoint["ar_rmse"] ** 2 - endpoint["tf_rmse"] ** 2
    first = g0[:, -1].astype(np.float64) - truth[16].astype(np.float64)
    first_rmse = float(np.sqrt(np.mean(first * first)))
    g0_rms, g0_cos = _latent_metrics(records["G0"]["encoded_condition"], gt16, condition_mask)
    tf_rms, tf_cos = _latent_metrics(records["TF2"]["encoded_condition"], gt32, condition_mask)
    ar_rms, ar_cos = _latent_metrics(records["AR2"]["encoded_condition"], gt32, condition_mask)
    summary = {"record_index": int(record_index), "seed_pair": list(seed_pair), "g0_rmse": first_rmse,
               **endpoint, "squared_error_identity_residual": squared_error_lhs - squared_error_rhs,
               "feedback_change_rms": float(np.sqrt(np.mean(feedback_change * feedback_change))),
               "g0_latent_rms": g0_rms, "tf2_latent_rms": tf_rms,
               "ar2_latent_rms": ar_rms, "g0_latent_cosine": g0_cos,
               "tf2_latent_cosine": tf_cos, "ar2_latent_cosine": ar_cos,
               "delta_feedback_latent_rms": ar_rms - tf_rms,
               "metric_scope": "second_chunk_final_frame", "condition_space": "FP32 VAE condition mask"}
    frames = []
    for offset in range(16):
        target = truth[17 + offset].astype(np.float64)
        tf_diff = tf[:, offset].astype(np.float64) - target
        ar_diff = ar[:, offset].astype(np.float64) - target
        tf_rmse = float(np.sqrt(np.mean(tf_diff * tf_diff)))
        ar_rmse = float(np.sqrt(np.mean(ar_diff * ar_diff)))
        frames.append({"record_index": int(record_index), "seed_pair": list(seed_pair),
                       "truth_frame_index": 17 + offset, "chunk_frame_index": offset + 1,
                       "tf_rmse": tf_rmse, "ar_rmse": ar_rmse,
                       "delta_feedback_rmse": ar_rmse - tf_rmse})
    return summary, frames
