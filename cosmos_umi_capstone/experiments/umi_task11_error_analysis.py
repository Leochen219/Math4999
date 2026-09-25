"""Float64 true-error spectra and paired feedback-error geometry for Task 11."""
from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

import numpy as np


def _finite_array(value: Any, *, name: str, dtype: Any = np.float64) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite and non-empty")
    return array


def array_sha256(value: Any) -> str:
    """Hash a canonical ndarray identity without retaining its contents in JSON."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii")); digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes()); digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def prediction_error(prediction: Any, truth: Any, *, mask: Any | None = None) -> np.ndarray:
    """Return prediction minus aligned truth in float64, optionally under a mask."""
    estimate = np.asarray(prediction)
    target = np.asarray(truth)
    if estimate.shape != target.shape or estimate.size == 0:
        raise ValueError("prediction and truth must have the same non-empty shape")
    if not np.issubdtype(estimate.dtype, np.number) or not np.issubdtype(target.dtype, np.number):
        raise ValueError("prediction and truth must be numeric arrays")
    estimate64 = _finite_array(estimate, name="prediction", dtype=np.float64)
    target64 = _finite_array(target, name="truth", dtype=np.float64)
    if mask is None:
        return np.ascontiguousarray((estimate64 - target64).reshape(-1), dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != estimate.shape or not np.any(selected):
        raise ValueError("mask must match prediction geometry and select at least one value")
    return np.ascontiguousarray(estimate64[selected] - target64[selected], dtype=np.float64)


def _svd_rank(singular_values: np.ndarray, shape: tuple[int, int]) -> tuple[int, float]:
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return 0, 0.0
    tolerance = float(np.finfo(np.float64).eps * max(shape) * singular_values[0])
    return int(np.count_nonzero(singular_values > tolerance)), tolerance


def _k_for_energy(cumulative: np.ndarray, target: float) -> int | None:
    if cumulative.size == 0:
        return None
    return int(np.searchsorted(cumulative, target, side="left") + 1)


def error_spectrum(columns: Any) -> dict[str, Any]:
    """Compute an uncentered thin SVD summary while preserving error amplitude."""
    matrix = _finite_array(columns, name="error matrix", dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("error matrix must have shape [features, columns]")
    singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    numerical_rank, tolerance = _svd_rank(singular, matrix.shape)
    energy = singular * singular
    total = float(np.sum(energy, dtype=np.float64))
    if total == 0.0:
        cumulative = np.zeros_like(energy)
        return {
            "status": "ZERO_ERROR_MATRIX", "feature_count": int(matrix.shape[0]),
            "column_count": int(matrix.shape[1]), "zero_column_indices": list(range(matrix.shape[1])),
            "singular_values": singular.tolist(), "cumulative_squared_energy": cumulative.tolist(),
            "numerical_rank": numerical_rank, "numerical_rank_tolerance": tolerance,
            "k90": None, "k95": None, "k99": None, "effective_rank": None,
            "truncation_relative_frobenius_residual_by_rank": [None] * (singular.size + 1),
        }
    probabilities = energy / total
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    positive = probabilities[probabilities > 0.0]
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive), dtype=np.float64)))
    truncation = [float(np.sqrt(max(0.0, np.sum(energy[k:], dtype=np.float64) / total)))
                  for k in range(singular.size + 1)]
    norms = np.linalg.norm(matrix, axis=0)
    return {
        "status": "OK", "feature_count": int(matrix.shape[0]), "column_count": int(matrix.shape[1]),
        "zero_column_indices": np.flatnonzero(norms == 0.0).astype(int).tolist(),
        "column_norms": norms.tolist(), "singular_values": singular.tolist(),
        "cumulative_squared_energy": cumulative.tolist(), "numerical_rank": numerical_rank,
        "numerical_rank_tolerance": tolerance, "k90": _k_for_energy(cumulative, 0.90),
        "k95": _k_for_energy(cumulative, 0.95), "k99": _k_for_energy(cumulative, 0.99),
        "effective_rank": effective_rank,
        "truncation_relative_frobenius_residual_by_rank": truncation,
    }


def _projection_residual(basis: np.ndarray, vector: np.ndarray) -> tuple[float | None, float, np.ndarray]:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return None, norm, np.zeros_like(vector)
    projected = basis @ (basis.T @ vector) if basis.shape[1] else np.zeros_like(vector)
    error = float(np.linalg.norm(vector - projected))
    return error / norm, norm, projected


def leave_one_episode_out(columns: Any, episode_ids: Sequence[Any], *,
                          ranks: Sequence[int] = (1, 2, 4, 8, 10)) -> dict[str, Any]:
    """Project each two-seed held-out episode onto the other ten error columns.

    Uncentered and centered bases are fit on training columns only. The centered
    relative residual uses ``heldout - training_mean`` as its denominator; the
    raw reconstruction residual uses ``||heldout||`` as its denominator.
    """
    matrix = _finite_array(columns, name="error matrix", dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(episode_ids):
        raise ValueError("episode labels must match error-matrix columns")
    if matrix.shape[1] != 12:
        raise ValueError("Task 11 LOEO requires twelve columns, two for each episode")
    labels = np.asarray(episode_ids)
    unique = list(dict.fromkeys(labels.tolist()))
    if len(unique) != 6 or any(int(np.count_nonzero(labels == item)) != 2 for item in unique):
        raise ValueError("Task 11 LOEO requires six episodes with both seed columns each")
    requested = tuple(ranks)
    if not requested or any(isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or rank < 1
                            for rank in requested) or len(set(requested)) != len(requested):
        raise ValueError("requested ranks must be distinct positive integers")

    folds: list[dict[str, Any]] = []
    summaries: dict[tuple[str, int | None], list[tuple[Any, int | None, dict[str, float | None]]]] = {}
    for heldout_episode in unique:
        heldout = np.flatnonzero(labels == heldout_episode)
        training = np.flatnonzero(labels != heldout_episode)
        train = np.asarray(matrix[:, training], dtype=np.float64)
        train_mean = np.mean(train, axis=1, dtype=np.float64)
        centered_train = train - train_mean[:, None]
        uncentered_u, uncentered_s, _ = np.linalg.svd(train, full_matrices=False)
        centered_u, centered_s, _ = np.linalg.svd(centered_train, full_matrices=False)
        train_rank, train_tol = _svd_rank(uncentered_s, train.shape)
        centered_rank, centered_tol = _svd_rank(centered_s, centered_train.shape)
        train_energy = uncentered_s * uncentered_s
        train_total = float(np.sum(train_energy, dtype=np.float64))
        train_cumulative = np.cumsum(train_energy / train_total) if train_total > 0.0 else np.zeros_like(train_energy)
        training_k95 = _k_for_energy(train_cumulative, 0.95)
        if training_k95 is not None:
            training_k95 = None if train_rank == 0 else min(training_k95, train_rank)
        centered_energy = centered_s * centered_s
        centered_total = float(np.sum(centered_energy, dtype=np.float64))
        centered_cumulative = (np.cumsum(centered_energy / centered_total)
                               if centered_total > 0.0 else np.zeros_like(centered_energy))
        training_centered_k95 = _k_for_energy(centered_cumulative, 0.95)
        if training_centered_k95 is not None:
            training_centered_k95 = None if centered_rank == 0 else min(training_centered_k95, centered_rank)

        rank_specs: list[tuple[str, int | None]] = [("requested", int(rank)) for rank in requested]
        if training_k95 is not None:
            rank_specs.append(("training_k95", int(training_k95)))
        if training_centered_k95 is not None:
            rank_specs.append(("training_centered_k95", int(training_centered_k95)))
        rank_rows: list[dict[str, Any]] = []
        for rank_kind, requested_rank in rank_specs:
            effective_uncentered = 0 if requested_rank is None else min(requested_rank, train_rank)
            effective_centered = 0 if requested_rank is None else min(requested_rank, centered_rank)
            basis_uncentered = uncentered_u[:, :effective_uncentered]
            basis_centered = centered_u[:, :effective_centered]
            seed_rows: list[dict[str, Any]] = []
            for column_index in heldout:
                target = np.asarray(matrix[:, column_index], dtype=np.float64)
                centered_target = target - train_mean
                raw_relative, raw_norm, _ = _projection_residual(basis_uncentered, target)
                centered_relative, centered_norm, centered_projected = _projection_residual(basis_centered, centered_target)
                centered_error = float(np.linalg.norm(centered_target - centered_projected))
                raw_reconstruction_error = centered_error
                raw_reconstruction_relative = (None if raw_norm == 0.0 else raw_reconstruction_error / raw_norm)
                seed_rows.append({
                    "column_index": int(column_index), "heldout_norm": raw_norm,
                    "centered_target_norm": centered_norm,
                    "raw_relative_residual": raw_relative,
                    "centered_relative_residual": centered_relative,
                    "centered_reconstruction_error_norm": centered_error,
                    "raw_reconstruction_relative_residual": raw_reconstruction_relative,
                })
            keys = ("raw_relative_residual", "centered_relative_residual",
                    "raw_reconstruction_relative_residual")
            episode_mean = {
                key: (None if not [row[key] for row in seed_rows if row[key] is not None]
                      else float(np.mean([row[key] for row in seed_rows if row[key] is not None], dtype=np.float64)))
                for key in keys
            }
            rank_rows.append({
                "rank_kind": rank_kind, "requested_rank": requested_rank,
                "effective_rank_uncentered": effective_uncentered,
                "effective_rank_centered": effective_centered,
                "rank_capped": (requested_rank is None or effective_uncentered != requested_rank
                                or effective_centered != requested_rank),
                "rank_capped_uncentered": requested_rank is None or effective_uncentered != requested_rank,
                "rank_capped_centered": requested_rank is None or effective_centered != requested_rank,
                "seed_rows": seed_rows, "episode_mean": episode_mean,
            })
            summary_rank = requested_rank if rank_kind == "requested" else None
            summaries.setdefault((rank_kind, summary_rank), []).append((
                heldout_episode.item() if isinstance(heldout_episode, np.generic) else heldout_episode,
                requested_rank, episode_mean,
            ))
        folds.append({
            "held_out_episode": heldout_episode.item() if isinstance(heldout_episode, np.generic) else heldout_episode,
            "held_out_column_indices": heldout.astype(int).tolist(),
            "training_column_indices": training.astype(int).tolist(),
            "training_column_count": int(training.size), "heldout_column_count": int(heldout.size),
            "training_mean_sha256": array_sha256(np.ascontiguousarray(train_mean, dtype=np.float64)),
            "training_mean_norm": float(np.linalg.norm(train_mean)),
            "training_numerical_rank": train_rank, "training_numerical_rank_tolerance": train_tol,
            "training_centered_numerical_rank": centered_rank,
            "training_centered_numerical_rank_tolerance": centered_tol,
            "training_k95": training_k95, "training_centered_k95": training_centered_k95,
            "rank_results": rank_rows,
        })
    episode_summary = []
    for (rank_kind, requested_rank), rows in summaries.items():
        metric_summary: dict[str, Any] = {}
        for metric in ("raw_relative_residual", "centered_relative_residual",
                       "raw_reconstruction_relative_residual"):
            values = [row[2][metric] for row in rows if row[2][metric] is not None]
            metric_summary[metric] = None if not values else float(np.mean(values, dtype=np.float64))
            metric_summary[metric + "_episode_count"] = len(values)
        selected_ranks = [{"held_out_episode": row[0], "selected_rank": row[1]} for row in rows]
        numeric_selected = [row[1] for row in rows if row[1] is not None]
        episode_summary.append({"rank_kind": rank_kind, "requested_rank": requested_rank,
                                "episode_count": len(rows),
                                "selected_rank_min": min(numeric_selected) if numeric_selected else None,
                                "selected_rank_max": max(numeric_selected) if numeric_selected else None,
                                "selected_ranks_by_episode": selected_ranks, **metric_summary})
    return {"status": "COMPLETE", "fold_count": len(folds), "folds": folds,
            "six_episode_summary_of_episode_means": episode_summary,
            "projection_interpretation": "optimal representation of known held-out error columns; not a trained prediction"}


def error_decomposition(baseline_error: Any, feedback_change: Any, *,
                        combined_error: Any | None = None) -> dict[str, Any]:
    """Compare observed combined error with the independent b/p expansion."""
    baseline = _finite_array(baseline_error, name="baseline error", dtype=np.float64).reshape(-1)
    change = _finite_array(feedback_change, name="feedback change", dtype=np.float64).reshape(-1)
    if baseline.shape != change.shape:
        raise ValueError("baseline error and feedback change must have identical flattened size")
    combined = (baseline + change if combined_error is None else
                _finite_array(combined_error, name="combined error", dtype=np.float64).reshape(-1))
    if combined.shape != baseline.shape:
        raise ValueError("combined error must have the same flattened size as baseline and feedback change")
    n = int(baseline.size)
    baseline_mse = float(np.mean(baseline * baseline, dtype=np.float64))
    feedback_mse = float(np.mean(change * change, dtype=np.float64))
    cross = float(2.0 * np.dot(baseline, change) / n)
    combined_mse = float(np.mean(combined * combined, dtype=np.float64))
    difference = combined_mse - baseline_mse
    denominator = float(np.linalg.norm(baseline) * np.linalg.norm(change))
    cosine = None if denominator == 0.0 else float(np.dot(baseline, change) / denominator)
    return {"feature_count": n, "baseline_mse": baseline_mse,
            "feedback_change_mse": feedback_mse, "cross_term_2dot_over_n": cross,
            "combined_error_mse": combined_mse, "squared_error_difference": difference,
            "identity_residual": difference - (feedback_mse + cross), "cosine": cosine}


def rgb_endpoint_metrics(prediction: Any, truth: Any) -> dict[str, float]:
    """Evaluate a final RGB frame in float64 while preserving its FP32 source contract."""
    estimate, target = np.asarray(prediction), np.asarray(truth)
    if (estimate.dtype != np.float32 or target.dtype != np.float32 or estimate.shape != target.shape
            or estimate.ndim != 3 or estimate.shape[0] != 3):
        raise ValueError("endpoint RGB must be matching float32 CHW arrays")
    if not np.isfinite(estimate).all() or not np.isfinite(target).all():
        raise ValueError("endpoint RGB must be finite")
    if estimate.min() < 0.0 or estimate.max() > 1.0 or target.min() < 0.0 or target.max() > 1.0:
        raise ValueError("endpoint RGB must lie in [0,1]")
    difference = estimate.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(difference * difference, dtype=np.float64))
    rmse = float(np.sqrt(mse))
    return {"mse": mse, "rmse": rmse,
            "mae": float(np.mean(np.abs(difference), dtype=np.float64)),
            "psnr_db": float("inf") if rmse == 0.0 else float(-20.0 * math.log10(rmse))}
