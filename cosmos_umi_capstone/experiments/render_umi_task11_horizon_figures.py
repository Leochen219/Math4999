"""Render static, source-bound Task 11 five-horizon endpoint figures.

This offline renderer reads only ``task11_horizon_metrics.json``. It validates
the two-schedule endpoint pairing and exact TF/AR error-geometry fields before
writing PNG/SVG figures, captions, source rows, and a hash manifest to a new
output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


HORIZONS = (1, 2, 3, 4, 5)
PAIR_HORIZONS = (2, 3, 4, 5)
SCHEDULE_COLORS = ("#2563eb", "#ea580c")
NEUTRAL = "#475569"
TOLERANCE = 5e-8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _close(actual: Any, expected: Any, *, what: str) -> None:
    actual_value = _finite_number(actual, name=f"{what} actual")
    expected_value = _finite_number(expected, name=f"{what} expected")
    if not math.isclose(actual_value, expected_value, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
        raise ValueError(f"{what} does not match its source rows")


def _row_key(row: dict[str, Any], *, include_mode: bool = True) -> tuple[Any, ...]:
    base = (int(row["schedule_index"]), int(row["horizon_chunks"]))
    return base + ((str(row["mode"]),) if include_mode else ())


def _index_unique_rows(rows: Any, *, expected_keys: set[tuple[Any, ...]], name: str,
                       include_mode: bool = True) -> dict[tuple[Any, ...], dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{name} must be a list")
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    try:
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{name} rows must be objects")
            key = _row_key(row, include_mode=include_mode)
            if key in indexed:
                raise ValueError(f"{name} contains duplicate rows")
            indexed[key] = row
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} rows have an invalid schema: {exc}") from exc
    if set(indexed) != expected_keys:
        raise ValueError(f"{name} do not cover the expected endpoint rows")
    return indexed


def _expected_endpoint_keys() -> set[tuple[int, int, str]]:
    return {
        (schedule, horizon, mode)
        for schedule in range(2)
        for horizon in HORIZONS
        for mode in (("G0",) if horizon == 1 else ("TF", "AR"))
    }


def _expected_pair_keys() -> set[tuple[int, int]]:
    return {(schedule, horizon) for schedule in range(2) for horizon in PAIR_HORIZONS}


def _check_identity(data: dict[str, Any]) -> list[list[int]]:
    if data.get("status") != "PASS":
        raise ValueError("expected a PASS task11_horizon_metrics.json")
    if data.get("primary_horizon_axis") != "generated chunk index 1 through 5":
        raise ValueError("unexpected primary horizon axis")
    if data.get("interpretation") != "two preregistered seeds; means are descriptive, not population confidence intervals":
        raise ValueError("unexpected two-seed interpretation metadata")
    record_index = data.get("record_index")
    if isinstance(record_index, bool) or not isinstance(record_index, int) or record_index < 1:
        raise ValueError("record_index must be a positive integer")
    identity_hash = data.get("run_identity_sha256")
    if not isinstance(identity_hash, str) or len(identity_hash) != 64 or any(c not in "0123456789abcdef" for c in identity_hash):
        raise ValueError("run_identity_sha256 must be a lowercase SHA-256 digest")
    schedules = data.get("seed_schedules")
    if (not isinstance(schedules, list) or len(schedules) != 2
            or any(not isinstance(schedule, list) or len(schedule) != 5 for schedule in schedules)):
        raise ValueError("expected exactly two five-seed schedules")
    parsed: list[list[int]] = []
    for schedule in schedules:
        converted = []
        for seed in schedule:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("seed schedules must contain nonnegative integer seeds")
            converted.append(seed)
        parsed.append(converted)
    if data.get("seconds_axis") != "nominally inferred at 5 Hz; source has no actual timestamp feature":
        raise ValueError("unexpected seconds-axis metadata")
    return parsed


def _validate_analysis(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[list[int]]]:
    if not isinstance(data, dict):
        raise ValueError("analysis JSON root must be an object")
    schedules = _check_identity(data)
    endpoints = _index_unique_rows(
        data.get("endpoint_metrics_per_seed"), expected_keys=_expected_endpoint_keys(),
        name="endpoint rows", include_mode=True,
    )
    for (schedule_index, horizon, mode), row in endpoints.items():
        if row.get("schedule_index") != schedule_index or row.get("horizon_chunks") != horizon or row.get("mode") != mode:
            raise ValueError("endpoint row key fields are inconsistent")
        expected_seed = schedules[schedule_index][horizon - 1]
        if row.get("seed") != expected_seed:
            raise ValueError("endpoint row seed does not match its seed schedule")
        if row.get("endpoint_frame_index") != horizon * 16:
            raise ValueError("endpoint row frame index does not match the 16-frame horizon")
        for field in ("rmse", "condition_latent_rms"):
            _finite_number(row.get(field), name=f"endpoint {field}")
        if row["rmse"] < 0 or row["condition_latent_rms"] < 0:
            raise ValueError("endpoint RMSE and masked latent RMS must be nonnegative")

    mean_keys = {(horizon, mode) for horizon in HORIZONS
                 for mode in (("G0",) if horizon == 1 else ("TF", "AR"))}
    means: dict[tuple[int, str], dict[str, Any]] = {}
    mean_rows = data.get("descriptive_two_seed_means")
    if not isinstance(mean_rows, list):
        raise ValueError("descriptive mean rows must be a list")
    for row in mean_rows:
        if not isinstance(row, dict):
            raise ValueError("descriptive mean rows must be objects")
        try:
            key = (int(row["horizon_chunks"]), str(row["mode"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("descriptive mean rows have an invalid schema") from exc
        if key in means:
            raise ValueError("descriptive mean rows contain duplicates")
        means[key] = row
    if set(means) != mean_keys:
        raise ValueError("descriptive mean rows do not cover the expected endpoint modes")
    for (horizon, mode), mean_row in means.items():
        selected = [row for (schedule, h, m), row in endpoints.items() if h == horizon and m == mode]
        if mean_row.get("seed_count") != 2:
            raise ValueError("descriptive mean seed_count must equal two")
        for field in ("rmse", "condition_latent_rms"):
            expected = sum(float(row[field]) for row in selected) / 2.0
            _close(mean_row.get(field), expected, what=f"descriptive mean {field} for h{horizon}/{mode}")

    expected_pair_keys = _expected_pair_keys()
    delta_rows = _index_unique_rows(
        data.get("tf_ar_endpoint_error_deltas"), expected_keys=expected_pair_keys,
        name="paired delta rows", include_mode=False,
    )
    for key, delta in delta_rows.items():
        schedule_index, horizon = key
        if delta.get("seed") != schedules[schedule_index][horizon - 1]:
            raise ValueError("paired delta seed does not match its schedule")
        tf = endpoints[(schedule_index, horizon, "TF")]
        ar = endpoints[(schedule_index, horizon, "AR")]
        _close(delta.get("rgb_rmse_ar_minus_tf"), float(ar["rmse"]) - float(tf["rmse"]),
               what=f"paired delta RGB RMSE for schedule {schedule_index}, h{horizon}")
        _close(delta.get("condition_latent_rms_ar_minus_tf"),
               float(ar["condition_latent_rms"]) - float(tf["condition_latent_rms"]),
               what=f"paired delta mask-RMS for schedule {schedule_index}, h{horizon}")

    geometry_by_space = {}
    for space, source_key in (
        ("rgb", "tf_ar_error_geometry_rgb"),
        ("condition_masked_latent", "tf_ar_error_geometry_condition_masked_latent"),
    ):
        indexed = _index_unique_rows(data.get(source_key), expected_keys=expected_pair_keys,
                                     name=f"{space} error geometry", include_mode=False)
        for key, row in indexed.items():
            schedule_index, horizon = key
            if row.get("seed") != schedules[schedule_index][horizon - 1]:
                raise ValueError(f"{space} error geometry seed does not match its schedule")
            required = (
                "tf_squared_error", "ar_squared_error", "cross_term_2dot_over_n",
                "feedback_change_squared", "squared_error_difference", "identity_residual",
            )
            values = {field: _finite_number(row.get(field), name=f"{space} error geometry {field}")
                      for field in required}
            if values["tf_squared_error"] < 0 or values["ar_squared_error"] < 0 or values["feedback_change_squared"] < 0:
                raise ValueError(f"{space} error geometry squared errors must be nonnegative")
            observed_difference = values["ar_squared_error"] - values["tf_squared_error"]
            _close(values["squared_error_difference"], observed_difference,
                   what=f"{space} error geometry observed MSE difference")
            tf_endpoint = endpoints[(schedule_index, horizon, "TF")]
            ar_endpoint = endpoints[(schedule_index, horizon, "AR")]
            endpoint_metric = "rmse" if space == "rgb" else "condition_latent_rms"
            endpoint_name = "endpoint RMSE" if space == "rgb" else "endpoint mask-RMS"
            _close(values["tf_squared_error"], float(tf_endpoint[endpoint_metric]) ** 2,
                   what=f"{space} TF MSE does not match its {endpoint_name}")
            _close(values["ar_squared_error"], float(ar_endpoint[endpoint_metric]) ** 2,
                   what=f"{space} AR MSE does not match its {endpoint_name}")
            residual = values["squared_error_difference"] - (
                values["cross_term_2dot_over_n"] + values["feedback_change_squared"])
            _close(values["identity_residual"], residual,
                   what=f"{space} error geometry identity residual")
            if space == "condition_masked_latent":
                count = row.get("masked_element_count")
                if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                    raise ValueError("condition-masked latent geometry needs a positive masked_element_count")
        geometry_by_space[space] = indexed

    return endpoints, means, {"deltas": delta_rows, "geometry": geometry_by_space}, schedules


def build_figure_source_data(data: dict[str, Any], *, analysis_sha256: str) -> dict[str, Any]:
    """Validate source metrics and return the exact rows consumed by the figures."""
    if len(analysis_sha256) != 64 or any(c not in "0123456789abcdef" for c in analysis_sha256):
        raise ValueError("analysis_sha256 must be a lowercase SHA-256 digest")
    endpoints, means, paired, schedules = _validate_analysis(data)
    endpoint_rows = [endpoints[key] for key in sorted(endpoints)]
    mean_rows = [means[key] for key in sorted(means)]
    delta_rows = [paired["deltas"][key] for key in sorted(paired["deltas"])]
    geometry_rows = {
        space: [paired["geometry"][space][key] for key in sorted(paired["geometry"][space])]
        for space in ("rgb", "condition_masked_latent")
    }

    endpoint_plot_series = []
    for schedule_index in range(2):
        # The single G0 observation at h1 is retained once in endpoint_rows.
        # TF and AR visual traces begin at h2, so h1 is not duplicated as two
        # apparent mode-specific measurements.
        for mode in ("TF", "AR"):
            endpoint_plot_series.append({
                "schedule_index": schedule_index,
                "mode": mode,
                "points": [endpoints[(schedule_index, h, mode)] for h in PAIR_HORIZONS],
            })
    endpoint_mean_series = []
    for mode in ("TF", "AR"):
        endpoint_mean_series.append({
            "mode": mode,
            "points": [means[(h, mode)] for h in PAIR_HORIZONS],
        })

    delta_mean_rows = []
    for horizon in PAIR_HORIZONS:
        values = [paired["deltas"][(schedule, horizon)] for schedule in range(2)]
        delta_mean_rows.append({
            "horizon_chunks": horizon,
            "rgb_rmse_ar_minus_tf": sum(float(row["rgb_rmse_ar_minus_tf"]) for row in values) / 2.0,
            "condition_latent_rms_ar_minus_tf": sum(float(row["condition_latent_rms_ar_minus_tf"]) for row in values) / 2.0,
        })

    return {
        "schema": "umi-task11-horizon-figure-source-v1",
        "source_analysis_sha256": analysis_sha256,
        "source_record_index": data["record_index"],
        "source_run_identity_sha256": data["run_identity_sha256"],
        "seed_schedules": schedules,
        "scope": {
            "episode_count": 1,
            "seed_schedule_count": 2,
            "claim": "one episode with two seed schedules; means are descriptive, not population estimates or confidence intervals",
            "comparison": "endpoint prediction errors are relative to real trajectory frames under future-pose motion conditions",
            "horizons": "five observed generated-chunk endpoints only; line segments are visual guides, not fitted growth laws",
            "h1": "one G0 endpoint per schedule; TF and AR mode-specific endpoints start at h2",
            "latent": "condition-mask RMS/error geometry uses image-condition latent mask slots; future poses are action conditions",
            "mse_identity": "observed AR-minus-TF MSE difference equals the FP64 cross term plus feedback-change squared, up to the recorded identity residual",
        },
        "endpoint_rows": endpoint_rows,
        "descriptive_mean_rows": mean_rows,
        "endpoint_plot_series_h2_h5": endpoint_plot_series,
        "endpoint_descriptive_mean_series_h2_h5": endpoint_mean_series,
        "paired_delta_rows": delta_rows,
        "paired_delta_descriptive_means": delta_mean_rows,
        "decomposition_rows": geometry_rows,
    }


def _caption_text(source: dict[str, Any]) -> str:
    return (
        "# Task 11 five-horizon figure captions\n\n"
        f"Source record index: `{source['source_record_index']}`. Run identity SHA-256: "
        f"`{source['source_run_identity_sha256']}`. Source metrics SHA-256: "
        f"`{source['source_analysis_sha256']}`.\n\n"
        "Scope is one episode with two seed schedules (two preregistered schedules). The plotted two-seed means "
        "are descriptive only; there are no population confidence intervals or population claims. Endpoint errors are relative to real "
        "trajectory frames under future-pose motion conditions. The five observed chunk endpoints are shown without "
        "extrapolation or smoothing; connecting segments are visual guides, not fitted growth laws. No exponential "
        "or population-level growth claim is made.\n\n"
        "Figure 1. Truth-relative decoded RGB endpoint RMSE (normalized RGB units) and condition-mask latent RMS "
        "(latent units) versus generated horizon h1–h5, in separate panels. At h1 the single G0 endpoint for each "
        "schedule is plotted once; TF and AR endpoint traces begin at h2. At h2–h5, solid/circle denotes TF and "
        "dashed/square denotes AR. Dark-gray traces are descriptive two-schedule means. Latent values use the "
        "saved image-condition latent mask; future poses are action conditions.\n\n"
        "Figure 2. Matched same-schedule AR minus TF endpoint RMSE differences at h2–h5, with a visible zero line. "
        "The latent panel is a difference in condition-mask RMS (not RMSE). Negative differences are retained; the "
        "dark-gray line is the descriptive mean of the two paired schedules.\n\n"
        "Figure 3. Per-schedule squared-error decomposition at h2–h5, shown separately for decoded RGB and the "
        "condition-masked latent carrier. With `b = TF − truth` and `p = AR − TF`, the observed `MSE_AR − MSE_TF` "
        "is compared with `mean(2 b p)` and `mean(p²)`. The independently computed identity residual is retained "
        "in `figure_source_data.json`; it is not hidden by defining the observed difference from the right-hand side.\n"
    )


def _set_plot_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": "#475569",
        "axes.labelcolor": "#172033",
        "text.color": "#172033",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "axes.grid": True,
        "grid.color": "#dbe2ea",
        "grid.linewidth": 0.65,
        "grid.alpha": 0.85,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "svg.hashsalt": "umi-task11-horizon-figures-v1",
    })


def _save_figure(fig: Any, output_dir: Path, stem: str) -> dict[str, str]:
    result = {}
    for extension in ("png", "svg"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=220 if extension == "png" else None,
                    bbox_inches="tight", metadata={"Date": None})
        result[path.name] = _sha256(path)
    return result


def _render_endpoint_figure(source: dict[str, Any], output_dir: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))
    metric_configs = (
        ("rmse", "Decoded RGB endpoint RMSE", "RMSE (normalized RGB units)"),
        ("condition_latent_rms", "Condition-mask latent endpoint RMS", "Masked latent RMS"),
    )
    mode_style = {"TF": ("-", "o"), "AR": ("--", "s")}
    x_pair = list(PAIR_HORIZONS)
    for ax, (metric, title, y_label) in zip(axes, metric_configs):
        for series in source["endpoint_plot_series_h2_h5"]:
            schedule = int(series["schedule_index"])
            mode = series["mode"]
            linestyle, marker = mode_style[mode]
            ax.plot(x_pair, [float(row[metric]) for row in series["points"]],
                    color=SCHEDULE_COLORS[schedule], linestyle=linestyle, marker=marker,
                    linewidth=1.65, markersize=4.4, alpha=0.95, zorder=3)
        for mean_series in source["endpoint_descriptive_mean_series_h2_h5"]:
            mode = mean_series["mode"]
            linestyle, marker = mode_style[mode]
            ax.plot(x_pair, [float(row[metric]) for row in mean_series["points"]],
                    color=NEUTRAL, linestyle=linestyle, marker=marker,
                    linewidth=2.5, markersize=5.1, zorder=4)

        for schedule, color in enumerate(SCHEDULE_COLORS):
            g0 = next(row for row in source["endpoint_rows"]
                      if row["schedule_index"] == schedule and row["horizon_chunks"] == 1)
            ax.scatter([1], [float(g0[metric])], marker="D", s=44,
                       facecolor="white", edgecolor=color, linewidth=1.4, zorder=5)
        mean_g0 = next(row for row in source["descriptive_mean_rows"]
                       if row["horizon_chunks"] == 1)
        ax.scatter([1], [float(mean_g0[metric])], marker="D", s=49,
                   facecolor=NEUTRAL, edgecolor="white", linewidth=0.7, zorder=6)
        ax.set_title(title)
        ax.set_xlabel("Generated horizon (chunks)")
        ax.set_xticks(HORIZONS)
        ax.set_xlim(0.8, 5.2)
        ax.set_ylabel(y_label)
        ax.grid(axis="x", visible=False)

    legend_handles = [
        Line2D([], [], color=SCHEDULE_COLORS[0], linestyle="-", marker="o", label="Schedule 0 · TF"),
        Line2D([], [], color=SCHEDULE_COLORS[0], linestyle="--", marker="s", label="Schedule 0 · AR"),
        Line2D([], [], color=SCHEDULE_COLORS[1], linestyle="-", marker="o", label="Schedule 1 · TF"),
        Line2D([], [], color=SCHEDULE_COLORS[1], linestyle="--", marker="s", label="Schedule 1 · AR"),
        Line2D([], [], color=NEUTRAL, linestyle="-", marker="o", linewidth=2.5, label="Two-schedule mean · TF"),
        Line2D([], [], color=NEUTRAL, linestyle="--", marker="s", linewidth=2.5, label="Two-schedule mean · AR"),
        Line2D([], [], color=NEUTRAL, marker="D", markerfacecolor="white", linestyle="None",
               label="G0 endpoint at h1 (one per schedule)"),
    ]
    fig.suptitle("Truth-relative endpoint error across five observed horizons", y=0.985, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.925,
             "One episode · two seed schedules · h1 is a single shared G0 mode, not separate TF/AR observations",
             ha="center", fontsize=8.4)
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=4, frameon=False, fontsize=8.1)
    fig.tight_layout(rect=(0.025, 0.15, 0.975, 0.89), w_pad=1.4)
    files = _save_figure(fig, output_dir, "endpoint_error")
    plt.close(fig)
    return files


def _render_delta_figure(source: dict[str, Any], output_dir: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9))
    configs = (
        ("rgb_rmse_ar_minus_tf", "Matched endpoint RMSE difference · RGB", "AR − TF RMSE (normalized RGB units)"),
        ("condition_latent_rms_ar_minus_tf", "Matched endpoint RMS difference · latent",
         "AR − TF condition-mask RMS (latent units)"),
    )
    by_schedule = {schedule: [row for row in source["paired_delta_rows"]
                              if row["schedule_index"] == schedule]
                   for schedule in range(2)}
    means_by_horizon = {row["horizon_chunks"]: row for row in source["paired_delta_descriptive_means"]}
    marker_for = ("o", "s")
    for ax, (field, title, y_label) in zip(axes, configs):
        ax.axhline(0.0, color=NEUTRAL, linewidth=1.05, linestyle="--", zorder=1)
        for schedule in range(2):
            rows = sorted(by_schedule[schedule], key=lambda row: row["horizon_chunks"])
            ax.plot([row["horizon_chunks"] for row in rows], [float(row[field]) for row in rows],
                    color=SCHEDULE_COLORS[schedule], marker=marker_for[schedule], linewidth=1.7,
                    markersize=4.7, label=f"Schedule {schedule}", zorder=3)
        ax.plot(PAIR_HORIZONS, [float(means_by_horizon[h][field]) for h in PAIR_HORIZONS],
                color=NEUTRAL, marker="D", linewidth=2.45, markersize=5.0,
                label="Descriptive two-schedule mean", zorder=4)
        ax.set_title(title)
        ax.set_xlabel("Generated horizon (chunks)")
        ax.set_ylabel(y_label)
        ax.set_xticks(PAIR_HORIZONS)
        ax.set_xlim(1.8, 5.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Matched autoregressive minus teacher-forced endpoint error", y=0.98,
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.91, "Same schedule and horizon paired · zero reference shown · negative outcomes retained · no interval",
             ha="center", fontsize=8.6)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.035),
               ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0.025, 0.13, 0.975, 0.87), w_pad=1.4)
    files = _save_figure(fig, output_dir, "paired_rmse_delta")
    plt.close(fig)
    return files


def _render_geometry_figure(source: dict[str, Any], output_dir: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.1), sharex=True)
    spaces = (
        ("rgb", "Decoded RGB", "MSE contribution (RGB units²)"),
        ("condition_masked_latent", "Condition-masked latent", "MSE contribution (latent units²)"),
    )
    term_specs = (
        ("cross_term_2dot_over_n", "Cross term · mean(2bp)", "#2563eb", "o", "-"),
        ("feedback_change_squared", "Feedback change · mean(p²)", "#ea580c", "s", "--"),
        ("squared_error_difference", "Observed ΔMSE · AR − TF", NEUTRAL, "D", ":"),
    )
    for row_index, schedule in enumerate(range(2)):
        for col_index, (space, space_title, y_label) in enumerate(spaces):
            ax = axes[row_index, col_index]
            records = sorted(
                [record for record in source["decomposition_rows"][space]
                 if record["schedule_index"] == schedule],
                key=lambda record: record["horizon_chunks"],
            )
            ax.axhline(0.0, color="#94a3b8", linewidth=0.8, zorder=1)
            for field, label, color, marker, linestyle in term_specs:
                ax.plot([record["horizon_chunks"] for record in records],
                        [float(record[field]) for record in records], color=color,
                        marker=marker, linestyle=linestyle, linewidth=1.8, markersize=4.6,
                        label=label, zorder=3)
            ax.set_title(f"Schedule {schedule} · {space_title}")
            ax.set_xticks(PAIR_HORIZONS)
            ax.set_xlim(1.8, 5.2)
            if row_index == 1:
                ax.set_xlabel("Generated horizon (chunks)")
            ax.set_ylabel(y_label)
    handles = [Line2D([], [], color=color, marker=marker, linestyle=linestyle,
                      linewidth=1.8, label=label)
               for _, label, color, marker, linestyle in term_specs]
    fig.suptitle("Exact teacher-forced / autoregressive squared-error geometry", y=0.99,
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.95,
             "Per schedule and horizon: observed ΔMSE is compared with cross term + feedback-change squared",
             ha="center", fontsize=8.5)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=3, frameon=False, fontsize=8.4)
    fig.tight_layout(rect=(0.03, 0.09, 0.97, 0.92), h_pad=1.4, w_pad=1.5)
    files = _save_figure(fig, output_dir, "mse_error_geometry")
    plt.close(fig)
    return files


def render_figures(analysis_path: Path, output_dir: Path) -> dict[str, Any]:
    """Render PNG/SVG exports from a frozen metrics JSON into a new directory."""
    analysis_path = Path(analysis_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    data = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis_sha256 = _sha256(analysis_path)
    source = build_figure_source_data(data, analysis_sha256=analysis_sha256)
    _set_plot_style()
    output_dir.mkdir(parents=True)
    source_path = output_dir / "figure_source_data.json"
    source_path.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    caption_path = output_dir / "figure_caption.md"
    caption_path.write_text(_caption_text(source), encoding="utf-8")

    figure_hashes = {}
    figure_hashes.update(_render_endpoint_figure(source, output_dir))
    figure_hashes.update(_render_delta_figure(source, output_dir))
    figure_hashes.update(_render_geometry_figure(source, output_dir))
    renderer_sha256 = _sha256(Path(__file__).resolve())
    manifest = {
        "schema": "umi-task11-horizon-figure-manifest-v1",
        "source_analysis_sha256": analysis_sha256,
        "source_run_identity_sha256": source["source_run_identity_sha256"],
        "renderer_sha256": renderer_sha256,
        "files": {
            "figure_source_data.json": _sha256(source_path),
            "figure_caption.md": _sha256(caption_path),
            **figure_hashes,
        },
        "scope": source["scope"],
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "PASS",
        "output_dir": str(output_dir.resolve()),
        "source_analysis_sha256": analysis_sha256,
        "renderer_sha256": renderer_sha256,
        "figure_manifest_sha256": _sha256(manifest_path),
        "files": manifest["files"],
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_json", type=Path, help="frozen task11_horizon_metrics.json input")
    parser.add_argument("output_dir", type=Path, help="new directory for PNG/SVG and source companions")
    args = parser.parse_args(argv)
    print(json.dumps(render_figures(args.analysis_json, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
