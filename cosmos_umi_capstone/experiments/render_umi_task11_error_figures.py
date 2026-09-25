"""Render reviewed static figures and a compact archive for Task 11 true errors.

The renderer uses only compact ``analysis.json`` statistics; the six large
error matrices stay in the remote, manifest-bound attempt directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile


ENSEMBLES = ("E1", "ETF2", "EAR2")
SPACES = ("latent", "rgb")
ENSEMBLE_COLORS = {"E1": "#2563eb", "ETF2": "#ea580c", "EAR2": "#52525b"}
RAW_COLOR = "#2563eb"
CENTERED_COLOR = "#ea580c"
FIXED_RANKS = (1, 2, 4, 8, 10)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plot_data(data: dict[str, Any]) -> dict[str, Any]:
    spectral: dict[str, dict[str, Any]] = {}
    loeo: dict[str, dict[str, Any]] = {}
    adaptive: dict[str, dict[str, list[int]]] = {}
    for ensemble in ENSEMBLES:
        spectral[ensemble] = {}
        loeo[ensemble] = {}
        adaptive[ensemble] = {}
        for space in SPACES:
            item = data["ensembles"][ensemble][space]
            spectrum = item["spectrum"]
            spectral[ensemble][space] = {
                "cumulative_squared_energy": [float(v) for v in spectrum["cumulative_squared_energy"]],
                "truncation_relative_frobenius_residual_by_rank": [float(v) for v in spectrum["truncation_relative_frobenius_residual_by_rank"]],
            }
            folds = item["leave_one_episode_out"]["folds"]
            per_rank: dict[str, list[dict[str, float | int]]] = {}
            centered_k95: list[int] = []
            uncentered_k95: list[int] = []
            for fold in folds:
                centered_k95.append(int(fold["training_centered_k95"]))
                uncentered_k95.append(int(fold["training_k95"]))
                for result in fold["rank_results"]:
                    if result["rank_kind"] != "requested" or result["requested_rank"] not in FIXED_RANKS:
                        continue
                    rank = int(result["requested_rank"])
                    episode = int(fold["held_out_episode"])
                    point = {
                        "held_out_episode": episode,
                        "raw_relative_residual": float(result["episode_mean"]["raw_relative_residual"]),
                        "centered_relative_residual": float(result["episode_mean"]["centered_relative_residual"]),
                    }
                    per_rank.setdefault(str(rank), []).append(point)
            rank_rows = []
            for rank in FIXED_RANKS:
                points = per_rank[str(rank)]
                if len(points) != 6:
                    raise ValueError(f"expected six held-out episodes for {ensemble}/{space}/rank {rank}")
                rank_rows.append({
                    "requested_rank": rank,
                    "episode_points": points,
                    "six_episode_mean_raw": fmean(p["raw_relative_residual"] for p in points),
                    "six_episode_mean_centered": fmean(p["centered_relative_residual"] for p in points),
                })
            loeo[ensemble][space] = rank_rows
            adaptive[ensemble][space] = {
                "training_k95_by_held_out_episode": uncentered_k95,
                "training_centered_k95_by_held_out_episode": centered_k95,
            }

    return {
        "schema": "umi-task11-error-figure-source-v1",
        "source_analysis_sha256": None,
        "source_manifest_sha256": data["source"]["manifest_sha256"],
        "spectral": spectral,
        "loeo_fixed_ranks": loeo,
        "adaptive_k95_reported_separately": adaptive,
        "scope": {
            "columns": "12 actual error columns from 6 episodes × 2 held-out seeds",
            "spaces": {"latent": "masked condition-latent carrier", "rgb": "decoded RGB endpoint"},
            "loeo_episode_points": "each point is the mean of two held-out seed columns for one episode; six episodes are shown, with a descriptive six-episode mean and no population interval",
            "loeo_raw_denominator": "norm of the held-out uncentered error vector",
            "loeo_centered_denominator": "norm of the held-out error vector minus training-only mean",
            "adaptive_k95": "fold-specific uncentered and centered training-only ranks are recorded separately, not mixed into fixed-rank curves",
        },
    }


def render_figures(analysis_path: Path, output_dir: Path) -> dict[str, Any]:
    """Render PNG/SVG exports and source rows; output directory must be new."""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    data = json.loads(analysis_path.read_text(encoding="utf-8"))
    if data.get("schema") != "umi-task11-error-analysis-v1" or data.get("status") != "COMPLETE":
        raise ValueError("expected a COMPLETE umi-task11-error-analysis-v1 analysis JSON")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

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
        "svg.hashsalt": "umi-task11-error-figures-v1",
    })

    output_dir.mkdir(parents=True)
    plot_data = _plot_data(data)
    plot_data["source_analysis_sha256"] = _sha256(analysis_path)
    source_data_path = output_dir / "figure_source_data.json"
    source_data_path.write_text(json.dumps(plot_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    definitions_text = (
        "# Task 11 true-error figure companion\n\n"
        "These panels summarize the immutable Task 10 true-error matrices for six episodes and two seed schedules "
        "(12 columns per ensemble and space). E1 is the G0 prediction minus its aligned 16-frame-condition ground truth; "
        "ETF2 is the TF2 prediction minus its aligned 32-frame-condition ground truth; EAR2 is the AR2 prediction "
        "minus that same 32-frame-condition ground truth. Latent panels use the saved condition-mask carrier; RGB panels "
        "use the decoded endpoint.\n\n"
        "Cumulative energy is the cumulative squared singular-value share of the uncentered actual-error matrix. "
        "Truncation is the relative Frobenius norm residual, not an energy fraction. The LOEO panels show each of six "
        "held-out episode means (each averages its two seed columns) and the descriptive six-episode mean at fixed ranks "
        "1, 2, 4, 8, and 10. Raw residual denominators are the held-out error norms; centered sensitivity denominators "
        "are the norms after subtracting a training-only mean. No population interval is implied.\n\n"
        "In every centered LOEO training fold, ten centered training columns have numerical rank at most 9; therefore "
        "requested rank 10 is numerically capped for centered sensitivity. Fold-specific adaptive training-k95 ranks "
        "are retained separately in `figure_source_data.json` and are not mixed into the fixed-rank curves. The plots "
        "describe these twelve-column ensembles only; they are not full-Jacobian spectra and do not combine horizons, "
        "feedback modes, or spaces.\n\n"
        f"Source Task 10 manifest SHA-256: `{data['source']['manifest_sha256']}`. "
        f"Source analysis JSON SHA-256: `{_sha256(analysis_path)}`.\n"
    )
    caption_path = output_dir / "figure_caption.md"
    caption_path.write_text(definitions_text, encoding="utf-8")

    colors = ENSEMBLE_COLORS
    markers = {"E1": "o", "ETF2": "s", "EAR2": "^"}
    panel_title = {"latent": "Masked condition-latent carrier", "rgb": "Decoded RGB endpoint"}
    files: dict[str, str] = {}

    # Spectral energy: true SVD squared-energy accumulation, in separate spaces.
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), sharey=True)
    for ax, space in zip(axes, SPACES):
        for ensemble in ENSEMBLES:
            y = plot_data["spectral"][ensemble][space]["cumulative_squared_energy"]
            ax.plot(range(1, 13), y, color=colors[ensemble], marker=markers[ensemble], linewidth=1.8, markersize=4.3, label=ensemble)
        ax.axhline(0.95, color="#334155", linewidth=1.0, linestyle="--", label="95% reference")
        ax.set_title(panel_title[space])
        ax.set_xlabel("Leading singular directions")
        ax.set_xticks([1, 2, 4, 6, 8, 10, 12])
        ax.set_xlim(1, 12)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Cumulative squared singular-value energy")
    fig.suptitle("Cumulative squared singular-value energy", y=0.98, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.915, "Uncentered SVD · 12 actual error columns from 6 episodes × 2 seeds; not a full-Jacobian spectrum", ha="center", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=(0.02, 0.12, 0.98, 0.89))
    for ext in ("png", "svg"):
        path = output_dir / f"cumulative_energy.{ext}"
        fig.savefig(path, dpi=220 if ext == "png" else None, bbox_inches="tight", metadata={"Date": None})
        files[path.name] = _sha256(path)
    plt.close(fig)

    # Truncation: relative Frobenius norm residual, deliberately not energy.
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), sharey=True)
    for ax, space in zip(axes, SPACES):
        for ensemble in ENSEMBLES:
            y = plot_data["spectral"][ensemble][space]["truncation_relative_frobenius_residual_by_rank"]
            ax.plot(range(13), y, color=colors[ensemble], marker=markers[ensemble], linewidth=1.8, markersize=4.3, label=ensemble)
        ax.set_title(panel_title[space])
        ax.set_xlabel("Retained singular directions")
        ax.set_xticks([0, 1, 2, 4, 6, 8, 10, 12])
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 1.04)
    axes[0].set_ylabel("Relative Frobenius norm residual")
    fig.suptitle("Truncation residual by retained rank", y=0.98, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.915, "Residual is a norm ratio (not squared energy) · 12-column uncentered error matrix", ha="center", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=(0.02, 0.12, 0.98, 0.89))
    for ext in ("png", "svg"):
        path = output_dir / f"truncation_norm_residual.{ext}"
        fig.savefig(path, dpi=220 if ext == "png" else None, bbox_inches="tight", metadata={"Date": None})
        files[path.name] = _sha256(path)
    plt.close(fig)

    # LOEO: each open marker is one episode (two seed columns averaged); filled
    # line markers are the six-episode descriptive mean. No CI/population claim.
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 9.2), sharex=True, sharey=True)
    ranks = list(FIXED_RANKS)
    jitter = [-0.12, -0.072, -0.024, 0.024, 0.072, 0.12]
    for row, ensemble in enumerate(ENSEMBLES):
        for col, space in enumerate(SPACES):
            ax = axes[row, col]
            records = plot_data["loeo_fixed_ranks"][ensemble][space]
            for metric, color, marker, linestyle in (
                ("raw_relative_residual", RAW_COLOR, "o", "-"),
                ("centered_relative_residual", CENTERED_COLOR, "^", "--"),
            ):
                means = []
                for rank_index, record in enumerate(records):
                    points = record["episode_points"]
                    vals = [float(p[metric]) for p in points]
                    means.append(fmean(vals))
                    xs = [ranks[rank_index] + jitter[i] for i in range(6)]
                    ax.scatter(xs, vals, s=20, marker=marker, facecolors="white", edgecolors=color, linewidths=0.9, alpha=0.88, zorder=3)
                ax.plot(ranks, means, color=color, marker=marker, markersize=4.8, linewidth=1.7, linestyle=linestyle, zorder=4)
            ax.set_title(f"{ensemble} · {panel_title[space]}")
            ax.set_xticks(ranks)
            ax.set_xlim(0.5, 10.5)
            ax.set_ylim(0.84, 1.03)
            ax.grid(axis="x", visible=False)
            if col == 0:
                ax.set_ylabel("Relative residual")
            if row == 2:
                ax.set_xlabel("Requested retained rank")
    legend_handles = [
        Line2D([], [], color=RAW_COLOR, marker="o", markerfacecolor="white", linestyle="None", label="Raw: episode points"),
        Line2D([], [], color=RAW_COLOR, marker="o", linestyle="-", label="Raw: six-episode mean"),
        Line2D([], [], color=CENTERED_COLOR, marker="^", markerfacecolor="white", linestyle="None", label="Centered: episode points"),
        Line2D([], [], color=CENTERED_COLOR, marker="^", linestyle="--", label="Centered: six-episode mean"),
    ]
    fig.suptitle("Leave-one-episode-out residuals at requested ranks", y=0.985, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.95, "Six held-out episodes; each episode point averages its two held-out seeds. No interval is implied.", ha="center", fontsize=8.5)
    fig.text(0.5, 0.026, "Raw denominator: held-out error norm · Centered sensitivity denominator: centered held-out target norm. Adaptive training-k95 ranks are reported separately.", ha="center", fontsize=8)
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0.02, 0.115, 0.98, 0.92), h_pad=1.5, w_pad=1.2)
    for ext in ("png", "svg"):
        path = output_dir / f"loeo_episode_residuals.{ext}"
        fig.savefig(path, dpi=220 if ext == "png" else None, bbox_inches="tight", metadata={"Date": None})
        files[path.name] = _sha256(path)
    plt.close(fig)

    renderer_path = Path(__file__).resolve()
    manifest = {
        "schema": "umi-task11-figure-bundle-v2",
        "source_analysis_sha256": _sha256(analysis_path),
        "source_manifest_sha256": data["source"]["manifest_sha256"],
        "figure_source_data_sha256": _sha256(source_data_path),
        "figure_caption_sha256": _sha256(caption_path),
        "renderer_sha256": _sha256(renderer_path),
        "files": dict(sorted(files.items())),
        "chart_map": {
            "cumulative_energy": "two-panel line-marker chart; separate masked latent and decoded RGB; twelve directions, six episodes by two seeds, 95% reference",
            "truncation_norm_residual": "two-panel line-marker chart; relative Frobenius norm residual, explicitly not energy",
            "loeo_episode_residuals": "3 by 2 small multiples; six episode points plus descriptive mean at fixed ranks; raw and centered denominators distinguished; adaptive ranks separate",
        },
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def create_light_bundle(source_attempt: Path, figure_dir: Path) -> dict[str, Any]:
    """Bundle compact report outputs without copying large error matrices."""
    archive_path = figure_dir / "task11_true_error_light_bundle.zip"
    manifest_path = figure_dir / "light_bundle_manifest.json"
    sidecar_path = figure_dir / "task11_true_error_light_bundle.sha256"
    if any(path.exists() for path in (archive_path, manifest_path, sidecar_path)):
        raise FileExistsError("light bundle outputs already exist")

    source_attempt = source_attempt.resolve()
    source_manifest_path = source_attempt / "MANIFEST.sha256"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"missing source result manifest: {source_manifest_path}")
    source_manifest_rows = {}
    for line in source_manifest_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        source_manifest_rows[name.strip()] = digest

    selected_source = (
        "analysis.json",
        "episode_column_means.csv",
        "error_columns.csv",
        "error_decomposition.csv",
        "review_report.md",
    )
    files: dict[str, str] = {}
    for name in selected_source:
        path = source_attempt / name
        if not path.is_file() or source_manifest_rows.get(name) != _sha256(path):
            raise ValueError(f"source result file missing or not bound by source MANIFEST: {name}")
        files[name] = _sha256(path)
    for name in ("run_identity.json", "run_status.json"):
        path = source_attempt.parent / name
        if not path.is_file():
            raise FileNotFoundError(f"missing run provenance file: {path}")
        files[name] = _sha256(path)
    status = json.loads((source_attempt.parent / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE":
        raise ValueError("light bundle requires a COMPLETE source run")

    figure_manifest_path = figure_dir / "figure_manifest.json"
    source_data_path = figure_dir / "figure_source_data.json"
    figure_manifest = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
    if figure_manifest["source_analysis_sha256"] != source_manifest_rows["analysis.json"]:
        raise ValueError("figure bundle is not bound to the source analysis JSON")
    if _sha256(source_data_path) != figure_manifest["figure_source_data_sha256"]:
        raise ValueError("figure source data hash mismatch")
    files["figures/figure_source_data.json"] = _sha256(source_data_path)
    caption_path = figure_dir / "figure_caption.md"
    if not caption_path.is_file() or _sha256(caption_path) != figure_manifest["figure_caption_sha256"]:
        raise ValueError("figure caption missing or hash mismatch")
    files["figures/figure_caption.md"] = _sha256(caption_path)
    for name, digest in figure_manifest["files"].items():
        path = figure_dir / name
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"figure file missing or hash mismatch: {name}")
        files[f"figures/{name}"] = digest
    files["figures/figure_manifest.json"] = _sha256(figure_manifest_path)

    excluded = []
    for name, digest in sorted(source_manifest_rows.items()):
        if name.lower().endswith(".npy"):
            path = source_attempt / name
            if not path.is_file() or _sha256(path) != digest:
                raise ValueError(f"remote-only matrix missing or hash mismatch: {name}")
            excluded.append({"name": name, "sha256": digest, "size_bytes": path.stat().st_size})
    if len(excluded) != 6:
        raise ValueError(f"expected six remote-only error matrices, found {len(excluded)}")

    bundle_manifest = {
        "schema": "umi-task11-light-bundle-v2",
        "source_attempt": str(source_attempt),
        "source_attempt_status": "COMPLETE",
        "source_result_manifest_sha256": _sha256(source_manifest_path),
        "source_analysis_json_sha256": source_manifest_rows["analysis.json"],
        "figure_manifest_sha256": _sha256(figure_manifest_path),
        "included_files": dict(sorted(files.items())),
        "excluded_remote_only_error_matrices": excluded,
        "scope_note": "This compact bundle is not a replacement for the full source attempt; six hashed error matrices remain remote-only.",
    }
    manifest_path.write_text(json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with ZipFile(archive_path, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in selected_source:
            archive.write(source_attempt / name, arcname=name)
        for name in ("run_identity.json", "run_status.json"):
            archive.write(source_attempt.parent / name, arcname=name)
        archive.write(source_data_path, arcname="figures/figure_source_data.json")
        archive.write(caption_path, arcname="figures/figure_caption.md")
        archive.write(figure_manifest_path, arcname="figures/figure_manifest.json")
        for name in figure_manifest["files"]:
            archive.write(figure_dir / name, arcname=f"figures/{name}")
        archive.write(manifest_path, arcname="light_bundle_manifest.json")
    archive_digest = _sha256(archive_path)
    sidecar_path.write_text(f"{archive_digest}  {archive_path.name}\n", encoding="ascii")
    return {**bundle_manifest, "archive_sha256": archive_digest, "archive_size_bytes": archive_path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-attempt-dir", type=Path, help="completed attempt directory; creates a compact ZIP beside the figures")
    args = parser.parse_args()
    manifest = render_figures(args.analysis_json, args.output_dir)
    bundle = create_light_bundle(args.source_attempt_dir, args.output_dir) if args.source_attempt_dir else None
    print(json.dumps({"output_dir": str(args.output_dir), "manifest": manifest, "light_bundle": bundle}, indent=2))


if __name__ == "__main__":
    main()
