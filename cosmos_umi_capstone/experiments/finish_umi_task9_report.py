"""Render Task 9's precomputed numerical summary without a model or raw tensors."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def render(summary_path: Path, consistency_path: Path, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import numpy as np

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with consistency_path.open(encoding="utf-8", newline="") as stream:
        consistency = list(csv.DictReader(stream))
    if summary.get("status") != "COMPLETE" or len(consistency) != 3 * summary["direction_count"]:
        raise ValueError("summary or per-direction consistency rows are incomplete")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("report output directory must be new and empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary_path, output_dir / "scan_summary.json")
    shutil.copy2(consistency_path, output_dir / "half_step_consistency.csv")
    report = ["# UMI Task 9: Bridge response-spectrum stratum", "",
              f"Scene: `{json.dumps(summary['scene_identity']['scene'], sort_keys=True)}`; "
              f"diffusion seed: `{summary['seed']}`.",
              "32 independent training directions, eight independent held-out directions; "
              "all 40 directions have a paired half-step check.", "",
              "| Output space | Half-step passes | k90/k95/k99 | Effective rank | Unresolved energy | "
              "Held-out residual at k95 (mean/max) |",
              "|---|---:|---:|---:|---:|---:|"]
    with (output_dir / "singular_values.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("space", "index", "singular_value", "resolved_cumulative_energy"))
        for space, result in summary["spaces"].items():
            singular = np.asarray(result["singular_values"], dtype=np.float64)
            cumulative = np.asarray(result["cumulative_energy"], dtype=np.float64)
            for index, value in enumerate(singular):
                writer.writerow((space, index + 1, float(value),
                                 float(cumulative[index]) if index < len(cumulative) else ""))
            k95 = result["k95"]
            residuals = result["heldout_relative_residual_by_k"]
            selected = [] if k95 is None or residuals is None else [
                value for value in residuals[k95 - 1] if value is not None]
            mean = "N/A" if not selected else f"{np.mean(selected):.4f}"
            worst = "N/A" if not selected else f"{np.max(selected):.4f}"
            effective = result["effective_rank"]
            unresolved = result["unresolved_energy_fraction"]
            report.append(f"| {space} | {result['half_step_pass_count']}/{result['half_step_total']} | "
                          f"{result['k90']}/{k95}/{result['k99']} | "
                          f"{'N/A' if effective is None else f'{effective:.4f}'} | "
                          f"{'N/A' if unresolved is None else f'{unresolved:.4g}'} | {mean}/{worst} |")
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
            index = np.arange(1, singular.size + 1)
            if singular.size and singular[0] > 0:
                axes[0].semilogy(index, singular / singular[0], marker="o", markersize=3)
            axes[0].set(xlabel="Index", ylabel="Singular value / largest", title="Uncentered response spectrum")
            if cumulative.size:
                axes[1].plot(index[:cumulative.size], cumulative, marker="o", markersize=3)
            for target in (0.90, 0.95, 0.99):
                axes[1].axhline(target, linestyle="--", linewidth=0.7, color="gray")
            axes[1].set(xlabel="Truncation rank k", ylabel="Resolved cumulative energy", ylim=(0, 1.02))
            if residuals:
                valid = [[value for value in row if value is not None] for row in residuals]
                means = [np.mean(row) if row else np.nan for row in valid]
                maxima = [np.max(row) if row else np.nan for row in valid]
                axes[2].plot(np.arange(1, len(means) + 1), means, label="mean")
                axes[2].plot(np.arange(1, len(maxima) + 1), maxima, label="max")
                axes[2].legend()
            axes[2].set(xlabel="Truncation rank k", ylabel="Held-out projection residual", ylim=(0, 1.05))
            fig.suptitle(f"{space} | {result['interpretation']}")
            fig.tight_layout()
            fig.savefig(output_dir / f"{space}_spectrum.png", dpi=150)
            fig.savefig(output_dir / f"{space}_spectrum.svg")
            plt.close(fig)
    all_pass = all(value["half_step_pass_count"] == value["half_step_total"]
                   for value in summary["spaces"].values())
    report += ["", "## Interpretation", "",
               ("All sampled directions pass the paired half-step diagnostic; this remains empirical "
                "local-linearity evidence, not a proof of a Jacobian." if all_pass else
                "At least one output space fails the complete 40-direction half-step gate. "
                "Those spaces have finite-amplitude response spectra, not a validated Jacobian spectrum."),
               "The training directions' squared-singular-value energy can be concentrated while "
               "the independent held-out directions have large projection residuals; training "
               "compression alone does not establish a general low-rank response operator.",
               "Baseline repeat RMS is reported separately from a heuristic FP32-scale resolution floor. "
               "Neither floor is a rigorous bound for the entire inference chain.",
               "The three output spaces are distinct: predicted latent, actual FP32 re-encoded feedback "
               "condition latent, and final float RGB. The 32-column observed rank cannot exceed 32. "
               "No scene/seed matrices are pooled.", "", "## Failed half-step directions", ""]
    for space in summary["spaces"]:
        failed = [row["direction_id"] for row in consistency
                  if row["space"] == space and row["pass"] == "False"]
        report.append(f"- `{space}`: {', '.join(failed) if failed else 'none'}.")
    report += ["", "Numerical generation and the summary/CSV were completed on the server. "
               "Figures and this report were rendered locally from those unchanged lightweight files "
               "because the CUDA runtime does not include matplotlib; no model generation was repeated.", ""]
    (output_dir / "experiment_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--consistency", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    render(args.summary, args.consistency, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
