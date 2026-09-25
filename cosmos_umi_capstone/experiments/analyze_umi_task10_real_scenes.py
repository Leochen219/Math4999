"""Offline, episode-clustered analysis of the 48 saved Task 10 calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from task8_frozen.run_umi_task8_experiment import Task8SampleStore
from task8_frozen.task8_formal import _array_hash, _canonical_hash
from task8_frozen.umi_task8_runtime import Task8InputAdapter
from umi_task10_real_scenes import (LOCKED_RECORDS, SEED_PAIRS, aggregate_episodes,
                                    analyze_stratum_records, build_call_plan)
from run_umi_task10_real_scenes import _load_preflight, _sha256_file, _write_json_atomic


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty metric table")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def _plot_results(root: Path, summaries: list[dict[str, Any]], image_rows: list[tuple[int, tuple[int, int], np.ndarray, np.ndarray, np.ndarray]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = root / "analysis" / "figures"
    plots.mkdir(parents=True, exist_ok=True)
    by_record = {index: [] for index in LOCKED_RECORDS}
    for row in summaries:
        by_record[row["record_index"]].append(row["delta_feedback_rmse"])
    figure, axis = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(LOCKED_RECORDS))
    for seed_slot in range(2):
        axis.scatter(positions + (seed_slot - .5) * .16,
                     [by_record[index][seed_slot] for index in LOCKED_RECORDS],
                     label=f"seed pair {SEED_PAIRS[seed_slot]}")
    axis.plot(positions, [np.mean(by_record[index]) for index in LOCKED_RECORDS], "k_", markersize=20,
              label="episode mean")
    axis.axhline(0, color="gray", linewidth=1)
    axis.set_xticks(positions, [str(index) for index in LOCKED_RECORDS])
    axis.set_xlabel("Bridge original record index")
    axis.set_ylabel("AR2 − TF2 endpoint RGB RMSE")
    axis.set_title("Feedback effect across six real trajectories")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plots / "episode_feedback_delta.png", dpi=160)
    figure.savefig(plots / "episode_feedback_delta.svg")
    plt.close(figure)
    figure, axes = plt.subplots(6, 6, figsize=(13, 14))
    ordered = sorted(image_rows, key=lambda row: (LOCKED_RECORDS.index(row[0]), SEED_PAIRS.index(row[1])))
    for row_index, (record, pair, truth, tf, ar) in enumerate(ordered):
        row = LOCKED_RECORDS.index(record)
        block = 0 if pair == SEED_PAIRS[0] else 3
        for column, (name, image) in enumerate((("Truth", truth), ("TF2", tf), ("AR2", ar))):
            axis = axes[row, block + column]
            axis.imshow(np.moveaxis(np.clip(image, 0, 1), 0, -1))
            axis.set_title(f"{record} {pair} {name}", fontsize=7)
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(plots / "real_scene_comparison.png", dpi=130)
    plt.close(figure)


def _manifest(root: Path) -> None:
    manifest = root / "MANIFEST.sha256"
    with manifest.open("w", encoding="utf-8") as stream:
        for path in sorted(item for item in root.rglob("*") if item.is_file()
                           and item != manifest and item.name != "review_bundle.zip"):
            stream.write(f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}\n")


def validate_saved_seed(record: dict[str, Any], expected_seed: int) -> None:
    generation = record.get("generation") or {}
    noise = generation.get("noise_evidence") or {}
    if (noise.get("seed") != expected_seed
            or noise.get("prepare_seed") != expected_seed
            or generation.get("sampler_generator_seeds") != [expected_seed] * 30):
        raise ValueError("actual saved seed evidence differs from locked call plan")


def analyze(root: Path) -> dict[str, Any]:
    status_path = root / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "GENERATION_COMPLETE" or status.get("completed_samples") != 48:
        raise ValueError("all 48 generation calls must complete before scientific analysis")
    identity = json.loads((root / "run_identity.json").read_text(encoding="utf-8"))
    if identity.get("record_indices") != list(LOCKED_RECORDS) or identity.get("seed_pairs") != [list(x) for x in SEED_PAIRS]:
        raise ValueError("run identity disagrees with preregistered episode/seed matrix")
    analysis_root = root / "analysis"
    if analysis_root.exists() and any(analysis_root.iterdir()):
        raise FileExistsError("analysis output already exists; use a new attempt directory")
    analysis_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    all_frames: list[dict[str, Any]] = []
    images: list[tuple[int, tuple[int, int], np.ndarray, np.ndarray, np.ndarray]] = []
    provenance: list[dict[str, Any]] = []
    for index in LOCKED_RECORDS:
        report, input_path = _load_preflight(root, index)
        batch = Task8InputAdapter.from_preflight(input_path, root / "preflight" / f"record_{index:02d}" / "task8_preflight.json")
        condition_path = root / "records" / f"record_{index:02d}" / "ground_truth_conditions.npz"
        receipt_path = condition_path.with_name("ground_truth_conditions.sha256.json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        evidence_path = condition_path.with_name("ground_truth_condition_evidence.json")
        if (_sha256_file(condition_path) != receipt.get("npz_sha256")
                or _sha256_file(evidence_path) != receipt.get("evidence_sha256")
                or receipt.get("record_index") != index
                or receipt.get("input_sha256") != _sha256_file(input_path)
                or receipt.get("vae_sha256") != identity["vae_sha256"]):
            raise ValueError("ground truth condition receipt hash mismatch")
        with np.load(condition_path, allow_pickle=False) as saved:
            gt16 = np.array(saved["gt_condition_x16"], copy=True)
            gt32 = np.array(saved["gt_condition_x32"], copy=True)
            mask = np.array(saved["condition_mask"], dtype=bool, copy=True)
        provenance.append({"record_index": index, "episode_id": report["selected"]["episode_id"],
                           "file_path": report["selected"]["file_path"],
                           "language": report["selected"]["language"],
                           "inputs_npz_sha256": _sha256_file(input_path),
                           "gt_condition_sha256": _sha256_file(condition_path)})
        for pair in SEED_PAIRS:
            formal = root / "records" / f"record_{index:02d}" / f"seeds_{pair[0]}_{pair[1]}" / "formal"
            formal_status = json.loads((formal / "run_status.json").read_text(encoding="utf-8"))
            if formal_status.get("status") != "COMPLETE" or formal_status.get("formal_calls") != 4:
                raise ValueError(f"record {index} seed pair {pair} formal status is incomplete")
            if formal_status.get("plan") != [dict(row) for row in build_call_plan(*pair)]:
                raise ValueError(f"record {index} seed pair {pair} has an incorrect call plan")
            binding = formal_status.get("binding") or {}
            if (binding.get("model") != identity["checkpoint_sha256"]
                    or binding.get("vae") != identity["vae_sha256"]
                    or binding.get("data") != _sha256_file(input_path)
                    or binding.get("actions") != _array_hash(batch.actions)
                    or binding.get("noise") != _canonical_hash({"policy": "prediction-region-seeded",
                          "seeds": [pair[0], pair[0], pair[1], pair[1]], "pair": "TF2-AR2"})):
                raise ValueError(f"record {index} seed pair {pair} has a mismatched source binding")
            store = Task8SampleStore(formal / "samples")
            records = {call: store.load_record(call) for call in ("G0", "G0_repeat", "TF2", "AR2")}
            for call in ("G0", "G0_repeat", "TF2", "AR2"):
                expected_action = batch.actions[0 if call.startswith("G0") else 1]
                if not np.array_equal(records[call]["action"], expected_action):
                    raise ValueError(f"record {index} seed pair {pair} {call} action differs from preflight")
                expected_seed = pair[0 if call.startswith("G0") else 1]
                validate_saved_seed(records[call], expected_seed)
            summary, frame_rows = analyze_stratum_records(records, batch.rgb, gt16, gt32, mask,
                                                           record_index=index, seed_pair=pair)
            summaries.append(summary)
            all_frames.extend(frame_rows)
            images.append((index, pair, np.array(batch.rgb[32], copy=True),
                           np.array(records["TF2"]["generated_rgb"][:, -1], copy=True),
                           np.array(records["AR2"]["generated_rgb"][:, -1], copy=True)))
            del records
        del batch, gt16, gt32, mask
    aggregate = aggregate_episodes(summaries)
    episode_means = np.array([row["mean_delta_feedback_rmse"] for row in aggregate["episodes"]], dtype=np.float64)
    rng = np.random.default_rng(20260925)
    boot = np.mean(rng.choice(episode_means, size=(10000, len(episode_means)), replace=True), axis=1)
    aggregate["episode_mean_delta_feedback_rmse"] = float(np.mean(episode_means))
    aggregate["exploratory_cluster_bootstrap_95pct_mean_interval"] = [float(x) for x in np.quantile(boot, [.025, .975])]
    aggregate["bootstrap_seed"] = 20260925
    aggregate["inference_unit"] = "episode; two seed schedules averaged within episode"
    aggregate["scope"] = "six preregistered Bridge train-shard trajectories; observed-future-pose motion conditions"
    _write_json_atomic(analysis_root / "comparison.json", _json_safe(aggregate))
    _write_json_atomic(analysis_root / "provenance.json", {"episodes": provenance})
    _write_csv(analysis_root / "stratum_metrics.csv", [_json_safe(row) for row in summaries])
    _write_csv(analysis_root / "per_frame_metrics.csv", all_frames)
    _plot_results(root, summaries, images)
    report_lines = ["# Task 10: Real-scene paired feedback error reproduction", "",
                    "Six preregistered Bridge trajectories; two independent paired seed schedules each.",
                    "Actions contain future observed poses. This is not control-only open-loop prediction.", "",
                    f"Episode-level mean Δfeedback RMSE: {aggregate['episode_mean_delta_feedback_rmse']:+.8f}.",
                    f"Episode median: {aggregate['median_episode_delta_feedback_rmse']:+.8f}.",
                    f"Positive/negative/zero episode means: {aggregate['positive_episode_count']}/"
                    f"{aggregate['negative_episode_count']}/{aggregate['zero_episode_count']}.",
                    "Exploratory cluster bootstrap 95% interval for episode mean: "
                    f"{aggregate['exploratory_cluster_bootstrap_95pct_mean_interval']}. ",
                    "Six episodes do not justify a population-level or causal claim.", "",
                    "Primary metric is the second chunk's final frame only; the 16 per-frame rows are secondary.",
                    "Float32 RGB and FP32 VAE condition latents are quantitative; PNG is display-only.", "",
                    "| Record | Mean Δfeedback RGB RMSE | Seed-pair deltas |", "|---:|---:|---|" ]
    for row in aggregate["episodes"]:
        report_lines.append(f"| {row['record_index']} | {row['mean_delta_feedback_rmse']:+.8f} | "
                            f"{[round(x, 8) for x in row['seed_pair_deltas']]} |")
    report_lines += ["", "The feedback effect is a paired comparison under the same a1 and second-chunk noise.",
                     "It is not a model-only error decomposition, a Lipschitz bound, or a long-horizon stability result.", ""]
    (analysis_root / "experiment_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    _manifest(root)
    bundle = analysis_root / "review_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [root / "run_identity.json", root / "run_status.json", root / "MANIFEST.sha256"]:
            archive.write(path, path.relative_to(root).as_posix())
        for path in analysis_root.rglob("*"):
            if path.is_file() and path != bundle:
                archive.write(path, path.relative_to(root).as_posix())
        for path in Path(__file__).resolve().parent.glob("*task10*.py"):
            archive.write(path, f"code/{path.name}")
        for path in sorted((Path(__file__).resolve().parent / "task8_frozen").glob("*.py")):
            archive.write(path, f"code/task8_frozen/{path.name}")
        source = Path(__file__).with_name("umi_task9_bridge_preflight.py")
        archive.write(source, f"code/{source.name}")
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args(argv)
    result = analyze(Path(args.run_root).resolve())
    print(json.dumps({"status": "ANALYSIS_COMPLETE", "episode_count": result["episode_count"],
                      "episode_mean_delta_feedback_rmse": result["episode_mean_delta_feedback_rmse"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
