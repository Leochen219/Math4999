#!/usr/bin/env python3
"""Export only the reviewed Task 11 decomposition scalars for the demo."""

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_SHA256 = "9290a21548f9de2a6131a12b3b521d56ef90ba536b91dd72653ec6e762350602"
PUBLIC_REVISION = "8f41d976ae699eae5d10392823d3e0f060196968"
PUBLIC_SOURCE = (
    "https://github.com/Leochen219/Math4999/blob/"
    f"{PUBLIC_REVISION}/cosmos_umi_capstone/results/task11_horizon_error_run01/"
    "horizon_figures/figure_source_data.json"
)


def build_fixture(source_path: Path) -> dict:
    raw = source_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"source SHA-256 mismatch: {digest}")
    source = json.loads(raw)
    rows = []
    for metric in ("rgb", "condition_masked_latent"):
        for item in source["decomposition_rows"][metric]:
            row = {
                "id": f"{metric}-schedule{item['schedule_index']}-h{item['horizon_chunks']}",
                "metric": metric,
                "scheduleIndex": item["schedule_index"],
                "seed": item["seed"],
                "horizonChunks": item["horizon_chunks"],
                "tfSquaredError": item["tf_squared_error"],
                "arSquaredError": item["ar_squared_error"],
                "feedbackChangeSquared": item["feedback_change_squared"],
                "feedbackChangeRmse": item["feedback_change_rmse"],
                "errorFeedbackCosine": item["error_feedback_cosine"],
                "crossTerm2DotOverN": item["cross_term_2dot_over_n"],
                "squaredErrorDifference": item["squared_error_difference"],
                "identityResidual": item["identity_residual"],
            }
            if "masked_element_count" in item:
                row["maskedElementCount"] = item["masked_element_count"]
            rows.append(row)

    if len(rows) != 16:
        raise ValueError(f"expected 16 scalar rows, found {len(rows)}")
    scope = source["scope"]
    return {
        "schema": "bounded-rollout-lab-task11-scalars-v1",
        "provenance": {
            "source": PUBLIC_SOURCE,
            "sourceRevision": PUBLIC_REVISION,
            "sourceSha256": digest,
            "sourceRecordIndex": source["source_record_index"],
            "sourceRunIdentitySha256": source["source_run_identity_sha256"],
            "sourceAnalysisSha256": source["source_analysis_sha256"],
        },
        "scope": {
            "episodeCount": scope["episode_count"],
            "seedScheduleCount": scope["seed_schedule_count"],
            "horizons": "five observed generated-chunk endpoints; this export contains h2-h5 only",
            "futurePoseConditioning": "future observed poses enter Bridge motion conditions",
            "metrics": {
                "rgb": "RGB endpoint squared error decomposition",
                "condition_masked_latent": "condition-mask latent endpoint squared error decomposition",
            },
            "precisionMetadata": "unknown",
            "claim": scope["claim"],
            "mseIdentity": scope["mse_identity"],
        },
        "rows": rows,
    }


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=here.parent / "cosmos_umi_capstone/results/task11_horizon_error_run01/horizon_figures/figure_source_data.json",
        help="path to the public source figure JSON in the repository",
    )
    parser.add_argument("--output", type=Path, default=here / "data/task11-scalars.json")
    args = parser.parse_args()
    fixture = build_fixture(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(fixture['rows'])} reviewed scalar rows to {args.output}")


if __name__ == "__main__":
    main()
