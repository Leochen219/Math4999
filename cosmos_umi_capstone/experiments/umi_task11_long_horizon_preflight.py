"""Fixed-record, five-action-chunk CPU preflight for Task 11.

This is an additive extension. It reuses Task 9's pinned TFRecord parser and
official BridgeOrig action conversion without changing any Task 4--10
defaults. No Torch, model, or network work is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from .umi_task11_long_horizon import (CHUNK_LENGTH, HORIZON_CHUNKS, OBSERVATION_COUNT,
                                          RECORD_INDEX, TRANSITION_COUNT, HorizonTrajectory,
                                          file_sha256)
    from . import umi_task9_bridge_preflight as task9
except ImportError:  # direct invocation from experiments/
    from umi_task11_long_horizon import (CHUNK_LENGTH, HORIZON_CHUNKS, OBSERVATION_COUNT,
                                         RECORD_INDEX, TRANSITION_COUNT, HorizonTrajectory,
                                         file_sha256)
    import umi_task9_bridge_preflight as task9


EXPECTED_RECORD_COUNT = 52
EXPECTED_RECORD15_FRAME_COUNT = 89
EXPECTED_SHARD_SHA256 = "04e67b7ba8c66a1990d72f9fb2d0861d1203623c74d9d2c4d4678f928ba4f579"
NOMINAL_FPS = 5.0
PRIMARY_IMAGE_KEY = "steps/observation/image_0"


class LongHorizonPreflightError(ValueError):
    """Raised when the fixed Task 11 source window is not admissible."""


def _sha_array_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def build_task11_trajectory(episode: Any, stats: Any) -> tuple[HorizonTrajectory, dict[str, Any]]:
    """Validate record 15 and construct its first 81 real frames/80 actions."""
    if getattr(episode, "index", None) != RECORD_INDEX:
        raise LongHorizonPreflightError("Task 11 is locked to fixed record 15")
    states = np.asarray(getattr(episode, "states", None))
    actions = np.asarray(getattr(episode, "actions", None))
    if states.dtype != np.float32 or actions.dtype != np.float32:
        raise LongHorizonPreflightError("Bridge state/action payloads must be float32")
    if states.ndim != 2 or states.shape[1:] != (7,) or actions.shape != states.shape:
        raise LongHorizonPreflightError("state/action rows must be temporally aligned [T,7]")
    if len(states) != EXPECTED_RECORD15_FRAME_COUNT:
        raise LongHorizonPreflightError(f"fixed record 15 must have 89 observations, got {len(states)}")
    if len(states) < OBSERVATION_COUNT or not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise LongHorizonPreflightError("record 15 lacks 81 finite observations and aligned actions")
    if getattr(episode, "has_language", False) is not True:
        raise LongHorizonPreflightError("record 15 has no language metadata")
    language = task9._language_value(episode)

    first = np.flatnonzero(np.asarray(episode.is_first))
    last = np.flatnonzero(np.asarray(episode.is_last))
    if first.tolist() != [0] or last.tolist() != [len(states) - 1]:
        raise LongHorizonPreflightError("record 15 first/last sequence flags do not identify one complete episode")
    if len(episode.image0) != len(states) or any(not isinstance(blob, bytes) or not blob for blob in episode.image0):
        raise LongHorizonPreflightError("primary image_0 payload is not aligned with all state observations")
    image_counts = {name: len(values) for name, values in episode.images.items()}
    if not image_counts or any(count != len(states) for count in image_counts.values()):
        raise LongHorizonPreflightError("camera payload sequences are not aligned with state/action frames")

    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - dependency diagnostic
        raise LongHorizonPreflightError("Pillow is required to decode the official image_0 stream") from error
    rgb_frames: list[np.ndarray] = []
    frame_proofs: list[dict[str, Any]] = []
    for index, encoded in enumerate(episode.image0[:OBSERVATION_COUNT]):
        proof = task9.jpeg_content_proof(encoded)
        if proof.get("shape") != [256, 256, 3] or int(proof.get("pixel_max", 0)) <= int(proof.get("pixel_min", 0)):
            raise LongHorizonPreflightError(f"image_0 frame {index} is not nonconstant RGB 256x256")
        with Image.open(io.BytesIO(encoded)) as image:
            image.load()
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)
        rgb_frames.append(rgb.astype(np.float32) / np.float32(255.0))
        frame_proofs.append(proof)
    rgb_float32 = np.ascontiguousarray(np.stack(rgb_frames), dtype=np.float32)
    if rgb_float32.shape != (OBSERVATION_COUNT, 3, 256, 256):
        raise LongHorizonPreflightError("decoded Task 11 RGB tensor has unexpected geometry")

    adapter = task9.BridgeOrigAdapter()
    raw_chunks: list[np.ndarray] = []
    normalized_chunks: list[np.ndarray] = []
    action_evidence: list[dict[str, Any]] = []
    for chunk_index, start in enumerate(range(0, TRANSITION_COUNT, CHUNK_LENGTH)):
        raw, _initial_pose = adapter.build_raw_action(
            states[start : start + CHUNK_LENGTH + 1], actions[start : start + CHUNK_LENGTH]
        )
        normalized = task9.normalize_quantile(raw, stats.q01, stats.q99)
        if raw.shape != (CHUNK_LENGTH, 10) or normalized.shape != (CHUNK_LENGTH, 10):
            raise LongHorizonPreflightError(f"official action chunk {chunk_index} is not [16,10]")
        if raw.dtype != np.float32 or normalized.dtype != np.float32:
            raise LongHorizonPreflightError("official action adapter changed float32 representation")
        if not np.isfinite(raw).all() or not np.isfinite(normalized).all():
            raise LongHorizonPreflightError(f"official action chunk {chunk_index} is non-finite")
        raw_chunks.append(raw)
        normalized_chunks.append(normalized)
        action_evidence.append({
            "chunk_index": chunk_index,
            "start_frame": start,
            "end_transition_exclusive": start + CHUNK_LENGTH,
            "future_pose_frame": start + CHUNK_LENGTH,
            "raw_sha256": _sha_array_bytes(raw),
            "normalized_sha256": _sha_array_bytes(normalized),
            "raw_min": raw.min(axis=0).astype(float).tolist(),
            "raw_max": raw.max(axis=0).astype(float).tolist(),
            "normalized_min": float(normalized.min()),
            "normalized_max": float(normalized.max()),
        })
    raw_array = np.ascontiguousarray(np.stack(raw_chunks), dtype=np.float32)
    normalized_array = np.ascontiguousarray(np.stack(normalized_chunks), dtype=np.float32)
    trajectory = HorizonTrajectory(
        record_index=RECORD_INDEX,
        episode_id=str(getattr(episode, "episode_id", "")),
        rgb=rgb_float32,
        states=np.ascontiguousarray(states[:OBSERVATION_COUNT], dtype=np.float32),
        original_actions=np.ascontiguousarray(actions[:TRANSITION_COUNT], dtype=np.float32),
        raw_actions=raw_array,
        normalized_actions=normalized_array,
        prompt=language,
    )
    evidence = {
        "record_index": RECORD_INDEX,
        "episode_id": episode.episode_id,
        "source_file_path": episode.file_path,
        "source_frame_count": len(states),
        "frames_used": OBSERVATION_COUNT,
        "transitions_used": TRANSITION_COUNT,
        "language": language,
        "camera": {
            "feature_key": PRIMARY_IMAGE_KEY,
            "official_mapping": "Octo Bridge config: image_primary=image_0; POS_EULER actions",
            "metadata_flag": bool(episode.has_image_flags.get("image_0", episode.has_image0)),
            "metadata_payload_contradiction": bool(not episode.has_image_flags.get("image_0", episode.has_image0)),
            "decoded_shapes": sorted({tuple(row["shape"]) for row in frame_proofs}),
            "unique_encoded_frames": len({row["sha256"] for row in frame_proofs}),
            "all_frames_rgb_256x256": all(row["shape"] == [256, 256, 3] for row in frame_proofs),
            "all_frames_nonconstant": all(row["pixel_max"] > row["pixel_min"] for row in frame_proofs),
            "first_81_encoded_jpeg_sha256": [row["sha256"] for row in frame_proofs],
            "rgb_float32_sha256": _sha_array_bytes(rgb_float32),
        },
        "temporal_alignment": {
            "frame_indexes": list(range(OBSERVATION_COUNT)),
            "state_action_sequence_aligned": True,
            "is_first_indexes": first.tolist(),
            "is_last_indexes": last.tolist(),
            "timestamp_feature_present": False,
            "nominal_fps": NOMINAL_FPS,
            "sample_period_seconds_nominal_inferred": 1.0 / NOMINAL_FPS,
            "wall_clock_alignment_proven": False,
            "action_at_t_uses_future_observed_pose_t_plus_1": True,
        },
        "normalizer": {"sha256": stats.sha256, "dimension": 10,
                       "formula": "2*(raw-q01)/max(q99-q01,1e-8)-1"},
        "action_chunks": action_evidence,
        "action_provenance": {
            "adapter": "pinned BridgeOrigLeRobotDataset backward_framewise / rot6d",
            "state_components": "translation state[:3], Euler XYZ state[3:6], no rescale",
            "transforms": ["DEFAULT_ROTATION", "TCP_TO_FLANGE", "BRIDGE_TO_OPENCV"],
            "gripper_source": "original action[:,6]",
            "chunk_count": HORIZON_CHUNKS,
            "raw_action_sha256": _sha_array_bytes(raw_array),
            "normalized_action_sha256": _sha_array_bytes(normalized_array),
        },
        "warnings": (["episode_metadata/has_image_0 is false but image_0 payload decodes as the pinned official primary stream"]
                     if not episode.has_image_flags.get("image_0", episode.has_image0) else []),
    }
    return trajectory, evidence


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_preflight_bundle(trajectory: HorizonTrajectory, evidence: Mapping[str, Any],
                           output_dir: str | Path, *, source_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically publish immutable CPU input arrays and admission evidence."""
    root = Path(output_dir).resolve()
    npz_path, report_path = root / "task11_trajectory.npz", root / "task11_preflight.json"
    if npz_path.exists() or report_path.exists():
        raise FileExistsError("Task 11 preflight artifacts are immutable; choose a new output directory")
    _atomic_npz(npz_path, {
        "rgb_float32": trajectory.rgb,
        "states_float32": trajectory.states,
        "actions_original": trajectory.original_actions,
        "actions_raw": trajectory.raw_actions,
        "actions_normalized": trajectory.normalized_actions,
    })
    report = {
        "status": "PREFLIGHT_PASSED_GENERATION_NOT_PERFORMED",
        "generation_performed": False,
        "record_index": RECORD_INDEX,
        "source_identity": dict(source_identity),
        "selected": dict(evidence),
        "trajectory_npz": {
            "path": str(npz_path), "sha256": file_sha256(npz_path),
            "arrays": {
                "rgb_float32": list(trajectory.rgb.shape),
                "states_float32": list(trajectory.states.shape),
                "actions_original": list(trajectory.original_actions.shape),
                "actions_raw": list(trajectory.raw_actions.shape),
                "actions_normalized": list(trajectory.normalized_actions.shape),
            },
        },
    }
    _atomic_json(report_path, report)
    report["report_sha256"] = file_sha256(report_path)
    return report


def load_task11_trajectory(npz_path: str | Path, report_path: str | Path) -> HorizonTrajectory:
    """Load only a complete, hash-bound Task 11 preflight bundle."""
    path, metadata_path = Path(npz_path).resolve(), Path(report_path).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "PREFLIGHT_PASSED_GENERATION_NOT_PERFORMED" or metadata.get("generation_performed") is not False:
        raise LongHorizonPreflightError("Task 11 preflight did not pass without generation")
    if metadata.get("record_index") != RECORD_INDEX:
        raise LongHorizonPreflightError("Task 11 preflight record identity is not fixed record 15")
    bundle = metadata.get("trajectory_npz")
    if not isinstance(bundle, Mapping) or bundle.get("sha256") != file_sha256(path):
        raise LongHorizonPreflightError("Task 11 trajectory NPZ hash differs from the preflight report")
    with np.load(path, allow_pickle=False) as arrays:
        expected = {"rgb_float32", "states_float32", "actions_original", "actions_raw", "actions_normalized"}
        if set(arrays.files) != expected:
            raise LongHorizonPreflightError("Task 11 trajectory NPZ has missing or unexpected arrays")
        return HorizonTrajectory(
            record_index=RECORD_INDEX,
            episode_id=str(metadata["selected"]["episode_id"]),
            rgb=np.asarray(arrays["rgb_float32"]),
            states=np.asarray(arrays["states_float32"]),
            original_actions=np.asarray(arrays["actions_original"]),
            raw_actions=np.asarray(arrays["actions_raw"]),
            normalized_actions=np.asarray(arrays["actions_normalized"]),
            prompt=str(metadata["selected"]["language"]),
        )


def run_long_horizon_preflight(dataset_root: str | Path, output_dir: str | Path,
                               normalizer_stats_path: str | Path,
                               official_parity_report: str | Path) -> dict[str, Any]:
    """Verify transfer/hash/CRC evidence and publish the fixed real window."""
    root = Path(dataset_root).resolve()
    shards = sorted(root.glob("*.tfrecord*"))
    if len(shards) != 1:
        raise LongHorizonPreflightError(f"expected exactly one TFRecord shard, got {len(shards)}")
    shard = shards[0]
    status_path = root / "download_status.json"
    if not status_path.is_file():
        raise LongHorizonPreflightError("download_status.json is required for the COMPLETE transfer gate")
    try:
        transfer = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LongHorizonPreflightError("download_status.json is invalid") from error
    if not isinstance(transfer, Mapping) or transfer.get("status") != "COMPLETE":
        raise LongHorizonPreflightError("dataset transfer status must be exactly COMPLETE")
    manifest = task9._verify_manifest(root)
    if not manifest.get("all_match") or not manifest.get("files"):
        raise LongHorizonPreflightError("every file listed by MANIFEST.sha256 must match")
    shard_file = manifest["files"].get(shard.name)
    if not isinstance(shard_file, Mapping) or shard_file.get("match") is not True:
        raise LongHorizonPreflightError("the selected TFRecord shard is not hash-bound by MANIFEST.sha256")
    if shard_file.get("actual_sha256") != EXPECTED_SHARD_SHA256:
        raise LongHorizonPreflightError("the selected TFRecord shard differs from the pinned Task 10 subset")
    stats = task9.load_normalizer_stats(normalizer_stats_path)
    parity_path = Path(official_parity_report).resolve()
    try:
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LongHorizonPreflightError("pinned official CPU parity report is missing or invalid") from error
    normalization = parity.get("normalization") if isinstance(parity, Mapping) else None
    if (not isinstance(parity, Mapping) or parity.get("status") != "PASS"
            or parity.get("action_exact_equal") is not True
            or parity.get("initial_pose_exact_equal") is not True
            or not isinstance(normalization, Mapping)
            or normalization.get("saved_exact_equal") is not True):
        raise LongHorizonPreflightError("pinned official action/normalizer parity report did not pass")
    episode = None
    record_summaries: list[dict[str, Any]] = []
    for record in task9.iter_tfrecord_records(shard, verify_crc=True):
        decoded = task9.decode_bridge_episode(record)
        record_summaries.append({"record_index": decoded.index, "frame_count": len(decoded.states),
                                 "state_action_aligned": len(decoded.states) == len(decoded.actions)})
        if decoded.index == RECORD_INDEX:
            episode = decoded
    indexes = [row["record_index"] for row in record_summaries]
    if len(indexes) != EXPECTED_RECORD_COUNT or indexes != list(range(EXPECTED_RECORD_COUNT)):
        raise LongHorizonPreflightError(f"full shard CRC scan must prove records 0..51, got {len(indexes)}")
    if not all(row["state_action_aligned"] for row in record_summaries):
        raise LongHorizonPreflightError("one or more CRC-checked records have misaligned state/action rows")
    if episode is None:
        raise LongHorizonPreflightError("fixed record 15 is absent after full shard CRC scan")
    trajectory, evidence = build_task11_trajectory(episode, stats)
    evidence["full_shard_scan"] = {"record_count": len(record_summaries), "crc_checked": True,
                                    "record_indexes": indexes,
                                    "all_state_action_aligned": all(row["state_action_aligned"] for row in record_summaries)}
    evidence["shard"] = {"name": shard.name, "bytes": shard.stat().st_size,
                         "sha256": file_sha256(shard), "manifest_sha256": shard_file["actual_sha256"]}
    evidence["transfer"] = {"status": transfer["status"], "status_file": str(status_path),
                            "status_file_sha256": file_sha256(status_path)}
    source_identity = {
        "dataset_root": str(root), "manifest_sha256": file_sha256(root / "MANIFEST.sha256"),
        "normalizer_path": str(Path(normalizer_stats_path).resolve()), "normalizer_sha256": stats.sha256,
        "official_parity_path": str(parity_path), "official_parity_sha256": file_sha256(parity_path),
        "official_parity_status": parity["status"],
        "source_parser": "umi_task9_bridge_preflight.iter_tfrecord_records + decode_bridge_episode",
        "record_count": len(record_summaries), "crc_checked": True,
    }
    return write_preflight_bundle(trajectory, evidence, output_dir, source_identity=source_identity)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--normalizer-stats", required=True)
    parser.add_argument("--official-parity-report", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_long_horizon_preflight(args.dataset_root, args.output_dir, args.normalizer_stats,
                                        args.official_parity_report)
    print(json.dumps({"status": report["status"], "record_index": report["record_index"],
                      "trajectory_npz": report["trajectory_npz"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
