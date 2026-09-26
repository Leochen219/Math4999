"""CPU tests for Task 11's separate 81-frame real-trajectory preflight."""
from __future__ import annotations

import io
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from umi_task11_long_horizon_preflight import (LongHorizonPreflightError,
                                               build_task11_trajectory,
                                               load_task11_trajectory,
                                               file_sha256,
                                               write_preflight_bundle)
from umi_task9_bridge_preflight import BridgeEpisode, NormalizerStats
from umi_task11_long_horizon import HorizonTrajectory


def _episode(*, index: int = 15, frame_count: int = 89, first: int = 0,
             last: int | None = None, camera_flag: bool = False) -> BridgeEpisode:
    from PIL import Image

    last = frame_count - 1 if last is None else last
    states = np.zeros((frame_count, 7), dtype=np.float32)
    states[:, 0] = np.arange(frame_count, dtype=np.float32) * 0.001
    actions = np.zeros((frame_count, 7), dtype=np.float32)
    frames = []
    for frame in range(frame_count):
        pixels = np.zeros((256, 256, 3), dtype=np.uint8)
        pixels[..., 0] = np.arange(256, dtype=np.uint8)[None, :]
        pixels[..., 1] = np.arange(256, dtype=np.uint8)[:, None]
        pixels[..., 2] = frame
        stream = io.BytesIO()
        Image.fromarray(pixels, "RGB").save(stream, format="JPEG", quality=95)
        frames.append(stream.getvalue())
    return BridgeEpisode(index=index, episode_id=11, states=states, actions=actions,
                         image0=frames, images={name: list(frames) for name in
                             ("image_0", "image_1", "image_2", "image_3")},
                         language=["sweep into pile"] * frame_count, has_image0=camera_flag,
                         has_image_flags={"image_0": camera_flag, "image_1": True,
                                          "image_2": True, "image_3": True},
                         has_language=True,
                         is_first=np.asarray([int(i == first) for i in range(frame_count)], dtype=np.int64),
                         is_last=np.asarray([int(i == last) for i in range(frame_count)], dtype=np.int64),
                         is_terminal=np.zeros(frame_count, dtype=np.int64), file_path="fixture/out.npy",
                         extra_keys=())


class LongHorizonPreflightTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("scipy"),
                         "pinned official Euler transform requires the already-installed SciPy runtime")
    def test_builds_five_official_chunks_from_first_81_real_frames_and_keeps_flag_conflict(self):
        stats = NormalizerStats(np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32), "c" * 64)
        trajectory, evidence = build_task11_trajectory(_episode(), stats)
        self.assertEqual(trajectory.record_index, 15)
        self.assertEqual(trajectory.rgb.shape, (81, 3, 256, 256))
        self.assertEqual(trajectory.states.shape, (81, 7))
        self.assertEqual(trajectory.original_actions.shape, (80, 7))
        self.assertEqual(trajectory.raw_actions.shape, (5, 16, 10))
        self.assertEqual(trajectory.normalized_actions.shape, (5, 16, 10))
        self.assertTrue(evidence["camera"]["metadata_payload_contradiction"])
        self.assertEqual(evidence["camera"]["feature_key"], "steps/observation/image_0")
        self.assertEqual(evidence["temporal_alignment"]["timestamp_feature_present"], False)
        self.assertEqual(evidence["temporal_alignment"]["frame_indexes"], list(range(81)))
        self.assertEqual([row["start_frame"] for row in evidence["action_chunks"]], [0, 16, 32, 48, 64])
        self.assertEqual([row["end_transition_exclusive"] for row in evidence["action_chunks"]],
                         [16, 32, 48, 64, 80])

    def test_rejects_nonfixed_record_short_window_or_misaligned_episode_boundary(self):
        stats = NormalizerStats(np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32), "c" * 64)
        with self.assertRaisesRegex(LongHorizonPreflightError, "fixed record 15"):
            build_task11_trajectory(_episode(index=14), stats)
        with self.assertRaisesRegex(LongHorizonPreflightError, "89"):
            build_task11_trajectory(_episode(frame_count=80), stats)
        with self.assertRaisesRegex(LongHorizonPreflightError, "first/last"):
            build_task11_trajectory(_episode(last=80), stats)

    def test_preflight_npz_is_hash_bound_and_immutable(self):
        rgb = np.broadcast_to(np.arange(81, dtype=np.float32)[:, None, None, None], (81, 3, 2, 2)).copy() / 80
        trajectory = HorizonTrajectory(record_index=15, episode_id="11", rgb=rgb,
            states=np.zeros((81, 7), np.float32), original_actions=np.zeros((80, 7), np.float32),
            raw_actions=np.zeros((5, 16, 10), np.float32), normalized_actions=np.zeros((5, 16, 10), np.float32),
            prompt="sweep into pile")
        evidence = {"episode_id": 11, "language": "sweep into pile"}
        with tempfile.TemporaryDirectory() as temporary:
            report = write_preflight_bundle(trajectory, evidence, temporary, source_identity={"shard": "a" * 64})
            loaded = load_task11_trajectory(report["trajectory_npz"]["path"], Path(temporary) / "task11_preflight.json")
            np.testing.assert_array_equal(loaded.rgb, trajectory.rgb)
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                write_preflight_bundle(trajectory, evidence, temporary, source_identity={"shard": "a" * 64})
            npz_path = Path(report["trajectory_npz"]["path"])
            with np.load(npz_path, allow_pickle=False) as saved:
                arrays = {name: saved[name] for name in saved.files}
            arrays["states_float32"] = arrays["states_float32"].astype(np.float64)
            np.savez_compressed(npz_path, **arrays)
            report_path = Path(temporary) / "task11_preflight.json"
            metadata = json.loads(report_path.read_text(encoding="utf-8"))
            metadata["trajectory_npz"]["sha256"] = file_sha256(npz_path)
            report_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "float32"):
                load_task11_trajectory(npz_path, report_path)
            with npz_path.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(LongHorizonPreflightError, "hash"):
                load_task11_trajectory(npz_path, report_path)


if __name__ == "__main__":
    unittest.main()
