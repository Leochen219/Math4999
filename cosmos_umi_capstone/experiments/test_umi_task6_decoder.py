from __future__ import annotations

import unittest
from types import SimpleNamespace
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

import umi_task6_decoder as api


class DecoderTests(unittest.TestCase):
    def _raw(self, root):
        ids = [item["sample_id"] for item in api.select_decoder_latents()]
        for sample_id in ids:
            sample = root / "samples" / sample_id; sample.mkdir(parents=True)
            np.save(sample / "output_full.npy", np.zeros((3, 2, 2, 2), np.float32), allow_pickle=False)
            digest = hashlib.sha256((sample / "output_full.npy").read_bytes()).hexdigest()
            (sample / "sample.json").write_text(json.dumps({"sample_id": sample_id}), encoding="utf-8")
            (sample / "status.json").write_text(json.dumps({"status": "success", "artifact_sha256": {"output_full.npy": digest}}), encoding="utf-8")
        completed = [f"sample_{i:02d}" for i in range(32)]
        (root / "run_status.json").write_text(json.dumps({"status": "AWAITING_REVIEW", "completed_samples": completed, "group": {"state": "bridge_0", "seed": 0}}), encoding="utf-8")
        entries = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "MANIFEST.sha256": entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}\n")
        (root / "MANIFEST.sha256").write_text("".join(entries), encoding="ascii")

    def test_exactly_eight_latents_and_sixteen_serial_replays(self):
        logical = api.select_decoder_latents()
        plan = api.decoder_replay_plan()
        self.assertEqual(len(logical), 8)
        self.assertEqual(len(plan), 16)
        self.assertEqual({x["sample_id"] for x in logical}, {x["sample_id"] for x in plan})
        self.assertEqual({x["decode_precision"] for x in plan}, {"native_bf16", "temporary_fp32"})

    def test_uint8_simulation_does_not_round_trip_through_png(self):
        value = np.array([[0.001, 0.499, 1.2]], dtype=np.float32)
        result = api.simulate_uint8(value)
        np.testing.assert_array_equal(result, np.round(np.clip(value, 0, 1) * 255) / 255)

    def test_replay_restores_decoder_state_on_failure(self):
        class Runtime:
            decoder_state = "bf16"
            def decode_prediction_latent(self, latent, *, precision):
                self.decoder_state = precision
                raise RuntimeError("decode failed")
            def restore_decoder_state(self): self.decoder_state = "bf16"
        runtime = Runtime()
        with self.assertRaises(RuntimeError): api.replay_one(runtime, np.zeros((1,)), precision="temporary_fp32")
        self.assertEqual(runtime.decoder_state, "bf16")

    def test_replay_returns_float_final_frame(self):
        class Runtime:
            decoder_state = "bf16"
            def decode_prediction_latent(self, latent, *, precision):
                return np.zeros((3, 2, 4, 4), dtype=np.float32) + .25
            def restore_decoder_state(self): pass
        result = api.replay_one(Runtime(), np.zeros((2,)), precision="native_bf16")
        self.assertEqual(result["decoded_final_float32"].shape, (3, 4, 4))
        self.assertEqual(result["decoded_full_float32"].dtype, np.float32)

    def test_temporary_precision_state_is_restored_after_success(self):
        class Runtime:
            decoder_state = "bf16"
            def decode_prediction_latent(self, latent, *, precision):
                self.decoder_state = "fp32" if precision == "temporary_fp32" else "bf16"
                return np.zeros((3, 2, 2, 2), np.float32)
            def restore_decoder_state(self): self.decoder_state = "bf16"
        runtime = Runtime(); result = api.replay_one(runtime, np.zeros((2,)), precision="temporary_fp32")
        self.assertEqual(runtime.decoder_state, "bf16"); self.assertEqual(result["decoder_state_after"], "bf16")

    def test_end_to_end_is_serial_and_does_not_mutate_raw_tree(self):
        class Runtime:
            decoder_state = "bf16"
            active = 0
            maximum = 0
            def decode_prediction_latent(self, latent, *, precision):
                self.active += 1; self.maximum = max(self.maximum, self.active)
                value = np.zeros((3, 2, 2, 2), np.float32) + (0.1 if precision == "native_bf16" else 0.2)
                self.active -= 1; return value
            def restore_decoder_state(self): pass
        class Encoder:
            def __call__(self, frame): return np.asarray(frame, dtype=np.float32).mean(keepdims=True)
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"; self._raw(raw)
            before = {p.relative_to(raw): (p.read_bytes(), p.stat().st_mtime_ns) for p in raw.rglob("*") if p.is_file()}
            decoder = Path(temporary) / "decoder"
            runtime = Runtime(); result = api.run_task6_decoder_replays(runtime, raw, encoder=Encoder(), decoder_root=decoder)
            self.assertEqual(result["status"], "COMPLETE"); self.assertEqual(result["decoder_calls"], 16); self.assertEqual(runtime.maximum, 1)
            after = {p.relative_to(raw): (p.read_bytes(), p.stat().st_mtime_ns) for p in raw.rglob("*") if p.is_file()}
            self.assertEqual(before, after); self.assertTrue((decoder / "MANIFEST.sha256").is_file())

    def test_encoder_is_mandatory(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(Exception): api.run_task6_decoder_replays(object(), Path(temporary) / "raw")

    def test_resume_rejects_tampered_success(self):
        class Runtime:
            decoder_state = "bf16"
            def decode_prediction_latent(self, latent, *, precision): return np.zeros((3, 2, 2, 2), np.float32)
            def restore_decoder_state(self): pass
        class Encoder:
            def __call__(self, frame): return np.asarray(frame, np.float32).mean(keepdims=True)
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"; self._raw(raw); decoder = Path(temporary) / "decoder"
            api.run_task6_decoder_replays(Runtime(), raw, encoder=Encoder(), decoder_root=decoder)
            target = next(decoder.glob("*__native_bf16/decoded_final_float32.npy")); target.write_bytes(b"tampered")
            with self.assertRaises(Exception): api.run_task6_decoder_replays(Runtime(), raw, encoder=Encoder(), decoder_root=decoder, resume=True)


if __name__ == "__main__": unittest.main()
