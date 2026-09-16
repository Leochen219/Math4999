from __future__ import annotations

import unittest
from types import SimpleNamespace
import hashlib
import json
import tempfile
from unittest import mock
from pathlib import Path

import numpy as np

import umi_task6_decoder as api


class DecoderTests(unittest.TestCase):
    def _raw(self, root):
        ids = [item["sample_id"] for item in api.build_generation_plan()]
        for sample_id in ids:
            sample = root / "samples" / sample_id; sample.mkdir(parents=True)
            np.save(sample / "output_full.npy", np.zeros((3, 2, 2, 2), np.float32), allow_pickle=False)
            np.save(sample / "predicted_latent.npy", np.zeros((1, 48, 4, 16, 16), np.float32), allow_pickle=False)
            digest = hashlib.sha256((sample / "output_full.npy").read_bytes()).hexdigest(); predicted_digest = hashlib.sha256((sample / "predicted_latent.npy").read_bytes()).hexdigest()
            (sample / "sample.json").write_text(json.dumps({"sample_id": sample_id}), encoding="utf-8")
            (sample / "status.json").write_text(json.dumps({"status": "success", "artifact_sha256": {"output_full.npy": digest, "predicted_latent.npy": predicted_digest}}), encoding="utf-8")
        completed = ids
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
            def actual_identity(self): return {"model_state": "runtime-v1", "decoder_state": "bf16"}
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
            def actual_identity(self): return {"model_state": "runtime-v1", "decoder_state": "bf16"}
            def decode_prediction_latent(self, latent, *, precision):
                return np.zeros((3, 2, 4, 4), dtype=np.float32) + .25
            def restore_decoder_state(self): pass
        result = api.replay_one(Runtime(), np.zeros((2,)), precision="native_bf16")
        self.assertEqual(result["decoded_final_float32"].shape, (3, 4, 4))
        self.assertEqual(result["decoded_full_float32"].dtype, np.float32)

    def test_temporary_precision_state_is_restored_after_success(self):
        class Runtime:
            decoder_state = "bf16"
            def actual_identity(self): return {"model_state": "runtime-v1", "decoder_state": "bf16"}
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
            def actual_identity(self): return {"model_state": "runtime-v1", "decoder_state": "bf16"}
            def decode_prediction_latent(self, latent, *, precision):
                self.active += 1; self.maximum = max(self.maximum, self.active)
                value = np.zeros((3, 2, 2, 2), np.float32) + (0.1 if precision == "native_bf16" else 0.2)
                self.active -= 1; return value
            def restore_decoder_state(self): pass
        class Encoder:
            def identity(self): return {"encoder_state": "encoder-v1", "code": "fixture"}
            def __call__(self, frame): return np.asarray(frame, dtype=np.float32).mean(keepdims=True)
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"; self._raw(raw)
            before = {p.relative_to(raw): (p.read_bytes(), p.stat().st_mtime_ns) for p in raw.rglob("*") if p.is_file()}
            decoder = Path(temporary) / "decoder"
            runtime = Runtime(); result = api.run_task6_decoder_replays(runtime, raw, encoder=Encoder(), decoder_root=decoder)
            self.assertEqual(result["status"], "COMPLETE"); self.assertEqual(result["decoder_calls"], 16); self.assertEqual(runtime.maximum, 1)
            after = {p.relative_to(raw): (p.read_bytes(), p.stat().st_mtime_ns) for p in raw.rglob("*") if p.is_file()}
            self.assertEqual(before, after); self.assertTrue((decoder / "MANIFEST.sha256").is_file())

    def test_raw_manifest_remains_valid_after_decoder_monitor_writes_root_telemetry(self):
        import run_umi_task6_experiment as runner
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"; self._raw(raw)
            safe = lambda: {"gpu_used_gib": 0, "gpu_free_gib": 100, "ram_available_gib": 600,
                            "rss_gib": 0, "swap_used_gib": 0, "disk_free_gib": 20}
            monitor = runner.ResourceMonitor(raw, gpu_sampler=safe, ram_sampler=safe, disk_sampler=safe)
            monitor.start(); monitor.stop()
            raw_status = raw / "run_status.json"; raw_status.write_text('{"status":"AWAITING_REVIEW","completed_samples":' + json.dumps([item["sample_id"] for item in api.build_generation_plan()]) + ',"group":{"state":"bridge_0","seed":0},"decoder_status":"RESOURCE_STOP"}')
            manifest_sha, status = api._verify_raw_task6(raw)
            self.assertTrue(manifest_sha); self.assertEqual(len(status["completed_samples"]), 32)

    def test_encoder_is_mandatory(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(Exception): api.run_task6_decoder_replays(object(), Path(temporary) / "raw")

    def test_class_only_runtime_and_encoder_identities_are_rejected(self):
        class Runtime:
            decoder_state = "bf16"
        class Encoder:
            def __call__(self, frame): return np.asarray(frame, np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"; self._raw(raw)
            with self.assertRaises(Exception): api.run_task6_decoder_replays(Runtime(), raw, encoder=Encoder(), decoder_root=Path(temporary) / "decoder")

    def test_resume_rejects_tampered_success(self):
        class Runtime:
            decoder_state = "bf16"
            def decode_prediction_latent(self, latent, *, precision): return np.zeros((3, 2, 2, 2), np.float32)
            def restore_decoder_state(self): pass
            def actual_identity(self): return {"model_state": "runtime-v1", "decoder_state": "bf16"}
        class Encoder:
            def identity(self): return {"encoder_state": "encoder-v1", "code": "fixture"}
            def __call__(self, frame): return np.asarray(frame, np.float32).mean(keepdims=True)
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"; self._raw(raw); decoder = Path(temporary) / "decoder"
            api.run_task6_decoder_replays(Runtime(), raw, encoder=Encoder(), decoder_root=decoder)
            target = next(decoder.glob("*__native_bf16/decoded_final_float32.npy")); target.write_bytes(b"tampered")
            with self.assertRaises(Exception): api.run_task6_decoder_replays(Runtime(), raw, encoder=Encoder(), decoder_root=decoder, resume=True)

    def test_resume_rejects_changed_runtime_or_encoder_content_identity(self):
        class Runtime:
            decoder_state = "bf16"
            def __init__(self, version): self.version = version
            def actual_identity(self): return {"model_state": self.version, "decoder_state": "bf16"}
            def decode_prediction_latent(self, latent, *, precision): return np.zeros((3, 2, 2, 2), np.float32)
            def restore_decoder_state(self): pass
        class Encoder:
            def __init__(self, version): self.version = version
            def identity(self): return {"encoder_state": self.version, "code": "fixture"}
            def __call__(self, frame): return np.asarray(frame, np.float32).mean(keepdims=True)
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"; self._raw(raw); decoder = Path(temporary) / "decoder"
            api.run_task6_decoder_replays(Runtime("weights-v1"), raw, encoder=Encoder("vae-v1"), decoder_root=decoder)
            with self.assertRaises(Exception): api.run_task6_decoder_replays(Runtime("weights-v2"), raw, encoder=Encoder("vae-v1"), decoder_root=decoder, resume=True)
            with self.assertRaises(Exception): api.run_task6_decoder_replays(Runtime("weights-v1"), raw, encoder=Encoder("vae-v2"), decoder_root=decoder, resume=True)

    def test_raw_manifest_rejects_unlisted_resource_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"; self._raw(raw)
            (raw / "unexpected.bin").write_bytes(b"not in the evidence manifest")
            with self.assertRaises(Exception): api._verify_raw_task6(raw)

    def test_partial_decoder_run_resumes_without_replaying_successes(self):
        class Runtime:
            decoder_state = "bf16"
            def __init__(self, fail_after=None): self.calls = 0; self.fail_after = fail_after
            def actual_identity(self): return {"model_state": "runtime-v1", "decoder_state": "bf16"}
            def decode_prediction_latent(self, latent, *, precision):
                self.calls += 1
                if self.fail_after is not None and self.calls > self.fail_after: raise RuntimeError("planned stop")
                return np.zeros((3, 2, 2, 2), np.float32)
            def restore_decoder_state(self): self.decoder_state = "bf16"
        class Encoder:
            def identity(self): return {"encoder_state": "encoder-v1", "code": "fixture"}
            def __call__(self, frame): return np.asarray(frame, np.float32).mean(keepdims=True)
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"; self._raw(raw); decoder = Path(temporary) / "decoder"
            first = Runtime(3); stopped = api.run_task6_decoder_replays(first, raw, encoder=Encoder(), decoder_root=decoder)
            self.assertEqual(stopped["status"], "BLOCKED"); self.assertEqual(first.calls, 4)
            completed_dirs = [p for p in decoder.iterdir() if p.is_dir() and p.name.endswith(("__native_bf16", "__temporary_fp32"))]
            before = {p.name: (p / "record.json").stat().st_mtime_ns for p in completed_dirs}
            second = Runtime(); resumed = api.run_task6_decoder_replays(second, raw, encoder=Encoder(), decoder_root=decoder, resume=True)
            self.assertEqual(resumed["status"], "COMPLETE"); self.assertEqual(second.calls, 13)
            self.assertEqual(before, {p.name: (p / "record.json").stat().st_mtime_ns for p in completed_dirs})

    def test_encoder_failure_restores_cache_and_keeps_primary_error(self):
        class Encoder:
            state = "clean"
            def identity(self): return {"encoder_state": "encoder-v1", "code": "fixture"}
            def reset_cache(self): self.state = "clean"
            def __call__(self, frame): self.state = "dirty"; raise RuntimeError("encoder failed")
        with self.assertRaises(RuntimeError) as context:
            api.reencode_frame(np.zeros((3, 2, 2), np.float32), Encoder())
        self.assertEqual(str(context.exception), "encoder failed")

    def test_decoder_uses_public_task6_runtime_adapter_decoder_seam(self):
        import umi_task6_runtime as runtime_api
        carrier = np.ones((1,48,5,16,16), np.float32); mask = np.zeros_like(carrier, bool); mask[:, :, 0] = True
        bank = np.zeros((3,) + carrier.shape, np.float32); bank[:, mask] = 1.0
        directions = runtime_api._derive_frozen_directions_unpinned(bank, mask); hashes = {key: runtime_api._array_sha(value) for key, value in directions.items()}
        inputs = runtime_api.Task6Inputs(carrier, [0], mask, bank, action=np.zeros((16,10), np.float32), prompt="decoder", direction_hashes=hashes)
        class Resident:
            def __init__(self): self.inputs = inputs; self.decoder_state = "bf16"
            def actual_identity(self): return {"model_state": "resident-v1", "decoder_state": "bf16"}
            def decode_prediction_latent(self, latent, *, precision): self.decoder_state = precision; return np.zeros((3,2,2,2), np.float32)
            def restore_decoder_state(self): self.decoder_state = "bf16"
        resident = Resident(); adapter = runtime_api.Task6RuntimeAdapter(resident, inputs)
        class Encoder:
            def identity(self): return {"encoder_state": "encoder-v1", "code": "fixture"}
            def __call__(self, frame): return np.asarray(frame, np.float32).mean(keepdims=True)
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"; self._raw(raw); result = api.run_task6_decoder_replays(adapter, raw, encoder=Encoder(), decoder_root=Path(temporary)/"decoder")
            self.assertEqual(result["status"], "COMPLETE"); self.assertEqual(resident.decoder_state, "bf16")

    def test_adapter_falls_back_to_validated_task5_replay_for_model_ops_runtime(self):
        import umi_task6_runtime as runtime_api
        carrier = np.ones((1, 48, 5, 16, 16), np.float32); mask = np.zeros_like(carrier, bool); mask[:, :, 0] = True
        bank = np.zeros((3,) + carrier.shape, np.float32); bank[:, mask] = 1.0
        directions = runtime_api._derive_frozen_directions_unpinned(bank, mask); hashes = {key: runtime_api._array_sha(value) for key, value in directions.items()}
        inputs = runtime_api.Task6Inputs(carrier, [0], mask, bank, action=np.zeros((16, 10), np.float32), prompt="decoder", direction_hashes=hashes)
        class OfficialLike:
            def __init__(self):
                self.inputs = inputs; self.model = object(); self.ops = object()
        resident = OfficialLike(); adapter = runtime_api.Task6RuntimeAdapter(resident, inputs)
        observed = {}
        def validated(runtime, latent, *, precision):
            observed["runtime"] = runtime; observed["latent"] = latent; observed["precision"] = precision
            return {"decoder_normalized_full_output": np.zeros((3, 2, 2, 2), np.float32)}
        with mock.patch("umi_task5_decoder.replay_decode", side_effect=validated):
            decoded = adapter.decode_prediction_latent(np.ones((1, 48, 5, 16, 16), np.float32), precision="temporary_fp32")
        self.assertIs(observed["runtime"], resident); self.assertEqual(observed["precision"], "fp32")
        self.assertEqual(decoded.shape, (3, 2, 2, 2))


if __name__ == "__main__": unittest.main()
