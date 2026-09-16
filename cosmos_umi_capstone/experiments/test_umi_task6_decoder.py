from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

import umi_task6_decoder as api


class DecoderTests(unittest.TestCase):
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


if __name__ == "__main__": unittest.main()
