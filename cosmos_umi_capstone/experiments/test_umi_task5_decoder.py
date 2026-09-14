"""Non-GPU guards for Task 5B decoder capability selection."""
from __future__ import annotations

import unittest
from types import SimpleNamespace


class Task5DecoderTests(unittest.TestCase):
    def test_only_verified_non_amp_bf16_wan_path_is_eligible_for_fp32_replay(self):
        import umi_task5_decoder as api
        unsupported = SimpleNamespace(model=SimpleNamespace(tokenizer_vision_gen=None), ops=SimpleNamespace(torch=None))
        self.assertFalse(api.decoder_support_info(unsupported)["supported_fp32_replay"])
        inner = SimpleNamespace(decode=lambda value: value)
        wan = SimpleNamespace(model=inner, dtype="bfloat16", is_amp=False)
        supported = SimpleNamespace(model=SimpleNamespace(tokenizer_vision_gen=SimpleNamespace(model=wan)), ops=SimpleNamespace(torch=object()))
        self.assertTrue(api.decoder_support_info(supported)["supported_fp32_replay"])
        wan.is_amp = True
        self.assertFalse(api.decoder_support_info(supported)["supported_fp32_replay"])
        del wan.is_amp
        self.assertTrue(api.decoder_support_info(supported)["supported_fp32_replay"])


if __name__ == "__main__":
    unittest.main()
