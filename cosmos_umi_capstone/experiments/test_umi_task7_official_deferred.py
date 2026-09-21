"""Focused contract tests for full-generation deferred decode."""
import gc
import importlib
import unittest
import weakref

import numpy as np

import test_umi_precision_official as official_fixture
from umi_precision_runtime import EvidenceError, validate_capture


class OfficialDeferredDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = importlib.import_module("umi_precision_official")

    def fixture(self, seed=0):
        # Reuse the approved existing official-shaped CPU fixture without
        # inheriting its whole test class into this focused module.
        helper = official_fixture.OfficialTests("test_true_seams_capture_dtype_noise_reset_baseline_and_output")
        helper.api = self.api
        runtime, model = helper.fixture(seed=seed)
        self.addCleanup(helper.doCleanups)
        return runtime, model

    def test_deferred_full_generation_runs_all_steps_without_decode(self):
        runtime, model = self.fixture()
        runtime.settings["num_steps"] = 30
        spec = {"sample_id": "deferred", "group": "B", "alpha": 0.0, "sign": 0}
        native = runtime.execute(spec, runtime.inputs, scope="full")
        self.assertEqual(model.decode_count, 1)
        deferred = runtime.execute(spec, runtime.inputs, scope="full", decode_policy="deferred")
        self.assertEqual(model.decode_count, 1)
        self.assertEqual(len(deferred["steps"]), 30)
        self.assertEqual(len([row for row in deferred["tensor_evidence"] if row["role"] == "sampler_update"]), runtime.settings["num_steps"])
        np.testing.assert_array_equal(native["output_full"], deferred["output_full"])
        validate_capture(deferred, spec, runtime.inputs, "full", require_decoded=False)
        self.assertNotIn("decoded_final", deferred)

    def test_deferred_policy_is_explicit_and_rejects_module_scope(self):
        runtime, _ = self.fixture()
        spec = {"sample_id": "deferred", "group": "B", "alpha": 0.0, "sign": 0}
        with self.assertRaises(ValueError):
            runtime.execute(spec, runtime.inputs, scope="full", decode_policy="wat")
        with self.assertRaises(ValueError):
            runtime.execute(spec, runtime.inputs, scope="module", decode_policy="deferred")

    def test_deferred_cleanup_restores_original_net_and_releases_working_copy(self):
        runtime, model = self.fixture()
        original_net = model.net
        captured = {}
        original_deepcopy = self.api.copy.deepcopy

        def capture(value, memo=None):
            result = original_deepcopy(value) if memo is None else original_deepcopy(value, memo)
            if value is original_net:
                captured["working"] = weakref.ref(result)
            return result

        self.api.copy.deepcopy = capture
        self.addCleanup(setattr, self.api.copy, "deepcopy", original_deepcopy)
        spec = {"sample_id": "deferred", "group": "B", "alpha": 0.0, "sign": 0}
        runtime.execute(spec, runtime.inputs, scope="full", decode_policy="deferred")
        self.assertIs(model.net, original_net)
        gc.collect()
        self.assertIsNone(captured["working"]())

    def test_deferred_exception_restores_request_state_and_never_decodes(self):
        runtime, model = self.fixture()
        original_net = model.net
        model.net.hidden_cast = True
        spec = {"sample_id": "deferred-fail", "group": "B", "alpha": 0.0, "sign": 0}
        with self.assertRaisesRegex(EvidenceError, "hidden"):
            runtime.execute(spec, runtime.inputs, scope="full", decode_policy="deferred")
        self.assertIs(model.net, original_net)
        self.assertEqual(model.decode_count, 0)

    def test_deferred_validator_rejects_dummy_decoded_frame_and_missing_evidence(self):
        runtime, _ = self.fixture()
        spec = {"sample_id": "deferred", "group": "B", "alpha": 0.0, "sign": 0}
        record = runtime.execute(spec, runtime.inputs, scope="full", decode_policy="deferred")
        record["decoded_final"] = np.zeros((3, 2, 2), dtype=np.float32)
        with self.assertRaises(EvidenceError):
            validate_capture(record, spec, runtime.inputs, "full", require_decoded=False)
        record.pop("decoded_final")
        record["tensor_evidence"] = [row for row in record["tensor_evidence"] if row["role"] != "network_condition"]
        with self.assertRaises(EvidenceError):
            validate_capture(record, spec, runtime.inputs, "full", require_decoded=False)


if __name__ == "__main__":
    unittest.main()
