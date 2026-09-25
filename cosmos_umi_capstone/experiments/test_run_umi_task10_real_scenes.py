"""CPU checks for the multi-episode execution gates."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_umi_task10_real_scenes import (_sha256_file, _verified_truth_conditions,
                                         _matrix_complete, parse_args, validate_preflight_report,
                                         validate_run_identity)
from analyze_umi_task10_real_scenes import validate_saved_seed


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.valid = {"status": "PREFLIGHT_PASSED_GENERATION_NOT_PERFORMED", "record_count": 52,
                 "selected": {"record_index": 14, "language": "move the pot",
                              "inputs_npz": {"sha256": "a" * 64, "actions_normalized_shape": [2, 16, 10],
                                             "rgb_float32_shape": [33, 3, 256, 256]}},
                 "camera": {"feature_key": "steps/observation/image_0",
                            "all_window_frames_rgb_256x256": True,
                            "all_window_frames_nonconstant": True,
                            "unique_encoded_frames": 33},
                 "manifest_sha256_match": True,
                 "stop_reasons": ["Model generation remains outside this generation-free preflight and requires independent main-agent admission."],
                 "official_parity_artifact": {"status": "PASS", "action_exact_equal": True,
                                               "initial_pose_exact_equal": True,
                                               "normalization": {"saved_exact_equal": True}},
                 "temporal_alignment": {"sequence_index_order_proven": True,
                                        "frequency_continuity_proven_from_official_source": True,
                                        "flags_prove_first_last_boundaries": True}}
    def test_preflight_rejects_substitution_or_missing_camera_proof(self):
        valid = self.valid
        validate_preflight_report(valid, 14)
        changed = {**valid, "selected": {**valid["selected"], "record_index": 8}}
        with self.assertRaisesRegex(ValueError, "record"):
            validate_preflight_report(changed, 14)
        changed = {**valid, "camera": {"feature_key": "steps/observation/image_0",
                                       "all_window_frames_rgb_256x256": False}}
        with self.assertRaisesRegex(ValueError, "camera"):
            validate_preflight_report(changed, 14)

    def test_preflight_refuses_unproven_official_action_parity(self):
        changed = {**self.valid, "official_parity_artifact":
                   {**self.valid["official_parity_artifact"], "action_exact_equal": False}}
        with self.assertRaisesRegex(ValueError, "parity"):
            validate_preflight_report(changed, 14)

    def test_preflight_refuses_unproven_temporal_alignment(self):
        changed = {**self.valid, "temporal_alignment":
                   {**self.valid["temporal_alignment"], "sequence_index_order_proven": False}}
        with self.assertRaisesRegex(ValueError, "temporal"):
            validate_preflight_report(changed, 14)

    def test_resume_identity_refuses_seed_or_input_change(self):
        frozen = {"dataset_sha256": "a", "code_sha256": "b", "record_indices": [1, 2, 3, 6, 7, 14],
                  "seed_pairs": [[0, 1], [2, 3]]}
        validate_run_identity(frozen, dict(frozen))
        changed = {**frozen, "seed_pairs": [[0, 1], [2, 4]]}
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_run_identity(frozen, changed)

    def test_saved_seed_requires_prepare_and_all_sampler_seeds(self):
        row = {"generation": {"noise_evidence": {"seed": 2, "prepare_seed": 2},
                              "sampler_generator_seeds": [2] * 30}}
        validate_saved_seed(row, 2)
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_saved_seed(row, 0)
        row["generation"]["sampler_generator_seeds"][29] = 3
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_saved_seed(row, 2)

    def test_gt_resume_receipt_is_bound_to_episode_input_and_vae(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            npz = root / "ground_truth_conditions.npz"
            evidence = root / "ground_truth_condition_evidence.json"
            npz.write_bytes(b"fixture")
            evidence.write_text("{}", encoding="utf-8")
            receipt = {"npz_sha256": _sha256_file(npz), "evidence_sha256": _sha256_file(evidence),
                       "record_index": 1, "input_sha256": "a" * 64, "vae_sha256": "b" * 64}
            (root / "ground_truth_conditions.sha256.json").write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "receipt/hash"):
                _verified_truth_conditions(None, None, root, record_index=2,
                                           input_sha256="a" * 64, vae_sha256="b" * 64)

    def test_formal_cli_selects_one_locked_episode(self):
        common = ["--stage", "formal", "--run-root", "r", "--dataset-root", "d",
                  "--framework-root", "f", "--checkpoint", "c", "--vae", "v",
                  "--official-parity-report", "p"]
        self.assertEqual(parse_args(common + ["--record-index", "6"]).record_index, 6)
        with self.assertRaises(SystemExit):
            parse_args(common + ["--record-index", "8"])

    def test_matrix_never_promotes_sample_count_without_complete_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run_identity.json").write_text(json.dumps({"checkpoint_sha256": "m", "vae_sha256": "v"}), encoding="utf-8")
            dummy = root / "input.npz"
            dummy.write_bytes(b"input")
            from umi_task10_real_scenes import LOCKED_RECORDS, SEED_PAIRS
            for index in LOCKED_RECORDS:
                for pair in SEED_PAIRS:
                    formal = root / "records" / f"record_{index:02d}" / f"seeds_{pair[0]}_{pair[1]}" / "formal"
                    formal.mkdir(parents=True)
                    status = "FAILED" if index == 14 and pair == (2, 3) else "COMPLETE"
                    (formal / "run_status.json").write_text(json.dumps({"status": status,
                       "binding": {"model": "m", "vae": "v", "data": _sha256_file(dummy)}}), encoding="utf-8")
            with patch("run_umi_task10_real_scenes._load_preflight", return_value=({}, dummy)), \
                 patch("run_umi_task10_real_scenes._completed_pair", side_effect=lambda folder, plan, binding:
                       json.loads((folder / "run_status.json").read_text(encoding="utf-8"))["status"] == "COMPLETE"):
                self.assertFalse(_matrix_complete(root))


if __name__ == "__main__":
    unittest.main()
