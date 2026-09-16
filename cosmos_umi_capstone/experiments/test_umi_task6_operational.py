import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class OperationalTask6Tests(unittest.TestCase):
    def setUp(self):
        import umi_task6_operational as op
        self.op = op

    def test_catalog_bridge0_is_exactly_pinned(self):
        from umi_task6_primitives import STATE_CATALOG, SOURCE_COMMIT
        asset = STATE_CATALOG["bridge_0"]
        self.assertEqual(asset.source_commit, SOURCE_COMMIT)
        self.assertEqual(asset.action_path, "inputs/action/bridge_20260501_0.json")
        self.assertEqual(asset.video_path, "inputs/action/bridge_20260501_0.mp4")

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "assets" / "task6_bridge0" / "bridge_20260501_0.mp4").is_file(), "local reviewed bridge asset is not present")
    def test_local_bridge_pair_hash_and_action_shape(self):
        root = Path(__file__).resolve().parents[1] / "assets" / "task6_bridge0"
        evidence = self.op.verify_pinned_bridge_assets(root / "bridge_20260501_0.json", root / "bridge_20260501_0.mp4")
        self.assertEqual(evidence["action_shape"], [16, 10]); self.assertEqual(evidence["fps"], 5)

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "assets" / "task6_bridge0" / "bridge_20260501_0.mp4").is_file(), "local reviewed bridge asset is not present")
    def test_asset_bundle_refuses_overwrite(self):
        root = Path(__file__).resolve().parents[1] / "assets" / "task6_bridge0"
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "bundle"
            self.op.prepare_bridge_upload_bundle(root / "bridge_20260501_0.json", root / "bridge_20260501_0.mp4", out)
            with self.assertRaises(FileExistsError):
                self.op.prepare_bridge_upload_bundle(root / "bridge_20260501_0.json", root / "bridge_20260501_0.mp4", out)

    def test_contract_never_accepts_self_attested_only(self):
        contract = self.contract()
        with self.assertRaises(self.op.OperationalEvidenceError): self.op.validate_launch_contract(contract)

    def test_contract_rejects_wrong_group_and_observed_identity(self):
        contract = self.contract(); observed = dict(contract)
        contract["group"] = {"state": "bridge_384", "seed": 0}
        with self.assertRaises(self.op.OperationalEvidenceError): self.op.validate_launch_contract(contract, observed=observed)

    def test_static_contract_requires_all_task5_and_bridge_hashes(self):
        contract = self.contract(); contract["task5"] = {"manifest_sha256": "a" * 64}
        with self.assertRaises(self.op.OperationalEvidenceError): self.op.validate_launch_contract(contract, observed=contract)
        contract = self.contract(); contract["bridge_asset_hashes"]["video"] = "0" * 64
        with self.assertRaises(self.op.OperationalEvidenceError): self.op.validate_launch_contract(contract, observed=contract)
        contract = self.contract(); observed = dict(contract); observed["cuda_version"] = "wrong"
        with self.assertRaises(self.op.OperationalEvidenceError): self.op.validate_launch_contract(contract, observed=observed)

    def test_task5_direction_extraction_requires_manifest_and_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "samples").mkdir()
            for direction in ("v0", "v1", "v2"):
                sample = root / "samples" / f"{direction}_alpha_00_plus"; sample.mkdir()
                arr = np.zeros((1, 48, 5, 16, 16), np.float32)
                mask = np.zeros_like(arr, dtype=bool); mask[:, :, 0] = True; arr[mask] = 1.0
                np.save(sample / "direction.npy", arr)
                np.save(sample / "mask.npy", mask)
                (sample / "sample.json").write_text("{}")
                hashes = {name: hashlib.sha256((sample / name).read_bytes()).hexdigest() for name in ("direction.npy", "mask.npy", "sample.json")}
                (sample / "status.json").write_text(json.dumps({"status": "success", "artifact_sha256": hashes}))
            plan = {"status": "complete", "plan": [{"sample_id": f"s{i}"} for i in range(32)]}; (root / "task5_plan.json").write_text(json.dumps(plan))
            (root / "status.json").write_text(json.dumps({"status": "complete", "formal_successful": 32}))
            entries = []
            for path in sorted(root.rglob("*")):
                if path.is_file(): entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
            (root / "MANIFEST.sha256").write_text("\n".join(entries) + "\n")
            result = self.op.extract_task5_directions(root)
            self.assertEqual(result["bank"].shape, (3, 1, 48, 5, 16, 16)); self.assertEqual(set(result["direction_sha256"]), {"v0", "v1", "v2", "u01", "u12"})

    def test_task5_manifest_rejects_nested_manifest_and_extra_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "task5_plan.json").write_text("{}")
            (root / "samples").mkdir(); (root / "samples" / "nested").mkdir()
            (root / "samples" / "nested" / "MANIFEST.sha256").write_text("bad")
            (root / "MANIFEST.sha256").write_text("\n")
            with self.assertRaises(self.op.OperationalEvidenceError): self.op.extract_task5_directions(root)

    def test_loader_spec_requires_module_function(self):
        with self.assertRaises(self.op.OperationalEvidenceError): self.op.import_callable("not-a-spec")

    def contract(self):
        return {"framework_commit": "1" * 40, "asset_source_commit": "2b17a2413bd86b2cf9b03823637108851e4ddf2d", "checkpoint_identity": {"sha256": "x"},
                "vae_sha256": "a" * 64, "torch_version": "2.10.0+cu130", "cuda_version": "13.0", "code_bundle_sha256": "b" * 64,
                "bridge_asset_hashes": {"action": "a" * 64, "video": "b" * 64}, "task5": {"manifest_sha256": "c" * 64},
                "group": {"state": "bridge_0", "seed": 0}, "prompt": self.op.BRIDGE0_PROMPT,
                "action": [[0.0] * 10 for _ in range(16)], "settings": {"num_steps": 30, "guidance": 1.0, "shift": 10.0, "batch_size": 1, "autocast": False, "tf32": False, "diffusion_cache": False},
                "cache_flags": {"autocast": False, "tf32": False, "diffusion_cache": False},
                "seed_routes": {"model": 0, "prepare": 0, "sampler": 0, "scheduler": 0},
                "geometry": {"carrier_shape": [1, 48, 5, 16, 16], "condition_indexes": [0], "predicted_indexes": [1, 2, 3, 4], "mask_shape": [1, 48, 5, 16, 16]}}


if __name__ == "__main__": unittest.main()
