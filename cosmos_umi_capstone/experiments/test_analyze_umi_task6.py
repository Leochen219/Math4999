from __future__ import annotations

import json
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import analyze_umi_task6 as api
import umi_task5_primitives as p5


def _linear_records():
    shape = (1, 1, 3, 2, 2); mask = np.zeros(shape, bool); mask[:, :, 0] = True
    bank = np.zeros((3,) + shape, np.float32)
    bank[0, :, :, 0, 0, 0] = 2; bank[1, :, :, 0, 0, 1] = 2; bank[2, :, :, 0, 1, 0] = 2
    frozen = p5.freeze_task5_directions(bank, mask)
    y0 = np.zeros((1, 1, 2, 2, 2), np.float32)
    z_bar = np.ones(shape, np.float32); rgb0 = np.zeros((3,2,2), np.float32)
    def evidence(sample_id, *, direction, delta, kind, alpha, sign, output, rgb):
        consumed = np.add(z_bar, delta, dtype=np.float32)
        return {"output_full": output, "predicted_latent": np.asarray(output, np.float32).copy(), "decoded_final": rgb, "z_bar": z_bar.copy(), "mask": mask.copy(), "direction": direction.copy(), "actual_delta_fp32": np.subtract(consumed, z_bar, dtype=np.float32), "target_delta_fp32": delta.copy(), "consumed_input_fp32": consumed, "s_z": 1.0, "spec": {"sample_id": sample_id, "kind": kind, "state": "bridge_0", "seed": 0, "model_seed": 0, "alpha": alpha, "sign": sign, **({"direction_id": sample_id.split("_alpha", 1)[0]} if kind == "perturbation" else {})}, "group": {"state": "bridge_0", "seed": 0}, "seed": 0, "model_seed": 0}
    records = {}
    for name in ("baseline_pre", "baseline_post"):
        records[f"bridge_0__seed_0__{name}"] = evidence(name, direction=np.zeros(shape,np.float32), delta=np.zeros(shape,np.float32), kind="baseline", alpha=0.0, sign=0, output=y0.copy(), rgb=rgb0.copy())
    weights = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 1, 2, 2)
    for direction_id, direction in frozen["directions"].items():
        scalar = float(np.sum(direction[:, :, 0] * weights[:, :, 0]))
        for ordinal, alpha in enumerate((1e-3, 3e-3, 1e-2)):
            for sign, label in ((1,"plus"),(-1,"minus")):
                delta = sign * alpha * direction
                records[f"bridge_0__seed_0__{direction_id}_alpha_{ordinal:02d}_{label}"] = {
                    "output_full": y0 + sign * alpha * scalar,
                    "decoded_final": np.full((3,2,2), sign * alpha * scalar, np.float32),
                    **evidence(f"{direction_id}_alpha_{ordinal:02d}_{label}", direction=direction, delta=delta, kind="perturbation", alpha=alpha, sign=sign, output=y0 + sign * alpha * scalar, rgb=np.full((3,2,2), sign * alpha * scalar, np.float32))}
    return records, frozen


class AnalysisTests(unittest.TestCase):
    def test_exact_linear_prefixed_task6_fixture_has_six_and_24_checks(self):
        records, frozen = _linear_records()
        result = api.analyze_task6_records(records, plan_detail={"s_z": 1.0, "combination_coefficients": {"c01": frozen["c01"], "c12": frozen["c12"]}}, strict=True)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(len(result["additivity"]), 6)
        self.assertEqual(len(result["predictions"]), 24)

    def test_plan_detail_is_derived_from_nontrivial_saved_directions(self):
        records, frozen = _linear_records()
        detail = api._derive_plan_detail(api._normalize_records(records))
        self.assertAlmostEqual(detail["combination_coefficients"]["c01"], frozen["c01"])
        self.assertAlmostEqual(detail["combination_coefficients"]["c12"], frozen["c12"])

    def test_public_core_derives_nontrivial_coefficients_without_plan_detail(self):
        records, frozen = _linear_records()
        result = api.analyze_task6_records(records, strict=True)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertAlmostEqual(result["derivation"]["combination_coefficients"]["c01"], frozen["c01"])

    def test_core_requires_consumed_input_evidence(self):
        records, frozen = _linear_records()
        del records["bridge_0__seed_0__v0_alpha_00_plus"]["consumed_input_fp32"]
        with self.assertRaises(ValueError): api.analyze_task6_records(records, strict=True)

    def test_normalized_id_collision_and_missing_spec_model_seed_fail_closed(self):
        records, frozen = _linear_records()
        with self.assertRaises(ValueError):
            api._normalize_records({"baseline_pre": records["bridge_0__seed_0__baseline_pre"], "x__baseline_pre": records["bridge_0__seed_0__baseline_pre"]})
        records["bridge_0__seed_0__baseline_pre"]["spec"].pop("model_seed")
        with self.assertRaises(ValueError): api.analyze_task6_records(records, strict=True)

    def test_missing_and_stopped_are_not_reported_as_scientific_failures(self):
        result = api.analyze_task6_records({"baseline_pre": {}}, strict=False)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["summary"]["prediction_total"], 24)

    def test_raw_manifest_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "x.txt").write_text("x", encoding="utf-8")
            (root / "MANIFEST.sha256").write_text(f"{'0'*64}  x.txt\n", encoding="ascii")
            with self.assertRaises(ValueError): api.verify_raw_manifest(root)

    def test_difference_metrics_is_field_specific_for_shape_and_nonfinite(self):
        self.assertEqual(api.difference_metrics(np.zeros((2,)), np.zeros((3,)))["reason"], "shape_mismatch")
        self.assertEqual(api.difference_metrics(np.array([np.nan]), np.zeros((1,)))["reason"], "nonfinite")

    def test_cross_group_table_marks_absent_groups_not_run(self):
        rows = api.summarize_cross_groups({})
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["status"] == "NOT_RUN" for row in rows))

    def test_tampered_package_is_not_returned_as_idempotent(self):
        records, frozen = _linear_records()
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"; raw.mkdir(); (raw / "MANIFEST.sha256").write_text("", encoding="ascii")
            result = api.analyze_task6_records(records, plan_detail={"combination_coefficients": {"c01": frozen["c01"], "c12": frozen["c12"]}}, strict=True)
            # A raw-less package is sufficient to exercise package tamper detection.
            output = Path(temporary) / "analysis"; api.write_task6_artifacts(result, output)
            (output / "experiment_report.md").write_text("tampered", encoding="utf-8")
            with self.assertRaises(FileExistsError): api.write_task6_artifacts(result, output)

    def test_float64_reduction_does_not_change_raw_fixture(self):
        records, frozen = _linear_records(); before = records["bridge_0__seed_0__baseline_pre"]["output_full"].copy()
        api.analyze_task6_records(records, plan_detail={"combination_coefficients": {"c01": frozen["c01"], "c12": frozen["c12"]}})
        np.testing.assert_array_equal(before, records["bridge_0__seed_0__baseline_pre"]["output_full"])

    def test_package_is_outside_raw_and_manifest_valid(self):
        records, frozen = _linear_records()
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"; raw.mkdir()
            completed = []
            for key, record in records.items():
                sid = key.split("__")[-1]; sample = raw / "samples" / sid; sample.mkdir(parents=True); completed.append(sid)
                arrays = {name: value for name, value in record.items() if isinstance(value, np.ndarray)}
                for name, value in arrays.items(): np.save(sample / f"{name}.npy", value, allow_pickle=False)
                (sample / "sample.json").write_text(json.dumps({"sample_id": sid}), encoding="utf-8")
                hashes = {f"{name}.npy": hashlib.sha256((sample / f"{name}.npy").read_bytes()).hexdigest() for name in arrays}
                (sample / "status.json").write_text(json.dumps({"status": "success", "artifact_sha256": hashes}), encoding="utf-8")
            status = {"status": "AWAITING_REVIEW", "completed_samples": completed, "group": {"state": "bridge_0", "seed": 0}}
            (raw / "run_status.json").write_text(json.dumps(status), encoding="utf-8")
            entries = []
            for path in sorted(raw.rglob("*")):
                if path.is_file(): entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(raw).as_posix()}\n")
            (raw / "MANIFEST.sha256").write_text("".join(entries), encoding="ascii")
            result = api.analyze_task6_records(records, plan_detail={"combination_coefficients": {"c01": frozen["c01"], "c12": frozen["c12"]}}, strict=True)
            output = Path(temporary) / "analysis"; published = api.write_task6_artifacts(result, output, raw_root=raw)
            self.assertTrue((output / "review_bundle.zip").is_file()); self.assertEqual(api.verify_analysis_manifest(output)["sha256"], Path(published["manifest"]).read_bytes() and api.verify_analysis_manifest(output)["sha256"])
            self.assertTrue(api.write_task6_artifacts(result, output, raw_root=raw)["idempotent"])

    def test_fresh_review_bundles_are_byte_deterministic(self):
        records, frozen = _linear_records(); result = api.analyze_task6_records(records, strict=True)
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "a"; second = Path(temporary) / "b"
            api.write_task6_artifacts(result, first); api.write_task6_artifacts(result, second)
            self.assertEqual((first / "review_bundle.zip").read_bytes(), (second / "review_bundle.zip").read_bytes())

    def test_public_runner_store_reload_and_analysis_derives_coefficients(self):
        import run_umi_task6_experiment as runner
        import umi_task6_runtime as runtime_api
        carrier = np.ones((1, 48, 5, 16, 16), np.float32); mask = np.zeros_like(carrier, bool); mask[:, :, 0] = True
        bank = np.zeros((3,) + carrier.shape, np.float32); bank[:, mask] = 1.0
        directions = runtime_api._derive_frozen_directions_unpinned(bank, mask); hashes = {key: runtime_api._array_sha(value) for key, value in directions.items()}
        inputs = runtime_api.Task6Inputs(carrier, [0], mask, bank, action=np.zeros((16,10), np.float32), prompt="fixture", state="bridge_0", seed=0, direction_hashes=hashes)
        class Runtime:
            def __init__(self): self.inputs = inputs; self.provenance = {"source_commit": "2b17a2413bd86b2cf9b03823637108851e4ddf2d"}
            def actual_identity(self): return {"runtime": "public-e2e"}
            def execute(self, spec, runtime_inputs, *, scope="full"):
                full = runtime_inputs.for_spec(spec); predicted = full[:, :, 1:, ...].copy(); value = float(np.mean(full))
                return {"output_full": predicted, "decoded_final": np.full((3,2,2), value, np.float32)}
            def cleanup(self): pass
        runtime = Runtime(); adapter = runtime_api.Task6RuntimeAdapter(runtime, inputs)
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            direction_path = Path(temporary) / "directions.npy"; np.save(direction_path, bank, allow_pickle=False)
            asset_path = Path(temporary) / "asset.bin"; asset_path.write_bytes(b"fixture asset")
            config = {"environment": {"name": "fixture"}, "provenance": {"source_commit": "2b17a2413bd86b2cf9b03823637108851e4ddf2d"}, "asset_hashes": {"asset": hashlib.sha256(asset_path.read_bytes()).hexdigest()}, "asset_paths": {"asset": str(asset_path)}, "carrier_shape": list(carrier.shape), "carrier_hash": inputs.identity()["z0"], "condition_indexes": [0], "predicted_indexes": [1,2,3,4], "mask_shape": list(mask.shape), "mask_hash": inputs.geometry.metadata()["mask_sha256"], "action_shape": [16,10], "action_hash": inputs.identity()["action"], "prompt": inputs.prompt, "direction_hashes": hashes, "direction_bank_path": str(direction_path), "direction_bank_file_hash": hashlib.sha256(direction_path.read_bytes()).hexdigest(), "settings": {"num_steps":30,"guidance":1.0,"shift":10.0,"autocast":False,"tf32":False,"diffusion_cache":False,"batch_size":1}, "seed_config": {"seed":0,"prepare":0,"sampler":0,"scheduler":0}, "observed_runtime_identity": runtime.actual_identity(), "observed_input_identity": inputs.identity()}
            preflight = Path(temporary) / "preflight.json"; preflight.write_text(json.dumps(config), encoding="utf-8")
            runner.execute_task6(runner.parse_args(["--phase", "preflight", "--run-dir", str(run_dir), "--preflight-json", str(preflight)]), runtime=runtime, inputs=inputs)
            safe = lambda: {"gpu_used_gib":0,"gpu_free_gib":100,"gpu_reserved_gib":0,"ram_available_gib":600,"rss_gib":0,"swap_used_gib":0,"disk_free_gib":20}
            runner.run_resource_smoke(run_dir, lifecycle={"pre_load":lambda:None,"load":lambda:(runtime,inputs),"cleanup":lambda:None,"unload":lambda:None}, samplers={key:safe for key in ("gpu","ram","disk")})
            runner.accept_resource_smoke(run_dir, hashes=json.loads((run_dir / "run_status.json").read_text())["hashes"])
            monitor = runner.ResourceMonitor(run_dir, gpu_sampler=safe, ram_sampler=safe, disk_sampler=safe)
            status = runner.run_pilot(adapter, inputs, run_dir, monitor=monitor)
            self.assertEqual(status["status"], "AWAITING_REVIEW")
            reloaded = api._load_records(run_dir)
            result = api.analyze_task6_records(reloaded, strict=True)
            self.assertEqual(len(reloaded), 32); self.assertEqual(len(result["fits"]), 5); self.assertNotEqual(result["derivation"]["combination_coefficients"]["c01"], 1.0)

    def test_decoder_analysis_reads_sixteen_records_and_emits_five_spaces(self):
        import umi_task6_decoder as decoder_api
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); raw = root / "raw"; dec = root / "decoder"; mask = np.ones((1,1,2,2), bool)
            logicals = ["baseline_pre", "baseline_post"] + [f"v0_alpha_{i:02d}_{s}" for i in range(3) for s in ("plus", "minus")]
            for logical in logicals:
                sample = raw / "samples" / logical; sample.mkdir(parents=True)
                delta = 0.0 if logical.startswith("baseline") else (1 if logical.endswith("plus") else -1) * float(p5.ALPHAS[int(logical.split("alpha_")[1][:2])])
                np.save(sample / "mask.npy", mask, allow_pickle=False); np.save(sample / "actual_delta_fp32.npy", np.full(mask.shape, delta, np.float32), allow_pickle=False)
                (sample / "sample.json").write_text(json.dumps({"sample_id": logical}), encoding="utf-8"); (sample / "status.json").write_text(json.dumps({"status":"success","artifact_sha256":{}}), encoding="utf-8")
            for spec in decoder_api.decoder_replay_plan():
                replay = dec / spec["replay_id"]; replay.mkdir(parents=True); logical = spec["sample_id"].split("__")[-1]; factor = 1.0 if logical.endswith("plus") else -1.0 if logical.endswith("minus") else 0.0
                arrays = {"decoder_input_full_latent": np.full((1,1,2,2), .1 + factor * .01, np.float32), "predicted_latent": np.full((1,1,4,2,2), .15 + factor*.01, np.float32), "decoded_full_float32": np.full((3,2,2,2), .2 + factor*.01, np.float32), "decoded_final_float32": np.full((3,2,2), .2 + factor*.01, np.float32), "direct_float_input": np.full((3,2,2), .2 + factor*.01, np.float32), "uint8_simulated_input": np.full((3,2,2), .2 + factor*.01, np.float32), "direct_condition_latent_float32": np.full((1,1,2,2), .1 + factor*.01, np.float32), "uint8_condition_latent_float32": np.full((1,1,2,2), .1 + factor*.005, np.float32)}
                for name, value in arrays.items(): np.save(replay / f"{name}.npy", value, allow_pickle=False)
                (replay / "record.json").write_text(json.dumps({"status":"success","precision":spec["decode_precision"],"spec":spec}), encoding="utf-8")
            (dec / "decoder_config.json").write_text(json.dumps({"raw_manifest_sha256":"raw","raw_group":{"state":"bridge_0","seed":0}}), encoding="utf-8"); (dec / "status.json").write_text(json.dumps({"status":"COMPLETE"}), encoding="utf-8")
            entries = []
            for path in sorted(dec.rglob("*")):
                if path.is_file(): entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(dec).as_posix()}\n")
            (dec / "MANIFEST.sha256").write_text("".join(entries), encoding="ascii")
            result = api.analyze_decoder_replays(raw, dec)
            self.assertEqual(result["status"], "COMPLETE"); self.assertEqual(result["decoder_calls"], 16); self.assertEqual(set(result["metrics_spaces"]), {"prediction_latent", "native_rgb", "fp32_rgb", "direct_float_condition_latent", "uint8_sim_condition_latent"}); self.assertTrue(result["tensors"])
            summaries = [row for row in result["space_metrics"] if row.get("status") == "SUMMARY"]; self.assertTrue(summaries); self.assertTrue(all("plus_slope" in row and "next_secant_cosine" not in row for row in summaries))
            # A partial decoder evidence tree remains analyzable only as an
            # explicit field-level N/A; it must not silently lose its summary.
            missing = root / "decoder_missing"
            shutil.copytree(dec, missing)
            next(missing.glob("*baseline_pre__native_bf16/decoded_final_float32.npy")).unlink()
            entries = []
            for path in sorted(missing.rglob("*")):
                if path.is_file() and path.name != "MANIFEST.sha256":
                    entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(missing).as_posix()}\n")
            (missing / "MANIFEST.sha256").write_text("".join(entries), encoding="ascii")
            partial = api.analyze_decoder_replays(raw, missing)
            self.assertEqual(partial["status"], "COMPLETE")
            self.assertTrue(any(row.get("status") == "N/A" and row.get("reason") == "missing_pair" and row.get("candidate_status") == "N/A" for row in partial["space_metrics"]))


if __name__ == "__main__": unittest.main()
