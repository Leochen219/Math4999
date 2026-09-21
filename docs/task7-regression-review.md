# Main regression evidence before operational runner

Code snapshot ae316fc (approved encoder, deferred full-generation seam, feedback runtime). Tar SHA256 abdc98ceb5d1c50e08934d932326376af52971d66c3c512b23a0ed1fa1d94437 verified on local and remote. Remote isolated root /root/autodl-tmp/task7-cpu-check-20260921-01/frozen-ae316fc/cosmos_umi_capstone. No framework or historical source modifications.

Task4-6 selected full regression modules:
test_umi_precision_{torch,storage,runtime,primitives,official,identity,calibration}; test_run_umi_precision_experiment; test_analyze_umi_precision_contrast; test_umi_task5_{runtime,primitives,decoder}; test_run_umi_task5_experiment; test_analyze_umi_task5; test_umi_task6_{runtime,primitives,operational,decoder}; test_run_umi_task6_experiment; test_run_umi_task6_decoder_only; test_analyze_umi_task6. Commands use python -m unittest and explicit names expanded from this list.

1. Local bundled Python: 339 tests/124.273s, two errors (missing matplotlib), eighteen skips (missing Torch/assets/platform). Not counted as full PASS.
2. Initial remote overlay: 339 tests/43.547s, four errors: two missing matplotlib; two correctly detected mixed-source hashes due to layering changed runtime over historical analysis directory. Deployment test setup invalid for full source-bound suite; not a runtime parity result. Replaced with complete frozen tree instead of weakening hashes.
3. Complete frozen tree, existing experiment venv, CUDA_VISIBLE_DEVICES empty, all listed modules except calibration: 326 tests/35.634s, OK with two asset-presence skips. Actual Torch CPU tests included; no model/GPU use.
4. Existing /root/miniconda3/bin/python (3.12 with matplotlib), same frozen source, test_umi_precision_calibration: 13 tests/2.893s, all PASS. No dependency installation.
5. Isolated tree assets/task6_bridge0 symlink to existing reviewed /root/autodl-tmp/task6-code-61ddae0/cosmos_umi_capstone/assets/task6_bridge0, source unchanged. Experiment venv reran test_umi_task6_operational.OperationalTask6Tests.test_local_bridge_pair_hash_and_action_shape and .test_asset_bundle_refuses_overwrite: 2 tests/0.024s, all PASS.

Union covers all 339 distinct regression tests passing, using two pre-existing interpreters; do not describe it as one interpreter's single all-green run. Task7 focused encoder/deferred/runtime suite separately 24/24 PASS (see task2b review). Operational runner and analyzer are not yet included in this snapshot or certified by these runs.
