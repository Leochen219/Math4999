# Task 1 implementation report

## Result

Implemented deterministic Task 6 primitive contracts, resource-policy evaluation,
canonical terminal status payload construction, and explicit seed routing through
the existing official precision runtime seams. No model inference, network, SSH,
GPU, or asset loading was performed.

## RED evidence

1. Before adding production code:

   `C:\Users\hongy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest cosmos_umi_capstone/experiments/test_umi_task6_primitives.py`

   Failed with `ModuleNotFoundError: No module named 'umi_task6_primitives'`.

2. Before seed implementation:

   `...python.exe -m unittest test_umi_precision_official.OfficialTests.test_seed_one_reaches_prepare_sampler_scheduler_and_changes_only_noise_path`

   Failed with `TypeError: OfficialPrecisionRuntime.__init__() got an unexpected keyword argument 'model_seed'`.

These failures were caused by the intentionally missing production interfaces.

## GREEN evidence

Focused suite:

`...python.exe -m unittest test_umi_task6_primitives.py test_umi_precision_official.py test_umi_precision_runtime.py test_umi_task5_primitives.py test_umi_task5_runtime.py`

`Ran 49 tests in 25.044s` — `OK`.

Additional Task 5/precision identity, storage, torch, decoder, analyzer, and
launcher coverage:

`...python.exe -m unittest test_umi_task5_decoder.py test_analyze_umi_task5.py test_run_umi_task5_experiment.py test_umi_precision_identity.py test_umi_precision_storage.py test_umi_precision_torch.py`

`Ran 23 tests` — `OK (skipped=10)` (Torch-only skips are environment documented).

The broad 116-test precision/Task 5 attempt had 1 historical CRLF-bound
calibration-asset failure and 3 historical Matplotlib/calibration-artifact
environment failures, plus 10 documented skips. No failure implicated these
changes; historical calibration code/assets were not modified.

## Files

- `cosmos_umi_capstone/experiments/umi_task6_primitives.py`: immutable state and six-group catalog, pilot selector, frame preprocessing, action validation/hash helpers, exact generation/replay plans, resource gates, and canonical run status builder.
- `cosmos_umi_capstone/experiments/test_umi_task6_primitives.py`: focused primitive tests.
- `cosmos_umi_capstone/experiments/umi_precision_official.py`: optional non-negative `model_seed`/`seed` parameter routed through prepare, sampler, scheduler generator, and generation seams; seed included in actual noise provenance.
- `cosmos_umi_capstone/experiments/umi_precision_runtime.py`: seed-aware call plans, capture validation, and runtime plan/diagnostic propagation.
- `cosmos_umi_capstone/experiments/test_umi_precision_official.py`: seed-0 identity and seed-1 seam regression fixture.

## Commit

`f16c8f8` — `Implement Task 6 deterministic primitives and seed routing`

## Self-review

- `git diff --check` passed.
- Import-time dependencies remain NumPy/stdlib only for Task 6 primitives; OpenCV is not imported.
- Existing seed-0 defaults and Task 4/5 tests remain green.
- Resource evaluation is pure, fail-closed for NaN snapshots, and selects a hard-stop code before warnings.
- The primitive preprocessing uses deterministic NumPy nearest-neighbor resize and reflection padding. Task 2 should bind this helper to the verified official asset/preprocessing convention and independently verify asset provenance before inference.
- `umi_reference` paths are provenance labels for the Task 5 first frame/action chunk; Task 2 must resolve them against the validated Task 5 archive rather than treating them as downloadable Bridge assets.

