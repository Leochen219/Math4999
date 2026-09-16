# Task 6: Cross-context local-linearity reproduction

## Goal

Implement and execute the approved Task 6 experiment without changing the validated Task 4/5 semantics. Luna implements and runs each bounded task; the controller independently reviews source, hashes, test evidence, resource gates, and scientific conclusions. The first execution is restricted to the `bridge_0 / seed 0` pilot and must stop afterward.

## Global constraints

- Work only on branch `codex/task6-cross-context-linearity` in its isolated worktree.
- Preserve Task 4 C semantics: post-VAE FP32 condition perturbation, FP32 network and UniPC accumulation, 30 denoise steps, guidance 1, shift 10, autocast off, TF32 off, diffusion cache off, batch size 1.
- Preserve Task 5 perturbations exactly: `alpha=[0.001,0.003,0.01]` and byte-identical `v0`, `v1`, `v2`, `u01`, `u12`. Never resample, tune, or select directions after observing results.
- States are `umi_reference`, `bridge_0`, and `bridge_384`; seeds are 0 and 1. A group has exactly 32 generation calls. The approved first pilot is only `bridge_0 / seed 0`.
- `umi_reference` uses the Task 5 first frame, action chunk 0, and prompt. Bridge assets come from NVIDIA `cosmos-dependencies` commit `2b17a2413bd86b2cf9b03823637108851e4ddf2d` at `inputs/action/bridge_20260501_{0,384}.{mp4,json}`. They are official paired evaluation/example assets, not claimed training data.
- Extract video frame 0 and apply the official aspect-preserving resize plus reflection padding to `256x256`. Each action is `(16,10)`. Runtime evidence must show carrier `[1,48,5,16,16]`, condition indexes `[0]`, predicted indexes `[1,2,3,4]`, and the Task 5 mask geometry.
- Seed must reach the actual prepare/noise, sampler, and scheduler seams. Default seed 0 must remain byte-compatible with Task 5.
- Decoder coverage per group is the two baselines plus `v0` at all three positive and negative amplitudes: eight final prediction latents, each decoded via native BF16 and temporary FP32, for 16 decoder-only replays. Each decoded float final frame is re-encoded directly and after explicit uint8 quantization simulation.
- Never use MP4 for quantitative comparisons. Keep prediction latent, RGB, and `E(D(G))` condition-latent spaces separate.
- Scientific thresholds stay fixed: realized input direction cosine >= 0.99; plus/minus cosine <= -0.99; response > 10x baseline floor (or strictly nonzero if the floor is zero); both one-sided slopes in `[0.8,1.2]` with R-squared >= 0.98; adjacent centered-secant cosine >= 0.95 and relative RMS change <= 0.25; all six additivity relative errors <= 0.10; all 24 holdout relative errors <= 0.10.
- Resource monitoring is fail-closed and sequential. No parallel generation, decoder replay, or second resident model. No automatic retry after OOM/resource stop. No deletion of prior experiments.
- GPU: start used <= 1 GiB; smoke peak allocated <= 35 GiB and NVML used <= 45 GiB. Warn at used > 60 GiB or free < 35 GiB. Stop at used > 75 GiB, free < 20 GiB, PyTorch reserved > 65 GiB, CUDA OOM, or cleanup-baseline growth > 2 GiB for two consecutive samples.
- RAM: start available >= 500 GiB. Warn below 400 GiB or RSS > 100 GiB. Stop below 300 GiB available, RSS > 160 GiB, any swap use, or cleanup-baseline growth > 10 GiB for two consecutive samples.
- Disk: pilot start free >= 10 GiB. Estimate remaining as `mean_success_sample_bytes * remaining_samples * 1.3`; recompute every four samples. Warn below 8 GiB or forecast completion below 6 GiB. Stop starting new samples below 5 GiB. Full-matrix launch requires free space greater than `1.3 * remaining_matrix_estimate + 5 GiB`.
- Poll NVML every second and RAM/disk every five seconds. Persist `gpu_samples.csv`, `ram_samples.csv`, `disk_samples.csv`, per-sample allocator snapshots, and a final `run_status.json` including reason, sample counts, final resources, and code/model/config/direction/input/noise hashes.
- Each sample writes to an independent temporary directory, hashes and validates, then atomically renames. Resume never overwrites successful samples; incomplete attempts move to numbered attempt directories.
- The pilot executes 32 generation calls, 16 decoder replays, float and uint8-simulated re-encoding, creates all reports, then records `AWAITING_REVIEW` and stops even if every scientific gate passes.
- Remote output uses the lowest unused `/root/autodl-tmp/cosmos-experiments/2026-09-16/umi_task6_cross_context_linearity_runNN/`. Local archive uses the matching lowest unused `C:\Users\hongy\Desktop\Semester\Capstone\2026-09-16\task6_cross_context_linearity_runNN`.
- Do not push GitHub, delete remote raw data, run SVD, or start the other five groups in this execution.

## Task 1: Deterministic seed, state, preprocessing, and resource primitives

Use test-driven development. Add failing tests first, run them to record RED, then implement the smallest production code and rerun GREEN.

1. Add `cosmos_umi_capstone/experiments/test_umi_task6_primitives.py` and `umi_task6_primitives.py`.
2. Define the immutable three-state catalog with pinned source commit and exact asset paths. Define the six state/seed groups, but expose a pilot selector that returns only `bridge_0 / seed 0`.
3. Implement deterministic frame-0 preprocessing helpers for aspect-preserving resize and reflection padding to `256x256`. Tests must cover wide, tall, square, odd-dimension, dtype/range, and reflection-padding edge cases without requiring OpenCV at import time.
4. Implement action parsing/validation for exact `(16,10)`, finite numeric values, and stable hashes. Keep asset loading separate from pure validation.
5. Implement the exact 32-call generation plan and 16-call decoder replay plan. Assert Task 5 amplitudes and direction names verbatim.
6. Implement pure resource policy/snapshot evaluation with warning and hard-stop reasons for all GPU, RAM, swap, disk, forecast, and consecutive-growth gates from Global constraints. A hard stop must dominate warnings and return a machine-stable reason code.
7. Implement a canonical terminal `run_status.json` payload builder that includes completed/failed/skipped logical samples, last resource snapshots, and code/model/config/direction/input/noise hashes.
8. Parameterize the existing official precision runtime seed. The public/default behavior remains seed 0, but explicit seed 1 must be observed at actual prepare/noise, sampler, and scheduler seams. Do not weaken any existing identity or compatibility checks.
9. Add focused regression tests proving seed 0 retains the previous identity and seed 1 changes only the predicted-noise seed/hash path while base latent, condition, action, prompt, settings, and direction identities remain fixed.

Acceptance: new primitive tests are green; the existing official/runtime/Task 5 focused tests are green apart from documented environment-only skips; no inference or network access occurs.

## Task 2: Task 6 runtime, atomic runner, preflight, and monitoring

Use test-driven development. Read Task 1 interfaces rather than duplicating them.

1. Add `cosmos_umi_capstone/experiments/test_umi_task6_runtime.py`, `umi_task6_runtime.py`, `test_run_umi_task6_experiment.py`, and `run_umi_task6_experiment.py`.
2. Adapt the validated Task 5 runtime through narrow interfaces so a single resident model can run any approved state and seed sequentially while preserving Task 4 C FP32 behavior. Do not copy the model or retain more than the current sample tensors in memory.
3. Add a generation-free preflight that validates environment/provenance, asset hashes, action `(16,10)`, runtime carrier/mask/index geometry, Task 5 direction hashes, seed routing configuration, and cache-off/FP32 settings. It must fail before model generation on any mismatch.
4. Load the five frozen directions from a validated Task 5 direction bank. Refuse a missing, extra, resampled, reordered, hash-mismatched, wrong-shape, wrong-mask, or non-unit direction.
5. Implement exact group-local baselines and the 30 signed perturbations for 32 generation calls. Same state/action/prompt/base latent/mask/directions and same seed must be bound to every call in one group.
6. Add `umi_reference / seed 0` reuse verification using exactly four engineering-equivalence calls: two baselines and `v0, alpha=0.001` positive/negative. Reuse requires bitwise equality of all required raw outputs; otherwise report that all 32 calls must be rerun. This helper does not execute the full reference group.
7. Implement a baseline-only resource smoke mode. It records pre-load, loaded, per-call, post-call, post-cleanup, and unloaded snapshots, estimates real per-sample disk cost, applies every hard gate, and emits a reviewable decision. It never auto-continues into the pilot.
8. Implement background CSV monitors with one-second GPU cadence and five-second RAM/disk cadence, plus per-sample allocator snapshots. Monitor failures are hard failures. Ensure monitors terminate and flush on normal exit, exception, interrupt, and resource stop.
9. Implement per-sample temporary directories, atomic publication, SHA256 manifests, strict compatibility-bound resume, numbered incomplete attempts, and append-only invocation history. Successful samples are immutable.
10. Implement the pilot state machine: `PREFLIGHT -> RESOURCE_SMOKE -> AWAITING_RESOURCE_REVIEW -> PILOT -> ANALYSIS -> AWAITING_REVIEW`. The default CLI may prepare/preflight, but generation requires explicit `--phase resource-smoke` or `--phase pilot`. Pilot accepts only `bridge_0 / seed 0`, refuses any other group, and stops after its 32 successful generation samples.
11. Write `run_status.json` on every terminal/paused outcome, including OOM and signal interruption. Never auto-retry or start the decoder concurrently.

Acceptance: exact sample counts and call order are proven by tests; simulated resource stops preserve all completed samples; resume is strict and immutable; existing Task 4/5 focused regression tests stay green.

## Task 3: Decoder round trip, scientific analysis, reports, and package

Use test-driven development. Consume successful Task 2 artifacts read-only.

1. Add `cosmos_umi_capstone/experiments/test_umi_task6_decoder.py`, `umi_task6_decoder.py`, `test_analyze_umi_task6.py`, and `analyze_umi_task6.py`.
2. Select exactly eight prediction latents: both group baselines plus `v0` positive and negative at all three amplitudes. Run native BF16 and temporary FP32 decode serially, producing exactly 16 decoder replay records. Verify decoder state/cache restoration on success and failure.
3. For each decoded float final frame, perform direct float re-encoding and explicit `round(clamp(x,0,1)*255)/255` uint8 simulation followed by re-encoding. Save realized inputs and outputs before any visualization encoding.
4. Compute float64 reductions from float32 differences separately for prediction latent, native/FP32 RGB, float round trip, and uint8-simulated round trip. Preserve N/A with field-specific reasons for zero denominators, missing data, nonfinite values, or stopped runs.
5. Recompute every Task 5 gate per group with unchanged thresholds: realized input geometry, baseline floor, one-sided slopes/R-squared, centered-secant consistency, six additivity errors, and 24 held-out prediction errors. Report maximum additivity and holdout error plus failing directions and amplitudes.
6. Compute cross-seed/state derivative cosine and gain differences only when both compared groups exist; label them descriptive with no pass threshold. A pilot must report the other groups as not run rather than failed.
7. Add the mathematical note for `G_{k,s}(z_k+delta)=G_{k,s}(z_k)+J_{k,s}(z_k)delta+R_{k,s}(delta)` and explain centered finite-difference error `O(h^2)+O(u/h)`, finite-precision limitations, and why empirical passage is not a proof of Jacobian existence or a Lipschitz upper bound.
8. Produce JSON/CSV summaries, failure-amplitude tables, decoder/round-trip tables, PNG+SVG plots, `experiment_report.md`, `resource_report.md`, logs, source/config/provenance snapshots, `MANIFEST.sha256`, and deterministic `review_bundle.zip`. Large raw tensors stay outside the bundle.
9. Analysis and packaging must be atomic, idempotent for an already matching complete package, reject stale/tampered raw artifacts, and never mutate raw samples.

Acceptance: synthetic exact-linear fixtures pass all gates; nonlinear, noisy, quantized, missing, stopped, and zero-denominator fixtures are reported correctly; exact counts are enforced; Task 4/5 analysis regressions remain green.

## Task 4: Remote Linux verification and bridge_0 seed-0 pilot

This task begins only after Tasks 1-3 pass task review and a whole-branch review. It is operational and must not change scientific thresholds or source behavior.

1. Verify remote framework/checkpoint/VAE hashes, framework commit, CUDA environment, interpreter `torch 2.10.0+cu130`, GPU idle state, RAM, swap, disk, and absence of competing experiment processes. Stop on mismatch.
2. Fetch the four pinned Bridge assets to a local staging directory, verify Git LFS entities and SHA256, extract/validate frame 0 locally, then upload the small verified bundle and reviewed code. Do not download models.
3. Run the full remote Linux test suite and record exact pass/skip/fail counts. Any real regression blocks inference.
4. Run the generation-free preflight and independently verify its provenance, shapes, indexes, mask, direction hashes, actions, seeds, and settings.
5. Run one baseline resource smoke only. Stop at `AWAITING_RESOURCE_REVIEW`. The controller reviews resource CSVs, allocator metrics, cleanup recovery, and disk forecast before authorizing pilot continuation.
6. If and only if the smoke gates pass, run `bridge_0 / seed 0` sequentially: 32 generations, then 16 decoder replays, then float/uint8 re-encoding and analysis. Monitor resource logs continuously and stop fail-closed on any hard gate.
7. Independently recompute counts, hashes, gate metrics, manifests, and headline conclusions from raw artifacts. Confirm final status `AWAITING_REVIEW` and that no other group was started.
8. Copy the lightweight review bundle and the complete approved pilot archive to the local dated directory, verify SHA256 after transfer, and leave remote raw data intact.

Acceptance: execution either produces a fully reviewable pilot with exact counts and resource compliance or a truthful stopped run with all evidence preserved. In both cases it stops before the remaining five groups.
