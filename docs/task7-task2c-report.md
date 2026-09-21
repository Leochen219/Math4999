# Task 7 Task 2c — operational staged runner

## Scope

Implemented the source-bound Task 7 runner in
`cosmos_umi_capstone/experiments/run_umi_task7_experiment.py`. The runner keeps
historical Task 6 raw and decoder trees read-only, uses one reviewed
`OfficialRuntimeFactory`/`Task6RuntimeAdapter` resident runtime, wraps its
loaded tokenizer with the reviewed Task 1 `FeedbackEncoder`, and passes the
underlying `OfficialPrecisionRuntime` to the approved Task 2b
`FeedbackRuntime`.

`preflight` never imports the model loader or calls inference. Live stages run
strict preload resource checks before factory construction. Factory z0, mask,
and v0 are byte-identity checked against the frozen source artifacts; the
historical z0 is never re-encoded or replaced.

## CLI

```text
python run_umi_task7_experiment.py \
  --stage preflight|smoke|A|B|C \
  --run-dir RUN_DIR [--resume] \
  --raw-root TASK6_RAW_ROOT --decoder-root TASK6_DECODER_ROOT \
  --framework-root FRAMEWORK --checkpoint CHECKPOINT --vae VAE \
  --action ACTION_JSON --video VIDEO --launch-contract CONTRACT_JSON \
  --task5-root TASK5_ROOT --gpu-index 0
```

Formal stages are explicit and never advance automatically. `smoke` is an
engineering attempt and is excluded from formal A/B/C counts. A is 16 source
frame encodes (eight native and eight temporary FP32); B is eight trajectories
at steps 0/1 (16 G/D/E calls), and C is two shared baselines plus six actual
step-1 rays at beta .1/.2/.4 and both signs (38 calls per G/D/E). C requires an
external analyzer gate bound to this source and code digest. Zero rays are
recorded as `SKIPPED_C`, never fabricated.

## Source and artifact schema

Preflight validates all eight `baseline_pre`,
`v0_alpha_00/01/02_plus/minus`, and `baseline_post` pairs:

- each source status/record artifact hash is checked;
- raw `output_full.npy` equals decoder `decoder_input_full_latent.npy` by hash;
- Task 6 plan provenance binds prompt, action, bridge_0/seed0, geometry, and
  actual seed/model-seed metadata;
- one immutable z0/mask is shared, baseline consumed input equals z0, and
  every perturbation reconstructs observed float32 z0 +/- alpha*s_z*v0;
- six perturbation direction arrays are byte-identical v0, baseline direction
  evidence is zero when present, and decoder condition shape is the exact
  condition-only temporal shape (not merely equal element count).

Successful new samples contain `record.json`, independently stored `.npy`
arrays, and `status.json` with artifact SHA-256 hashes, required artifact set,
and observed G/D/E counts. Failed attempts retain partial evidence under
numbered `.attempt.NNN`/`.failed-attempt.NNN` directories. A process lock
prevents a second runner. Resume validates the complete binding (source,
Task-7 code, config, model, VAE, framework, z0, mask, v0, and noise policy),
rechecks every successful artifact, and refuses automatic retry of failed or
resource-stopped runs.

## Resource policy

The runner reuses Task 6 `ResourceMonitor` and `evaluate_resources`. It requires
actual GPU, RAM, cgroup, disk, RSS, and swap fields before live start; missing
or non-finite fields are hard stops. Task 7 applies the approved preload GPU
<=1 GiB/disk >=10 GiB limits, smoke peak allocated <=35 GiB/NVML <=45 GiB,
GPU >75/free <20/reserved >65 stops, cgroup <20 warning/<10 stop, host RAM
available <300/RSS >160/swap >0 stops, cleanup-growth stops, and the strict
1.3*measured-success-bytes*remaining + 5 GiB forecast before launching a
sample. There is no deletion or retry path after an OOM or monitor failure.

Live orchestration now takes a run-wide lock before preload, pins the launch to
GPU 0 with `HF_HUB_OFFLINE=1`, and keeps factory unload plus monitor stop in a
single cleanup path. After factory construction, the approved
`observe_live_launch`/`validate_launch_contract` path is run and its evidence is
persisted before the first G call. Formal samples persist synchronous `pre_call`,
`post_call`, and `post_cleanup` monitor captures when the production monitor is
used; Python/Torch sample state is released and peak stats are reset between
samples; accumulated allocated/reserved/NVML peaks are derived from monitor rows,
and ambiguous post-load all-zero Torch allocation telemetry is a hard stop.
Smoke requires historical first-G latent parity, a complete post-call resource
gate, the 35/45 GiB accumulated peak limits, measured artifact publication,
and final cleanup capture before `SMOKE_COMPLETE` is written. Owned monitor
telemetry is stored under stage/attempt-specific directories so later stages or
resumes do not overwrite earlier evidence.

Failure accounting retains returned G/D/E counts through smoke parity/peak
gates, publication, factory unload, and monitor shutdown. Root `run_status.json`
stores those live counts separately from `error_operation_counts`, preserves
the formal `stage_status`, and records the original smoke feedback exception
capture (including partial diagnostics) without replacing it with zero counts.
Setup-boundary evidence (including observed loader-tokenizer encode calls and
failures at the `factory.loader_payload_encoder.encode` boundary) is persisted
under stage/attempt-scoped paths such as `setup/smoke/setup_evidence.json` and
`setup/a-attempt-001/setup_evidence.json`; smoke/stage/root status records
reference the immutable path. The observer snapshots `__dict__` ownership for
both the factory loader and tokenizer encode slot, removes temporary shadows
when a class method was originally inherited, preserves instance overrides,
and fails closed on installation/restoration errors. Setup observations remain
separate from formal G/D/E counts. A smoke completion is changed to a terminal
cleanup failure if shutdown/unload fails, so it cannot be reused.

Complete-stage resume accepts preserved `KeyboardInterrupt`/`InterruptedError`
attempts only when each entry is plan-bound, structurally valid, count-valid,
duplicate-free, and matched to a preserved failed-attempt artifact. Genuine
failed/resource-stop attempts, malformed or unbound entries, orphaned attempt
evidence, and inconsistent totals remain terminal/rejected. A verified
COMPLETE resume is read-only and performs no executor/factory calls.

The C analyzer gate must contain these exact lineage fields in addition to the
existing scientific/source/code fields:

```text
a_run_status_sha256
a_samples_manifest_sha256
b_run_status_sha256
b_samples_manifest_sha256
```

`*_run_status_sha256` is the SHA-256 of `stages/{A,B}/run_status.json`.
`*_samples_manifest_sha256` is the SHA-256 of canonical JSON (sorted keys,
compact separators) mapping each listed completed/skipped sample ID to the
SHA-256 of that sample's `status.json`. The runner recomputes both A and B
values before loading C and rejects foreign or incomplete lineage.

## Follow-up: real Decoder record compatibility

The immutable Decoder source format uses `record.json` (not `status.json`) at
each temporary-FP32 replay directory. The runner now reads that file through
the existing strict artifact-manifest checker, requiring top-level
`status: "success"`, every required `.npy` artifact and every manifest hash to
match. It also binds `spec.sample_id`, `spec.replay_id`, `spec.seed`,
`spec.state`, and `spec.decode_precision` to the requested
`bridge_0__seed_0__<name>` / `...__temporary_fp32` replay. Missing, non-success,
hash-tampered, or precision-mismatched records fail closed; no fallback status
is fabricated.

## Verification

Using the required bundled NumPy interpreter:

```text
python -m py_compile run_umi_task7_experiment.py
python -m unittest test_run_umi_task7_experiment
Ran 46 tests; OK
python -m unittest test_run_umi_task7_experiment test_umi_task7_runtime \
    test_umi_task7_encoder test_umi_task7_official_deferred
Ran 56 tests; OK (skipped=2)
```

The dependency aggregate is 56 tests with 2 existing optional real-Torch skips
(`test_run_umi_task7_experiment test_umi_task7_runtime test_umi_task7_encoder
test_umi_task7_official_deferred`). No model, GPU, download, environment
installation, or remote mutation was performed. Focused runner tests cover
CLI bindings, exact plans/counts, strict resource failures, protected-root
containment, source-contract historical action hashing, atomic artifact
references, interrupted resume accounting, all-16 A orchestration, A encoder
and post-parity failure counts, B own-step resume, approved full-carrier
feedback evidence, all-38 C ray/probe inputs, C skip resume accounting, real
Task-6 `ResourceMonitor` fake samplers, strict Torch telemetry/peak reset,
stage-scoped monitor paths, smoke cleanup failure status, and resume identity
guards. Round3 adds weak-reference output release, live smoke/formal root
counter preservation, setup encoder observation/restoration, no-resume smoke
lockout, strict COMPLETE counter reconstruction/duplicate rejection, and
foreign A/B gate lineage rejection. Round4 adds original smoke exception-capture
preservation, exact class-vs-instance observer restoration, non-overwriting
setup evidence paths with status references, and verified interrupted-attempt
COMPLETE resume/no-op behavior.

The independent analyzer remains responsible for scientific A/B/C decisions;
the runner only consumes an explicitly bound C gate.

## Release boundary

This worker did not perform a model, GPU, remote, or live-framework call. The
CPU integration tests use injected factory/monitor seams and therefore do not
claim that the installed CUDA runtime, NVML sampler, or historical source tree
has passed a release smoke. Main must independently run the immutable-source
preflight and the real smoke with the released contract, review actual
operation/resource evidence, and only then release formal A/B/C stages.
