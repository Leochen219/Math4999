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

## Verification

Using the required bundled NumPy interpreter:

```text
python -m py_compile run_umi_task7_experiment.py
python -m unittest test_run_umi_task7_experiment test_umi_task7_runtime \
    test_umi_task7_encoder test_umi_task7_official_deferred -v
Ran 20 tests in 0.315s
OK (skipped=2)
```

The two skips are the existing optional real-Torch fixtures; no model, GPU,
download, environment installation, or remote mutation was performed. Focused
runner tests cover CLI bindings, exact stage plans/counts, strict resource
failures, root containment, atomic failed-attempt preservation, observed
failed invocation counts, and generation-free preflight.

The independent analyzer remains responsible for scientific A/B/C decisions;
the runner only consumes an explicitly bound C gate.

## Current concerns / release boundary

This worker did not perform a model, GPU, remote, or live-framework call. The
default `main` path now constructs the real Task 6 `OfficialRuntimeFactory`,
performs the strict preload check, wraps its returned tokenizer with
`FeedbackEncoder`, and binds the underlying official runtime; that wiring is
not empirically validated here because the approved runtime/model environment
is intentionally out of scope for this CPU worker. The seven runner tests use
an injected CPU monitor and executor, so they do not claim that a live factory
build, NVML/cgroup sampler, smoke first-G parity, or CUDA cleanup succeeds.
The smoke path still needs main’s live review to confirm the recorded first-G
latent is checked against the historical baseline before acceptance.

The immutable-source validator has strict checks for the reviewed Task 6 plan,
raw/decoder hashes, z0/mask/v0 geometry, and perturbation reconstruction, but
was not run against the remote historical source tree in this worker. Main
should independently run preflight/smoke with the released source/contract,
review actual resource and operation evidence, and only then release A/B/C.
