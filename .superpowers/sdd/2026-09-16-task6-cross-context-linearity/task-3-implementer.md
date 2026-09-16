# Task 3 implementer report

## Scope

Implemented the offline decoder round-trip and scientific analysis layer, with
the review round closing the evidence-boundary and resume gaps:

- `cosmos_umi_capstone/experiments/umi_task6_decoder.py`
- `cosmos_umi_capstone/experiments/analyze_umi_task6.py`
- focused tests in `test_umi_task6_decoder.py` and `test_analyze_umi_task6.py`

The decoder selects exactly eight logical latents and expands them to exactly
16 serial native-BF16/temporary-FP32 replay calls in a separate derived root.
It requires an encoder, restores and validates decoder state/cache in
`finally`, performs direct-float and explicit uint8-simulation re-encoding,
uses atomic per-replay publication and a decoder manifest, and never invokes
generation. The analyzer consumes validated raw and decoder manifests
read-only, preserves the raw tree, delegates the fixed Task 5 six-additivity /
24-holdout gates while explicitly recording FP32 subtraction/FP64 reductions,
keeps the five decoder/round-trip spaces separate, emits all-six-group
`NOT_RUN` rows, and stages an idempotent package with deterministic ZIP and
manifest verification.

## TDD evidence

### RED

Before adding the Task 3 modules, the planned imports were absent:

```text
python -m unittest test_umi_task6_decoder test_analyze_umi_task6
ModuleNotFoundError: No module named 'umi_task6_decoder'
```

### GREEN

From the worktree root:

```text
python -m unittest discover -s cosmos_umi_capstone/experiments -p 'test_umi_task6_decoder.py' -v
Ran 7 tests ... OK
python -m unittest discover -s cosmos_umi_capstone/experiments -p 'test_analyze_umi_task6.py' -v
Ran 8 tests ... OK
python -m py_compile cosmos_umi_capstone/experiments/umi_task6_decoder.py cosmos_umi_capstone/experiments/analyze_umi_task6.py
git diff --check
```

Task 4/5 focused regressions (from `cosmos_umi_capstone/experiments`):

```text
python -m unittest test_umi_task5_decoder test_umi_task5_primitives test_analyze_umi_task5 test_umi_precision_runtime test_umi_precision_official
Ran 42 tests ... OK
```

`git diff --check` and compile checks pass.  No network, remote inference,
asset download, SVD, or raw-run mutation was performed.

## Boundaries and concerns

The official decoder is intentionally reached through the existing verified
Task 5 low-level seam when a runtime does not expose the preferred
`decode_prediction_latent` adapter. The real Task 4/6 integration must inject
the loaded runtime and condition encoder at Task 4 execution time. This task
does not claim scientific results without a completed pilot; missing/stopped
groups are reported as incomplete/not-run evidence. The controller should
independently review the Task 5 delegation against the official runtime before
remote execution.

## Review round 2 hardening

The implementation now derives `s_z` and `c01/c12` from saved `z_bar`, mask,
and frozen direction tensors rather than defaulting scientific values. Task 6
residuals, responses, finite differences, additivity, and holdout predictions
use explicit FP32 subtraction and FP64 reductions; Task 5 source is unchanged.
Decoder inputs validate the completed 32-sample raw status and raw manifest,
and the derived decoder root binds raw manifest, group, runtime, encoder, plan,
config, and artifact hashes. Status and config are refreshed into the decoder
manifest after every published replay, allowing strict partial resume without
overwriting successes.

Additional verification:

```text
python -m unittest discover -s cosmos_umi_capstone/experiments -p 'test_umi_task6_decoder.py' -v
Ran 8 tests ... OK
python -m unittest discover -s cosmos_umi_capstone/experiments -p 'test_analyze_umi_task6.py' -v
Ran 9 tests ... OK
python -m py_compile cosmos_umi_capstone/experiments/umi_task6_decoder.py cosmos_umi_capstone/experiments/analyze_umi_task6.py
git diff --check
```

The public analysis path rejects stopped/partial generation runs and missing,
invalid, or unbound decoder evidence. Packages bind raw and decoder manifests,
fixed configuration, and hashes of relevant Task 6/5 runtime sources. No
remote execution or Task 4 work was performed.

## Review round 3 implementation

The public Task 6 core now performs its own point, derivative, slope/R2,
additivity, and 24-row holdout calculations; it no longer imports or calls the
Task 5 analyzer. Every tensor subtraction is explicitly FP32 and only norms,
cosines, fits, and other reductions use FP64. Plan detail is exposed in the
result and is derived from saved `z_bar`, mask, and frozen direction evidence;
nontrivial combination coefficients are not defaulted. Decoder analysis now
includes the prediction-latent, native RGB, temporary-FP32 RGB, and both
condition-latent round-trip spaces, records N/A reasons, and persists centered
finite-difference tensors. Decoder code identity, condition-encoder failure
restoration, strict raw inventory, and byte-stable README ZIP metadata were
also hardened. New tests cover strict raw extra-file rejection, partial
decoder resume without replaying successes, encoder failure restoration,
public coefficient derivation, and fresh-package ZIP determinism.

Static verification in this Windows worktree:

```text
python -m py_compile cosmos_umi_capstone/experiments/*.py  [per-file invocation: PASS]
git diff --check  PASS
```

The local Windows interpreter has no NumPy installation, so the expanded
NumPy-dependent Task 3 and full regression suites must be run by the parent in
the configured experiment interpreter. No remote/model/SVD action was taken.
