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

## Verification correction and round 3 follow-up

The initial post-round-3 compile claim was corrected after the bundled
interpreter exposed an indentation error in decoder-space analysis. The error
was fixed before the current commit. The decoder now reports per-space
one-sided slopes/R2, adjacent centered-secant cosine/relative change, and
candidate/floor failure reasons. Prediction-latent metrics are emitted once
from the native replay pair rather than duplicated as independent precision
spaces. Actual consumed deltas are validated from saved consumed input when
available and are not incorrectly required to be bitwise equal to requested
target deltas. Public analysis validates normalized sample identity and rejects
missing RGB evidence; carrier-shaped `output_full` fallbacks are sliced by the
validated predicted frame indexes.

Verified with bundled interpreter:

```text
C:\Users\hongy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m py_compile <all experiments/*.py>
PASS

python -m unittest discover -s . -p 'test_umi_task6_decoder.py' -v
Ran 11 tests in 5.082s ... OK
python -m unittest discover -s . -p 'test_analyze_umi_task6.py' -v
Ran 11 tests in 1.689s ... OK
python -m unittest discover -s . -p 'test_umi_task6_runtime.py' -v
Ran 18 tests in 1.630s ... OK
```

The broad discover run completed 345 tests; Task 6/Task 5/precision-runtime
and primitive tests passed, while 24 unrelated pre-existing precision-contrast
and calibration-environment tests failed because this bundled environment
lacks the checked-in calibration hash/line-ending state and matplotlib. Those
failures do not involve the changed Task 6 files. No remote/model/SVD action
was taken.

Focused regression after the correction (bundled interpreter, from
`cosmos_umi_capstone/experiments`):

```text
python -m unittest test_umi_task6_decoder test_analyze_umi_task6 test_umi_task6_runtime test_umi_task6_primitives test_umi_task5_decoder test_umi_task5_primitives test_umi_task5_runtime test_analyze_umi_task5 test_umi_precision_runtime test_umi_precision_official test_umi_precision_primitives test_umi_precision_storage
Ran 132 tests in 35.034s ... OK
```

Latest implementation commit before this report-only update:
`cf2e674480dd1291042291ff06e917d66c2833bc`.

## Review round 4 evidence boundary

The public schema now requires every exact logical sample to carry complete
spec, group, seed, model-seed, `z_bar`, mask, frozen direction, target delta,
consumed input, and realized delta evidence. Missing consumed input, duplicate
normalized IDs, extra IDs, or mismatched identities fail closed. Predicted
latent fallback is now restricted to an exact four-frame block; carrier-shaped
outputs are sliced only by the validated runtime prediction indexes. The
adapter exposes a decoder seam that delegates to the resident runtime, and the
decoder persists the four-frame `predicted_latent` separately from any full
decode carrier. A public end-to-end test now performs real preflight, resource
smoke, pilot runner/store publication of all 32 records, reload, and strict
coefficient analysis. A real 16-record decoder tree test covers all five named
spaces, realized input steps, derivative tensors, slopes/R2, adjacent secant
metrics, and summary gates.

Bundled interpreter verification after these changes:

```text
py_compile all experiments/*.py: PASS
Task6 decoder + analysis + runtime focused suite: Ran 46 tests in 25.560s ... OK
Focused Task6/Task5/precision runtime regression: Ran 138 tests in 51.871s ... OK
```

Latest implementation commit: `16d2fb26a63e2fc6b394e3ab76485173171a01a5`.

The focused decoder fixture also verifies that removing a required baseline
field produces a field-specific `N/A` summary (`reason=missing_pair`) rather
than dropping the scientific space or fabricating a slope.

## Review round 5 evidence boundary

The adapter now routes runtimes exposing only the official `model`/`ops`
surface through the validated Task 5 `replay_decode` seam, while retaining
the existing direct seam for testable resident runtimes. Decoder analysis
requires the exact v2 replay schema and 16 IDs/specs, terminal call count,
artifact hashes, decoder code identity (including `umi_task5_decoder.py`),
and a hash-validated complete 32-sample raw run. Runtime and encoder resume
bindings are rejected when absent or class-only. Sample IDs now bind alpha
ordinal/value and sign, and manifest exclusions are root-relative so nested
MANIFEST/lock files are evidence rather than silently ignored. The decoder
fixture uses realized alpha-linear responses and asserts five named spaces,
unit slopes/R2, near-unit secant cosine, and near-zero secant change; missing
raw input evidence yields explicit `N/A/missing_pair` rows.

Bundled interpreter verification after round 5:

```text
py_compile changed experiments: PASS
Task6 decoder + analysis + runtime focused suite: Ran 48 tests ... OK
Focused Task6/Task5/precision runtime regression: Ran 140 tests ... OK
```

Round 5 also adds an OfficialPrecisionRuntime-like adapter test (model/ops
only) proving delegation to Task 5 `replay_decode`, forged decoder-plan
rejection, exact realized-alpha linear decoder evidence, and raw-manifest
binding before decoder analysis. No remote/model execution was performed.
