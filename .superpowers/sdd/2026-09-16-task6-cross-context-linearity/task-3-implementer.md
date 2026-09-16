# Task 3 implementer report

## Scope

Implemented only the offline decoder round-trip and scientific analysis layer:

- `cosmos_umi_capstone/experiments/umi_task6_decoder.py`
- `cosmos_umi_capstone/experiments/analyze_umi_task6.py`
- focused tests in `test_umi_task6_decoder.py` and `test_analyze_umi_task6.py`

The decoder selects exactly eight logical latents and expands them to exactly
16 serial native-BF16/temporary-FP32 replay calls.  It restores the decoder
state/cache in `finally`, performs direct-float and explicit uint8-simulation
re-encoding, and never invokes generation.  The analyzer consumes validated
raw arrays read-only, delegates the fixed Task 5 six-additivity/24-holdout
gates, keeps spaces separate, writes float32 tensor copies with float64
reductions, and stages an idempotent manifest-bound review package.

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
Ran 4 tests ... OK
python -m unittest discover -s cosmos_umi_capstone/experiments -p 'test_analyze_umi_task6.py' -v
Ran 4 tests ... OK
python -m py_compile cosmos_umi_capstone/experiments/umi_task6_decoder.py cosmos_umi_capstone/experiments/analyze_umi_task6.py
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
`decode_prediction_latent` adapter.  The real Task 4/6 integration must inject
the loaded runtime and condition encoder at Task 4 execution time.  This task
does not claim scientific results without a completed pilot; missing/stopped
groups are reported as incomplete/not-run evidence.
