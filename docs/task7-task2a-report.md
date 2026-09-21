# Task 7 Task 2a — full-generation deferred decode seam

## Scope

This bounded dependency adds an explicit `decode_policy` to the existing
`OfficialPrecisionRuntime.execute` API. `native` is the unchanged default;
`deferred` is accepted only for `scope="full"`. Deferred execution still runs
the complete configured denoiser and scheduler loop, publishes all existing
FP32/BF16 boundary evidence and the full predicted latent, then restores the
request-local network/hooks/cache state without invoking `model.decode` or
retaining a decoder frame. `validate_capture(..., require_decoded=False)` is
the corresponding strict validator and rejects a missing policy or a dummy
decoded frame.

The existing `scope="module"` step-zero diagnostic remains unchanged and is
not treated as deferred full generation. Task 7's later request-local runtime
can therefore call the deferred full path and perform exactly one separate
FP32 decoder/encoder phase after denoiser cleanup.

## Focused RED/GREEN evidence

RED (before the implementation):

```text
TypeError: OfficialPrecisionRuntime.execute() got an unexpected keyword argument 'decode_policy'
```

GREEN:

```text
python -m unittest experiments.test_umi_task7_official_deferred -v
Ran 5 tests in 0.085s
OK
```

The focused module reuses the existing official CPU fixture and checks
configured 30-step generation, 30 scheduler updates, output equality with
native generation, zero deferred decode calls, policy/scope rejection,
validator rejection of dummy frames/missing evidence, original-net identity,
working-copy collection, and exception-safe restoration.

## Files

- `cosmos_umi_capstone/experiments/umi_precision_official.py`
- `cosmos_umi_capstone/experiments/umi_precision_runtime.py`
- `cosmos_umi_capstone/experiments/test_umi_task7_official_deferred.py`

No GPU/model/upload call was made. The real Torch/CUDA decoder operation
evidence and dynamic seed/condition binding remain Task 2b work after main
reviews this seam.
