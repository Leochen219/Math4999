# Task 7 Task 2b — bounded dynamic feedback runtime

## Scope and implementation

Implemented `cosmos_umi_capstone/experiments/umi_task7_runtime.py` as a
request-local adapter around one loaded `OfficialPrecisionRuntime`. The
adapter does not load a model or create a second resident model, and it never
uses `scope="module"`.

The public API is:

```python
runtime = FeedbackRuntime(
    official_runtime,
    encoder=FeedbackEncoder(tokenizer_vision_gen),
    z0=frozen_task6_full_carrier,
    mask=official_runtime.inputs.geometry.mask,
    condition_indexes=official_runtime.inputs.geometry.condition_indexes,
    v0=frozen_task6_v0,
    seed=0,
)

condition_only = runtime.extract_condition(full_carrier)
record = runtime.step(condition_only, step_index=0)  # step_index is 0 or 1
```

`extract_condition(fullcarrier)` and `embed_condition(condition_only)` use
the authoritative temporal axis/indexes and reject flattening or mismatched
shapes. Embedding starts from the immutable historical `z0`, replaces only
the condition coordinates, and asserts bytewise exterior preservation.

Each `step` binds an identity-bearing dynamic inputs object to the resident
official runtime for the duration of the request. `for_spec` returns the exact
request full carrier. The selected runtime seeds are the fixed pair
`[0, 1]`, selected by `step_index`; the requested seed is observed at actual
prepare, sampler, and scheduler boundaries. Preparation resolves the
installed `arch_invariant_rand` (the official `cosmos_framework.utils.misc`
function in production, an explicitly attached function only in the fixture),
draws at the full carrier shape/dtype/device before flattening, blends the
official mask layout, and records a prediction-region-only noise hash. Seed 0
must match the resident Task 6 prepared noise outside the condition mask.

The production path requires the loaded model's explicit `tensor_kwargs_fp32`
mapping, including both `dtype` and `device`; it fails closed when either is
absent. Tests that exercise missing official noise use an isolated mock
`cosmos_framework.utils.misc` module, so an installed fallback cannot
accidentally satisfy the boundary test. The capture records readbacks from
the actual clean, initial, and reference slots after preparation rather than
treating the requested carrier as evidence.

The generation call is exactly:

```python
official_runtime.execute(spec, dynamic_inputs,
                          scope="full", decode_policy="deferred")
```

The runtime requires all 30 denoiser/scheduler steps and explicit FP32 G
dispatch/backend/condition/noise evidence. A record distinguishes the one G
generation invocation (`operation_counts["G"] == 1`) from
`denoiser_steps == 30`; it also records actual scheduler seeds and all
per-step consumed condition masks.

The deferred full latent is passed once to the decoder; the predicted temporal
slice is retained separately. Production FP32 decoding reuses Task 1's
precision snapshot, backend guard, dispatch observer, state-dtype evidence,
and cache cleanup. Conversion happens before the dispatch observer, so BF16 →
FP32 state conversion is not misreported as model compute. Decoder input is
the full latent, one inner decode invocation is required, and normalization
uses float64 arithmetic followed by float32 storage with explicit clamp/layout
metadata. The last decoded frame is passed once to the Task 1
`FeedbackEncoder` as `precision="temporary_fp32"`; its condition-only output
and separately embedded full carrier are both preserved.

Returned record schema (arrays are independent current-request CPU copies):

```text
step_index, seed, request_seed
condition_input, condition_input_full
generation                    # complete official deferred capture
full_latent, predicted_latent
prediction_noise, prediction_noise_hash
denoiser_steps
decoder_input_full, decoder_raw_output, decoded_last_rgb, normalization
encoded_condition, encoded_carrier
encoder_arrays.actual_encoder_input
actual.prepared_condition, actual.first_condition, actual.last_condition,
  actual.initial_condition, actual.reference_condition, actual.condition_steps
encoder                       # Task 1 precision/evidence payload
evidence.operation_counts     # G=1, D=1, E=1
evidence.arch_invariant_rand
evidence.prepare_seed, evidence.sampler_seeds
next_consumption_check        # "deferred_to_next_step"
elapsed_seconds
```

Failures retain partial capture/counters on the exception and restore the
resident runtime's patched preparation/input/seed state. Cleanup attempts all
components and does not retry inference. Decoder restoration also restores
exact Task 1 tensor slots, constants, dtype, training metadata, backend flags,
hooks, and decoder caches.

## Tests and evidence

Genuine RED, before the replacement runtime existed:

```text
python -m unittest test_umi_task7_runtime -v
setUpClass ... ModuleNotFoundError: No module named 'umi_task7_runtime'
Ran 0 tests ... FAILED (errors=1)
```

Focused fixture tests after implementation:

```text
python -m unittest test_umi_task7_runtime -v
Ran 7 tests in 0.140s
OK (skipped=1)
```

The skip is the real-Torch decoder fixture because the bundled Windows
interpreter has NumPy but no PyTorch. The fixture is present and runs when the
existing Torch CPU interpreter is available; it checks one decode, positive
FP32 dispatch evidence, BF16 state byte restoration, training restoration,
and removal of the temporary `inner.decode` instance attribute.

The focused run also checks exact `_prepared_for_call` instance-attribute
ownership: an absent shadow is removed after both success and failure.

Dependency/regression-focused run:

```text
python -m unittest test_umi_task7_runtime test_umi_task7_encoder \
    test_umi_task7_official_deferred test_umi_precision_official \
    test_umi_precision_runtime
Ran 49 tests in 24.666s
OK (skipped=2)
```

The second skip is the existing Task 1 real-Torch fixture. `py_compile` of the
runtime and focused tests exited 0. No model, GPU, model download, framework
edit, or environment installation was performed. The authorized remote CPU
upload was not performed from this worker; therefore no remote real-Torch
result is claimed here.

The untracked Task 7 runner draft was not modified; runner integration remains
Task 2c. This report does not claim whole Task 2 completion or live-stage
completion.
