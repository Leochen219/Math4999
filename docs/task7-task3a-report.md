# Task3a numerical API report

Status: DONE for the bounded pure-NumPy numerical API.  Task3b may add
artifact loading, engineering evidence, CLI, CSV/plot/report generation, and
must preserve these calculations.  No model, GPU, remote filesystem, upload,
dependency installation, or raw-data mutation was used.

## API and data contract

All tensor arguments are materialized `numpy.ndarray` values with
`dtype=numpy.float32`, finite values, and nonempty shape.  A condition mask is
an explicit boolean array with the same shape as the input carrier and at
least one selected coordinate.  `output_mask` is optional when output space
has the same shape as the input; it is required when the observed output has a
different shape.  Invalid engineering arrays raise `EngineeringDataError`,
not a scientific `FAIL`.

* `fp32_difference(left, right) -> ndarray[float32]` performs explicit FP32
  subtraction.  `rms64(value, mask=None) -> float` and
  `cosine64(left, right, mask=None) -> float | None` use FP64 reductions;
  zero cosine denominators return `None`.
* `byte_equal(left, right) -> bool` compares dtype, shape, and contiguous
  bytes, so `+0.0` and `-0.0` differ.  `float32_difference`,
  `arrays_byte_equal`, and `compute_input_geometry` are readable aliases.
* `input_geometry(actual_plus, actual_minus, direction, mask) -> dict` returns
  independent `h_plus`/`h_minus`, signed direction cosines, opposite cosine,
  and exact outside-mask flags.  It also includes Task5-compatible key names
  (`plus_input_cosine`, `minus_input_cosine`, and
  `plus_minus_input_cosine`).
* `one_direction_window(amplitudes, direction, plus_inputs, minus_inputs,
  plus_outputs, minus_outputs, baseline_output, mask, *, floor=0.0,
  output_mask=None) -> dict` requires exactly three strictly increasing
  amplitudes.  It computes one-sided log-log slope/R2 against each side's
  actual length and the actual-step central quotient
  `(F_plus - F_minus) / (h_plus + h_minus)`.  Adjacent relative change uses
  the earlier quotient as denominator.  It returns raw point/geometry/secant
  scalars, reasons, `PASS`/`FAIL`, and `additivity_status=NOT_TESTED`.
* `propagation_metrics(delta0, delta1, delta2, mask, *, floor0=0.0,
  floor1=0.0, floor2=0.0) -> dict` computes condition-mask RMS values,
  `A1`, `A2`, and `incremental=A2/A1`, with `None` and reasons for zero
  denominators plus per-step floor/reliability flags.
* `fixed_beta_prediction(*, actual_delta1, beta_plus_input,
  beta_minus_input, beta_plus_output, beta_minus_output, baseline_output,
  actual_delta2, mask, beta=0.1, local_window_pass=None,
  response_reliable=None, output_mask=None) -> dict` enforces beta `.1`,
  forms the actual-step quotient from the supplied beta=.1 pair, and predicts
  `delta2_hat=q * RMS(delta1)`.  `Eprop` is descriptive when unreliable;
  `PASS` requires both explicit reliability flags and the `Eprop <= .10` gate.
  The returned `reliable` field reports local geometry/noise/evaluation
  reliability independently of prediction status: an above-threshold Eprop
  is a reliable scientific `FAIL`, while an undefined zero evaluation
  denominator is `N/A` and unreliable.  The actual `delta1` ray is used
  directly; no projection onto `v0` occurs.
* `evaluate_fixed_beta_predictions(rays, *, q_by_ray, mask,
  local_window_pass=None, response_reliable=None, output_mask=None) -> list`
  requires six independent ray records and returns all six rows.  Flags may be
  per-record or six-element sequences, and only bool/`numpy.bool_`/`None` are
  accepted, so one failed ray cannot hide another.  A zero ray or zero
  evaluation denominator is explicitly unreliable with a reason.

Natural aliases are exported for Task3b (`analyze_one_direction_window`,
`compute_propagation`, `predict_fixed_beta`, and `fit_loglog`).

## Tests and evidence

The focused command, run from `cosmos_umi_capstone/experiments`, was:

```text
C:\Users\hongy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest test_analyze_umi_task7 -v
```

Final result: **16 tests, 16 passed, 0 failures**.  Coverage includes the
two-coordinate affine oracle with a changed second-step ray, FP32 tolerance
and FP64 reductions, multidimensional cosine, byte-level signed-zero identity,
actual asymmetric input lengths, current-q denominator, nonlinear failed
window with no dropped endpoint, zero/high floors, rounding disappearance,
zero propagation denominators, wrong mask/shape/dtype/nonfinite engineering
inputs, fixed-beta rejection of a tempting beta=.2, explicit reliability
flags, retention of all six prediction rows, explicit zero evaluation
denominator reasons, zero-ray reliability, strict per-ray flag typing, and
separation of reliability from Eprop PASS/FAIL.

The first RED run was genuine: before implementation, importing the focused
test failed with `ModuleNotFoundError: No module named 'analyze_umi_task7'`.
After the implementation and fixture correction (the oracle has
`M1 @ (2,1) = (1,5)`), the focused suite produced `Ran 16 tests ... OK`.
The bundled interpreter also completed:

```text
python.exe -m py_compile analyze_umi_task7.py test_analyze_umi_task7.py
```

No broad test discovery was run by this bounded task.  Initial scoped commit
before this report's self-reference was `8a85396`; the final amended commit
is reported to the parent after the amendment.
