# UMI analytic precision calibration

This is a CPU-only method-calibration layer; it does not load Cosmos, SSH, start GPU generation, or download/upgrade dependencies.

## Fixed setup

- Matrix: fixed `256x128` FP32 values, seed `20260913`, scale `0.088388347648318433`.
- Input/direction: deterministic `x` and unit-L2 `v`; `v` norm is `0.99999999999999989`.
- The same represented FP32 weight values are used in FP64 and FP32 paths; the FP64 cast is lossless.
- Input-quantized mode is Task 1 BF16 round-to-nearest-even represented again as FP32.
- Low-precision compute is explicitly `float32` rounding simulation, not native BF16 execution.
- Differences are promoted to FP64 before subtraction and RMS reduction.

## Scan

The scan uses `h=logspace(-7,-1,25)` for one-sided and centered differences. Values and N/A reasons are in `metrics.csv` and `metrics.json`.

| mode | scheme | finite points | min relative error | max relative error |
|---|---|---:|---:|---:|
| `fp64_reference` | `one_sided` | 25 | 1.8569772835536596e-08 | 0.0033300954175304203 |
| `fp64_reference` | `centered` | 25 | 3.673248356746442e-11 | 2.750526517474188e-05 |
| `input_quantized_bf16` | `one_sided` | 25 | 0.24250553358567792 | 258.1238284694862 |
| `input_quantized_bf16` | `centered` | 25 | 0.13280681691676088 | 128.96187609537 |
| `low_precision_compute_fp32` | `one_sided` | 25 | 0.0003854379581327838 | 12.424059027639254 |
| `low_precision_compute_fp32` | `centered` | 25 | 2.234265866758706e-05 | 6.712664602928491 |

## Interpretation boundary

This calibration measures a known cubic CPU function and is not evidence about Cosmos L/M/eta values. FP32 is not ground truth; the FP64 path is a declared numerical reference with the same FP32 represented W. A zero baseline response does not imply eta=0. One direction does not establish a full Jacobian or low-rank structure. A missing finite-difference window does not prove non-differentiability. The conditional bound in the separate math note does not require a 1/h or U-shaped curve.

See `experiments/umi_precision_calibration_math.md` for the step-by-step conditional error-bound proof and assumptions.
