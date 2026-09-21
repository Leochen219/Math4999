# Task 7 Task 3 — offline saved-evidence analysis

The analyzer in `cosmos_umi_capstone/experiments/analyze_umi_task7.py` is
CPU-only and model-free. It consumes only the released `Task7SampleStore`
records and `validate_task7_sources` source contract from the runner. The
analyzer has its own `analysis_code_sha256`; it is not required to equal the
frozen runner code digest.

## CLI

```text
python cosmos_umi_capstone/experiments/analyze_umi_task7.py \
  --run-dir RUN_DIR --raw-root TASK6_RAW_ROOT --decoder-root TASK6_DECODER_ROOT \
  --stage A|B|C|all [--output-dir NEW_ANALYSIS_DIR]
```

`--stage A` is the usable A-only path and does not require B/C outputs. The
default `all` analyzes A, then B when complete, and emits an explicit C skip
when A/B gates do not permit C. A scientific `FAIL` is a valid measured
result; an engineering failure is separate and never represented as zeros.
Successful analysis destinations are never overwritten; use a new explicit
directory for a rerun.

## Evidence and precision contract

Every stage/sample manifest and every record artifact reference is verified by
the runner's loader before analysis. Full-carrier source arrays are validated
against the authoritative boolean mask and sliced to condition-only
coordinates. Dtype, shape, finite values, and contiguous bytes are checked
before quantitative work. B requires exact condition-chain evidence,
per-step operation counts, paired seed-0/seed-1 prediction-noise identities,
historical step-0 full-latent parity, A temporary-FP32 encoder parity, and
byte-exact baseline-pre/post replay. C additionally binds the analyzer gate to
the current A/B run-status and sample-manifest digests.

Neural G/D/E computation is reported as FP32. Approved P range normalization
may use a float64 intermediate followed by FP32 storage; this is not relabeled
as an all-FP32 scalar path. Tensor differences use explicit FP32 subtraction;
RMS, cosine, regression, and floor reductions use FP64 over the condition
mask. Response floors are computed from saved baseline-post minus
baseline-pre tensors, never caller-provided flags.

## Outputs

The analysis destination contains `analysis_summary.json`,
`a_comparison.csv`, `b_propagation.csv`, `c_prediction.csv` (or an explicit C
skip), small audit tensors under `analysis_tensors/`, `task7_report.md`, and
diagnostic PNG+SVG figures when an existing Matplotlib runtime is available.
`MANIFEST.sha256` excludes itself and `review_bundle.zip`; the light bundle
excludes models, keys, video, and oversized full-chunk arrays.

The report records the fixed three-amplitude A window, actual signed step
lengths, central secant, condition-mask RMS, floor/zero-denominator reasons,
controlled repeated action and seeds `[0,1]`, and the conditional numerical
note `O(h^2)+O(u/h)` from `docs/task7-mathematical-interpretation.md`. It does
not claim accuracy, Jacobian existence, global stability, or rank.

## Verification

Focused numerical regression:

```text
python -m unittest test_analyze_umi_task7
```

The synthetic tests exercise the approved affine/nonlinear numerical API and
explicit reliability failures. Synthetic artifact tests are labelled test
evidence and are not model/GPU results; live source preflight and formal
runner release remain separate gates.
