# Task 7 Task 1 — actual FP32 feedback encoder

## Changes

- Added `experiments/umi_task7_encoder.py` with the documented
  `FeedbackEncoder`, `encode_feedback_frame`, and
  `temporary_encoder_precision` APIs.
- The seam validates floating RGB input, performs only the required
  `[0,1]` → `[1,3,1,H,W]` / `[-1,1]` preprocessing, calls the loaded encoder,
  and returns independent CPU arrays for the source, normalized input, actual
  scaled latent, and public output.
- Temporary FP32 conversion operates on the same loaded parameters and
  buffers, converts wrapper dtype/normalization constants, leaves integer
  buffers untouched, and restores exact tensor slots, constants, dtype, train
  state, backend flags, and cache state in `finally` blocks.
- A request-local `TorchDispatchMode` records operation dtypes and rejects
  hidden BF16 compute on the temporary FP32 path. Inner encoder input/output
  dtypes are captured before any projection to NumPy.
- Added `experiments/test_umi_task7_encoder.py` covering native regression,
  normalization, FP32 conversion/restoration, hidden BF16 rejection, input
  validation, failures, and repeated-call cleanup.

## Commands and results

Baseline (before Task 7 files):

```text
python -m unittest discover -p "test_*.py"
NO TESTS RAN (root discovery does not recurse into experiments)

python -m unittest discover experiments -p "test_*.py"
Ran 441 tests in 196.711s
FAILED (errors=2, skipped=19)
```

The two baseline errors are the pre-existing calibration artifact tests,
which require the unavailable `matplotlib`. The baseline suite also prints
pre-existing CLI error output for `--overwrite` and the Task 4 direction-seed
preflight test; these did not appear as additional failures.

Focused local checks after implementation:

```text
python -m py_compile experiments/umi_task7_encoder.py experiments/test_umi_task7_encoder.py
exit 0

python -m unittest experiments.test_umi_task7_encoder -v
Ran 1 test in 0.000s
OK (skipped=1)
```

Post-change full discovery (including the new test module):

```text
python -m unittest discover experiments -p "test_*.py"
Ran 442 tests in 197.308s
FAILED (errors=2, skipped=20)
```

The failure set is unchanged: both errors are the same missing-`matplotlib`
calibration artifact tests; the additional skip is the new real-Torch fixture.

The bundled Windows Python has NumPy but no PyTorch, so the real Torch CPU
fixtures are skipped locally. Attempts to use the authorized remote aliases
`seetacloud-umi` and `remote3.13` from this worker failed at DNS resolution;
no remote model/GPU/upload action was performed here. A remote Torch CPU run
is required before claiming integration-green status.

## Source and API contract

The implementation follows the existing `ConditionEncoderAdapter` in
`umi_task6_cosmos_loader.py` and the request-local conversion/restore pattern
in `umi_task5_decoder.py`. The reviewed installed source
`/root/autodl-tmp/cosmos-framework-task6-clean/.../wan2pt2_vae_4x16x16.py`
has the critical shape of `encode`: it saves the incoming dtype, casts the
video to the VAE native dtype, calls the inner model with `self.scale`, and
casts the latent back to the incoming dtype. Therefore public float32 output
alone is not accepted as FP32 evidence; the inner boundary and Torch dispatch
operations are observed.

```python
api = FeedbackEncoder(tokenizer_vision_gen, device="cuda")
record = api.encode(frame_rgb_float32, precision="native")
record = api.encode(frame_rgb_float32, precision="temporary_fp32")
```

`record["arrays"]` contains independent NumPy arrays. `record["evidence"]`
contains JSON-safe precision, shape, dtype, operation, cast, identity, and
cleanup evidence. The API does not resize, write PNG/MP4, load weights, or
retain hooks/payloads between calls.

## Caveats

- The remote real-Torch CPU validation remains outstanding because this worker
  cannot resolve the supplied SSH aliases. No higher-precision recovery claim
  is made; FP32 is an exact dtype conversion of the loaded BF16 values.
- The seam intentionally does not implement Stage A orchestration, GPU
  execution, decoder replay, or upload.

## Round 1 review fixes

Implemented in commit `b55c6ce` on top of `35064b3`.

The round-1 review identified four state/evidence issues, all covered by new
focused tests:

- Official-style `clear_decoder_cache()` is called before and after every
  request. Fixed-length cache lists of `None` slots are recognized as clear
  without calling `list.clear()` and destroying their structure.
- Precision restoration now attempts every cleanup component, surfaces
  restoration failures, and verifies parameter/buffer slots, storage/dtype/
  shape plus compact value probes, normalization-constant identities, and
  training metadata after restoration. A silent no-op restore fails closed.
- Encoder invocation binds signatures before execution and never retries a
  method after its body raises `TypeError` or `ValueError`.
- Temporary FP32 calls fail closed if `TorchDispatchMode` is unavailable or
  observes no operations. Evidence now records per-call floating parameter,
  buffer, and normalization-constant dtypes.

Focused remote CPU validation after these fixes:

```text
ssh seetacloud-umi
cd /root/autodl-tmp/task7-cpu-check-20260921-01
CUDA_VISIBLE_DEVICES="" \
  PYTHONPATH=/root/autodl-tmp/task7-cpu-check-20260921-01:/root/autodl-tmp/task6-code-439e845/cosmos_umi_capstone/experiments \
  /root/autodl-tmp/cosmos-framework/.venv/bin/python -m unittest test_umi_task7_encoder -v
Ran 12 tests in 0.037s
OK
```

This run used only CPU fixture doubles; it did not load model weights, invoke
GPU execution, or upload artifacts beyond the two scoped source/test files.
