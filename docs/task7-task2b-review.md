# Main independent Task2b review

Reviewed runtime and tests at frozen 0298bd4, not the earlier mutable upload snapshot. Initial scoped verdict: changes required; no model/GPU release.

Confirmed source architecture: full/deferred 30-step G; dynamic inputs binding; step-index seeds [0,1]; prediction-only noise comparison; one FP32 full-latent decoder; condition-only encoder output dynamically embedded in immutable full carrier; generation cleanup before decoding. Actual production decode uses precision snapshot and dispatch evidence, restores dtype/training/method ownership.

Round 1 findings:
1. Remote real-Torch CPU suite has 22/23 passing. Missing-noise fixture imports installed real Cosmos noise when fixture attribute is removed; NumPy mock dtype string then reaches Torch.to and causes TypeError. Explicitly isolate module resolution in negative test and reject invalid production dtype before official noise invocation.
2. Runtime restore unconditionally assigns inherited _prepared_for_call as an instance shadow. Preserve original attribute ownership on success and failure.
3. actual.prepared_condition is target.copy(), not readback. Capture clean/reference/initial values from actual rebuilt prepared tuple and assert intended condition matches them before recording evidence.
4. Encoder arrays are discarded after encode; retain actual encoder input and preprocessing arrays so later artifacts use observed values rather than reconstruction.

Sent together to original Luna implementer for one scoped fix round. No threshold changes or extra inference.

## Round 1 resolution — a89171e

All four findings addressed. Scoped diff reviewed: missing kwargs fail before RNG; negative import test isolated; inherited prepare method restored with delattr; prepared clean/initial/reference readbacks checked; encoder arrays retained. Main awaited frozen upload, verified exact source SHA256 2b640a2dc9c46c71091b10b76487156e36e138eccf59adb97e0ca6e4721fdfd9 and test SHA256 554214a371c8003ec9b0dadbb2ea41a72a5733fd1445dfd1e917c74e9c3bc4ce locally/remotely, then independently reran 24 tests in 0.559s: all PASS, no skips, existing real Torch CPU.

Spec verdict: approved for Task2b bounded adapter. Quality verdict: approved. Live model integration remains a smoke gate, not inferred from fixtures. Full operational provenance/resources/resume belong to Task2c. No Task7 GPU/model calls made. Minor: constructor seed aliases are redundant because authoritative step_index fixes seeds; runner must use step_index rather than infer arbitrary seed support.

Validation command (existing remote Torch CPU, CUDA_VISIBLE_DEVICES empty): python -m unittest test_umi_task7_runtime test_umi_task7_encoder test_umi_task7_official_deferred -v. Result before fixes: 23 tests, one error in missing-noise fixture; decoder actual-Torch compute/restoration test and all encoder/deferred tests pass. An earlier upload-before-worker-freeze indentation error was a mutable deployment race, not accepted code evidence; final frozen 0298bd4 upload was awaited before this run.
