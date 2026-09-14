# Experiment Timeline

## Task 0: reproducible execution boundary

The remote experiment environment was fixed to CUDA 13.0 with Torch
2.10.0+cu130, GPU 0, cached model assets, fixed prompts/actions/seeds, and
hash-based provenance. Full experiment outputs remain archived outside this
repository.

## Task 1: official UMI two-chunk rollout

The official forward-dynamics path was exercised with a (2, 16, 10) action
array and chunk seed vector [0, 1]. Each chunk generated 16 frames. The run
produced per-chunk MP4 files and a concatenated rollout, while recording timing,
peak memory, video metadata, and logs.

## Task 2: zero-perturbation repeat

Two independent two-chunk calls with identical inputs, actions, prompt, model
configuration and chunk seeds produced exactly equal decoded uint8 frames. The
decoded uint8 RMS noise floor was zero. This establishes a repeatability
baseline for that execution path, not a float-latent noise bound.

## Task 3: pixel-space finite-amplitude scan

A 64-direction by four-amplitude pixel-space scan was saved in float form.
No direction met the finite-difference consistency gate. That result was
treated as an interface/precision diagnostic rather than evidence that the
model is globally nonlinear: image preprocessing, VAE encoding, BF16
conversion, clamping, and sampling can attenuate tiny pixel perturbations.

## Task 4: post-VAE interface and precision contrast

The measurement boundary moved to the official VAE-conditioned latent carrier.
The bridge finds condition and prediction positions at runtime, captures
evidence before the network, and pairs prediction-region noise across calls.
The cache-off FP32 condition path was the stable reference. In Task 4 group C,
the positive and negative predicted-latent slopes were 1.0071 and 1.0021,
with R² values 0.999964 and 0.999992. Native RGB did not satisfy the same
linear-response gate.

## Task 5: directional local-linearity study

Task 5 reused Task 4's exact base latent, mask, action and initial noise, then
tested v0, v1, v2, and normalized combinations u01 and u12 at amplitudes
[1e-3, 3e-3, 1e-2].

- 32/32 formal full-generation calls succeeded.
- All 5/5 predicted-latent directions passed the preregistered local-linearity
  gate; positive/negative slopes were approximately 0.997 to 1.005 and all
  R² values were at least 0.999997.
- 6/6 additivity checks passed; the largest relative additivity residual was
  0.00206.
- 24/24 held-out finite-amplitude predictions passed; the largest relative
  error was 0.0908 under the preregistered 0.1 threshold.
- Sixteen decoder-only replays succeeded. Native BF16 matched the original
  RGB output exactly; the FP32 decoder path had a near-linear v0 RGB response,
  while native BF16 did not.

The independent review found no issue that changes these bounded conclusions.
