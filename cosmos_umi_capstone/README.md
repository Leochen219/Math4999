# Cosmos UMI Capstone Experiments

Source-only companion code for Hongyu Chen's MATH 4999 Capstone,
Numerical Stability and Error Propagation in Autoregressive World-Model
Rollouts.

This directory contains the reusable experiment bridge, analysis code, and
unit tests from the post-VAE precision and directional-linearity studies. It
does **not** include the Cosmos framework, model weights, checkpoints,
datasets, generated videos, raw tensors, server configuration, SSH keys, or
experiment archives.

## What is included

- experiments/umi_fd_post_vae_*.py: post-VAE condition-latent bridge,
  scanner, analyser, and tests.
- experiments/umi_precision_*.py: precision-path instrumentation,
  reproducibility checks, evidence capture, and analysis.
- experiments/umi_task5_*.py, run_umi_task5_experiment.py, and
  analyze_umi_task5.py: Task 5's multi-direction finite-amplitude
  linearity, additivity, held-out prediction, and decoder precision replay.
- experiments/test_*.py: CPU-oriented unit and contract tests.
- A small, hash-verified CPU calibration fixture required by the precision
  analyser tests. It contains no model output, checkpoint, scene data, or
  server credential.
- docs/: scope, chronology, reproducibility constraints, and result
  interpretation.

## Important scope

The code wraps an externally installed official Cosmos runtime. Obtain that
framework and its model assets separately under their applicable licence; this
repository intentionally does not redistribute them. The scripts require a
validated model/checkpoint path before full GPU inference can run.

The formal Task 5 result is local to one initial scene, action chunk 0, model
seed 0, five latent directions, and three finite amplitudes. It provides
evidence for local predicted-latent linearity and additivity at that point; it
does not prove global linearity, a global Lipschitz bound, multi-step
stability, cross-scene generalisation, or low rank.

## Quick CPU test

The unit tests are designed to exercise geometry, evidence and analysis logic
without loading model weights. From the experiments directory, run:

    python -m unittest discover -p "test_*.py"

Install requirements-dev.txt before running the complete test suite; the
calibration-artifact tests regenerate PNG and SVG evidence using matplotlib.
Some tests also require NumPy and PyTorch. GPU execution additionally requires
the official Cosmos environment described in docs/reproducibility.md.

## Full-run safety

Before a full run, set an explicit, empty run directory; never point a run at
an existing completed experiment. Preserve status.json, logs, input hashes,
GPU samples, raw float tensors, and a SHA256 manifest. Use generated MP4 only
for visualisation, not as a quantitative input or feedback frame.

## Repository layout

    cosmos_umi_capstone/
    ├── README.md
    ├── docs/
    │   ├── experiment_timeline.md
    │   ├── findings_and_limitations.md
    │   └── reproducibility.md
    └── experiments/
        ├── umi_fd_post_vae_*.py
        ├── umi_precision_*.py
        ├── umi_task5_*.py
        ├── run_umi_*.py / analyze_umi_*.py
        └── test_*.py
